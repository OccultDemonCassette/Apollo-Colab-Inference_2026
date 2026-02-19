"""
inference.py — Apollo Audio Restoration · Fork 5 (Unified)

Merged features:
  Fork 1 (jarredou)       — core chunked overlap-windowed inference engine
  Fork 2 (ibratabian17)   — configurable SR, output subtype, gain staging,
                            fade_sec, CUDA device selection, model as function arg
  Fork 3 (Losses)         — shlex-safe subprocess calls (design principle applied
                            here as clean argparse usage), multi-format awareness
  Fork 4 (Qupci)          — 32-bit FLOAT default output, base_model.py patch
                            (applied separately), on-demand download pattern

Usage (single file):
  python inference.py --in_wav input.mp3 --out_wav output.wav \\
      --ckpt model/apollo_model_uni.ckpt --config configs/config_apollo_uni.yaml \\
      --chunk_size 19 --overlap 2

Usage (batch directory):
  python inference.py --in_dir /path/to/input/ --out_dir /path/to/output/ \\
      --ckpt model/apollo_model_uni.ckpt --config configs/config_apollo_uni.yaml \\
      --chunk_size 19 --overlap 2 --out_subtype FLOAT
"""

import os
import sys
import glob
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
import yaml
from ml_collections import ConfigDict
from tqdm.auto import tqdm

import look2hear.models

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Supported input formats for batch/directory mode
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".opus", ".ogg", ".aiff", ".aif"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_config(config_path: str) -> ConfigDict:
    """Load a YAML model config into a ConfigDict."""
    with open(config_path, "r") as f:
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(file_path: str, sr: int = 44100) -> tuple[torch.Tensor, int]:
    """
    Load audio from any librosa-supported format.

    Always resamples to `sr`. Returns a (channels, samples) float32 tensor
    and the actual loaded sample rate.  Mono inputs are expanded to shape
    (1, samples) so the rest of the pipeline always sees 2-D tensors.
    """
    audio, samplerate = librosa.load(file_path, mono=False, sr=sr)
    print(f"  INPUT  → shape={audio.shape}  sr={samplerate} Hz  file={os.path.basename(file_path)}")
    tensor = torch.from_numpy(audio)
    # Ensure at least 2-D: (channels, samples)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor, samplerate


def save_audio(file_path: str, audio: np.ndarray, samplerate: int = 44100, subtype: str = "FLOAT") -> None:
    """
    Write audio to a WAV file.

    `audio` is expected to be shape (channels, samples); soundfile wants
    (samples, channels), hence the transpose.

    Supported subtypes: FLOAT (32-bit, default), PCM_16, PCM_24.
    FLOAT is the recommended default — it avoids clipping if the model
    outputs values slightly outside [-1, 1] and preserves full dynamic range.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    sf.write(file_path, audio.T, samplerate, subtype=subtype)


# ---------------------------------------------------------------------------
# dB gain utility
# ---------------------------------------------------------------------------

def db_gain(audio: torch.Tensor | np.ndarray, gain_db: float):
    """Apply a linear gain equivalent to `gain_db` decibels."""
    if gain_db == 0.0:
        return audio
    factor = 10.0 ** (gain_db / 20.0)
    return audio * factor


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _get_windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Build a rectangular window with zeroed leading/trailing edges.

    The fade regions use linspace(1,1,...) and linspace(0,0,...) — producing
    flat constant values — so the 'fade' is really just a hard zero-mask on
    the tail of each chunk.  This blanks out the failure-prone tail of model
    output without introducing a true cross-fade artefact.

    First and last chunks override the window edges to 1 inside the main loop,
    so they are never masked.
    """
    fadein  = torch.linspace(1, 1, fade_size)   # constant 1 — no actual ramp
    fadeout = torch.linspace(0, 0, fade_size)   # constant 0 — tail blanked
    window = torch.ones(window_size)
    window[:fade_size]  *= fadein
    window[-fade_size:] *= fadeout
    return window


# ---------------------------------------------------------------------------
# Single-chunk inference
# ---------------------------------------------------------------------------

def process_chunk(chunk: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Run one audio chunk through the model on GPU, return result on CPU."""
    chunk = chunk.unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(chunk)
    return out.squeeze(0).squeeze(0).cpu()


# ---------------------------------------------------------------------------
# Core inference — one file
# ---------------------------------------------------------------------------

def run_inference(
    input_wav: str,
    output_wav: str,
    model: torch.nn.Module,
    samplerate: int,
    chunk_size: int,
    overlap: int,
    fade_sec: float,
    gain_in_db: float,
    gain_out_db: float,
    out_subtype: str,
) -> None:
    """
    Process a single audio file through the Apollo model.

    Parameters
    ----------
    input_wav    : Path to input audio (any librosa-supported format).
    output_wav   : Path to write the restored WAV.
    model        : Loaded Apollo model (already on CUDA).
    samplerate   : Target sample rate for load & save (Hz).
    chunk_size   : Processing chunk length in seconds.
    overlap      : Overlap divisor — step = chunk_size / overlap.
                   overlap=2 → 50 % overlap, overlap=4 → 75 % overlap.
    fade_sec     : Length of the tail-blanking region in seconds.
    gain_in_db   : dB gain applied to audio *before* inference.
    gain_out_db  : dB gain applied to audio *after* inference.
    out_subtype  : soundfile WAV subtype: 'FLOAT', 'PCM_16', 'PCM_24'.
    """
    test_data, sr = load_audio(input_wav, sr=samplerate)

    # Optional input gain
    if gain_in_db != 0.0:
        test_data = db_gain(test_data, gain_in_db)

    C         = chunk_size * sr          # chunk length in samples
    step      = C // overlap             # hop size
    fade_size = int(fade_sec * sr)       # tail-blank length in samples
    border    = C - step                 # symmetric padding width

    print(f"  CONFIG → chunk={chunk_size}s ({C} smp)  step={step} smp  "
          f"overlap={overlap}  fade={fade_sec}s ({fade_size} smp)")

    # Reflect-pad both ends so the very first and last real samples are
    # centred in a chunk rather than at the boundary.
    if test_data.shape[1] > 2 * border and border > 0:
        test_data = torch.nn.functional.pad(test_data, (border, border), mode="reflect")

    window_template = _get_windowing_array(C, fade_size)

    result  = torch.zeros((1,) + tuple(test_data.shape), dtype=torch.float32)
    counter = torch.zeros((1,) + tuple(test_data.shape), dtype=torch.float32)

    i = 0
    total_samples = test_data.shape[1]
    pbar = tqdm(total=total_samples, desc="  Processing chunks", unit="smp", leave=False)

    while i < total_samples:
        part   = test_data[:, i : i + C]
        length = part.shape[-1]

        # Pad the final (possibly short) chunk
        if length < C:
            if length > C // 2 + 1:
                part = torch.nn.functional.pad(part, (0, C - length), mode="reflect")
            else:
                part = torch.nn.functional.pad(part, (0, C - length), mode="constant", value=0)

        out = process_chunk(part, model)

        # Clone the template so we can modify edge regions without mutating it
        window = window_template.clone()
        if i == 0:
            # First chunk — no tail-blank on the leading edge
            window[:fade_size] = 1.0
        if i + C >= total_samples:
            # Last chunk — no tail-blank on the trailing edge
            window[-fade_size:] = 1.0

        result[...,  i : i + length] += out[..., :length] * window[..., :length]
        counter[..., i : i + length] += window[..., :length]

        i += step
        pbar.update(step)

    pbar.close()

    # Weighted average of overlapping chunks
    final_output = result / counter
    final_output = final_output.squeeze(0).numpy()
    np.nan_to_num(final_output, copy=False, nan=0.0)

    # Remove the reflect padding added earlier
    if test_data.shape[1] > 2 * border and border > 0:
        final_output = final_output[..., border:-border]

    # Optional output gain
    if gain_out_db != 0.0:
        final_output = db_gain(final_output, gain_out_db)

    save_audio(output_wav, final_output, samplerate=sr, subtype=out_subtype)
    print(f"  OUTPUT → {output_wav}  (subtype={out_subtype})")


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, config: ConfigDict, cuda_device: str) -> torch.nn.Module:
    """
    Instantiate and load an Apollo model checkpoint onto GPU.

    The CUDA_VISIBLE_DEVICES env var is set here so callers can select a
    specific GPU without modifying the environment before running the script.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    feature_dim = config["model"]["feature_dim"]
    sr          = config["model"]["sr"]
    win         = config["model"]["win"]
    layer       = config["model"]["layer"]

    print(f"  Loading checkpoint: {ckpt_path}")
    model = look2hear.models.BaseModel.from_pretrain(
        ckpt_path, sr=sr, win=win, feature_dim=feature_dim, layer=layer
    ).cuda()
    model.eval()
    return model


def unload_model(model: torch.nn.Module) -> None:
    """Move model off GPU and free CUDA memory."""
    model.cpu()
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apollo Audio Restoration — Fork 5 (Unified Inference)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ────────────────────────────────────────────────────────────────
    io = p.add_argument_group("I/O (single-file mode)")
    io.add_argument("--in_wav",  type=str, default=None,
                    help="Path to a single input audio file.")
    io.add_argument("--out_wav", type=str, default=None,
                    help="Path for the single output WAV file.")

    batch = p.add_argument_group("I/O (batch/directory mode)")
    batch.add_argument("--in_dir",  type=str, default=None,
                       help="Input directory containing audio files.")
    batch.add_argument("--out_dir", type=str, default=None,
                       help="Output directory for restored WAV files.")

    # ── Model ──────────────────────────────────────────────────────────────
    mdl = p.add_argument_group("Model")
    mdl.add_argument("--ckpt",   type=str, required=True,
                     help="Path to model checkpoint (.ckpt / .bin).")
    mdl.add_argument("--config", type=str, default="configs/apollo.yaml",
                     help="Path to model config YAML.")

    # ── Processing ─────────────────────────────────────────────────────────
    proc = p.add_argument_group("Processing")
    proc.add_argument("--chunk_size", type=int,   default=19,
                      help="Chunk length in seconds. Use 19 for Universal model, 25 for others.")
    proc.add_argument("--overlap",    type=int,   default=2,
                      help="Overlap divisor. step = chunk_size / overlap. Higher = more overlap.")
    proc.add_argument("--fade_sec",   type=float, default=3.0,
                      help="Length of the chunk tail-blanking region in seconds.")
    proc.add_argument("--sr",         type=int,   default=44100,
                      help="Target sample rate for loading and saving audio.")

    # ── Audio quality ──────────────────────────────────────────────────────
    qual = p.add_argument_group("Audio quality")
    qual.add_argument("--out_subtype", type=str, default="FLOAT",
                      choices=["FLOAT", "PCM_16", "PCM_24"],
                      help="Output WAV bit depth. FLOAT (32-bit) is recommended to avoid clipping.")
    qual.add_argument("--gain_in",  type=float, default=0.0,
                      help="Input gain in dB applied before inference.")
    qual.add_argument("--gain_out", type=float, default=0.0,
                      help="Output gain in dB applied after inference.")

    # ── Hardware ───────────────────────────────────────────────────────────
    hw = p.add_argument_group("Hardware")
    hw.add_argument("--cuda", type=str, default="0",
                    help="CUDA device index (sets CUDA_VISIBLE_DEVICES).")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Validate I/O mode ─────────────────────────────────────────────────
    single_mode = args.in_wav  is not None or args.out_wav is not None
    batch_mode  = args.in_dir  is not None or args.out_dir is not None

    if single_mode and batch_mode:
        parser.error("Specify either single-file mode (--in_wav / --out_wav) "
                     "or batch mode (--in_dir / --out_dir), not both.")

    if not single_mode and not batch_mode:
        parser.error("You must specify either --in_wav + --out_wav (single-file) "
                     "or --in_dir + --out_dir (batch).")

    if single_mode:
        if args.in_wav is None or args.out_wav is None:
            parser.error("Single-file mode requires both --in_wav and --out_wav.")

    if batch_mode:
        if args.in_dir is None or args.out_dir is None:
            parser.error("Batch mode requires both --in_dir and --out_dir.")

    # ── Load config & model (once, shared across all files) ───────────────
    config = get_config(args.config)
    print(f"\n{'─'*60}")
    print(f"  Apollo Fork 5 — Unified Inference")
    print(f"{'─'*60}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Config     : {args.config}")
    print(f"  Chunk      : {args.chunk_size}s  Overlap: {args.overlap}  Fade: {args.fade_sec}s")
    print(f"  Sample rate: {args.sr} Hz  Subtype: {args.out_subtype}")
    print(f"  Gain in/out: {args.gain_in:+.1f} dB / {args.gain_out:+.1f} dB")
    print(f"  CUDA device: {args.cuda}")
    print(f"{'─'*60}\n")

    model = load_model(args.ckpt, config, args.cuda)

    # ── Resolve file list ─────────────────────────────────────────────────
    if single_mode:
        jobs = [(args.in_wav, args.out_wav)]
    else:
        in_dir  = Path(args.in_dir)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_files = []
        for ext in SUPPORTED_EXTENSIONS:
            audio_files.extend(in_dir.glob(f"*{ext}"))
            audio_files.extend(in_dir.glob(f"*{ext.upper()}"))
        audio_files = sorted(set(audio_files))  # deduplicate & sort

        if not audio_files:
            print(f"❌  No supported audio files found in '{in_dir}'.")
            print(f"    Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
            sys.exit(1)

        jobs = [
            (str(f), str(out_dir / (f.stem + ".wav")))
            for f in audio_files
        ]
        print(f"✅  Found {len(jobs)} file(s) in '{in_dir}'.\n")

    # ── Process ───────────────────────────────────────────────────────────
    total = len(jobs)
    for idx, (in_path, out_path) in enumerate(jobs, 1):
        print(f"[{idx}/{total}] {os.path.basename(in_path)}")
        try:
            run_inference(
                input_wav    = in_path,
                output_wav   = out_path,
                model        = model,
                samplerate   = args.sr,
                chunk_size   = args.chunk_size,
                overlap      = args.overlap,
                fade_sec     = args.fade_sec,
                gain_in_db   = args.gain_in,
                gain_out_db  = args.gain_out,
                out_subtype  = args.out_subtype,
            )
        except Exception as exc:
            print(f"  ⚠️  ERROR on '{in_path}': {exc}")
            if total == 1:
                raise   # re-raise in single-file mode so the user sees the traceback
            continue    # in batch mode, skip and continue with remaining files
        print()

    # ── Cleanup ───────────────────────────────────────────────────────────
    unload_model(model)

    if total > 1:
        print(f"{'─'*60}")
        print(f"✅  Batch complete — {total} file(s) processed.")
        print(f"    Output directory: {args.out_dir}")
        print(f"{'─'*60}")
    else:
        print(f"✅  Done — output saved to: {jobs[0][1]}")


if __name__ == "__main__":
    main()
