# 🎵 ACI_2026 — Unified Audio Restoration
### Apollo-Colab-Inference_2026
> A community fork of [jarredou/Apollo-Colab-Inference](https://github.com/jarredou/Apollo-Colab-Inference), which is itself built on [JusperLee/Apollo](https://github.com/JusperLee/Apollo): *Band-sequence Modeling for High-Quality Music Restoration in Compressed Audio.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OccultDemonCassette/Apollo-Colab-Inference_2026/blob/main/Apollo_Colab_Inference_2026.ipynb)

---

## What Is Apollo?

Apollo is a deep learning model trained to restore and enhance lossy compressed audio — primarily MP3 files at or below 128 kbps. It reconstructs high-frequency detail, reduces compression artefacts, and improves perceptual audio quality. Community-trained variants extend this capability to separated vocal stems and mixed music.

---

## About this edition:

Apollo_Colab_Inference_2026 (ACI_2026) is a unified merge of four community forks of the original jarredou Colab notebook. Rather than maintaining four separate notebooks with overlapping and sometimes conflicting improvements, ACI_2026 consolidates every meaningful feature into a single, cleanly coded implementation.

### Features merged from each fork

| Feature | Origin |
|---|---|
| Core chunked overlap-windowed inference engine | Fork 1 — [jarredou](https://github.com/jarredou/Apollo-Colab-Inference) |
| Configurable sample rate (`--sr`) | Fork 2 — ibratabian17 |
| Configurable output bit-depth (`--out_subtype`) | Fork 2 — ibratabian17 |
| Input / output dB gain staging (`--gain_in`, `--gain_out`) | Fork 2 — ibratabian17 |
| Configurable fade/tail-blank size (`--fade_sec`) | Fork 2 — ibratabian17 |
| Selectable CUDA device (`--cuda`) | Fork 2 — ibratabian17 |
| Directory batch mode for processing multiple files | Fork 3 — Losses |
| Multi-format input (WAV, MP3, FLAC, M4A, OPUS, OGG, AIFF) | Fork 3 — Losses |
| `shlex.quote()` path safety on all shell commands | Fork 3 — Losses |
| `weights_only=False` fix for PyTorch ≥ 2.0 | Fork 4 — Qupci |
| Robust `model_name` fallback for Baicai-style checkpoints | Fork 4 — Qupci |
| Baicai1145 Vocal MSST model support | Fork 4 — Qupci |
| On-demand model downloading (skip already-downloaded files) | Fork 4 — Qupci |
| PyTorch 2.5.1 + CUDA 12.4 pin for runtime stability | Fork 4 — Qupci |

### New in ACI_2026 (not present in any prior fork)

- **Model loaded once in batch mode** — prior forks respawned a fresh Python process per file; ACI_2026 loads the model once and reuses it across the entire batch, significantly reducing overhead for large collections.
- **Per-file error handling in batch mode** — a failed file prints a warning and is skipped; the batch does not abort.
- **`model.eval()` called explicitly** — disables dropout and batch normalisation training behaviour during inference (an inference best-practice omitted from all prior forks).
- **Window mutation bug fixed** — Fork 1/3/4 mutated the shared windowing array in-place on the first and last chunks; ACI_2026 clones the template per chunk, preventing a latent accumulation error when processing multiple files in the same process.
- **Output directory auto-created** — in both single-file and batch modes.
- **Structured `argparse` help output** — arguments are grouped by category (`I/O`, `Model`, `Processing`, `Audio quality`, `Hardware`) for readable `--help` output.
- **Mutual-exclusion validation** — single-file mode and batch mode flags cannot be mixed; clear error messages guide the user.

---

## Repository Structure

```
Apollo_Colab_Inference_2026/
│
├── Apollo_Colab_Inference_2026.ipynb   ← Google Colab notebook (start here)
├── inference.py               ← Unified inference script (all fork features merged)
├── base_model.py              ← Patched BaseModel (PyTorch ≥ 2.0 fix + Baicai support)
└── README.md                  ← This file
```

---

## Available Models

| Model | Best suited for | Recommended `chunk_size` |
|---|---|---|
| **MP3 Enhancer** | General lossy audio (music, speech, mixed) | 25 |
| **Lew Vocal Enhancer v1** | Pre-separated vocal stems | 25 |
| **Lew Vocal Enhancer v2 (beta)** | Pre-separated vocal stems (improved quality) | 25 |
| **Lew Universal Lossy Enhancer** | Mixed music and vocals | 19 |
| **Baicai1145 Vocal MSST** | Pre-separated vocal stems (community model) | 10–15 |

All Lew models are by [deton24](https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee).  
The Baicai1145 Vocal MSST model is hosted at [huggingface.co/baicai1145/Apollo-vocal-msst](https://huggingface.co/baicai1145/Apollo-vocal-msst).

---

## How to Use the Notebook in Google Colab

### Option A — Open directly from GitHub (recommended)

1. Click the **Open in Colab** badge at the top of this README, or navigate to:
   ```
   https://colab.research.google.com/github/OccultDemonCassette/Apollo-Colab-Inference_2026/blob/main/Apollo_Colab_Inference_2026.ipynb
   ```
2. Colab will open the notebook directly from your GitHub repo. No download required.

### Option B — Upload the `.ipynb` manually

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **File → Upload notebook**.
3. Select `Apollo_Colab_Inference_2026.ipynb` from your local machine.

### Option C — Open from Google Drive

1. Upload `Apollo_Colab_Inference_2026.ipynb` to your Google Drive.
2. Double-click the file in Drive — it will open automatically in Colab.

---

## Step-by-Step Usage Guide

### Step 1 — Set runtime to GPU

Before running any cells, ensure you have a GPU runtime:

- **Runtime → Change runtime type → Hardware accelerator → GPU → Save**

A T4 GPU (the free Colab tier) is sufficient for all models.

### Step 2 — Run Cell 1: Install

Click the ▶ button on **Cell 1 (⚙️ Install Dependencies & Patch Apollo Repo)**.

This cell will:
- Clone the upstream JusperLee/Apollo repository
- Apply the `base_model.py` patch (PyTorch ≥ 2.0 compatibility + Baicai support)
- Install the ACI_2026 `inference.py`
- Install Python dependencies (`omegaconf`, `ml_collections`)
- Reinstall PyTorch 2.5.1 with CUDA 12.4

> ⏱ **Estimated time: 3–5 minutes.** The PyTorch reinstall is the slow step. You only need to run this once per Colab session.

### Step 3 — (Optional) Run Cell 2: Mount Google Drive

Run **Cell 2** if your audio files are stored in Google Drive, or if you want outputs saved there.

After mounting, your Drive is accessible at `/content/drive/MyDrive/`.

**Skip this cell** if you plan to upload audio directly to the Colab file browser.

### Step 4 — Run Cell 3: Download Models

Run **Cell 3 (📥 Download Models)**.

- Check the boxes next to the models you want to download.
- Unchecked models are skipped entirely — no wasted bandwidth.
- Already-downloaded models are detected and skipped automatically on re-runs.
- **Baicai1145 Vocal MSST** is unchecked by default; enable it only if you need it.

> ⏱ Model file sizes range from ~200 MB to ~600 MB each.

### Step 5 — Process Audio

You have two options depending on whether you are processing one file or many.

---

#### Single File — Cell 4

Run **Cell 4 (🎧 Single File Inference)**.

**Configure the form fields:**

| Field | Description |
|---|---|
| `input_file` | Full path to your input audio file. Upload via the 📁 file browser on the left sidebar, then paste the path here. |
| `output_file` | Full path where the restored WAV will be saved. |
| `model` | Select from the dropdown — see the model table above. |
| `chunk_size` | Processing chunk length in seconds. See recommended values per model above. |
| `overlap` | Overlap divisor. `overlap=2` = 50% overlap (standard). `overlap=4` = 75% overlap (smoother joins, slower). |
| `fade_sec` | Length of the chunk tail-blanking region. Default `3` works well for all models. |
| `sample_rate` | Target sample rate in Hz. Default `44100`. |
| `out_subtype` | Output WAV bit depth. `FLOAT` (32-bit, recommended), `PCM_24`, or `PCM_16`. |
| `gain_in` | Level adjustment in dB applied **before** inference. Use `-3` to `+3` for headroom management. `0` = no change. |
| `gain_out` | Level adjustment in dB applied **after** inference. `0` = no change. |
| `cuda_device` | GPU device index. Leave as `0` for standard Colab sessions. |

---

#### Batch Directory — Cell 5

Run **Cell 5 (📁 Batch Directory Inference)**.

- Set `input_directory` to a folder containing your audio files.
- Set `output_directory` to a folder where restored WAVs will be saved.
- All other settings are identical to Cell 4.
- The model is loaded **once** and shared across all files — efficient for large batches.
- Failed files are skipped with a warning; the batch continues.
- All outputs are saved as `.wav` regardless of the input format.

**Supported input formats:** WAV, MP3, FLAC, M4A, OPUS, OGG, AIFF/AIF

---

## Uploading Audio Files to Colab

If you are not using Google Drive, you can upload audio directly to the Colab runtime:

1. Click the **📁 folder icon** in the left sidebar to open the file browser.
2. Click the **⬆ upload button** and select your files.
3. Files are uploaded to `/content/` by default — you can drag them into `/content/input_audio/` in the file browser, or set the full path in the notebook form.

> ⚠️ Files uploaded directly to Colab are lost when the runtime disconnects. Use Google Drive if you need persistent storage.

---

## Using Local CLI (Without Colab)

`inference.py` is a fully standalone command-line script. If you have a local GPU setup with the Apollo repo already cloned and `base_model.py` patched, you can run it directly:

**Single file:**
```bash
python inference.py \
    --in_wav  "path/to/input.mp3" \
    --out_wav "path/to/output.wav" \
    --ckpt    "model/apollo_model_uni.ckpt" \
    --config  "configs/config_apollo_uni.yaml" \
    --chunk_size 19 \
    --overlap 2 \
    --out_subtype FLOAT
```

**Batch directory:**
```bash
python inference.py \
    --in_dir  "path/to/input_folder/" \
    --out_dir "path/to/output_folder/" \
    --ckpt    "model/apollo_model_uni.ckpt" \
    --config  "configs/config_apollo_uni.yaml" \
    --chunk_size 19 \
    --overlap 2
```

**Full argument reference:**
```
I/O (single-file mode):
  --in_wav        Path to a single input audio file
  --out_wav       Path for the single output WAV file

I/O (batch/directory mode):
  --in_dir        Input directory containing audio files
  --out_dir       Output directory for restored WAV files

Model:
  --ckpt          Path to model checkpoint (.ckpt / .bin)  [required]
  --config        Path to model config YAML  [default: configs/apollo.yaml]

Processing:
  --chunk_size    Chunk length in seconds  [default: 19]
  --overlap       Overlap divisor (step = chunk_size / overlap)  [default: 2]
  --fade_sec      Chunk tail-blanking length in seconds  [default: 3.0]
  --sr            Target sample rate in Hz  [default: 44100]

Audio quality:
  --out_subtype   WAV bit depth: FLOAT, PCM_24, PCM_16  [default: FLOAT]
  --gain_in       Input gain in dB before inference  [default: 0.0]
  --gain_out      Output gain in dB after inference  [default: 0.0]

Hardware:
  --cuda          CUDA device index  [default: 0]
```

---

## Technical Notes

### On `chunk_size` and `overlap`

Apollo processes audio in fixed-length chunks that are stepped through the file with overlap. The step size is `chunk_size / overlap` seconds.

- **Higher `overlap`** (e.g. `4` or `8`) produces smoother transitions between chunks at the cost of longer processing time. It is rarely necessary to go above `4`.
- **Lower `chunk_size`** uses less VRAM per forward pass. If you encounter CUDA out-of-memory errors, reduce `chunk_size` first.
- The Lew Universal model was tuned at `chunk_size=19`. Using `25` with it will typically still work but may produce slightly different results.

### On `out_subtype`

- **`FLOAT`** (32-bit IEEE float) is the default and recommended format. It avoids integer clipping if the model output exceeds ±1.0, and preserves the full dynamic range for downstream processing.
- **`PCM_24`** is appropriate for final delivery files where 24-bit integer PCM is required.
- **`PCM_16`** is CD-quality integer PCM — usable, but any model output above ±1.0 will clip.

### On the `base_model.py` patch

The upstream Apollo `base_model.py` calls `torch.load()` without `weights_only=False`. From PyTorch 2.0 onward, `torch.load()` defaults to `weights_only=True`, which throws a `WeightsOnlyException` on checkpoint files that contain non-tensor Python objects (such as the `model_name` string or `infos` dictionary). The patched file adds `weights_only=False` to restore the pre-2.0 behaviour.

The patch also adds a three-tier `model_name` lookup to support Baicai1145-style checkpoints, which store the model name under `hyper_parameters` rather than at the top level of the checkpoint dict.

---

## Requirements

For local use, the following must be installed in your Python environment:

```
torch>=2.0.0
librosa
soundfile
tqdm
pyyaml
ml_collections
omegaconf
huggingface_hub
```

The Apollo repo itself must also be cloned and on the Python path:
```bash
git clone https://github.com/JusperLee/Apollo.git
cd Apollo
pip install omegaconf ml_collections
```

Then copy `base_model.py` from this repo into `look2hear/models/base_model.py` within the cloned Apollo directory.

---

## Credits & Acknowledgements

| Contributor | Contribution |
|---|---|
| [JusperLee](https://github.com/JusperLee) | Original Apollo model and codebase |
| [jarredou](https://github.com/jarredou/Apollo-Colab-Inference) | Fork 1 — Colab notebook, chunked inference engine, community model integration |
| [deton24](https://github.com/deton24) | Lew Vocal Enhancer and Universal models |
| [ibratabian17](https://github.com/ibratabian17) | Fork 2 — Advanced config options (SR, bit-depth, gain, fade, CUDA) |
| Losses | Fork 3 — Batch directory processing, multi-format input |
| [Qupci](https://github.com/Qupci) | Fork 4 — PyTorch ≥ 2.0 fix, Baicai checkpoint support, on-demand downloads |
| [baicai1145](https://huggingface.co/baicai1145) | Baicai1145 Vocal MSST community model |

---

## License

This repository contains notebook code and inference utilities only. The underlying Apollo model architecture and weights are subject to the licenses of their respective authors. Please refer to the [JusperLee/Apollo](https://github.com/JusperLee/Apollo) repository for the original license terms.
