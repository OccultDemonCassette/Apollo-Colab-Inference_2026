###
# base_model.py — Apollo BaseModel
#
# Original authors: Kai Li et al. (JusperLee/Apollo)
# Fork 5 patch: weights_only=False fix + robust model_name fallback
#   (enables PyTorch >= 2.0 compatibility and Baicai1145 Vocal MSST checkpoints)
#
# Changes from upstream:
#   - torch.load(..., weights_only=False) to avoid WeightsOnlyException on
#     checkpoint dicts that contain non-tensor Python objects.
#   - Three-tier model_name lookup:
#       1. top-level conf["model_name"]  (standard Apollo checkpoints)
#       2. conf["hyper_parameters"]["model_name"]  (Baicai1145-style checkpoints)
#       3. fallback to "Apollo"  (backward-compatibility catch-all)
###

import torch
import torch.nn as nn

from huggingface_hub import PyTorchModelHubMixin


def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


def pad_to_appropriate_length(x, lcm):
    values_to_pad = int(x.shape[-1]) % lcm
    if values_to_pad:
        appropriate_shape = x.shape
        padded_x = torch.zeros(
            list(appropriate_shape[:-1])
            + [appropriate_shape[-1] + lcm - values_to_pad],
            dtype=torch.float32,
        ).to(x.device)
        padded_x[..., : x.shape[-1]] = x
        return padded_x
    return x


class BaseModel(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/JusperLee/Apollo",
    pipeline_tag="audio-to-audio",
):
    def __init__(self, sample_rate, in_chan=1):
        super().__init__()
        self._sample_rate = sample_rate
        self._in_chan = in_chan

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sample_rate(self):
        return self._sample_rate

    @staticmethod
    def load_state_dict_in_audio(model, pretrained_dict):
        model_dict = model.state_dict()
        update_dict = {}
        for k, v in pretrained_dict.items():
            if "audio_model" in k:
                update_dict[k[12:]] = v
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        return model

    @staticmethod
    def from_pretrain(pretrained_model_conf_or_path, *args, **kwargs):
        from . import get

        # weights_only=False is required for checkpoint dicts that contain
        # non-tensor Python objects (model_name strings, infos dicts, etc.).
        # This became necessary with PyTorch >= 2.0 which changed the default
        # to weights_only=True.
        conf = torch.load(
            pretrained_model_conf_or_path,
            map_location="cpu",
            weights_only=False,
        )

        # Robust model_name resolution — handles three known checkpoint layouts:
        #   1. Standard Apollo:      conf["model_name"]
        #   2. Baicai1145 MSST:      conf["hyper_parameters"]["model_name"]
        #   3. Unknown / legacy:     default to "Apollo"
        model_name = conf.get("model_name")
        if model_name is None:
            hyper_params = conf.get("hyper_parameters") or {}
            model_name = hyper_params.get("model_name")
        if model_name is None:
            model_name = "Apollo"

        model_class = get(model_name)
        model = model_class(*args, **kwargs)

        # Resolve the state dict — handle two known checkpoint layouts:
        #
        #   A) Standard Apollo / jarredou checkpoints:
        #      conf["state_dict"] keys are bare:  "band_sequences.0.weight"
        #
        #   B) PyTorch Lightning checkpoints (e.g. Baicai1145 Vocal MSST):
        #      conf["state_dict"] keys are prefixed: "model.band_sequences.0.weight"
        #      PL wraps the model inside a LightningModule and saves weights with
        #      a "model." prefix. We strip it so load_state_dict() finds a match.
        raw_state_dict = conf["state_dict"]
        if all(k.startswith("model.") for k in raw_state_dict.keys()):
            raw_state_dict = {k[len("model."):]: v for k, v in raw_state_dict.items()}

        model.load_state_dict(raw_state_dict)
        return model

    def serialize(self):
        import pytorch_lightning as pl

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError
