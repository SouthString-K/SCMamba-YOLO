from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

ENV_USE_ENHANCE = "SCMAMBA_USE_ENHANCE"
ENV_ENHANCE_WEIGHTS = "SCMAMBA_ENHANCE_WEIGHTS"
ENV_ENHANCE_FREEZE = "SCMAMBA_ENHANCE_FREEZE"
LOGGER = logging.getLogger("SCMamba-YOLO")
DEFAULT_ENHANCE_HINT = (
    "Enhancement weights are optional but recommended. Please download the pretrained checkpoint "
    "for InteractNet from the enhancement release referenced in Enhancement-main/README.md."
)

_INTERACT_NET_CLS = None


def _truthy(value: str | None) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def configure_enhance_runtime(enable: bool, weights: str | None = None, freeze: bool = False):
    """Configure runtime flags through environment variables so train/val can share one implementation."""
    if enable:
        os.environ[ENV_USE_ENHANCE] = "1"
        if weights:
            os.environ[ENV_ENHANCE_WEIGHTS] = str(Path(weights).expanduser().resolve())
        else:
            os.environ.pop(ENV_ENHANCE_WEIGHTS, None)
        os.environ[ENV_ENHANCE_FREEZE] = "1" if freeze else "0"
    else:
        os.environ.pop(ENV_USE_ENHANCE, None)
        os.environ.pop(ENV_ENHANCE_WEIGHTS, None)
        os.environ.pop(ENV_ENHANCE_FREEZE, None)


def enhance_runtime_enabled() -> bool:
    return _truthy(os.getenv(ENV_USE_ENHANCE))


def enhance_runtime_frozen() -> bool:
    return _truthy(os.getenv(ENV_ENHANCE_FREEZE))


def _load_interactnet_class():
    global _INTERACT_NET_CLS
    if _INTERACT_NET_CLS is not None:
        return _INTERACT_NET_CLS

    repo_root = Path(__file__).resolve().parents[3]
    source_path = repo_root / "Enhancement-main" / "src" / "CDF_UIE_arch.py"
    if not source_path.exists():
        raise FileNotFoundError(f"Enhancement source file not found: {source_path}")

    spec = importlib.util.spec_from_file_location("scmamba_enhance_arch", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load enhancement module spec from: {source_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _INTERACT_NET_CLS = module.InteractNet
    return _INTERACT_NET_CLS


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        if keys and all(k.startswith("module.") for k in keys):
            checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}
    return checkpoint


class EnhanceFront(nn.Module):
    """
    A lightweight wrapper that inserts the InteractNet enhancement model before the detector backbone.

    YAML usage example:
        - [-1, 1, EnhanceFront, [3, 16]]

    Notes:
    - The first argument is kept as the output channel count for Ultralytics YAML compatibility.
    - The wrapped InteractNet returns `(enhanced, low_res_aux)`, and only the enhanced image is forwarded.
    """

    def __init__(self, c1: int, c2: int = 3, nc: int = 16, clamp_output: bool = True):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"EnhanceFront expects matching input/output channels, got c1={c1}, c2={c2}.")

        interact_net_cls = _load_interactnet_class()
        self.enhancer = interact_net_cls(nc=nc)
        self.clamp_output = clamp_output
        self.output_channels = c2
        self._maybe_load_runtime_weights()

    def _maybe_load_runtime_weights(self):
        if not enhance_runtime_enabled():
            return

        weights_path = os.getenv(ENV_ENHANCE_WEIGHTS)
        if not weights_path:
            LOGGER.warning(
                "EnhanceFront is enabled without pretrained enhancement weights. "
                "The model will run with random enhancement initialization. "
                f"{DEFAULT_ENHANCE_HINT}"
            )
            return

        resolved = Path(weights_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Enhancement weights not found: {resolved}\n{DEFAULT_ENHANCE_HINT}"
            )

        checkpoint = torch.load(resolved, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        missing, unexpected = self.enhancer.load_state_dict(state_dict, strict=False)
        LOGGER.info(
            f"Loaded enhancement weights from {resolved} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced, _ = self.enhancer(x)
        if self.clamp_output:
            enhanced = torch.clamp(enhanced, 0.0, 1.0)
        return enhanced
