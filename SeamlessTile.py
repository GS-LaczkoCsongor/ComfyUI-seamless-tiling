"""
ComfyUI Seamless Tiling Nodes

Custom nodes for generating seamless/tileable textures in ComfyUI.
Compatible with ComfyUI 0.10.0+ — uses model.clone() and wrapper patterns
instead of in-place model mutation.

Approach:
  - MODEL nodes: model.clone() + set_model_unet_function_wrapper() to
    temporarily apply circular padding around each UNet forward pass.
    The shared/cached model is never mutated.
  - VAE nodes: copy.deepcopy (VAE objects still support it) or temporary
    patching during a single decode call.
  - All Conv2d patching overrides `.forward` (public API) rather than
    `._conv_forward` (internal PyTorch method) for better compatibility.

Original concept: https://github.com/spinagon/ComfyUI-seamless-tiling
Asymmetric tiling: https://github.com/tjm35/asymmetric-tiling-sd-webui
"""

import copy
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _parse_tiling(tiling: str) -> tuple[bool, bool]:
    """Convert a tiling mode string to (tileX, tileY) booleans."""
    if tiling == "enable":
        return True, True
    elif tiling == "x_only":
        return True, False
    elif tiling == "y_only":
        return False, True
    return False, False


def _get_conv2d_padding(layer: Conv2d) -> tuple[int, int]:
    """Return (pad_h, pad_w) from a Conv2d layer."""
    pad = layer.padding
    if isinstance(pad, (tuple, list)):
        return int(pad[0]), int(pad[1])
    return int(pad), int(pad)


def _make_circular_forward(layer: Conv2d, tileX: bool, tileY: bool):
    """Build a replacement ``forward`` that uses circular/constant padding.

    Instead of relying on Conv2d's built-in padding (which only supports
    symmetric modes), we manually F.pad the input with per-axis modes and
    run the convolution with padding=0.
    """
    pad_h, pad_w = _get_conv2d_padding(layer)
    modeX = "circular" if tileX else "constant"
    modeY = "circular" if tileY else "constant"
    # F.pad order: (left, right, top, bottom)
    padX = (pad_w, pad_w, 0, 0)
    padY = (0, 0, pad_h, pad_h)
    stride = layer.stride
    dilation = layer.dilation
    groups = layer.groups

    def forward(self, input: Tensor) -> Tensor:
        x = F.pad(input, padX, mode=modeX)
        x = F.pad(x, padY, mode=modeY)
        return F.conv2d(x, self.weight, self.bias, stride, (0, 0), dilation, groups)

    return forward.__get__(layer, type(layer))


def _apply_circular_padding(module: torch.nn.Module, tileX: bool, tileY: bool):
    """Permanently replace .forward on every Conv2d in *module*."""
    for layer in module.modules():
        if isinstance(layer, Conv2d):
            pad_h, pad_w = _get_conv2d_padding(layer)
            if pad_h > 0 or pad_w > 0:
                layer.forward = _make_circular_forward(layer, tileX, tileY)


# ---------------------------------------------------------------------------
#  Nodes
# ---------------------------------------------------------------------------

class SeamlessTile:
    """Apply seamless tiling to a MODEL's UNet via circular padding.

    Uses ``model.clone()`` + ``set_model_unet_function_wrapper`` so the
    original cached model is never mutated.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tiling": (["enable", "x_only", "y_only", "disable"],),
                "copy_model": (["Make a copy", "Modify in place"],),
            },
        }

    CATEGORY = "conditioning"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"

    def run(self, model, copy_model, tiling):
        if copy_model == "Modify in place":
            model_copy = model
        else:
            model_copy = model.clone()

        tileX, tileY = _parse_tiling(tiling)

        # Nothing to do when tiling is disabled
        if not tileX and not tileY:
            return (model_copy,)

        def seamless_wrapper(apply_model_fn, args):
            """Temporarily patch every Conv2d in the diffusion model,
            run the original forward, then restore originals."""
            diffusion_model = model_copy.model.diffusion_model
            originals = {}
            try:
                for name, layer in diffusion_model.named_modules():
                    if isinstance(layer, Conv2d):
                        pad_h, pad_w = _get_conv2d_padding(layer)
                        if pad_h > 0 or pad_w > 0:
                            originals[name] = layer.forward
                            layer.forward = _make_circular_forward(layer, tileX, tileY)
                return apply_model_fn(args["input"], args["timestep"], **args["c"])
            finally:
                for name, layer in diffusion_model.named_modules():
                    if name in originals:
                        layer.forward = originals[name]

        model_copy.set_model_unet_function_wrapper(seamless_wrapper)
        return (model_copy,)


class CircularVAEDecode:
    """Decode latents with circular padding for seamless textures.

    Temporarily patches the VAE's Conv2d layers during the decode call
    and restores them immediately afterwards — no persistent mutation.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tiling": (["enable", "x_only", "y_only", "disable"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def decode(self, samples, vae, tiling):
        tileX, tileY = _parse_tiling(tiling)

        if not tileX and not tileY:
            return (vae.decode(samples["samples"]),)

        # Temporarily patch Conv2d layers for the duration of decode
        fsm = vae.first_stage_model
        originals = {}
        try:
            for name, layer in fsm.named_modules():
                if isinstance(layer, Conv2d):
                    pad_h, pad_w = _get_conv2d_padding(layer)
                    if pad_h > 0 or pad_w > 0:
                        originals[name] = layer.forward
                        layer.forward = _make_circular_forward(layer, tileX, tileY)
            result = vae.decode(samples["samples"])
        finally:
            for name, layer in fsm.named_modules():
                if name in originals:
                    layer.forward = originals[name]

        return (result,)


class MakeCircularVAE:
    """Return a VAE whose Conv2d layers permanently use circular padding.

    Uses ``copy.deepcopy`` (VAE objects support it) so the original
    shared VAE is not affected.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "tiling": (["enable", "x_only", "y_only", "disable"],),
                "copy_vae": (["Make a copy", "Modify in place"],),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "run"
    CATEGORY = "latent"

    def run(self, vae, tiling, copy_vae):
        if copy_vae == "Modify in place":
            vae_copy = vae
        else:
            vae_copy = copy.deepcopy(vae)

        tileX, tileY = _parse_tiling(tiling)

        if tileX or tileY:
            _apply_circular_padding(vae_copy.first_stage_model, tileX, tileY)

        return (vae_copy,)


class OffsetImage:
    """Roll/offset an image by a percentage — useful for checking seams."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "x_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1},
                ),
                "y_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image"

    def run(self, pixels, x_percent, y_percent):
        n, y, x, c = pixels.size()
        y = round(y * y_percent / 100)
        x = round(x * x_percent / 100)
        return (pixels.roll((y, x), (1, 2)),)
