"""
sd_webui_differential_diffusion/core.py
========================================
Differential Diffusion core algorithm.

Original paper: "Differential Diffusion: Giving Each Pixel Its Strength"
ComfyUI built-in: comfy_extras/nodes_differential_diffusion.py (PR #2876)

Changes from original reForge implementation:
  - Added: strength parameter (0.0~1.0) from ComfyUI PR #5709
  - Added: set_model_denoise_mask_function() API support (forward compatibility)
  - Added: extra_options in forward() signature (forward compatibility)
  - Removed: singleton / INIT class variable (not needed in WebUI)
"""

import inspect
import torch

# ---------------------------------------------------------------------------
# Frame inspection helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _get_sigmas_and_sampler(frame, target):
    found = frame.f_locals[target]
    if isinstance(found, torch.Tensor) and found[-1] < 0.1:
        return found, frame.f_code.co_name
    return False


def _find_outer_instance(target: str, target_type=None, callback=None):
    frame = inspect.currentframe()
    i = 0
    while frame and i < 100:
        if target in frame.f_locals:
            if callback is not None:
                res = callback(frame, target)
                if res:
                    return res
            else:
                found = frame.f_locals[target]
                if isinstance(found, target_type):
                    return found
        frame = frame.f_back
        i += 1
    return None


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class DifferentialDiffusionCore:
    """
    Per-sampling-run instance of Differential Diffusion.

    A fresh instance is created for each sampling run (txt2img / hires.fix),
    so sigma state is always reset correctly.

    strength parameter (ComfyUI PR #5709):
        1.0 → full differential diffusion (binary mask, original behaviour)
        0.0 → original continuous denoise_mask (no differential diffusion)
        0.5 → blend 50/50
    """

    _VARYING_SIGMAS_SAMPLERS = ["dpmpp_2s", "dpmpp_sde", "dpm_2", "heun", "restart"]

    def __init__(self, strength: float = 1.0) -> None:
        self.strength = float(max(0.0, min(1.0, strength)))
        self.sigmas: torch.Tensor | None = None
        self.thresholds: torch.Tensor | None = None
        self.mask_i = None
        self.valid_sigmas: bool = False
        self.sigmas_min: float = 0.0
        self.sigmas_max: float = 1.0
        self.thresholds_min_len: int = 0

    def _init_sigmas(self, sigma: torch.Tensor, denoise_mask: torch.Tensor) -> None:
        self.sigmas, sampler = (
            _find_outer_instance("sigmas", callback=_get_sigmas_and_sampler) or (None, "")
        )
        self.valid_sigmas = (
            not (
                "sample_" not in sampler
                or any(s in sampler for s in self._VARYING_SIGMAS_SAMPLERS)
            )
            or "generic" in sampler
        )
        if self.sigmas is None:
            self.sigmas = sigma[:1].repeat(2)
            self.sigmas[-1].zero_()

        self.sigmas_min = self.sigmas.min()
        self.sigmas_max = self.sigmas.max()
        self.thresholds = torch.linspace(
            1, 0, self.sigmas.shape[0],
            dtype=sigma.dtype, device=sigma.device,
        )
        self.thresholds_min_len = self.thresholds.shape[0] - 1

        if self.valid_sigmas:
            thresholds  = self.thresholds[:-1].reshape(-1, 1, 1, 1, 1)
            mask        = denoise_mask.unsqueeze(0)               # (1, B, C, H, W)
            binary_masks = (mask >= thresholds).to(denoise_mask.dtype)  # (N, B, C, H, W)

            if self.strength < 1.0:
                # Blend continuous mask with binary mask
                blended = (
                    denoise_mask.unsqueeze(0) * (1.0 - self.strength)
                    + binary_masks * self.strength
                )
                self.mask_i = iter(blended)
            else:
                self.mask_i = iter(binary_masks)

    def forward(
        self,
        sigma: torch.Tensor,
        denoise_mask: torch.Tensor,
        extra_options: dict | None = None,  # forward-compat with latest ComfyUI
        **kwargs,
    ) -> torch.Tensor:
        """Denoise mask function called by the sampler at each step."""

        if self.sigmas is None:
            self._init_sigmas(sigma, denoise_mask)

        # Fast path: pre-computed masks for fixed-step schedules
        if self.valid_sigmas:
            try:
                return next(self.mask_i)
            except StopIteration:
                self.valid_sigmas = False

        # Fallback: nearest-sigma threshold lookup
        if self.thresholds_min_len > 1:
            nearest_idx = (self.sigmas - sigma[0]).abs().argmin()
            if not self.thresholds_min_len > nearest_idx:
                nearest_idx = -2
            threshold = self.thresholds[nearest_idx]
        else:
            threshold = (sigma[0] - self.sigmas_min) / (self.sigmas_max - self.sigmas_min)

        binary_mask = (denoise_mask >= threshold).to(denoise_mask.dtype)

        if self.strength < 1.0:
            return denoise_mask * (1.0 - self.strength) + binary_mask * self.strength
        return binary_mask


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def apply_differential_diffusion(unet, strength: float = 1.0) -> None:
    """
    Apply DifferentialDiffusion denoise_mask_function to unet.

    Supports both:
      - set_model_denoise_mask_function() API (latest ComfyUI / future Forge)
      - Direct model_options assignment (current reForge)
    """
    dd = DifferentialDiffusionCore(strength=strength)
    if hasattr(unet, "set_model_denoise_mask_function"):
        unet.set_model_denoise_mask_function(dd.forward)
    else:
        unet.model_options["denoise_mask_function"] = dd.forward


def remove_differential_diffusion(unet) -> None:
    """Remove DifferentialDiffusion denoise_mask_function from unet."""
    unet.model_options.pop("denoise_mask_function", None)
