"""
sd-webui-DifferentialDiffusion
===============================
Location: extensions/sd-webui-DifferentialDiffusion/scripts/sd_webui_differential_diffusion.py

Paper: "Differential Diffusion: Giving Each Pixel Its Strength"
       https://github.com/exx8/differential-diffusion
ComfyUI: comfy_extras/nodes_differential_diffusion.py

Hook: model_options["denoise_mask_function"]
      Applied independently from CFG hooks — compatible with TCFG / SkimmedCFG / MaHiRo.

sorting_priority: 18.5
    Runs after the built-in reForge DifferentialDiffusion (18),
    ensuring this extension takes priority when both are present.

New vs reForge built-in:
    strength parameter (ComfyUI PR #5709) — blends continuous mask with binary mask.
    strength=1.0 → original behaviour (full binary mask)
    strength<1.0 → softer transition, useful when effect is too strong
"""

import logging
import os
import sys
import traceback
from functools import partial
from typing import Any

import gradio as gr
from modules import scripts, script_callbacks

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)
# ---------------------------------------------------------------------------

from sd_webui_differential_diffusion import (
    apply_differential_diffusion,
    remove_differential_diffusion,
)

logger = logging.getLogger(__name__)


def _has_forge_backend(p) -> bool:
    return hasattr(p, "sd_model") and hasattr(p.sd_model, "forge_objects")


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

class DifferentialDiffusionScript(scripts.Script):

    sorting_priority = 18.5

    def __init__(self):
        self.enabled  = False
        self.strength = 1.0

    def title(self) -> str:
        return "Differential Diffusion"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML(
                "<p><i>"
                "<b>Denoise Mask Function</b>: Applies a per-pixel denoising schedule "
                "based on mask density. Brighter mask areas denoise more; darker areas denoise less. "
                "Requires Forge backend."
                "</i></p>"
            )
            enabled = gr.Checkbox(label="Enable Differential Diffusion", value=False)
            strength = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.05, value=1.0,
                label="Strength",
                info="1.0 = full binary mask (original). Lower values blend with the continuous mask.",
            )

        # Infotext round-trip (PNG Info -> Send to txt2img / img2img).
        # Enable is bound through a callable matching the existing
        # "differential_diffusion" key (value "enabled"); a missing key resolves
        # to False so an image generated without it forces OFF, as required for
        # faithful same-seed reproduction. The strength uses a plain key. The
        # metadata write lives in process() (see below).
        #
        # NOTE: no enabled.change() listener is registered. The previous version
        # synced an instance flag via change(), but that value was always
        # overwritten by the UI args read during sampling, so it was dead code.
        self.infotext_fields = [
            (enabled,  lambda d: d.get("differential_diffusion", "") == "enabled"),
            (strength, "differential_diffusion_strength"),
        ]

        return [enabled, strength]

    # ------------------------------------------------------------------
    # Effective configuration (UI args + XYZ Grid override)
    # ------------------------------------------------------------------

    def _resolve(self, p, args):
        enabled  = bool(args[0])  if len(args) >= 1 else False
        strength = float(args[1]) if len(args) >= 2 else 1.0

        xyz = getattr(p, "_dd_xyz", {})
        if "enabled"  in xyz: enabled  = (xyz["enabled"] == "True")
        if "strength" in xyz: strength = float(xyz["strength"])

        return enabled, strength

    # ------------------------------------------------------------------
    # Metadata write (runs once before sampling so create_infotext captures it)
    # ------------------------------------------------------------------

    def process(self, p, *args):
        if len(args) < 1:
            return
        enabled, strength = self._resolve(p, args)
        if not enabled:
            return
        p.extra_generation_params.update({
            "differential_diffusion": "enabled",
            "differential_diffusion_strength": strength,
        })

    # ------------------------------------------------------------------
    # Hook application (correct timing for forge_objects.unet)
    # ------------------------------------------------------------------

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) < 1:
            logger.warning("[DifferentialDiffusion] process_before_every_sampling: missing args")
            return

        self.enabled, self.strength = self._resolve(p, args)

        if not self.enabled:
            return

        if not _has_forge_backend(p):
            msg = "[DifferentialDiffusion] Requires Forge backend."
            logger.warning(msg)
            print(msg, file=sys.stderr)
            return

        unet = p.sd_model.forge_objects.unet.clone()
        apply_differential_diffusion(unet, strength=self.strength)
        p.sd_model.forge_objects.unet = unet
        logger.debug("[DifferentialDiffusion] applied (strength=%.2f)", self.strength)


# ---------------------------------------------------------------------------
# XYZ Grid
# ---------------------------------------------------------------------------

def _set_xyz(p, x: Any, xs: Any, *, field: str) -> None:
    if not hasattr(p, "_dd_xyz"):
        p._dd_xyz = {}
    p._dd_xyz[field] = x


def _register_xyz() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return

    new_axes = [
        xyz_grid.AxisOption(
            "(Differential Diffusion) Enabled",
            str,
            partial(_set_xyz, field="enabled"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(Differential Diffusion) Strength",
            float,
            partial(_set_xyz, field="strength"),
        ),
    ]

    if not any(x.label.startswith("(Differential Diffusion)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(new_axes)


def _on_before_ui() -> None:
    try:
        _register_xyz()
    except Exception:
        print(
            f"[sd-webui-DifferentialDiffusion] XYZ Grid error:\n{traceback.format_exc()}"
        )


script_callbacks.on_before_ui(_on_before_ui)
