# sd-webui-DifferentialDiffusion

**EN** | [日本語](#日本語)

Differential Diffusion for Stable Diffusion WebUI (Forge-based).  
Applies a per-pixel denoising strength taken from a grayscale mask — brighter areas change more, darker areas change less — enabling smooth gradient inpainting and partial edits.

Original paper **"Differential Diffusion: Giving Each Pixel Its Strength"** by **exx8** — [GitHub](https://github.com/exx8/differential-diffusion) / ComfyUI built-in [PR #2876](https://github.com/comfyanonymous/ComfyUI/pull/2876)
> reForge ships a built-in Differential Diffusion. This extension runs at a higher priority (`sorting_priority` 18.5) and takes over when enabled, adding the `strength` parameter ([ComfyUI PR #5709](https://github.com/comfyanonymous/ComfyUI/pull/5709)).

Requires the **Forge backend** (`forge_objects`).

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-DifferentialDiffusion
```

---

## Strength

Differential Diffusion converts the continuous (grayscale) denoise mask into a per-step binary schedule. The `strength` slider blends that binary schedule back toward the raw continuous mask.

| strength | Behaviour |
| -------- | --------- |
| 1.0      | Full Differential Diffusion (binary mask) — original behaviour |
| 0.5      | 50 / 50 blend |
| 0.0      | Raw continuous mask — Differential Diffusion disabled |

Use it in img2img / inpaint with a grayscale mask. Lower `strength` when the effect is too strong and you want a softer transition.

---

## Algorithm

```
thresholds = linspace(1, 0, steps)          # one threshold per sampling step

at step i:
    binary = (denoise_mask >= thresholds[i])
    if strength < 1.0:
        mask = denoise_mask × (1 − strength) + binary × strength
    else:
        mask = binary
    → apply mask as the step's denoise_mask
```

A pixel starts denoising once the step threshold falls below its mask value.  
Bright pixels change across many steps (strong edit); dark pixels only near the end (weak edit) → a smooth gradient.

Hooked via `denoise_mask_function`, independent of the CFG pipeline — stacks freely with TCFG / SkimmedCFG / MaHiRo / FreSca.

---
---

# 日本語

**[English](#sd-webui-differentialdiffusion)** | 日本語

Forge 系 WebUI 向けの Differential Diffusion 実装。  
グレースケールのマスクからピクセル単位のノイズ除去強度を与え、明るい部分ほど大きく、暗い部分ほど小さく変化させることで、なめらかなグラデーション・インペイントや部分編集を可能にします。

原論文 **「Differential Diffusion: Giving Each Pixel Its Strength」** by **exx8** — [GitHub](https://github.com/exx8/differential-diffusion) / ComfyUI ビルトイン [PR #2876](https://github.com/comfyanonymous/ComfyUI/pull/2876)
> reForge には Differential Diffusion が組み込まれています。本拡張機能はより高い優先度（`sorting_priority` 18.5）で動作し、有効時に組み込みを上書きして `strength` パラメータ（[ComfyUI PR #5709](https://github.com/comfyanonymous/ComfyUI/pull/5709)）を追加します。

**Forge バックエンド**（`forge_objects`）が必要です。

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-DifferentialDiffusion
```

---

## Strength（強度）

Differential Diffusion は連続的な（グレースケールの）ノイズ除去マスクを、ステップごとのバイナリスケジュールへ変換します。`strength` スライダーは、そのバイナリスケジュールを元の連続マスク側へどれだけ戻すかを制御します。

| strength | 挙動 |
| -------- | --- |
| 1.0      | フル Differential Diffusion（バイナリマスク）＝オリジナル動作 |
| 0.5      | 50 / 50 ブレンド |
| 0.0      | 連続マスクそのまま＝Differential Diffusion 無効 |

img2img / インペイントでグレースケールのマスクと併用します。効果が強すぎる場合は `strength` を下げると、より柔らかい遷移になります。

---

## アルゴリズム

```
thresholds = linspace(1, 0, steps)          # サンプリング各ステップに1つ

ステップ i:
    binary = (denoise_mask >= thresholds[i])
    if strength < 1.0:
        mask = denoise_mask × (1 − strength) + binary × strength
    else:
        mask = binary
    → そのステップの denoise_mask として適用
```

各ピクセルは、ステップのしきい値が自身のマスク値を下回った時点からノイズ除去を開始します。  
明るいピクセルは多くのステップで変化し（強い編集）、暗いピクセルは終盤だけ変化する（弱い編集）ため、なめらかなグラデーションになります。

`denoise_mask_function` 経由でフックし、CFG パイプラインからは独立しているため、TCFG / SkimmedCFG / MaHiRo / FreSca と自由に併用できます。

---

## ライセンス

MIT License — Original work © exx8 (Differential Diffusion)
