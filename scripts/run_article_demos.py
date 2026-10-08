#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
run_article_demos.py — Demonstrations from arXiv:2307.02299:

Demo 1 (Figure 1): /ibia/ VCV synthesis — 4-step pipeline
  (a) Planning trajectories zv(t) and zc(t)
  (b) Flow of articulatory parameters (Body excursion toward /u/)
  (c) Formant trajectory F1-F2-F3
  (d) Output spectrogram and wave plot

Demo 5 (§3): Six consonant cluster pairs /bd/, /bg/, /db/, /dg/, /gb/, /gd/
  - Synthesize CVCCV with each cluster
  - Show selection vectors and articulatory parameters
  - Polar trajectory for each cluster

(The former Demo 3 "S-shaped F2 trajectories" was removed on
2026-10-08: the F2 pattern through the consonant is a V-shaped dip
(measured: F2 2276 -> 1630 -> 2276 Hz in /ibi/), not an S-shape —
see docs/manual.tex, Demo 1 section.)

Article parameters: T=16, K=10, Kvoy=30, Pexp=1, nu=-1
All output in US English.

Output: <repo>/output/article_demos/
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # noqa: E402

try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import polar_sync  # noqa: E402
from synthSYL import panphon_pipeline  # noqa: E402
from synthSYL.phonology import VOWELS_SYNTSYL, CONSONANTS_SYNTSYL  # noqa: E402
import vlam  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Config (article parameters)
# ─────────────────────────────────────────────────────────────────────
T_BASE = 16
T_STEP_MS = 10
SR_GESTURE = 100.0
K_DISPLAY = 30.0        # vocalic display curvature (Kvoy)
K_C_DISPLAY = 10.0       # consonantal display curvature (article K=10)
SR_DISPLAY = 1000.0      # display sample rate (10x engine for smooth curves)
FS_AUDIO = 20_000
GUI_LEN_MM = 195
VALRECT = 1.10  # author ruling 2026-10-07 (reduces the fizz on /i/)

OUT_BASE = REPO_ROOT / "output" / "article_demos"
DEMO1_DIR = OUT_BASE / "demo1_ibia"
DEMO5_DIR = OUT_BASE / "demo5_clusters"

for d in (DEMO1_DIR, DEMO5_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAEDA_LABELS = ["Jaw", "Body", "Dorsum", "Tip", "LipP", "LipH", "Hy"]
MAEDA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2"]

# Consonant cluster info (article §3)
CLUSTER_INFO = {
    "bd": {"Sc": "{1,2,3,6}", "note": "/b/ involved -> Sc={1,2,3,6}"},
    "bg": {"Sc": "{1,2,3,6}", "note": "/b/ involved, /g/ palatal"},
    "db": {"Sc": "{1,2,3,6}", "note": "/b/ involved -> Sc={1,2,3,6}"},
    "dg": {"Sc": "{1,2,3,4}", "note": "no /b/ -> Sc={1,2,3,4}"},
    "gb": {"Sc": "{1,2,3,6}", "note": "/b/ involved, /g/ palatal"},
    "gd": {"Sc": "{1,2,3,4}", "note": "no /b/ -> Sc={1,2,3,4}"},
}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def get_phoneme_inventory():
    vowels = {k: (v["rho"], v["theta"]) for k, v in VOWELS_SYNTSYL.items()}
    consonants = {k: (v["rho"], v["theta"]) for k, v in CONSONANTS_SYNTSYL.items()}
    return vowels, consonants


def run_pipeline_recorded(input_text, T=T_BASE, **kwargs):
    """Run panphon_pipeline under polar_sync instrumentation.

    The engine's ACTUAL block sequence is recorded so that the display
    branches (and any label timeline) are rebuilt from the same blocks
    the engine walked — see scripts/polar_sync.py and
    docs/DISPLAY_VS_ENGINE.md. The engine itself is untouched
    (Pval/formants/sig identical to a direct panphon_pipeline call).
    """
    blocks: list = []
    result = polar_sync.record_pipeline(
        panphon_pipeline, blocks, text=input_text, T=T, **kwargs)
    return result, blocks


def display_branches(result, blocks):
    """z_v / z_c via polar_sync.build_branches (original
    Syllable_Synthesis display anchoring: pd on the departure side,
    phase on pa, theta in [0, pi] / [-pi, 0] — teardrop z_c, K=30/10,
    display nu_v=-1 / nu_c=+1; z_v drawn with the same arcplot form)."""
    z_v, z_c, n_disp = polar_sync.build_branches(
        blocks, result.Pval.shape[0], t_step_ms=T_STEP_MS,
        sr_display=SR_DISPLAY, k_v=K_DISPLAY, k_c=K_C_DISPLAY,
        nu_v=-1, nu_c=1)
    return z_v, z_c, n_disp


# ═══════════════════════════════════════════════════════════════════
# DEMO 1: /ibia/ VCV synthesis (Figure 1 — 4-step pipeline)
# ═══════════════════════════════════════════════════════════════════
def run_demo1_ibia():
    """Figure 1: 4-step synthesis of /ibia/."""
    print("=" * 72)
    print("DEMO 1: /ibia/ VCV synthesis (Figure 1, 4-step pipeline)")
    print("=" * 72)

    # The article uses "ibia" = /i b i a/ = V-C-V-V
    # In synthSYL, "ibi a" would be two words; "ibia" is one word
    input_text = "ibi a"  # /ibi/ + /a/ = VC + V (with diphthong /ia/)
    # Actually, let's try "ibia" as a single word
    input_text = "ibia"

    print(f"  Input: {input_text}")
    print(f"  T={T_BASE}, K=10, Kvoy=30, Pexp=1")

    result, blocks = run_pipeline_recorded(input_text, T=T_BASE,
                                           verbose=True)
    if result is None:
        print("  Pipeline failed")
        return

    print(f"  Pval shape: {result.Pval.shape}")
    print(f"  Segments: {result.segments}")
    print(f"  Anchors: {len(result.anchors)}")
    for a in result.anchors:
        print(f"    {a.kind} i={a.i} pt={a.pt} hold={a.hold} "
              f"is_vo={a.is_vowel_onset} is_we={a.is_word_end}")

    # Synthesize audio
    state = vlam.VlamState.initial(GUI_LEN_MM)
    config = vlam.SynthConfig(play_audio=False)
    sr = vlam.synthwordfen(
        state=state, articulatory_params=result.Pval,
        word_tokens=["O", "V", "F"],
        f0_scale=1.0, soft_rect_s=VALRECT,
        duration_factor=T_BASE, envelope=result.envelope, config=config,
    )
    sig = sr.signal
    formants = sr.formants  # shape (n_frames, 3) = F1, F2, F3

    # Save WAV
    from scipy.io.wavfile import write as wav_write
    max_val = float(np.max(np.abs(sig))) if np.max(np.abs(sig)) > 1e-12 else 1.0
    sig_int16 = np.int16(32767 * sig / (1.01 * max_val))
    wav_path = DEMO1_DIR / "ibia.wav"
    wav_write(str(wav_path), FS_AUDIO, sig_int16)
    print(f"  WAV: {sig.shape} samples ({sig.shape[0]/FS_AUDIO*1000:.0f} ms)")
    print(f"  Formants: {formants.shape}")

    # Reconstruct polar branches from the RECORDED engine blocks
    # (original arcplot anchoring; arrays sampled at SR_DISPLAY)
    z_v, z_c, n_disp = display_branches(result, blocks)
    n_steps = result.Pval.shape[0]

    # ── Figure: 4 panels (a, b, c, d) ──
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), constrained_layout=True)

    # (a) Planning trajectories zv(t) and zc(t)
    ax = axes[0]
    t_ms_z = np.arange(n_disp) / (SR_DISPLAY / 1000.0)  # ms at display rate
    t_ms = np.arange(n_steps) * T_STEP_MS               # ms at engine rate
    valid_v = ~np.isnan(z_v.real)
    valid_c = ~np.isnan(z_c.real)
    ax.plot(t_ms_z[valid_v], z_v[valid_v].real, "-", color="#d62728",
            linewidth=1.5, label=r"$\Re(z_v)$ (vocalic)")
    ax.plot(t_ms_z[valid_v], z_v[valid_v].imag, "--", color="#d62728",
            linewidth=1, alpha=0.5, label=r"$\Im(z_v)$")
    if np.any(valid_c):
        seg_starts = np.where(valid_c & ~np.roll(valid_c, 1))[0]
        seg_ends = np.where(valid_c & ~np.roll(valid_c, -1))[0]
        for s, e in zip(seg_starts, seg_ends):
            ax.plot(t_ms_z[s:e+1], z_c[s:e+1].real, "-", color="#1f77b4",
                    linewidth=1.8)
            ax.plot(t_ms_z[s:e+1], z_c[s:e+1].imag, "--", color="#1f77b4",
                    linewidth=1, alpha=0.5)
        ax.plot([], [], "-", color="#1f77b4", linewidth=1.8,
                label=r"$\Re(z_c)$ (consonantal)")
    ax.set_ylabel("Complex value")
    ax.set_title("(a) Planning trajectories $z_v(t)$ and $z_c(t)$",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (ms)")

    # (b) Flow of articulatory parameters (trough effect visible)
    ax = axes[1]
    P = result.Pval
    for i in range(7):
        lw = 2.0 if i == 1 else 1.2  # Highlight Body (excursion toward /u/)
        alpha = 1.0 if i == 1 else 0.7
        ax.plot(t_ms, P[:, i], color=MAEDA_COLORS[i], linewidth=lw,
                alpha=alpha, label=MAEDA_LABELS[i])
    ax.set_ylabel("Maeda parameter")
    ax.set_title("(b) Articulatory parameters",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", ncol=7, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (ms)")

    # (c) Formant trajectory F1-F2-F3
    ax = axes[2]
    n_frames = formants.shape[0]
    t_formants = np.arange(n_frames) * T_STEP_MS
    ax.plot(t_formants, formants[:, 0], color="#2ca02c", linewidth=1.5,
            label="F1")
    ax.plot(t_formants, formants[:, 1], color="#d62728", linewidth=2.0,
            label="F2")
    ax.plot(t_formants, formants[:, 2], color="#1f77b4", linewidth=1.5,
            label="F3")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("(c) Formants F1-F2-F3",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (ms)")
    ax.set_ylim(0, 4000)

    # (d) Spectrogram and wave plot
    ax = axes[3]
    sig_f = sig.astype(float) / 32767.0 if sig.dtype == np.int16 else sig
    ax.specgram(sig_f, NFFT=256, Fs=FS_AUDIO, noverlap=192,
                cmap="magma", scale="dB")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("(d) Spectrogram + wave",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 4000)

    fig.suptitle(f"Figure 1 — four-step synthesis of /ibia/ (T={T_BASE})",
                 fontsize=13, fontweight="bold")

    fig_path = DEMO1_DIR / "ibia_4panel.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")
    print(f"  WAV saved: {wav_path}")

    # Also save formants as NPZ for later analysis
    np.savez(DEMO1_DIR / "ibia_formants.npz",
             formants=formants, Pval=P, z_v=z_v, z_c=z_c,
             t_ms=t_ms, t_z_ms=t_ms_z, segments=result.segments)


# ═══════════════════════════════════════════════════════════════════
# DEMO 5: Six consonant cluster pairs /bd/, /bg/, /db/, /dg/, /gb/, /gd/
# ═══════════════════════════════════════════════════════════════════
def run_demo5_clusters():
    """§3: Six consonant cluster pairs with selection vectors."""
    print("\n" + "=" * 72)
    print("DEMO 5: Six consonant cluster pairs /bd/, /bg/, /db/, /dg/, /gb/, /gd/")
    print("=" * 72)

    state = vlam.VlamState.initial(GUI_LEN_MM)
    config = vlam.SynthConfig(play_audio=False)
    vowels, consonants = get_phoneme_inventory()

    # CVCCV sequences: a + cluster + a (e.g., "abda" for /bd/)
    cluster_inputs = {
        "bd": "abda",
        "bg": "abga",
        "db": "adba",
        "dg": "adga",
        "gb": "agba",
        "gd": "agda",
    }

    all_results = []

    for cluster, input_text in cluster_inputs.items():
        info = CLUSTER_INFO[cluster]
        print(f"\n  [/{cluster}/] input={input_text}  Sc={info['Sc']}  ({info['note']})")

        result, blocks = run_pipeline_recorded(input_text, T=T_BASE,
                                               verbose=False)
        if result is None:
            print(f"    Failed")
            continue
        print(f"    Pval: {result.Pval.shape}  Segments: {result.segments}")

        sr = vlam.synthwordfen(
            state=state, articulatory_params=result.Pval,
            word_tokens=["O", "V", "F"],
            f0_scale=1.0, soft_rect_s=VALRECT,
            duration_factor=T_BASE, envelope=result.envelope, config=config,
        )
        sig = sr.signal
        formants = sr.formants

        # Save WAV
        max_val = float(np.max(np.abs(sig))) if np.max(np.abs(sig)) > 1e-12 else 1.0
        sig_int16 = np.int16(32767 * sig / (1.01 * max_val))
        from scipy.io.wavfile import write as wav_write
        wav_path = DEMO5_DIR / f"cluster_{cluster}.wav"
        wav_write(str(wav_path), FS_AUDIO, sig_int16)

        # Reconstruct polar from the recorded blocks (teardrop z_c)
        z_v, z_c, n_steps = display_branches(result, blocks)

        all_results.append((cluster, input_text, result, formants, z_v, z_c, info))

    # ── Figure 1: Maeda parameters for each cluster ──
    n = len(all_results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, (cluster, input_text, result, formants, z_v, z_c, info) in enumerate(all_results):
        ax = axes_flat[i]
        P = result.Pval
        t_ms = np.arange(P.shape[0]) * T_STEP_MS
        for j in range(7):
            ax.plot(t_ms, P[:, j], color=MAEDA_COLORS[j], linewidth=1.2,
                    label=MAEDA_LABELS[j])
        ax.set_title(f"/{cluster}/  Sc={info['Sc']}\n{info['note']}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Maeda param")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, ncol=4)

    for j in range(len(all_results), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Consonant cluster pairs — articulatory parameters\n"
        "arXiv:2307.02299 §3 (Berthommier 2023) — T=16, K=10, Kvoy=30, Pexp=1",
        fontsize=12, fontweight="bold")
    fig_path = DEMO5_DIR / "cluster_params.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\n  Params figure: {fig_path}")

    # ── Figure 2: Polar trajectories for each cluster ──
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, (cluster, input_text, result, formants, z_v, z_c, info) in enumerate(all_results):
        ax = axes_flat[i]
        ax.set_facecolor("#fafafa")
        for r in [0.5, 1.0, 1.2]:
            circle = Circle((0, 0), r, fill=False, linestyle=":",
                            linewidth=0.6, color="#888888", alpha=0.5)
            ax.add_patch(circle)
        ax.axhline(0, color="#cccccc", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="#cccccc", linewidth=0.5, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.2)

        # Plot inventory
        for k, (rho, theta) in vowels.items():
            x, y = rho * np.cos(theta), rho * np.sin(theta)
            ax.plot(x, y, "o", color="#d62728", markersize=3, alpha=0.2)
        for k, (rho, theta) in consonants.items():
            x, y = rho * np.cos(theta), rho * np.sin(theta)
            ax.plot(x, y, "s", color="#1f77b4", markersize=4, alpha=0.2)

        # Plot branches
        valid_v = ~np.isnan(z_v.real)
        ax.plot(z_v[valid_v].real, z_v[valid_v].imag, "-",
                color="#d62728", linewidth=1.4, alpha=0.85)
        valid_c = ~np.isnan(z_c.real)
        if np.any(valid_c):
            seg_starts = np.where(valid_c & ~np.roll(valid_c, 1))[0]
            seg_ends = np.where(valid_c & ~np.roll(valid_c, -1))[0]
            for s, e in zip(seg_starts, seg_ends):
                ax.plot(z_c[s:e+1].real, z_c[s:e+1].imag, "-",
                        color="#1f77b4", linewidth=1.8, alpha=0.9)

        ax.set_title(f"/{cluster}/  Sc={info['Sc']}",
                     fontsize=10, color="#003366", fontweight="bold")

    for j in range(len(all_results), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Consonant cluster pairs — polar trajectories\n"
        "(z_v red, K=30; z_c blue, K=10 — article §2.1)",
        fontsize=12, fontweight="bold")
    fig_path2 = DEMO5_DIR / "cluster_polar.png"
    fig.savefig(fig_path2, dpi=120)
    plt.close(fig)
    print(f"  Polar figure: {fig_path2}")

    # ── Figure 3: Spectrograms ──
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, (cluster, input_text, result, formants, z_v, z_c, info) in enumerate(all_results):
        ax = axes_flat[i]
        # Re-synth for spectrogram
        sr = vlam.synthwordfen(
            state=state, articulatory_params=result.Pval,
            word_tokens=["O", "V", "F"],
            f0_scale=1.0, soft_rect_s=VALRECT,
            duration_factor=T_BASE, envelope=result.envelope, config=config,
        )
        sig_f = sr.signal.astype(float)
        max_val = float(np.max(np.abs(sig_f))) if np.max(np.abs(sig_f)) > 1e-12 else 1.0
        sig_f = sig_f / max_val
        ax.specgram(sig_f, NFFT=256, Fs=FS_AUDIO, noverlap=192,
                    cmap="magma", scale="dB")
        ax.set_title(f"/{cluster}/  ({input_text})", fontsize=10)
        ax.set_ylim(0, 4000)
        if i == 0:
            ax.set_ylabel("Frequency (Hz)")
        if i >= len(all_results) - ncols:
            ax.set_xlabel("Time (s)")

    for j in range(len(all_results), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Consonant cluster pairs — spectrograms",
                 fontsize=12, fontweight="bold")
    fig_path3 = DEMO5_DIR / "cluster_spectrograms.png"
    fig.savefig(fig_path3, dpi=120)
    plt.close(fig)
    print(f"  Spectrogram figure: {fig_path3}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("ARTICLE DEMONSTRATIONS (arXiv:2307.02299, Berthommier 2023)")
    print(f"T={T_BASE}, K=10, Kvoy=30, Pexp=1, nu=-1 (article parameters)")
    print(f"Output: {OUT_BASE}")
    print("=" * 72)

    # Demo 1: /ibia/ VCV
    run_demo1_ibia()

    # Demo 5: Cluster pairs
    run_demo5_clusters()

    print("\n" + "=" * 72)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 72)
    print(f"  Demo 1 (/ibia/):      {DEMO1_DIR}")
    print(f"  Demo 5 (clusters):     {DEMO5_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
