#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
make_ibbi_polar_video.py — Visualisation polaire de la transition
ib ib → bi bi (Berthommier 2023, arXiv:2307.02299).

Associe la simulation ib bi sweep à la représentation polaire à deux
branches du dépôt covtl-pipeline (FBerthommier/covtl-pipeline).

Principe
--------
Pour chaque Tp (12 valeurs, 100 ms → 0 ms) :
  1. Re-démarre le pipeline synthSYL pour "ib ib" avec le Tp et δ couplé
     (helpers de trajectoire instrumentés : polar_sync.record_pipeline)
  2. Reconstruit les deux branches depuis la séquence de blocs ENREGISTRÉE
     (ancrage d'affichage original Syllable_Synthesis — pd côté départ,
     phase sur pa, θ∈[0,π]/[−π,0] — cf. docs/DISPLAY_VS_ENGINE.md) :
       z_v (rouge) : branche vocalique, même forme arcplot (K=30, ν=−1)
                     + plateaux stationnaires aux voyelles tenues
       z_c (bleu)  : branche consonantique, n'existe que pendant les
                     clusters /b/ — sub-arcs dep→C₁→...→Cₘ→arr
                     (K=10, ν=+1, article §2.1) en LARME (teardrop)
  3. Génère :
       - une figure polaire statique (.png)
       - une vidéo polaire dynamique (trajectoire qui se déroule avec
         trail rémanant ~0.75 s, point courant, cible active)
Concatène les 12 vidéos polaires en une vidéo progressive.

Sorties (dans <repo>/output/ibbi_sweep/polar/)
-------------------------------------------------------------
- static/polar_pXXX.png        : figure polaire statique par Tp
- comparison_polar.png         : 4×3 grille statique des 12 Tp
- segments/polar_pXXX.mp4      : vidéo polaire dynamique par Tp
- polar_progressive.mp4        : concaténation des 12 segments
- zvc_pXXX.npz                 : trajectoires z_v, z_c sauvegardées
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
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

try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
except Exception:
    pass
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import cv2  # noqa: E402

# Repo paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Engine-block-synchronized display reconstruction (original
# Syllable_Synthesis arcplot anchoring: pd on the departure side, phase
# on pa — teardrop z_c, arcplot z_v, display nu_v=-1 / nu_c=+1)
import polar_sync  # noqa: E402

from synthSYL import panphon_pipeline  # noqa: E402
from synthSYL.phonology import VOWELS_SYNTSYL, CONSONANTS_SYNTSYL  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
OUT_BASE = Path(__file__).resolve().parent.parent / "output" / "ibbi_sweep" / "polar"
STATIC_DIR = OUT_BASE / "static"
SEG_DIR = OUT_BASE / "segments"
ZVC_DIR = OUT_BASE / "zvc"

for d in (STATIC_DIR, SEG_DIR, ZVC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Sweep configuration (mirror run_ibbi_sweep.py)
# ─────────────────────────────────────────────────────────────────────
PAUSE_MS_VALUES = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 0]
TP_MAX_MS = 100.0
T_BASE = 5
T_STEP_MS = 10            # 1 Maeda step = 10 ms (T_S = 0.01 s)
SR_GESTURE = 100.0        # gesture rate (Hz) = 1/T_STEP_MS * 1000
K_DISPLAY = 30.0        # vocalic display curvature (Kvoy)
K_C_DISPLAY = 10.0      # consonantal display curvature (article §2.1: K=10)
FPS_RAW = 100.0
FPS_OUT = 25.0
FIG_SIZE = (7.2, 7.2)     # inches @ 100 dpi -> 720x720 px (square polar)
TRAIL_MS = 750            # fading trail length (ms)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def factor_for_pause_ms(pause_ms: float) -> float:
    return pause_ms / (T_BASE * 0.01 * 1000)  # pause_ms / 50


def delta_for_pause_ms(pause_ms: float) -> float:
    return 0.5 + 0.5 * (1.0 - pause_ms / TP_MAX_MS)


def form_label(pause_ms: float) -> str:
    if pause_ms > 60:
        return "ib ib  (schwa Ve audible)"
    if pause_ms > 20:
        return "ib ib  (schwa résiduel)"
    if pause_ms > 0:
        return "ib ib → bi bi  (transition)"
    return "bi bi  (fusion Vo = Ve = V)"


def get_phoneme_inventory():
    """Return (vowel_targets, consonant_targets) as dicts {key: (rho, theta)}."""
    vowels = {k: (v["rho"], v["theta"]) for k, v in VOWELS_SYNTSYL.items()}
    consonants = {k: (v["rho"], v["theta"]) for k, v in CONSONANTS_SYNTSYL.items()}
    return vowels, consonants


# ═══════════════════════════════════════════════════════════════════
# Polar branch reconstruction
# ═══════════════════════════════════════════════════════════════════
def display_branches(blocks, n_steps):
    """(z_v, z_c, n_steps) at the 100 Hz display grid from the RECORDED
    engine block sequence (polar_sync.record_pipeline), with the
    original Syllable_Synthesis display anchoring: pd on the departure
    side (the consonant on z_c approach legs), phase on the arrival
    point pa, theta in [0, pi] / [-pi, 0] — teardrop z_c (K=10), z_v
    with the same arcplot form (K=30), display nu_v=-1 / nu_c=+1."""
    return polar_sync.build_branches(
        blocks, n_steps, t_step_ms=T_STEP_MS, sr_display=SR_GESTURE,
        k_v=K_DISPLAY, k_c=K_C_DISPLAY, nu_v=-1, nu_c=1)


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════
def setup_polar_axes(ax):
    """Configure a polar axes with trigonometric orientation."""
    ax.set_facecolor("#fafafa")
    # Reference circles
    for r in [0.5, 1.0, 1.2]:
        circle = Circle((0, 0), r, fill=False, linestyle=":",
                        linewidth=0.6, color="#888888", alpha=0.5)
        ax.add_patch(circle)
    # Reference axes
    ax.axhline(0, color="#cccccc", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="#cccccc", linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r"$\Re(z)$", fontsize=9)
    ax.set_ylabel(r"$\Im(z)$", fontsize=9)
    ax.grid(True, alpha=0.2)
    # Theta labels
    for theta_deg, label in [(0, "0"), (90, "π/2"), (180, "π"),
                              (270, "3π/2")]:
        theta = np.deg2rad(theta_deg)
        x = 1.35 * np.cos(theta)
        y = 1.35 * np.sin(theta)
        ax.text(x, y, label, fontsize=8, color="#666666",
                ha="center", va="center")


def plot_inventory(ax, vowels, consonants):
    """Plot the phoneme inventory as background dots."""
    for k, (rho, theta) in vowels.items():
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        ax.plot(x, y, "o", color="#d62728", markersize=4, alpha=0.25)
        ax.text(x, y, k, color="#aa3333", fontsize=7, alpha=0.4,
                ha="center", va="bottom")
    for k, (rho, theta) in consonants.items():
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        ax.plot(x, y, "s", color="#1f77b4", markersize=5, alpha=0.25)
        ax.text(x, y, k, color="#3366aa", fontsize=7, alpha=0.4,
                ha="center", va="bottom")


def plot_static_polar(z_v, z_c, pause_ms, delta, out_path: Path,
                      vowels=None, consonants=None):
    """Static polar figure with both branches fully drawn."""
    fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)
    setup_polar_axes(ax)
    if vowels and consonants:
        plot_inventory(ax, vowels, consonants)

    # Vocalic branch (red)
    valid_v = ~np.isnan(z_v.real)
    ax.plot(z_v[valid_v].real, z_v[valid_v].imag, "-",
            color="#d62728", linewidth=1.6, alpha=0.85, label=r"$z_v$ (vocalic)")
    # Consonantal branch (blue)
    valid_c = ~np.isnan(z_c.real)
    if np.any(valid_c):
        # Split into segments (NaN breaks)
        seg_starts = np.where(valid_c & ~np.roll(valid_c, 1))[0]
        seg_ends = np.where(valid_c & ~np.roll(valid_c, -1))[0]
        for s, e in zip(seg_starts, seg_ends):
            ax.plot(z_c[s:e+1].real, z_c[s:e+1].imag, "-",
                    color="#1f77b4", linewidth=1.8, alpha=0.9)
        ax.plot([], [], "-", color="#1f77b4", linewidth=1.8,
                label=r"$z_c$ (consonantal)")

    # Anchor positions (large dots)
    # Use the start/end of each segment as anchor markers
    if np.any(valid_v):
        ax.plot(z_v[0].real, z_v[0].imag, "o",
                color="#aa0000", markersize=10, zorder=5)
        ax.plot(z_v[-1].real, z_v[-1].imag, "s",
                color="#aa0000", markersize=10, zorder=5)

    ax.set_title(
        f"Plan polaire — Tp = {pause_ms:.0f} ms    δ = {delta:.2f}\n"
        f"{form_label(pause_ms)}",
        fontsize=11, color="#003366", fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.text(0.5, 0.01,
             "Branches reconstruites via polar_arc "
             "(z_v K=30, z_c K=10 — article §2.1) — arXiv:2307.02299",
             ha="center", fontsize=8, color="#666666")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_comparison_polar(items: list, out_path: Path):
    """4×3 grid of static polar figures."""
    n = len(items)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 4.2 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    vowels, consonants = get_phoneme_inventory()

    for i, (pause_ms, delta, z_v, z_c) in enumerate(items):
        ax = axes_flat[i]
        setup_polar_axes(ax)
        plot_inventory(ax, vowels, consonants)
        valid_v = ~np.isnan(z_v.real)
        ax.plot(z_v[valid_v].real, z_v[valid_v].imag, "-",
                color="#d62728", linewidth=1.2, alpha=0.85)
        valid_c = ~np.isnan(z_c.real)
        if np.any(valid_c):
            seg_starts = np.where(valid_c & ~np.roll(valid_c, 1))[0]
            seg_ends = np.where(valid_c & ~np.roll(valid_c, -1))[0]
            for s, e in zip(seg_starts, seg_ends):
                ax.plot(z_c[s:e+1].real, z_c[s:e+1].imag, "-",
                        color="#1f77b4", linewidth=1.5, alpha=0.9)
        ax.set_title(f"Tp = {pause_ms:.0f} ms   δ = {delta:.2f}",
                     fontsize=10, color="#003366", fontweight="bold")
        ax.text(0.5, -0.15, form_label(pause_ms),
                transform=ax.transAxes, ha="center", fontsize=8,
                color="#444444")

    # Hide unused axes
    for j in range(len(items), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Plan polaire — transition ib ib → bi bi\n"
        "(branches z_v rouge, z_c bleu ; K=30, covtl-pipeline)",
        fontsize=12
    )
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Dynamic polar video rendering
# ═══════════════════════════════════════════════════════════════════
def render_polar_video_segment(z_v, z_c, pause_ms, delta,
                                out_mp4: Path,
                                wav_path: Optional[Path] = None,
                                vowels=None, consonants=None) -> Path:
    """Render a polar-only dynamic MP4 for one Tp value.

    The trajectory unfolds over time with a fading trail (~750 ms) and a
    current-position dot. If wav_path is given, the audio is dubbed in.
    """
    n_steps = len(z_v)
    duration_ms = n_steps * T_STEP_MS
    n_frames_out = max(1, int(round(duration_ms * FPS_OUT / 1000.0)))

    # Sample the trajectory at FPS_OUT
    # z_v and z_c are at SR_GESTURE = 100 Hz; we need FPS_OUT = 25 Hz
    step_per_out = int(round(SR_GESTURE / FPS_OUT))  # 4
    indices = np.arange(0, n_steps, step_per_out)
    if indices[-1] != n_steps - 1:
        indices = np.append(indices, n_steps - 1)
    z_v_out = z_v[indices]
    z_c_out = z_c[indices]
    n_out = len(indices)

    trail_samples = int(round(TRAIL_MS * SR_GESTURE / 1000 / step_per_out))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer: Optional[cv2.VideoWriter] = None

    if vowels is None or consonants is None:
        vowels, consonants = get_phoneme_inventory()

    print(f"  Rendering {n_out} polar frames @ {FPS_OUT} fps "
          f"(δ={delta:.2f}, Tp={pause_ms:.0f} ms)")

    t_start = time.time()
    for frame_idx in range(n_out):
        # Create figure
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        try:
            setup_polar_axes(ax)
            plot_inventory(ax, vowels, consonants)

            # Plot trail (fading)
            start = max(0, frame_idx - trail_samples)
            trail_v = z_v_out[start:frame_idx+1]
            trail_c = z_c_out[start:frame_idx+1]

            # Plot the past trail with fading alpha
            if len(trail_v) > 1:
                # Color the trail by recency (older = lighter)
                for k in range(len(trail_v) - 1):
                    alpha = 0.2 + 0.7 * (k / max(1, len(trail_v) - 1))
                    if not np.isnan(trail_v[k].real) and not np.isnan(trail_v[k+1].real):
                        ax.plot([trail_v[k].real, trail_v[k+1].real],
                                [trail_v[k].imag, trail_v[k+1].imag],
                                "-", color="#d62728", linewidth=1.4, alpha=alpha)
            if len(trail_c) > 1:
                for k in range(len(trail_c) - 1):
                    alpha = 0.2 + 0.7 * (k / max(1, len(trail_c) - 1))
                    if (not np.isnan(trail_c[k].real)
                            and not np.isnan(trail_c[k+1].real)):
                        ax.plot([trail_c[k].real, trail_c[k+1].real],
                                [trail_c[k].imag, trail_c[k+1].imag],
                                "-", color="#1f77b4", linewidth=2.0, alpha=alpha)

            # Current position dots
            if not np.isnan(z_v_out[frame_idx].real):
                ax.plot(z_v_out[frame_idx].real, z_v_out[frame_idx].imag,
                        "o", color="#d62728", markersize=10, zorder=6,
                        markeredgecolor="white", markeredgewidth=1.5)
            if not np.isnan(z_c_out[frame_idx].real):
                ax.plot(z_c_out[frame_idx].real, z_c_out[frame_idx].imag,
                        "o", color="#1f77b4", markersize=12, zorder=6,
                        markeredgecolor="white", markeredgewidth=1.5)

            ax.set_title(
                f"Tp = {pause_ms:.0f} ms    δ = {delta:.2f}    "
                f"{form_label(pause_ms)}\n"
                f"t = {frame_idx * T_STEP_MS * step_per_out:.0f} ms",
                fontsize=11, color="#003366", fontweight="bold"
            )

            # Capture frame
            fig.set_dpi(100)
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
            canvas = FigCanvas(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            img_array = np.array(buf, dtype=np.uint8)
            height, width = img_array.shape[:2]
            img_bgr = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)

            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    str(out_mp4), fourcc, float(FPS_OUT), (width, height)
                )
                print(f"  Video writer: {width}x{height} @ {FPS_OUT} fps "
                      f"-> {out_mp4.name}")

            video_writer.write(img_bgr)
        finally:
            plt.close(fig)

        if (frame_idx + 1) % 50 == 0 or frame_idx == n_out - 1:
            elapsed = time.time() - t_start
            fps_render = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
            print(f"    Frame {frame_idx+1}/{n_out} ({fps_render:.1f} fps)")

    if video_writer is not None:
        video_writer.release()

    # Dub audio if provided
    if wav_path is not None and wav_path.exists():
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe:
            dubbed = out_mp4.with_name(out_mp4.stem + "_dubbed.mp4")
            cmd = [
                ffmpeg_exe, "-y",
                "-i", str(out_mp4),
                "-i", str(wav_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "22050",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                "-pix_fmt", "yuv420p",
                str(dubbed),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                # Replace the silent video with the dubbed one
                out_mp4.unlink(missing_ok=True)
                dubbed.rename(out_mp4)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode() if e.stderr else str(e)
                print(f"  ⚠ Audio dubbing failed: {stderr[:200]}")
    return out_mp4


def concat_segments(segments: list[Path], out_path: Path) -> bool:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        return False
    list_file = SEG_DIR / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in segments:
            abs_p = str(p.resolve())
            f.write(f"file '{abs_p}'\n")
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    print(f"  Concatenating {len(segments)} segments...")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        cmd_reenc = [
            ffmpeg_exe, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-ar", "22050",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        try:
            subprocess.run(cmd_reenc, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            print(f"  ✗ Concat failed: {stderr[:300]}")
            return False


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    print("=" * 72)
    print("VISUALISATION POLAIRE ib ib → bi bi")
    print("=" * 72)
    print(f"  Tp sweep (ms) : {PAUSE_MS_VALUES}")
    print(f"  Display K     : z_v {K_DISPLAY:.0f}, z_c {K_C_DISPLAY:.0f} (article §2.1)")
    print(f"  Trail length   : {TRAIL_MS} ms")
    print(f"  Output         : {OUT_BASE}")
    print()

    vowels, consonants = get_phoneme_inventory()
    print(f"  Inventory      : {len(vowels)} vowels, {len(consonants)} consonants")
    print()

    collected = []
    segment_paths = []

    for i, pause_ms in enumerate(PAUSE_MS_VALUES):
        tag = f"p{int(round(pause_ms)):03d}"
        t0 = time.time()
        print(f"[{i+1}/{len(PAUSE_MS_VALUES)}] Tp = {pause_ms:.0f} ms")

        # Re-run pipeline with instrumented trajectory helpers so the
        # display branches are rebuilt from the engine's block sequence
        factor = factor_for_pause_ms(pause_ms)
        delta = delta_for_pause_ms(pause_ms)
        blocks = []
        result = polar_sync.record_pipeline(
            panphon_pipeline, blocks,
            text="ib ib",
            T=T_BASE,
            short_pause_duration_factor=factor,
            long_pause_duration_factor=max(factor, 1.0),
            delta_o=delta, delta_e=delta,
            verbose=False,
        )
        if result is None:
            print(f"  ✗ Pipeline failed")
            return 1

        # Reconstruct branches from the recorded blocks (100 Hz grid)
        z_v, z_c, n_steps = display_branches(blocks, result.Pval.shape[0])
        print(f"  z_v: {len(z_v)} samples   z_c: {np.sum(~np.isnan(z_c.real))} active samples")

        # Save z_v, z_c
        zvc_path = ZVC_DIR / f"zvc_{tag}.npz"
        np.savez(zvc_path, z_v=z_v, z_c=z_c,
                 pause_ms=pause_ms, factor=factor, delta=delta,
                 n_steps=n_steps)

        # Static figure
        static_path = STATIC_DIR / f"polar_{tag}.png"
        plot_static_polar(z_v, z_c, pause_ms, delta, static_path,
                          vowels=vowels, consonants=consonants)
        print(f"  Static fig : → {static_path.name}")

        collected.append((pause_ms, delta, z_v, z_c))

        # Dynamic video segment (audio dubbed from the sweep WAVs when present)
        wav_path = REPO_ROOT / "output" / "ibbi_sweep" / "wav" / f"synth_{tag}.wav"
        seg_mp4 = SEG_DIR / f"polar_{tag}.mp4"
        render_polar_video_segment(z_v, z_c, pause_ms, delta,
                                    seg_mp4, wav_path=wav_path,
                                    vowels=vowels, consonants=consonants)
        print(f"  MP4 segment : → {seg_mp4.name} "
              f"({os.path.getsize(seg_mp4)/1024:.0f} KB)")
        segment_paths.append(seg_mp4)

        print(f"  Done in {time.time()-t0:.1f}s")

    # Comparison figure
    comp_path = STATIC_DIR / "comparison_polar.png"
    plot_comparison_polar(collected, comp_path)
    print(f"\n✓ Comparison figure : {comp_path}")

    # Concatenate dynamic videos
    final_mp4 = OUT_BASE / "polar_progressive.mp4"
    print(f"\n[Final] Concatenating {len(segment_paths)} polar segments "
          f"-> {final_mp4.name}")
    if not concat_segments(segment_paths, final_mp4):
        print("  ✗ Concat failed")
        return 1
    size_mb = os.path.getsize(final_mp4) / (1024 * 1024)
    print(f"  ✓ Final video: {final_mp4}  ({size_mb:.2f} MB)")

    # Probe duration
    if shutil.which("ffprobe"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1",
             str(final_mp4)],
            capture_output=True, text=True
        )
        if probe.returncode == 0:
            dur = float(probe.stdout.strip())
            print(f"  Duration: {dur:.2f} s")

    print()
    print("=" * 72)
    print("RÉCAPITULATIF")
    print("=" * 72)
    print(f"  Figures statiques : {STATIC_DIR}/polar_pXXX.png ({len(collected)})")
    print(f"  Grille statique    : {comp_path}")
    print(f"  Segments vidéo     : {SEG_DIR}/polar_pXXX.mp4 ({len(segment_paths)})")
    print(f"  Vidéo finale       : {final_mp4}")
    print(f"  Trajectoires       : {ZVC_DIR}/zvc_pXXX.npz ({len(collected)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
