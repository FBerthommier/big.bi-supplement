#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
run_ibbi_ushape_sweep_en.py — ib ib → bi bi transition with a U-shaped
variation of Tp (pause duration), T=16, US English.

CONTINUOUS U-shape design (10 segments; author rulings 2026-10-07):
  Tp decreases to 0 (delta correlated: delta = 0.5 + 0.5*(1-Tp/160);
  the bottom of the U, Tp = 0, is the fused "bi bi"), then increases
  back to the 160-ms baseline. Segments flow directly into one
  another (no restarts).

The U-shape lets the listener clearly perceive:
  - "ib ib" at the start (full pause, audible schwa)
  - a CONTINUOUS acceleration into "bi bi" (no pause, fusion Vo=Ve=V)
  - "ib ib" again at the end (return to baseline)

Output: <repo>/output/ibbi_ushape_en/
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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # noqa: E402

try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import cv2  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import polar_sync  # noqa: E402
from synthSYL import panphon_pipeline  # noqa: E402
from synthSYL.phonology import VOWELS_SYNTSYL, CONSONANTS_SYNTSYL  # noqa: E402
import vlam  # noqa: E402
from vlam import VlamState, plot_vocal_tract  # noqa: E402
from VLAMvidmaker import freqevalNN_visual  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Paths & config
# ─────────────────────────────────────────────────────────────────────
OUT_BASE = Path(__file__).resolve().parent.parent / "output" / "ibbi_ushape_en"
NPZ_DIR = OUT_BASE / "npz"
WAV_DIR = OUT_BASE / "wav"
FIG_DIR = OUT_BASE / "figures"
POLAR_DIR = OUT_BASE / "polar"
POLAR_STATIC = POLAR_DIR / "static"
POLAR_SEG = POLAR_DIR / "segments"
ZVC_DIR = POLAR_DIR / "zvc"
DUAL_DIR = OUT_BASE / "dual"
DUAL_SEG = DUAL_DIR / "segments"

for d in (NPZ_DIR, WAV_DIR, FIG_DIR, POLAR_STATIC, POLAR_SEG, ZVC_DIR, DUAL_SEG):
    d.mkdir(parents=True, exist_ok=True)

# Configuration: T=16, U-shaped Tp
T_BASE = 16
T_STEP_MS = 10
SR_GESTURE = 100.0
K_DISPLAY = 30.0        # vocalic display curvature (Kvoy)
K_C_DISPLAY = 10.0      # consonantal display curvature (article §2.1: K=10)
FPS_OUT = 25.0
FS_AUDIO = 20_000
TRAIL_MS = 500          # fading trail length (ms) — short enough that
                        # past trajectories clearly yield to the current one
GUI_LEN_MM = 195
VALRECT = 1.10  # author ruling 2026-10-07 (was 0.75)
TP_MAX_MS = 160.0  # Tp = T = 16 steps = 160 ms at start/end

# CONTINUOUS U-shaped Tp ramp, 10 segments (author rulings 2026-10-07:
# continuous correlated Tp/delta variation — keep the U — then
# "ramener à 10 segments"). Tp decreases to 0 (delta correlated:
# delta = 0.5 + 0.5*(1 - Tp/160); the bottom of the U is the fused
# "bi bi"), then increases back to the 160-ms baseline. Segments flow
# directly into one another (no restarts).
TP_MS_VALUES = [160, 120, 80, 40, 0, 40, 80, 120, 160, 160]

MAEDA_LABELS = ["Jaw", "Body", "Dorsum", "Tip", "LipP", "LipH", "Hy"]
MAEDA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2"]


# ─────────────────────────────────────────────────────────────────────
# Helpers (US English labels)
# ─────────────────────────────────────────────────────────────────────
def factor_for_tp_ms(tp_ms: float) -> float:
    """Convert pause duration (ms) to short_pause_duration_factor."""
    return tp_ms / (T_BASE * T_STEP_MS)


def delta_for_tp_ms(tp_ms: float) -> float:
    """δ coupled to Tp: Tp=160 → δ=0.5 (schwa); Tp=0 → δ=1.0 (fusion)."""
    return 0.5 + 0.5 * (1.0 - tp_ms / TP_MAX_MS)


def form_label(tp_ms: float) -> str:
    if tp_ms >= 160:
        return "ib ib (clear baseline — audible schwa Ve)"
    if tp_ms > 60:
        return "ib ib (audible schwa)"
    if tp_ms > 20:
        return "ib ib (residual schwa)"
    if tp_ms > 0:
        return "ib ib → bi bi (transition)"
    return "bi bi (fusion Vo = Ve = V)"


def u_shape_position(idx: int, total: int) -> str:
    """Describe position in the U-shape."""
    if idx < 2:
        return "start baseline"
    if idx == total // 2:
        return "bottom of U (fusion)"
    if idx < total // 2:
        return f"descending {idx}/{total//2}"
    if idx >= total - 2:
        return "end baseline"
    return f"ascending {idx - total//2}/{total//2}"


def get_phoneme_inventory():
    vowels = {k: (v["rho"], v["theta"]) for k, v in VOWELS_SYNTSYL.items()}
    consonants = {k: (v["rho"], v["theta"]) for k, v in CONSONANTS_SYNTSYL.items()}
    return vowels, consonants


# ═══════════════════════════════════════════════════════════════════
# Pipeline + synth
# ═══════════════════════════════════════════════════════════════════
def run_pipeline(tp_ms: float):
    """(result, factor, delta, blocks) with the engine blocks recorded
    via polar_sync.

    Verbal transformation (article §2.2): when the time pressure cancels
    Tp (with delta = 1), x.VC.VC.x and xV.CV.x represent the SAME
    utterance — the percept switches to "bi bi". The syllable-graph
    transition is therefore applied at the fusion point: the input
    switches from "ib ib" (coda /b/) to "bi bi" (onset /b/) when
    Tp = 0, which is what makes the audio converge to the "bi bi"
    percept at the bottom of the U (keeping "ib ib" there leaves a
    residual word-boundary structure that never resyllabifies).
    """
    factor = factor_for_tp_ms(tp_ms)
    delta = delta_for_tp_ms(tp_ms)
    input_text = "bi bi" if tp_ms <= 0 else "ib ib"
    blocks: list = []
    result = polar_sync.record_pipeline(
        panphon_pipeline, blocks,
        text=input_text,
        T=T_BASE,
        short_pause_duration_factor=factor,
        long_pause_duration_factor=max(factor, 1.0),
        delta_o=delta, delta_e=delta,
        verbose=False,
    )
    if result is None:
        raise RuntimeError(f"Pipeline failed for tp_ms={tp_ms}")
    return result, factor, delta, blocks


def synthesize_wav(result, state, config, out_path: Path) -> np.ndarray:
    sr = vlam.synthwordfen(
        state=state,
        articulatory_params=result.Pval,
        word_tokens=["O", "V", "F"],
        f0_scale=1.0, soft_rect_s=VALRECT,
        duration_factor=T_BASE, envelope=result.envelope, config=config,
    )
    sig = sr.signal
    max_val = float(np.max(np.abs(sig))) if np.max(np.abs(sig)) > 1e-12 else 1.0
    sig_int16 = np.int16(32767 * sig / (1.01 * max_val))
    from scipy.io.wavfile import write as wav_write
    wav_write(str(out_path), FS_AUDIO, sig_int16)
    return sig_int16


# ═══════════════════════════════════════════════════════════════════
# Phoneme timeline
# ═══════════════════════════════════════════════════════════════════
def compute_phoneme_timeline(result, pause_ms: float):
    anchors = result.anchors
    nodes = result.nodes
    timeline = []
    prev_pt = None
    T_voy_ms = T_BASE * T_STEP_MS
    T_voy_arc_ms = 2 * T_voy_ms
    T_cons_ms = T_BASE * T_STEP_MS
    T_pause_ms = max(pause_ms, T_STEP_MS)

    for idx in range(len(anchors) - 1):
        A = anchors[idx]
        B = anchors[idx + 1]
        a_is_term = A.kind in ("pause", "synth")
        b_is_term = B.kind in ("pause", "synth")
        a_is_vowel = A.kind == "V"
        b_is_vowel = B.kind == "V"

        if A.hold and a_is_vowel and not A.is_vowel_onset and not A.is_word_end:
            n_frames = int(round(T_voy_ms * SR_GESTURE / 1000))
            seg_key = "?"
            if 0 <= A.i < len(nodes):
                seg_key = nodes[A.i].seg_key or "?"
            timeline.append((seg_key, n_frames))
            prev_pt = A.pt

        search_start = A.i + 1 if not A.is_vowel_onset else A.i
        search_end = B.i
        consonants = [nodes[k] for k in range(search_start, search_end)
                      if 0 <= k < len(nodes) and nodes[k].kind == "C"]
        m = len(consonants)

        if a_is_vowel and b_is_vowel:
            if m > 0:
                n_per_cons = int(round(T_cons_ms * SR_GESTURE / 1000))
                for k_idx, c in enumerate(consonants):
                    if k_idx == m - 1:
                        timeline.append((c.seg_key, 2 * n_per_cons))
                    else:
                        timeline.append((c.seg_key, n_per_cons))
            else:
                n_frames = int(round(T_voy_arc_ms * SR_GESTURE / 1000))
                timeline.append(("_", n_frames))
            prev_pt = B.pt
        elif a_is_vowel and b_is_term:
            n_frames = int(round(T_cons_ms * SR_GESTURE / 1000))
            timeline.append(("_", n_frames))
            prev_pt = A.pt
        elif a_is_term and b_is_vowel:
            if prev_pt is not None:
                n_frames = int(round(T_pause_ms * SR_GESTURE / 1000))
                timeline.append(("_", n_frames))
            else:
                timeline.append(("_", 1))
            prev_pt = B.pt

    n_total = sum(n for _, n in timeline)
    per_frame = ["?"] * n_total
    pos = 0
    for phoneme, n in timeline:
        for k in range(n):
            if pos + k < n_total:
                per_frame[pos + k] = phoneme
        pos += n
    n_steps = result.Pval.shape[0]
    if len(per_frame) < n_steps:
        last = per_frame[-1] if per_frame else "?"
        per_frame = per_frame + [last] * (n_steps - len(per_frame))
    elif len(per_frame) > n_steps:
        per_frame = per_frame[:n_steps]
    return per_frame, timeline


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════
def setup_polar_axes(ax):
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
    ax.set_xlabel(r"$\Re(z)$", fontsize=9)
    ax.set_ylabel(r"$\Im(z)$", fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_inventory(ax, vowels, consonants):
    for k, (rho, theta) in vowels.items():
        x, y = rho * np.cos(theta), rho * np.sin(theta)
        ax.plot(x, y, "o", color="#d62728", markersize=4, alpha=0.25)
        ax.text(x, y, k, color="#aa3333", fontsize=7, alpha=0.4, ha="center", va="bottom")
    for k, (rho, theta) in consonants.items():
        x, y = rho * np.cos(theta), rho * np.sin(theta)
        ax.plot(x, y, "s", color="#1f77b4", markersize=5, alpha=0.25)
        ax.text(x, y, k, color="#3366aa", fontsize=7, alpha=0.4, ha="center", va="bottom")


def plot_static_polar(z_v, z_c, tp_ms, delta, idx, out_path,
                      vowels=None, consonants=None):
    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    setup_polar_axes(ax)
    if vowels and consonants:
        plot_inventory(ax, vowels, consonants)
    valid_v = ~np.isnan(z_v.real)
    ax.plot(z_v[valid_v].real, z_v[valid_v].imag, "-",
            color="#d62728", linewidth=1.6, alpha=0.85, label=r"$z_v$ (vocalic)")
    valid_c = ~np.isnan(z_c.real)
    if np.any(valid_c):
        seg_starts = np.where(valid_c & ~np.roll(valid_c, 1))[0]
        seg_ends = np.where(valid_c & ~np.roll(valid_c, -1))[0]
        for s, e in zip(seg_starts, seg_ends):
            ax.plot(z_c[s:e+1].real, z_c[s:e+1].imag, "-",
                    color="#1f77b4", linewidth=1.8, alpha=0.9)
        ax.plot([], [], "-", color="#1f77b4", linewidth=1.8, label=r"$z_c$ (consonantal)")
    if np.any(valid_v):
        ax.plot(z_v[0].real, z_v[0].imag, "o", color="#aa0000", markersize=10, zorder=5)
        ax.plot(z_v[-1].real, z_v[-1].imag, "s", color="#aa0000", markersize=10, zorder=5)
    pos = u_shape_position(idx, len(TP_MS_VALUES))
    ax.set_title(
        f"Segment {idx+1}/{len(TP_MS_VALUES)} — {pos}\n"
        f"Tp = {tp_ms:.0f} ms    δ = {delta:.3f}\n"
        f"{form_label(tp_ms)}",
        fontsize=10, color="#003366", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.text(0.5, 0.01,
             "U-shaped Tp variation — arXiv:2307.02299 (Berthommier 2023)",
             ha="center", fontsize=8, color="#666666")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_comparison_polar(items, out_path):
    n = len(items)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 3.6 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    vowels, consonants = get_phoneme_inventory()
    for i, (tp_ms, delta, idx, z_v, z_c) in enumerate(items):
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
        ax.set_title(f"Seg {idx+1}: Tp={tp_ms:.0f} ms\nδ={delta:.3f}",
                     fontsize=9, color="#003366", fontweight="bold")
    for j in range(len(items), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(
        "Polar plot — U-shaped Tp variation (ib ib → bi bi → ib ib)\n"
        "arXiv:2307.02299 (Berthommier 2023) — T=16",
        fontsize=11)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Dynamic polar video segment
# ═══════════════════════════════════════════════════════════════════
def render_polar_segment(z_v, z_c, tp_ms, delta, idx, out_mp4,
                          wav_path=None, vowels=None, consonants=None):
    n_steps = len(z_v)
    step_per_out = int(round(SR_GESTURE / FPS_OUT))
    indices = np.arange(0, n_steps, step_per_out)
    if len(indices) == 0 or indices[-1] != n_steps - 1:
        indices = np.append(indices, n_steps - 1)
    n_out = len(indices)
    trail_samples = int(round(TRAIL_MS * SR_GESTURE / 1000 / step_per_out))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = None
    if vowels is None or consonants is None:
        vowels, consonants = get_phoneme_inventory()
    pos = u_shape_position(idx, len(TP_MS_VALUES))
    print(f"  Rendering {n_out} polar frames @ {FPS_OUT} fps "
          f"(seg {idx+1}, Tp={tp_ms:.0f} ms, δ={delta:.3f}, {pos})")
    t_start = time.time()
    for frame_idx in range(n_out):
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        try:
            setup_polar_axes(ax)
            plot_inventory(ax, vowels, consonants)
            start = max(0, frame_idx - trail_samples)
            trail_v = z_v[indices[start:frame_idx+1]]
            trail_c = z_c[indices[start:frame_idx+1]]
            for k in range(len(trail_v) - 1):
                alpha = 0.05 + 0.85 * (k / max(1, len(trail_v) - 1))
                if not np.isnan(trail_v[k].real) and not np.isnan(trail_v[k+1].real):
                    ax.plot([trail_v[k].real, trail_v[k+1].real],
                            [trail_v[k].imag, trail_v[k+1].imag],
                            "-", color="#d62728", linewidth=1.4, alpha=alpha)
            for k in range(len(trail_c) - 1):
                alpha = 0.05 + 0.85 * (k / max(1, len(trail_c) - 1))
                if not np.isnan(trail_c[k].real) and not np.isnan(trail_c[k+1].real):
                    ax.plot([trail_c[k].real, trail_c[k+1].real],
                            [trail_c[k].imag, trail_c[k+1].imag],
                            "-", color="#1f77b4", linewidth=2.0, alpha=alpha)
            if not np.isnan(z_v[indices[frame_idx]].real):
                ax.plot(z_v[indices[frame_idx]].real, z_v[indices[frame_idx]].imag,
                        "o", color="#d62728", markersize=10, zorder=6,
                        markeredgecolor="white", markeredgewidth=1.5)
            if not np.isnan(z_c[indices[frame_idx]].real):
                ax.plot(z_c[indices[frame_idx]].real, z_c[indices[frame_idx]].imag,
                        "o", color="#1f77b4", markersize=12, zorder=6,
                        markeredgecolor="white", markeredgewidth=1.5)
            ax.set_title(
                f"Seg {idx+1}/{len(TP_MS_VALUES)} — {pos}\n"
                f"Tp = {tp_ms:.0f} ms    δ = {delta:.3f}    "
                f"t = {indices[frame_idx] * T_STEP_MS:.0f} ms\n"
                f"{form_label(tp_ms)}",
                fontsize=10, color="#003366", fontweight="bold")
            fig.set_dpi(100)
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            img_array = np.array(buf, dtype=np.uint8)
            h, w = img_array.shape[:2]
            img_bgr = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)
            if video_writer is None:
                video_writer = cv2.VideoWriter(str(out_mp4), fourcc,
                                                float(FPS_OUT), (w, h))
            video_writer.write(img_bgr)
        finally:
            plt.close(fig)
        if (frame_idx + 1) % 50 == 0 or frame_idx == n_out - 1:
            elapsed = time.time() - t_start
            fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"    Frame {frame_idx+1}/{n_out} ({fps:.1f} fps)")
    if video_writer is not None:
        video_writer.release()
    if wav_path is not None and wav_path.exists():
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe:
            dubbed = out_mp4.with_name(out_mp4.stem + "_dubbed.mp4")
            cmd = [ffmpeg_exe, "-y", "-i", str(out_mp4), "-i", str(wav_path),
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ar", "22050",
                   "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                   "-pix_fmt", "yuv420p", str(dubbed)]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                out_mp4.unlink(missing_ok=True)
                dubbed.rename(out_mp4)
            except subprocess.CalledProcessError as e:
                print(f"  Warning: dub failed: {str(e.stderr)[:200]}")


def concat_segments(segments, out_path):
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        return False
    list_file = out_path.parent / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in segments:
            f.write(f"file '{p.resolve()}'\n")
    cmd = [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
           "-i", str(list_file), "-c", "copy", str(out_path)]
    print(f"  Concatenating {len(segments)} segments...")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        cmd_reenc = [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                     "-i", str(list_file),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                     "-c:a", "aac", "-b:a", "128k", "-ar", "22050",
                     "-pix_fmt", "yuv420p", str(out_path)]
        try:
            subprocess.run(cmd_reenc, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  Concat failed: {str(e.stderr)[:300]}")
            return False


# ═══════════════════════════════════════════════════════════════════
# Dual-panel video segment
# ═══════════════════════════════════════════════════════════════════
LEFT_FIG_SIZE = (8.0, 6.0)
RIGHT_FIG_SIZE = (6.0, 6.0)


def render_dual_segment(npz_path, zvc_path, wav_path, tp_ms, delta, idx,
                         result, out_mp4):
    Pval = np.load(npz_path)["Pval"].astype(float)
    if np.any(np.isnan(Pval)) or np.any(np.isinf(Pval)):
        Pval = np.nan_to_num(Pval)
    zvc = np.load(zvc_path)
    z_v, z_c = zvc["z_v"], zvc["z_c"]
    n_steps = len(z_v)

    # Synchronized labels: read back the engine-block-derived labels
    # stored in the zvc npz by main() (single source of truth)
    if "labels" in zvc:
        phoneme_per_frame = [str(x) for x in zvc["labels"]]
        lr, lt = zvc["label_rho"], zvc["label_theta"]
        label_targets = [(float(lr[i]), float(lt[i]))
                         if np.isfinite(lr[i]) and np.isfinite(lt[i])
                         else None for i in range(len(phoneme_per_frame))]
    else:
        phoneme_per_frame, _ = compute_phoneme_timeline(result, tp_ms)
        label_targets = [None] * len(phoneme_per_frame)

    state = VlamState.initial(vocal_tract_length_mm=GUI_LEN_MM)
    states = []
    for k in range(n_steps):
        state = freqevalNN_visual(Pval[k, :], state, VALRECT)
        states.append(state)

    step_per_out = int(round(SR_GESTURE / FPS_OUT))
    indices = np.arange(0, n_steps, step_per_out)
    if len(indices) == 0 or indices[-1] != n_steps - 1:
        indices = np.append(indices, n_steps - 1)
    n_out = len(indices)
    trail_samples = int(round(TRAIL_MS * SR_GESTURE / 1000 / step_per_out))

    vowels, consonants = get_phoneme_inventory()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = None
    pos = u_shape_position(idx, len(TP_MS_VALUES))
    print(f"  Rendering {n_out} dual frames @ {FPS_OUT} fps "
          f"(seg {idx+1}, Tp={tp_ms:.0f} ms, δ={delta:.3f}, {pos})")
    t_start = time.time()

    for frame_idx in range(n_out):
        gesture_idx = indices[frame_idx]
        current_phoneme = (phoneme_per_frame[gesture_idx]
                           if gesture_idx < len(phoneme_per_frame) else "?")
        current_target = (label_targets[gesture_idx]
                          if gesture_idx < len(label_targets) else None)

        fig_l, ax_l = plt.subplots(figsize=LEFT_FIG_SIZE)
        try:
            plot_vocal_tract(states[gesture_idx], ax=ax_l)
            fig_l.set_dpi(100)
        except Exception:
            pass

        fig_r, ax_r = plt.subplots(figsize=RIGHT_FIG_SIZE)
        try:
            setup_polar_axes(ax_r)
            plot_inventory(ax_r, vowels, consonants)
            start = max(0, frame_idx - trail_samples)
            trail_v = z_v[indices[start:frame_idx+1]]
            trail_c = z_c[indices[start:frame_idx+1]]
            for k in range(len(trail_v) - 1):
                alpha = 0.05 + 0.85 * (k / max(1, len(trail_v) - 1))
                if not np.isnan(trail_v[k].real) and not np.isnan(trail_v[k+1].real):
                    ax_r.plot([trail_v[k].real, trail_v[k+1].real],
                              [trail_v[k].imag, trail_v[k+1].imag],
                              "-", color="#d62728", linewidth=1.4, alpha=alpha)
            for k in range(len(trail_c) - 1):
                alpha = 0.05 + 0.85 * (k / max(1, len(trail_c) - 1))
                if (not np.isnan(trail_c[k].real)
                        and not np.isnan(trail_c[k+1].real)):
                    ax_r.plot([trail_c[k].real, trail_c[k+1].real],
                              [trail_c[k].imag, trail_c[k+1].imag],
                              "-", color="#1f77b4", linewidth=2.0, alpha=alpha)
            if not np.isnan(z_v[gesture_idx].real):
                ax_r.plot(z_v[gesture_idx].real, z_v[gesture_idx].imag,
                          "o", color="#d62728", markersize=10, zorder=6,
                          markeredgecolor="white", markeredgewidth=1.5)
            if not np.isnan(z_c[gesture_idx].real):
                ax_r.plot(z_c[gesture_idx].real, z_c[gesture_idx].imag,
                          "o", color="#1f77b4", markersize=12, zorder=6,
                          markeredgecolor="white", markeredgewidth=1.5)
            if current_phoneme and current_phoneme not in ("_", "?"):
                # Contextual target recorded from the engine blocks: the
                # label is drawn at the position the trajectory actually
                # aims at (not the generic inventory entry).
                if current_target is not None:
                    rho, theta = current_target
                    color = "#aa3333" if current_phoneme in vowels else "#3366aa"
                elif current_phoneme in vowels:
                    rho, theta = vowels[current_phoneme]
                    color = "#aa3333"
                elif current_phoneme in consonants:
                    rho, theta = consonants[current_phoneme]
                    color = "#3366aa"
                else:
                    rho, theta = None, None
                if rho is not None:
                    x = rho * np.cos(theta) * 1.15
                    y = rho * np.sin(theta) * 1.15
                    ax_r.text(x, y, f"/{current_phoneme}/",
                              fontsize=18, fontweight="bold",
                              color=color, ha="center", va="center",
                              bbox=dict(boxstyle="round,pad=0.3",
                                        facecolor="white",
                                        edgecolor=color, alpha=0.9))
            ax_r.set_title(f"Polar plot (t = {gesture_idx * T_STEP_MS:.0f} ms)",
                           fontsize=10, color="#003366")
        except Exception:
            pass

        fig_l.set_dpi(100)
        canvas_l = FigureCanvas(fig_l)
        canvas_l.draw()
        buf_l = canvas_l.buffer_rgba()
        img_l = np.array(buf_l, dtype=np.uint8)
        img_l_bgr = cv2.cvtColor(img_l[:, :, :3], cv2.COLOR_RGB2BGR)

        fig_r.set_dpi(100)
        canvas_r = FigureCanvas(fig_r)
        canvas_r.draw()
        buf_r = canvas_r.buffer_rgba()
        img_r = np.array(buf_r, dtype=np.uint8)
        img_r_bgr = cv2.cvtColor(img_r[:, :, :3], cv2.COLOR_RGB2BGR)

        plt.close(fig_l)
        plt.close(fig_r)

        h = max(img_l_bgr.shape[0], img_r_bgr.shape[0])
        w_l = int(img_l_bgr.shape[1] * h / img_l_bgr.shape[0])
        w_r = int(img_r_bgr.shape[1] * h / img_r_bgr.shape[0])
        img_l_resized = cv2.resize(img_l_bgr, (w_l, h))
        img_r_resized = cv2.resize(img_r_bgr, (w_r, h))
        sep = np.full((h, 10, 3), 255, dtype=np.uint8)
        combined = np.concatenate([img_l_resized, sep, img_r_resized], axis=1)

        title_h = 60
        title_bar = np.full((title_h, combined.shape[1], 3), 245, dtype=np.uint8)
        title = (f"Seg {idx+1}/{len(TP_MS_VALUES)} ({pos})    "
                 f"Tp = {tp_ms:.0f} ms    δ = {delta:.3f}    "
                 f"{form_label(tp_ms)}    |    phoneme: /{current_phoneme}/")
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            pil_img = PILImage.fromarray(title_bar)
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            except Exception:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), title, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(((combined.shape[1] - tw) // 2,
                       (title_h - th) // 2 - 2), title,
                      fill=(0, 51, 102), font=font)
            title_bar = np.array(pil_img)
        except ImportError:
            cv2.putText(title_bar, title, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 51, 102), 1, cv2.LINE_AA)

        final_frame = np.concatenate([title_bar, combined], axis=0)

        if video_writer is None:
            h_final, w_final = final_frame.shape[:2]
            video_writer = cv2.VideoWriter(str(out_mp4), fourcc,
                                            float(FPS_OUT), (w_final, h_final))
            print(f"  Video writer: {w_final}x{h_final} @ {FPS_OUT} fps -> {out_mp4.name}")

        video_writer.write(final_frame)

        if (frame_idx + 1) % 50 == 0 or frame_idx == n_out - 1:
            elapsed = time.time() - t_start
            fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"    Frame {frame_idx+1}/{n_out} ({fps:.1f} fps)")

    if video_writer is not None:
        video_writer.release()

    if wav_path.exists():
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe:
            dubbed = out_mp4.with_name(out_mp4.stem + "_dubbed.mp4")
            cmd = [ffmpeg_exe, "-y", "-i", str(out_mp4), "-i", str(wav_path),
                   "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "22050",
                   "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                   "-pix_fmt", "yuv420p", str(dubbed)]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                out_mp4.unlink(missing_ok=True)
                dubbed.rename(out_mp4)
                print(f"  Audio dubbed")
            except subprocess.CalledProcessError as e:
                print(f"  Dub failed: {str(e.stderr)[:200]}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    print("=" * 72)
    print("U-SHAPED Tp VARIATION: ib ib -> bi bi -> ib ib")
    print("arXiv:2307.02299 (Berthommier 2023) — US English, T=16")
    print("=" * 72)
    print(f"  Tp values (ms): {TP_MS_VALUES}")
    print(f"  U-shape: start Tp=160 → middle Tp=0 → end Tp=160")
    print(f"  delta coupled: delta = 0.5 + 0.5*(1 - Tp/160)")
    print(f"  Output: {OUT_BASE}")
    print()

    state = vlam.VlamState.initial(195)
    config = vlam.SynthConfig(play_audio=False)
    vowels, consonants = get_phoneme_inventory()

    polar_items = []
    polar_segments = []
    dual_segments = []

    for i, tp_ms in enumerate(TP_MS_VALUES):
        tag = f"s{i+1:02d}_tp{int(tp_ms):03d}"
        t0 = time.time()
        delta = delta_for_tp_ms(tp_ms)
        print(f"\n[{i+1}/{len(TP_MS_VALUES)}] Tp = {tp_ms:.0f} ms  delta = {delta:.3f}  "
              f"({u_shape_position(i, len(TP_MS_VALUES))})")

        result, factor, delta_actual, blocks = run_pipeline(tp_ms)
        labels, label_targets = polar_sync.build_labels(blocks,
                                                        result.seg_map)

        npz_path = NPZ_DIR / f"params_{tag}.npz"
        np.savez(npz_path, Pval=result.Pval, envelope=result.envelope,
                 tp_ms=tp_ms, delta=delta, factor=factor, T=T_BASE,
                 idx=i, u_position=u_shape_position(i, len(TP_MS_VALUES)))
        print(f"  Pval shape: {result.Pval.shape}  -> {npz_path.name}")

        wav_path = WAV_DIR / f"synth_{tag}.wav"
        sig = synthesize_wav(result, state, config, wav_path)
        print(f"  WAV: {sig.shape} samples ({sig.shape[0]/FS_AUDIO*1000:.0f} ms)")

        z_v, z_c, n_steps = polar_sync.build_branches(
            blocks, result.Pval.shape[0], t_step_ms=T_STEP_MS,
            sr_display=SR_GESTURE, k_v=K_DISPLAY, k_c=K_C_DISPLAY)
        zvc_path = ZVC_DIR / f"zvc_{tag}.npz"
        np.savez(zvc_path, z_v=z_v, z_c=z_c, tp_ms=tp_ms, delta=delta,
                 n_steps=n_steps, idx=i,
                 labels=np.array(labels, dtype="U8"),
                 label_rho=np.array([t[0] if t else np.nan
                                     for t in label_targets]),
                 label_theta=np.array([t[1] if t else np.nan
                                       for t in label_targets]))

        # Static polar
        static_path = POLAR_STATIC / f"polar_{tag}.png"
        plot_static_polar(z_v, z_c, tp_ms, delta, i, static_path,
                          vowels=vowels, consonants=consonants)
        print(f"  Static polar: -> {static_path.name}")

        polar_items.append((tp_ms, delta, i, z_v, z_c))

        # Dynamic polar segment
        seg_mp4 = POLAR_SEG / f"polar_{tag}.mp4"
        render_polar_segment(z_v, z_c, tp_ms, delta, i, seg_mp4,
                             wav_path=wav_path, vowels=vowels, consonants=consonants)
        polar_segments.append(seg_mp4)

        # Dual segment
        dual_mp4 = DUAL_SEG / f"dual_{tag}.mp4"
        render_dual_segment(npz_path, zvc_path, wav_path, tp_ms, delta, i,
                             result, dual_mp4)
        dual_segments.append(dual_mp4)

        print(f"  Done in {time.time()-t0:.1f}s")

    # Comparison polar grid
    comp_path = POLAR_STATIC / "comparison_polar.png"
    plot_comparison_polar(polar_items, comp_path)
    print(f"\nComparison polar figure: {comp_path}")

    # Concatenate polar videos
    final_polar = POLAR_DIR / "polar_progressive.mp4"
    print(f"\n[Final] Concatenating {len(polar_segments)} polar segments -> {final_polar.name}")
    if not concat_segments(polar_segments, final_polar):
        print("  Polar concat failed")
    else:
        size_mb = os.path.getsize(final_polar) / (1024 * 1024)
        print(f"  Final polar video: {final_polar}  ({size_mb:.2f} MB)")

    # Concatenate dual videos
    final_dual = DUAL_DIR / "dual_progressive.mp4"
    print(f"\n[Final] Concatenating {len(dual_segments)} dual segments -> {final_dual.name}")
    if not concat_segments(dual_segments, final_dual):
        print("  Dual concat failed")
    else:
        size_mb = os.path.getsize(final_dual) / (1024 * 1024)
        print(f"  Final dual video: {final_dual}  ({size_mb:.2f} MB)")

    # Probe durations
    if shutil.which("ffprobe"):
        for label, path in [("Polar", final_polar), ("Dual", final_dual)]:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True)
            if probe.returncode == 0:
                dur = float(probe.stdout.strip())
                print(f"  {label} duration: {dur:.2f} s ({dur/60:.1f} min)")

    print()
    print("=" * 72)
    print("SUMMARY — U-shaped Tp variation")
    print("=" * 72)
    print(f"  {'seg':>3}  {'Tp(ms)':>6}  {'delta':>5}  {'position':>20}  "
          f"{'Pval':>8}  {'sig(ms)':>7}  form")
    import subprocess as sp
    for i, tp_ms in enumerate(TP_MS_VALUES):
        tag = f"s{i+1:02d}_tp{int(tp_ms):03d}"
        delta = delta_for_tp_ms(tp_ms)
        pos = u_shape_position(i, len(TP_MS_VALUES))
        wav_file = WAV_DIR / f"synth_{tag}.wav"
        if wav_file.exists():
            r = sp.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(wav_file)], capture_output=True, text=True)
            sig_ms = float(r.stdout.strip()) * 1000 if r.returncode == 0 else 0
        else:
            sig_ms = 0
        npz = np.load(NPZ_DIR / f"params_{tag}.npz")
        print(f"  {i+1:>3}  {tp_ms:>6.0f}  {delta:>5.3f}  {pos:>20}  "
              f"{str(npz['Pval'].shape):>8}  {sig_ms:>7.0f}  {form_label(tp_ms)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
