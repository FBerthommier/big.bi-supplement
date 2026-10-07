#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
run_bigbi_polar_sweep.py — big.bi → bi.gbi transition with polar
coordinates, following arXiv:2307.02299 §4 (Berthommier 2023).

T=16 (didactic mode). The last bi.gbi segment is repeated 3 times for
perceptibility. All output in US English.

DOT FORM (author ruling 2026-10-07): big.bi is synthesized as the
dotted word "big.bi" — the '.' between /g/ and /b/ is a C.C syllable
boundary with NO pause (the reference Timit-to-Maeda semantics: the Ve
of "big" is coarticulated with the /i/ of "bi", the boundary anchor
weighted by COEFCEN). There is therefore NO Tp/pause dimension in this
sweep: δ alone drives the transformation. At δ < 1 the boundary
vocoid is a schwa-like @ (boundary anchor = δ·ρ_i); at δ = 1 it is the
full /i/ (fusion-ready), and bi.gbi — whose V.C dot is ignored, syllables
merged — realizes the fused /gb/ cluster (Sc={1,2,3,6}).

Sweep: 14 segments
  - 11 segments of "big.bi" at delta in {0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95, 0.995}
  - 3 segments of "bi.gbi" at delta = 1.00 (repeated for perceptibility)

Output: <repo>/output/bigbi_polar_sweep/
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
OUT_BASE = Path(__file__).resolve().parent.parent / "output" / "bigbi_polar_sweep"
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

# Sweep configuration following the article §4
# T=16 (didactic mode); δ from 0.70 to 1.00; bi.gbi repeated 3 times at the end
T_BASE = 16  # Maeda time step (was 5; 16 = didactic per the README)
T_STEP_MS = 10  # 1 step = 10 ms
SR_GESTURE = 100.0
K_DISPLAY = 30.0        # vocalic display curvature (Kvoy)
K_C_DISPLAY = 10.0       # consonantal display curvature (article K=10)
SR_DISPLAY = 1000.0      # display sample rate (10x engine for smooth curves)
FPS_OUT = 25.0
FS_AUDIO = 20_000
TRAIL_MS = 500          # fading trail length (ms) — short enough that
                        # past trajectories clearly yield to the current one
GUI_LEN_MM = 195
VALRECT = 1.10  # author ruling 2026-10-07 (was 0.75)

# 11 segments of "big.bi" (delta from 0.50 to 0.995) + 3 segments of
# "bi.gbi" (delta=1.0). DOT FORM: no pause, no Tp — delta alone drives
# the transformation (boundary anchor = delta * rho_i).
DELTA_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.995,
                1.00, 1.00, 1.00]
INPUT_TEXTS = (["big.bi"] * 11) + (["bi.gbi"] * 3)
DELTA_MIN = 0.50  # start at COEFCEN=0.5 (salient boundary schwa @)

MAEDA_LABELS = ["Jaw", "Body", "Dorsum", "Tip", "LipP", "LipH", "Hy"]
MAEDA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2"]


# ─────────────────────────────────────────────────────────────────────
# Helpers (US English labels)
# ─────────────────────────────────────────────────────────────────────
def factor_for_delta(delta: float) -> float:
    """Deprecated: the dot form has NO pause. Kept returning 0.0 for
    backward compatibility (make_bigbi_dual_video.py imports it)."""
    return 0.0


def tp_ms_for_delta(delta: float) -> float:
    """Deprecated: no Tp dimension in the dot-form sweep (the joint Tp
    sweep lost its meaning once big.bi lost its pause — author ruling
    2026-10-07). Always 0.0."""
    return 0.0


def form_label(delta: float, is_fused: bool = False, repeat_idx: int = 0) -> str:
    if is_fused:
        if repeat_idx > 0:
            return f"bi.gbi (repeat {repeat_idx+1}/3) — /gb/ cluster fused, Sc={{1,2,3,6}}"
        return "bi.gbi — /gb/ cluster fused, Sc={1,2,3,6}"
    if delta > 0.95:
        return "big.bi -> bi.gbi (boundary anchor = full V)"
    if delta > 0.85:
        return "big.bi (residual boundary schwa)"
    if delta >= 0.60:
        return "big.bi (audible boundary schwa)"
    return "big.bi (salient boundary schwa @)"  # delta <= 0.60: COEFCEN=0.5


def get_phoneme_inventory():
    vowels = {k: (v["rho"], v["theta"]) for k, v in VOWELS_SYNTSYL.items()}
    consonants = {k: (v["rho"], v["theta"]) for k, v in CONSONANTS_SYNTSYL.items()}
    return vowels, consonants


# ═══════════════════════════════════════════════════════════════════
# Pipeline + synth
# ═══════════════════════════════════════════════════════════════════
def run_pipeline(delta: float, input_text: str):
    """(result, factor, blocks) — blocks recorded via polar_sync so the
    display branches, the phoneme labels AND their polar positions come
    from the engine's actual block sequence (synchronized by design).

    DOT FORM: no pause factors — "big.bi"/"bi.gbi" are single dotted
    words. ``factor`` (returned 0.0) is kept for backward compatibility
    with make_bigbi_dual_video.py."""
    factor = 0.0
    blocks: list = []
    result = polar_sync.record_pipeline(
        panphon_pipeline, blocks,
        text=input_text,
        T=T_BASE,
        delta_o=delta, delta_e=delta,
        verbose=False,
    )
    if result is None:
        raise RuntimeError(f"Pipeline failed for delta={delta}, text={input_text}")
    return result, factor, blocks


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
# Phoneme timeline (for dual video labels)
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
# Plotting — static polar + panels (US English labels)
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


def plot_static_polar(z_v, z_c, delta, is_fused, repeat_idx, out_path,
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
    input_text = "bi.gbi" if is_fused else "big.bi"
    suffix = f" (repeat {repeat_idx+1}/3)" if (is_fused and repeat_idx > 0) else ""
    ax.set_title(
        f"Polar plot — {input_text}{suffix}    δ = {delta:.3f}\n"
        f"{form_label(delta, is_fused, repeat_idx)}",
        fontsize=11, color="#003366", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.text(0.5, 0.01,
             "Branches rebuilt from the engine blocks via polar_sync "
             "(original arcplot anchoring; z_v K=30, z_c K=10, "
             "nu_v=-1/nu_c=+1) — arXiv:2307.02299 §4 (dot form)",
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
    for i, (delta, is_fused, repeat_idx, z_v, z_c) in enumerate(items):
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
        input_text = "bi.gbi" if is_fused else "big.bi"
        suffix = f" r{repeat_idx+1}" if (is_fused and repeat_idx > 0) else ""
        ax.set_title(f"{input_text}{suffix}\nδ = {delta:.3f}",
                     fontsize=9, color="#003366", fontweight="bold")
    for j in range(len(items), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(
        "Polar plot — big.bi → bi.gbi transition (arXiv:2307.02299 §4)\n"
        "(z_v red, z_c blue; K=30, covtl-pipeline) — T=16, bi.gbi repeated 3×",
        fontsize=11)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Dynamic polar video segment
# ═══════════════════════════════════════════════════════════════════
def render_polar_segment(z_v, z_c, delta, is_fused, repeat_idx, out_mp4,
                          wav_path=None, vowels=None, consonants=None):
    n_steps = len(z_v)
    step_per_out = int(round(SR_DISPLAY / FPS_OUT))
    indices = np.arange(0, n_steps, step_per_out)
    if len(indices) == 0 or indices[-1] != n_steps - 1:
        indices = np.append(indices, n_steps - 1)
    n_out = len(indices)
    trail_samples = int(round(TRAIL_MS * SR_DISPLAY / 1000 / step_per_out))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = None
    if vowels is None or consonants is None:
        vowels, consonants = get_phoneme_inventory()
    input_text = "bi.gbi" if is_fused else "big.bi"
    suffix = f" (repeat {repeat_idx+1}/3)" if (is_fused and repeat_idx > 0) else ""
    print(f"  Rendering {n_out} polar frames @ {FPS_OUT} fps "
          f"(δ={delta:.3f}, {input_text}{suffix})")
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
                f"{input_text}{suffix}    δ = {delta:.3f}\n"
                f"{form_label(delta, is_fused, repeat_idx)}    "
                f"t = {indices[frame_idx] / (SR_DISPLAY / 1000.0):.0f} ms",
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
# Dual-panel (sagittal | polar) video segment
# ═══════════════════════════════════════════════════════════════════
LEFT_FIG_SIZE = (8.0, 6.0)
RIGHT_FIG_SIZE = (6.0, 6.0)


def render_dual_segment(npz_path, zvc_path, wav_path, delta, is_fused,
                         repeat_idx, input_text, result, out_mp4,
                         labels=None, label_targets=None):
    Pval = np.load(npz_path)["Pval"].astype(float)
    if np.any(np.isnan(Pval)) or np.any(np.isinf(Pval)):
        Pval = np.nan_to_num(Pval)
    zvc = np.load(zvc_path)
    z_v, z_c = zvc["z_v"], zvc["z_c"]
    n_steps = len(z_v)
    tp_ms = tp_ms_for_delta(delta) if not is_fused else 0.0

    if labels is None and "labels" in zvc:
        labels = [str(x) for x in zvc["labels"]]
        lr, lt = zvc["label_rho"], zvc["label_theta"]
        label_targets = [(float(lr[i]), float(lt[i]))
                         if np.isfinite(lr[i]) and np.isfinite(lt[i])
                         else None for i in range(len(labels))]

    state = VlamState.initial(vocal_tract_length_mm=GUI_LEN_MM)
    states = []
    for k in range(Pval.shape[0]):
        state = freqevalNN_visual(Pval[k, :], state, VALRECT)
        states.append(state)
    # z branches are sampled at SR_DISPLAY = 10 x the engine step rate
    gesture_div = int(round(SR_DISPLAY / SR_GESTURE))

    step_per_out = int(round(SR_DISPLAY / FPS_OUT))
    indices = np.arange(0, n_steps, step_per_out)
    if len(indices) == 0 or indices[-1] != n_steps - 1:
        indices = np.append(indices, n_steps - 1)
    n_out = len(indices)
    trail_samples = int(round(TRAIL_MS * SR_DISPLAY / 1000 / step_per_out))

    vowels, consonants = get_phoneme_inventory()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = None
    suffix = f" (repeat {repeat_idx+1}/3)" if (is_fused and repeat_idx > 0) else ""
    print(f"  Rendering {n_out} dual frames @ {FPS_OUT} fps "
          f"(δ={delta:.3f}, {input_text}{suffix})")
    t_start = time.time()

    for frame_idx in range(n_out):
        disp_idx = int(indices[frame_idx])          # display-rate index
        gesture_idx = min(disp_idx // gesture_div,  # engine-step index
                          Pval.shape[0] - 1)
        current_phoneme = (labels[gesture_idx]
                           if gesture_idx < len(labels) else "?")
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
            if not np.isnan(z_v[disp_idx].real):
                ax_r.plot(z_v[disp_idx].real, z_v[disp_idx].imag,
                          "o", color="#d62728", markersize=10, zorder=6,
                          markeredgecolor="white", markeredgewidth=1.5)
            if not np.isnan(z_c[disp_idx].real):
                ax_r.plot(z_c[disp_idx].real, z_c[disp_idx].imag,
                          "o", color="#1f77b4", markersize=12, zorder=6,
                          markeredgecolor="white", markeredgewidth=1.5)
            if current_phoneme and current_phoneme not in ("_", "?"):
                # Contextual target recorded from the engine (e.g. the
                # palatal /g/ in a front-vowel context): draw the label
                # at the position the trajectory actually aims at, not
                # at the generic inventory entry.
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
        title = (f"{input_text}{suffix}    δ = {delta:.3f}    "
                 f"{form_label(delta, is_fused, repeat_idx)}    |    "
                 f"phoneme: /{current_phoneme}/")
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
    print("TRANSITION big.bi -> bi.gbi  (arXiv:2307.02299 sec 4) — US English")
    print("=" * 72)
    print(f"  T = {T_BASE} (didactic mode)")
    print(f"  delta sweep : {DELTA_VALUES}")
    print(f"  Input texts : {INPUT_TEXTS}")
    print(f"  bi.gbi repeated 3 times at the end for perceptibility")
    print(f"  Output      : {OUT_BASE}")
    print()

    state = vlam.VlamState.initial(195)
    config = vlam.SynthConfig(play_audio=False)
    vowels, consonants = get_phoneme_inventory()

    polar_items = []
    polar_segments = []
    dual_segments = []

    # Track repeat index for bi.gbi
    bigbi_repeat_count = 0

    for i, (delta, input_text) in enumerate(zip(DELTA_VALUES, INPUT_TEXTS)):
        is_fused = (input_text == "bi.gbi")
        if is_fused:
            repeat_idx = bigbi_repeat_count
            bigbi_repeat_count += 1
        else:
            repeat_idx = 0

        # Use a tag that distinguishes repeats
        if is_fused and repeat_idx > 0:
            tag = f"d{int(round(delta * 1000)):03d}_r{repeat_idx+1}"
        else:
            tag = f"d{int(round(delta * 1000)):03d}"

        t0 = time.time()
        print(f"\n[{i+1}/{len(DELTA_VALUES)}] delta = {delta:.3f}  input = {input_text}"
              + (f"  (repeat {repeat_idx+1}/3)" if is_fused else ""))

        result, factor, blocks = run_pipeline(delta, input_text)
        tp_ms = tp_ms_for_delta(delta) if not is_fused else 0.0
        labels, label_targets = polar_sync.build_labels(blocks,
                                                        result.seg_map)

        npz_path = NPZ_DIR / f"params_{tag}.npz"
        np.savez(npz_path, Pval=result.Pval, envelope=result.envelope,
                 delta=delta, factor=factor, tp_ms=tp_ms,
                 input_text=input_text, is_fused=is_fused,
                 repeat_idx=repeat_idx, T=T_BASE)
        print(f"  Pval shape: {result.Pval.shape}  tp={tp_ms:.1f} ms  -> {npz_path.name}")

        wav_path = WAV_DIR / f"synth_{tag}.wav"
        sig = synthesize_wav(result, state, config, wav_path)
        print(f"  WAV: {sig.shape} samples ({sig.shape[0]/FS_AUDIO*1000:.0f} ms) -> {wav_path.name}")

        z_v, z_c, n_steps = polar_sync.build_branches(
            blocks, result.Pval.shape[0], t_step_ms=T_STEP_MS,
            sr_display=SR_DISPLAY, k_v=K_DISPLAY, k_c=K_C_DISPLAY)
        zvc_path = ZVC_DIR / f"zvc_{tag}.npz"
        np.savez(zvc_path, z_v=z_v, z_c=z_c, delta=delta, n_steps=n_steps,
                 input_text=input_text, is_fused=is_fused,
                 repeat_idx=repeat_idx,
                 labels=np.array(labels, dtype="U8"),
                 label_rho=np.array([t[0] if t else np.nan
                                     for t in label_targets]),
                 label_theta=np.array([t[1] if t else np.nan
                                       for t in label_targets]))
        print(f"  z_v: {len(z_v)} samples  z_c active: {np.sum(~np.isnan(z_c.real))}")

        # Static polar
        static_path = POLAR_STATIC / f"polar_{tag}.png"
        plot_static_polar(z_v, z_c, delta, is_fused, repeat_idx, static_path,
                          vowels=vowels, consonants=consonants)
        print(f"  Static polar: -> {static_path.name}")

        polar_items.append((delta, is_fused, repeat_idx, z_v, z_c))

        # Dynamic polar segment
        seg_mp4 = POLAR_SEG / f"polar_{tag}.mp4"
        render_polar_segment(z_v, z_c, delta, is_fused, repeat_idx, seg_mp4,
                             wav_path=wav_path, vowels=vowels, consonants=consonants)
        polar_segments.append(seg_mp4)

        # Dual segment
        dual_mp4 = DUAL_SEG / f"dual_{tag}.mp4"
        render_dual_segment(npz_path, zvc_path, wav_path, delta, is_fused,
                             repeat_idx, input_text, result, dual_mp4,
                             labels=labels, label_targets=label_targets)
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
    print("SUMMARY")
    print("=" * 72)
    print(f"  delta   input     Pval      sig(ms)   form")
    import subprocess as sp
    for i, (delta, input_text) in enumerate(zip(DELTA_VALUES, INPUT_TEXTS)):
        is_fused = (input_text == "bi.gbi")
        repeat_idx = i - 11 if is_fused else 0
        if is_fused and repeat_idx > 0:
            tag = f"d{int(round(delta * 1000)):03d}_r{repeat_idx+1}"
        else:
            tag = f"d{int(round(delta * 1000)):03d}"
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
        print(f"  {delta:.3f}   {input_text:8s}  {str(npz['Pval'].shape):10s}  "
              f"{sig_ms:6.0f}  {form_label(delta, is_fused, repeat_idx)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
