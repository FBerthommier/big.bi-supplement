#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
make_bigbi_dual_video.py — Vidéo bi-pane (sagittal | polaire) pour la
transition big.bi → bi.gbi (arXiv:2307.02299 §4).

Pour chaque δ (12 valeurs de 0.70 à 1.00) :
  1. Charge Pval, z_v, z_c
  2. Re-démarre le pipeline pour récupérer anchors + nodes
  3. Pour chaque frame à 25 fps : rend sagittal + polaire côte à côte
     avec phonème courant
  4. Ajoute l'audio
Concatène les 12 segments.
"""
from __future__ import annotations
import os, sys, shutil, subprocess, time, warnings
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from synthSYL import panphon_pipeline  # noqa: E402
from synthSYL.phonology import VOWELS_SYNTSYL, CONSONANTS_SYNTSYL  # noqa: E402
from vlam import VlamState, plot_vocal_tract  # noqa: E402
from VLAMvidmaker import freqevalNN_visual  # noqa: E402

# Import the reconstruction + timeline from the sweep script
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import run_bigbi_polar_sweep as sp  # noqa: E402

OUT_BASE = sp.OUT_BASE
NPZ_DIR = sp.NPZ_DIR
WAV_DIR = sp.WAV_DIR
ZVC_DIR = sp.ZVC_DIR
DUAL_DIR = OUT_BASE / "dual"
SEG_DIR = DUAL_DIR / "segments"
DUAL_DIR.mkdir(parents=True, exist_ok=True)
SEG_DIR.mkdir(parents=True, exist_ok=True)


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


def render_dual_segment(npz_path, zvc_path, wav_path, delta, is_fused,
                         input_text, result, out_mp4):
    # Delegate to the synchronized renderer of the sweep script
    # (engine-block-derived branches + labels + contextual label
    # positions + display-rate ball indexing).
    return sp.render_dual_segment(npz_path, zvc_path, wav_path, delta,
                                  is_fused, 0, input_text, result, out_mp4)



def concat_segments(segments, out_path):
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        return False
    list_file = SEG_DIR / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in segments:
            f.write(f"file '{p.resolve()}'\n")
    cmd = [ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
           "-i", str(list_file), "-c", "copy", str(out_path)]
    print(f"  Concatenating {len(segments)} dual segments...")
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
            print(f"  ✗ Concat failed: {str(e.stderr)[:300]}")
            return False


def main() -> int:
    print("=" * 72)
    print("VIDÉO BI-PANE big.bi → bi.gbi  (sagittal | polaire + phonème)")
    print("=" * 72)
    print(f"  δ sweep       : {sp.DELTA_VALUES}")
    print(f"  Output         : {DUAL_DIR}")
    print()

    segment_paths = []
    for i, (delta, input_text) in enumerate(zip(sp.DELTA_VALUES, sp.INPUT_TEXTS)):
        is_fused = (input_text == "bi.gbi")
        tag = f"d{int(round(delta * 1000)):03d}"
        t0 = time.time()
        print(f"[{i+1}/{len(sp.DELTA_VALUES)}] δ = {delta:.3f}  input = {input_text}")

        npz_path = NPZ_DIR / f"params_{tag}.npz"
        zvc_path = ZVC_DIR / f"zvc_{tag}.npz"
        wav_path = WAV_DIR / f"synth_{tag}.wav"
        seg_mp4 = SEG_DIR / f"dual_{tag}.mp4"
        if not npz_path.exists() or not zvc_path.exists() or not wav_path.exists():
            print(f"  ✗ Missing inputs")
            return 1

        factor = sp.factor_for_delta(delta)
        result, factor, blocks = sp.run_pipeline(delta, input_text)

        render_dual_segment(npz_path, zvc_path, wav_path, delta, is_fused,
                             input_text, result, seg_mp4)
        size_kb = os.path.getsize(seg_mp4) / 1024
        print(f"  ✓ Segment: {seg_mp4.name} ({size_kb:.0f} KB)  [{time.time()-t0:.1f}s]")
        segment_paths.append(seg_mp4)

    final_mp4 = DUAL_DIR / "dual_progressive.mp4"
    print(f"\n[Final] Concatenating {len(segment_paths)} dual segments → {final_mp4.name}")
    if not concat_segments(segment_paths, final_mp4):
        print("  ✗ Concat failed")
        return 1
    size_mb = os.path.getsize(final_mp4) / (1024 * 1024)
    print(f"  ✓ Final video: {final_mp4}  ({size_mb:.2f} MB)")

    if shutil.which("ffprobe"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(final_mp4)],
            capture_output=True, text=True)
        if probe.returncode == 0:
            dur = float(probe.stdout.strip())
            print(f"  Duration: {dur:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
