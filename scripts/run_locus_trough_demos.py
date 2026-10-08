#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
run_locus_trough_demos.py — Two additional demonstrations from
arXiv:2307.02299:

Demo 2 (Figure 3): Locus equations for /b/, /d/, /g/ across 8 vowels
  - F2_locus measured 30 ms after the consonant RELEASE, i.e. after the
    start of the C->V transition block (the article: "The Figure 3 is
    constructed by taking the F2 values 30 ms after the onset for the
    8 vowels and delta_o = 0.5" — "onset" = onset of the vowel at the
    release, per the locus-equation tradition of F2-onset measurement)
  - F2_vowel = mean steady-state F2 over the vowel plateau (envelope
    at full amplitude)
  - Plot F2_locus vs F2_vowel (locus equation: F2_locus = a*F2_vowel + b)
  - Uses delta_o = 0.5 (per article)
  - /g/ is velar for back vowels {u, o, O} and palatal for front vowels
    {i, e, è, y} AND /a/ (in the article's Figure 3 the /a/ point lies on
    the palatal branch), with the article positions:
    velar (rho=1.2, theta=pi/3), palatal (rho=1.1, theta=23*pi/12)

  Measurement timeline for a CV word (T = 16 frames of 10 ms):
    frames  0..T-1   initial arc (neutral -> Vo), envelope = 0
    frames T..2T-1   Vo -> C closure,             envelope = 0
    frames 2T..3T-1  C -> V transition (RELEASE at 2T), envelope rises
    frames 3T..4T-1  vowel plateau (envelope = 1)
  Hence F2_locus is taken at frame 2T + 3 (30 ms after the release) and
  F2_vowel is averaged over the plateau frames 3T..4T-1.

  An earlier version of this script derived the measurement frame from
  `anchor.i * T` (a NODE index, not a time) and therefore sampled F2
  during the silent Vo->C closure (frames T+3 / T+8). This produced a
  spurious /b/-/d/ inversion (/d/ slope 1.633). See docs/manual.tex,
  Demo 2 section, for the root-cause analysis and the sensitivity sweep.

Demo 4 (Figure 1b): Trough effect
  - The Body parameter (Maeda index 2, Pval column index 1) makes an
    unexpected front/back movement during /b/
  - This happens because /b/ is at (rho=1, theta=pi/3) = same position
    as /u/ in the complex plane
  - During /b/, the Body follows zc(t) toward /u/ while vowel
    articulators Sv={3,4,5,7} stay around /i/
  - The Body excursion is a PEAK (rise toward the /u/ direction:
    +1.25 from the /i/ plateau -2.25), not a dip: the "trough" of
    Lindblom et al. (2002) is a discontinuity in anticipatory
    coarticulation, and the sign of the simulated excursion follows
    the vowel context (peak over /i/, weak bump over /a/)

Article parameters: T=16, K=10, Kvoy=30, Pexp=1, nu=-1 (already set
in synthSYL/constants.py).

Output: <repo>/docs/figures/demo2_locus/
        <repo>/docs/figures/demo4_trough/
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────────────────────────────
# Paths: run from the repository (works on any clone, no /home/z deps)
# ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import polar_sync  # noqa: E402  (record_pipeline for the trough blocks)
try:
    from config import (  # noqa: E402
        T_BASE, T_STEP_MS, FS_AUDIO, GUI_LEN_MM, VALRECT,
        LOCUS_OFFSET_STEPS,
    )
except ImportError:  # standalone fallback if config.py is absent
    T_BASE = 16
    T_STEP_MS = 10
    FS_AUDIO = 20_000
    GUI_LEN_MM = 195
    VALRECT = 1.10  # author ruling 2026-10-07 (was 0.75)
    LOCUS_OFFSET_STEPS = 3

from synthSYL import panphon_pipeline  # noqa: E402
from synthSYL.types import SegmentDescription  # noqa: E402
from synthSYL.phonology import (  # noqa: E402
    VOWELS_SYNTSYL, CONSONANTS_SYNTSYL, SEGMENT_REGISTRY,
)
import vlam  # noqa: E402

SR_GESTURE = 100.0
K_DISPLAY = 30.0        # vocalic display curvature (Kvoy)
K_C_DISPLAY = 10.0       # consonantal display curvature (article K=10)

OUT_BASE = REPO_ROOT / "docs" / "figures"
DEMO2_DIR = OUT_BASE / "demo2"
DEMO4_DIR = OUT_BASE / "demo4"

for d in (DEMO2_DIR, DEMO4_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAEDA_LABELS = ["Jaw", "Body", "Dorsum", "Tip", "LipP", "LipH", "Hy"]
MAEDA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2"]

# Consonants to test (article: /b/, /d/, /g/)
CONSONANTS_TO_TEST = ["b", "d", "g"]

# 8 main vowels from synthSYL (article: /i, e, epsilon, a, o, O, u, y/)
VOWELS_TO_TEST = ["i", "e", "è", "a", "o", "O", "u", "y"]

# /g/ allophony (article §3 + Figure 3): velar for back vowels, palatal
# for front vowels. In the article's Figure 3 the /a/ point (F2v ~ 1780 Hz)
# lies on the palatal regression line, so /a/ is grouped with the front
# (palatal) vowels here.
G_VELAR_VOWELS = {"o", "O", "u"}
G_PALATAL_VOWELS = {"i", "e", "è", "é", "E", "y", "a"}

# VCV sequences for trough effect
TROUGH_VCV = [
    ("ibi", "/i/ - /b/ - /i/"),
    ("aba", "/a/ - /b/ - /a/"),
    ("idi", "/i/ - /d/ - /i/"),
    ("igi", "/i/ - /g/ - /i/"),
]


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def get_phoneme_inventory():
    vowels = {k: (v["rho"], v["theta"]) for k, v in VOWELS_SYNTSYL.items()}
    consonants = {k: (v["rho"], v["theta"]) for k, v in CONSONANTS_SYNTSYL.items()}
    return vowels, consonants


def set_consonant_position(cons: str, rho: float, theta: float) -> None:
    """Override SEGMENT_REGISTRY[cons] with article polar coordinates,
    preserving key/IPA/selection vector."""
    old = SEGMENT_REGISTRY[cons]
    SEGMENT_REGISTRY[cons] = SegmentDescription(
        key=old.key, IPA=old.IPA, rho=rho, theta=theta,
        is_vowel=old.is_vowel, params=old.params, features=old.features)


# ═══════════════════════════════════════════════════════════════════
# DEMO 2: Locus equations (Figure 3)
# ═══════════════════════════════════════════════════════════════════
def run_demo2_locus():
    """Figure 3: Locus equations for /b/, /d/, /g/ across 8 vowels."""
    print("=" * 72)
    print("DEMO 2: Locus equations (Figure 3) — with article consonant positions")
    print("=" * 72)
    print(f"  T={T_BASE}, K=10, Kvoy=30, Pexp=1 (article parameters)")
    print(f"  delta_o = 0.5  |  valrect (soft_rect_s) = {VALRECT}")
    print(f"  F2_locus = F2 at release + {LOCUS_OFFSET_STEPS} steps "
          f"({LOCUS_OFFSET_STEPS * T_STEP_MS} ms), release = start of C->V block")
    print(f"  F2_vowel = mean F2 over the vowel plateau (envelope = 1)")
    print(f"  Consonants: {CONSONANTS_TO_TEST}")
    print(f"  Vowels: {VOWELS_TO_TEST}")
    print()
    print("  Article consonant positions (§3):")
    print("    /b/: rho=1.0,  theta=pi/3     (was rho=1.2 in synthSYL)")
    print("    /d/: rho=1.2,  theta=3*pi/2   (was theta=23*pi/16 in synthSYL)")
    print("    /g/ velar (back vowels):    rho=1.2,  theta=pi/3")
    print("    /g/ palatal (front + /a/):  rho=1.1,  theta=23*pi/12")
    print()

    state = vlam.VlamState.initial(GUI_LEN_MM)
    config = vlam.SynthConfig(play_audio=False)

    original_registry = {k: SEGMENT_REGISTRY[k] for k in CONSONANTS_TO_TEST}
    sps = int(T_STEP_MS / 1000 * FS_AUDIO)   # audio samples per 10-ms step

    locus_data = {}

    try:
        for cons in CONSONANTS_TO_TEST:
            print(f"\n  Consonant /{cons}/:")
            locus_data[cons] = []

            for vowel in VOWELS_TO_TEST:
                input_text = cons + vowel
                print(f"    CV = {input_text:4s}  ", end="")

                # Article consonant positions
                if cons == 'b':
                    set_consonant_position('b', 1.0, np.pi / 3)
                elif cons == 'd':
                    set_consonant_position('d', 1.2, 3 * np.pi / 2)
                elif cons == 'g':
                    if vowel in G_PALATAL_VOWELS:
                        set_consonant_position('g', 1.1, 23 * np.pi / 12)
                        g_type = 'palatal'
                    else:
                        set_consonant_position('g', 1.2, np.pi / 3)
                        g_type = 'velar'
                    print(f"(/g/ {g_type})  ", end="")

                result = panphon_pipeline(
                    input_text, T=T_BASE,
                    delta_o=0.5, delta_e=0.5,
                    verbose=False,
                )
                if result is None:
                    print("FAILED")
                    continue

                sr = vlam.synthwordfen(
                    state=state, articulatory_params=result.Pval,
                    word_tokens=["O", "V", "F"],
                    f0_scale=1.0, soft_rect_s=VALRECT,
                    duration_factor=T_BASE, envelope=result.envelope,
                    config=config,
                )
                formants = sr.formants
                env_steps = result.envelope[::sps]

                # Release = last silent frame before the envelope rises
                # (start of the C->V transition block). For a CV word this
                # is frame 2T; detected from the envelope to stay robust.
                release = None
                for f in range(2, len(env_steps)):
                    if env_steps[f] > 1e-3 and env_steps[f - 1] <= 1e-3:
                        release = f - 1
                        break
                # Vowel plateau = frames at full amplitude
                on = np.nonzero(env_steps > 0.999)[0]

                if release is None or len(on) == 0:
                    print("no release/plateau found — FAILED")
                    continue

                locus_frame = min(release + LOCUS_OFFSET_STEPS, formants.shape[0] - 1)
                f2_locus = formants[locus_frame, 1]
                f2_vowel = float(np.mean(formants[on.min():on.max() + 1, 1]))

                # For /g/, record the type (palatal/velar)
                g_type = ''
                if cons == 'g':
                    g_type = ' (anterior)' if vowel in G_PALATAL_VOWELS \
                        else ' (posterior)'

                locus_data[cons].append((vowel, f2_vowel, f2_locus, g_type))
                print(f"F2_vowel={f2_vowel:.0f} Hz  "
                      f"F2_locus={f2_locus:.0f} Hz{g_type}")
    finally:
        # Restore the registry (demo 4 re-sets what it needs explicitly)
        SEGMENT_REGISTRY.update(original_registry)

    # ── Plot locus equations ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    consonant_colors = {"b": "#1f77b4", "d": "#ff7f0e", "g": "#2ca02c"}
    consonant_labels = {"b": "/b/", "d": "/d/", "g": "/g/"}

    for i, cons in enumerate(CONSONANTS_TO_TEST):
        ax = axes[i]
        data = locus_data[cons]

        if cons == 'g':
            ant_data = [(v, fv, fl, gt) for v, fv, fl, gt in data if '(anterior)' in gt]
            post_data = [(v, fv, fl, gt) for v, fv, fl, gt in data if '(posterior)' in gt]

            if ant_data:
                fv_arr = np.array([d[1] for d in ant_data])
                fl_arr = np.array([d[2] for d in ant_data])
                ax.scatter(fv_arr, fl_arr, c='#2ca02c', s=100, zorder=5,
                           edgecolors="white", linewidth=1.5, label='/g/ palatal (front + /a/)')
                if len(fv_arr) >= 2:
                    coeffs = np.polyfit(fv_arr, fl_arr, 1)
                    a, b = coeffs
                    x_fit = np.linspace(fv_arr.min() - 100, fv_arr.max() + 100, 100)
                    ax.plot(x_fit, a * x_fit + b, "--", color='#2ca02c', linewidth=1.5,
                            label=f'palatal: y={a:.3f}x+{b:.0f}')

            if post_data:
                fv_arr = np.array([d[1] for d in post_data])
                fl_arr = np.array([d[2] for d in post_data])
                ax.scatter(fv_arr, fl_arr, c='#d62728', s=100, zorder=5,
                           edgecolors="white", linewidth=1.5, label='/g/ velar (back)')
                if len(fv_arr) >= 2:
                    coeffs = np.polyfit(fv_arr, fl_arr, 1)
                    a, b = coeffs
                    x_fit = np.linspace(fv_arr.min() - 100, fv_arr.max() + 100, 100)
                    ax.plot(x_fit, a * x_fit + b, "--", color='#d62728', linewidth=1.5,
                            label=f'velar: y={a:.3f}x+{b:.0f}')
        else:
            f2_vowels = np.array([d[1] for d in data])
            f2_loci = np.array([d[2] for d in data])

            ax.scatter(f2_vowels, f2_loci, c=consonant_colors[cons], s=100, zorder=5,
                       edgecolors="white", linewidth=1.5)
            for j, (vowel, fv, fl, _) in enumerate(data):
                ax.annotate(f"/{vowel}/", (fv, fl), textcoords="offset points",
                            xytext=(5, 5), fontsize=9, color="#333333")

            if len(f2_vowels) >= 2:
                coeffs = np.polyfit(f2_vowels, f2_loci, 1)
                a, b = coeffs
                x_fit = np.linspace(f2_vowels.min() - 100, f2_vowels.max() + 100, 100)
                ax.plot(x_fit, a * x_fit + b, "--", color=consonant_colors[cons],
                        linewidth=1.5, alpha=0.7,
                        label=f"y = {a:.3f}x + {b:.0f}")

        # Identity line
        all_fv = np.array([d[1] for d in data])
        all_fl = np.array([d[2] for d in data])
        lim = [min(all_fv.min(), all_fl.min()) - 200, max(all_fv.max(), all_fl.max()) + 200]
        ax.plot(lim, lim, ":", color="#999999", linewidth=1, alpha=0.5, label="y = x")

        ax.set_xlabel("F2 vowel (Hz)", fontsize=11)
        ax.set_ylabel(f"F2 locus (Hz, {LOCUS_OFFSET_STEPS * T_STEP_MS} ms after release)", fontsize=11)
        ax.set_title(f"Locus equation for {consonant_labels[cons]}",
                     fontsize=13, fontweight="bold", color=consonant_colors[cons])
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle(
        "Figure 3: Locus equations for /b/, /d/, /g/ across 8 vowels\n"
        "arXiv:2307.02299 (Berthommier 2023) — "
        "T=16, K=10, Kvoy=30, Pexp=1, delta_o=0.5\n"
        "F2_locus 30 ms after consonant release; F2_vowel = plateau mean\n"
        "Article consonant positions: /b/ rho=1.0, /d/ theta=3pi/2, /g/ split (velar/palatal, /a/ palatal)",
        fontsize=11, fontweight="bold")
    fig_path = DEMO2_DIR / "locus_equations.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")

    # ── Combined plot ──
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for cons in CONSONANTS_TO_TEST:
        data = locus_data[cons]
        if cons == 'g':
            ant = [(d[1], d[2]) for d in data if '(anterior)' in d[3]]
            post = [(d[1], d[2]) for d in data if '(posterior)' in d[3]]
            if ant:
                fv, fl = zip(*ant)
                ax.scatter(fv, fl, c='#2ca02c', s=100, zorder=5,
                           edgecolors="white", linewidth=1.5, label='/g/ palatal (front + /a/)')
                if len(fv) >= 2:
                    c = np.polyfit(fv, fl, 1)
                    x = np.linspace(min(fv)-100, max(fv)+100, 100)
                    ax.plot(x, c[0]*x+c[1], "--", color='#2ca02c', linewidth=1.5, alpha=0.7)
            if post:
                fv, fl = zip(*post)
                ax.scatter(fv, fl, c='#d62728', s=100, zorder=5,
                           edgecolors="white", linewidth=1.5, label='/g/ velar (back)')
                if len(fv) >= 2:
                    c = np.polyfit(fv, fl, 1)
                    x = np.linspace(min(fv)-100, max(fv)+100, 100)
                    ax.plot(x, c[0]*x+c[1], "--", color='#d62728', linewidth=1.5, alpha=0.7)
        else:
            f2_vowels = np.array([d[1] for d in data])
            f2_loci = np.array([d[2] for d in data])
            ax.scatter(f2_vowels, f2_loci, c=consonant_colors[cons], s=100,
                       zorder=5, edgecolors="white", linewidth=1.5,
                       label=consonant_labels[cons])
            if len(f2_vowels) >= 2:
                coeffs = np.polyfit(f2_vowels, f2_loci, 1)
                x_fit = np.linspace(f2_vowels.min() - 100, f2_vowels.max() + 100, 100)
                ax.plot(x_fit, coeffs[0] * x_fit + coeffs[1], "--",
                        color=consonant_colors[cons], linewidth=1.5, alpha=0.7)

    all_f2 = np.concatenate([np.array([d[1] for d in locus_data[c]]) for c in CONSONANTS_TO_TEST])
    all_loci = np.concatenate([np.array([d[2] for d in locus_data[c]]) for c in CONSONANTS_TO_TEST])
    lim = [min(all_f2.min(), all_loci.min()) - 200, max(all_f2.max(), all_loci.max()) + 200]
    ax.plot(lim, lim, ":", color="#999999", linewidth=1, alpha=0.5, label="y = x")

    ax.set_xlabel("F2 vowel (Hz)", fontsize=12)
    ax.set_ylabel(f"F2 locus (Hz, {LOCUS_OFFSET_STEPS * T_STEP_MS} ms after release)", fontsize=12)
    ax.set_title("Locus equations — combined (with /g/ velar/palatal split)\n"
                 "(slope < 1 = coarticulation, slope = 1 = no coarticulation)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig_path2 = DEMO2_DIR / "locus_equations_combined.png"
    fig.savefig(fig_path2, dpi=120)
    plt.close(fig)
    print(f"  Combined figure: {fig_path2}")

    # Print summary
    print("\n  Locus equation summary:")
    print(f"  {'Cons':>5} {'Group':>12} {'Slope':>8} {'Intercept':>10} {'r2':>6}")
    for cons in CONSONANTS_TO_TEST:
        data = locus_data[cons]
        if cons == 'g':
            for group_name, group_filter in [('palatal', '(anterior)'), ('velar', '(posterior)')]:
                group_data = [(d[1], d[2]) for d in data if group_filter in d[3]]
                if len(group_data) >= 2:
                    fv = np.array([d[0] for d in group_data])
                    fl = np.array([d[1] for d in group_data])
                    coeffs = np.polyfit(fv, fl, 1)
                    a, b = coeffs
                    y_pred = a * fv + b
                    ss_res = np.sum((fl - y_pred) ** 2)
                    ss_tot = np.sum((fl - np.mean(fl)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    print(f"  /{cons}/  {group_name:>12} {a:>8.3f} {b:>10.0f} {r2:>6.3f}")
        else:
            f2_vowels = np.array([d[1] for d in data])
            f2_loci = np.array([d[2] for d in data])
            if len(f2_vowels) >= 2:
                coeffs = np.polyfit(f2_vowels, f2_loci, 1)
                a, b = coeffs
                y_pred = a * f2_vowels + b
                ss_res = np.sum((f2_loci - y_pred) ** 2)
                ss_tot = np.sum((f2_loci - np.mean(f2_loci)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                print(f"  /{cons}/  {'single':>12} {a:>8.3f} {b:>10.0f} {r2:>6.3f}")


# ═══════════════════════════════════════════════════════════════════
# DEMO 4: Trough effect (Figure 1b)
# ═══════════════════════════════════════════════════════════════════
def run_demo4_trough():
    """Figure 1b: Trough effect — Body parameter during /b/."""
    print("\n" + "=" * 72)
    print("DEMO 4: Trough effect (Figure 1b)")
    print("=" * 72)
    print(f"  The Body parameter (Pval column 1 = Maeda Body) makes an")
    print(f"  unexpected front/back movement during /b/ because /b/ is at")
    print(f"  (rho=1, theta=pi/3) = same position as /u/ in the complex plane.")
    print()

    # Article position for /b/ (§3): (rho=1, theta=pi/3) — same point as
    # /u/. The synthSYL default is rho=1.2; set the article value here so
    # that this demo does not depend on SEGMENT_REGISTRY state left over
    # from demo 2.
    set_consonant_position('b', 1.0, np.pi / 3)

    state = vlam.VlamState.initial(GUI_LEN_MM)
    config = vlam.SynthConfig(play_audio=False)
    vowels, consonants = get_phoneme_inventory()

    all_results = []

    for input_text, description in TROUGH_VCV:
        print(f"  [{input_text}] {description}")
        blocks: list = []
        result = polar_sync.record_pipeline(
            panphon_pipeline, blocks, text=input_text, T=T_BASE,
            verbose=False)
        if result is None:
            continue

        sr = vlam.synthwordfen(
            state=state, articulatory_params=result.Pval,
            word_tokens=["O", "V", "F"],
            f0_scale=1.0, soft_rect_s=VALRECT,
            duration_factor=T_BASE, envelope=result.envelope,
            config=config,
        )
        formants = sr.formants
        all_results.append((input_text, description, result, formants, blocks))
        print(f"    Pval: {result.Pval.shape}  Formants: {formants.shape}")

    # ── Figure 1: Body parameter (trough effect) for each VCV ──
    n = len(all_results)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, (input_text, desc, result, formants, blocks) in enumerate(all_results):
        ax = axes_flat[i]
        P = result.Pval
        t_ms = np.arange(P.shape[0]) * T_STEP_MS

        # Plot all 7 params with Body highlighted
        for j in range(7):
            if j == 1:  # Body (Pval column 1 = Maeda Body)
                ax.plot(t_ms, P[:, j], color="#ff7f0e", linewidth=3.0,
                        alpha=1.0, label=f"{MAEDA_LABELS[j]} (excursion)",
                        zorder=5)
            else:
                ax.plot(t_ms, P[:, j], color=MAEDA_COLORS[j], linewidth=1.0,
                        alpha=0.5, label=MAEDA_LABELS[j])

        # Consonant region shading: the RECORDED cluster blocks (the
        # frames the engine actually walked through the consonant),
        # not a hard-coded window.
        f0 = 0
        for b in blocks:
            if b["kind"] == "cluster":
                ax.axvspan(t_ms[f0], t_ms[f0 + b["n"] - 1],
                           alpha=0.15, color="#ff7f0e",
                           label="consonant region (cluster blocks)")
            f0 += b["n"]

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Maeda parameter value")
        ax.set_title(f"{input_text} — {desc}\n"
                     f"Body (orange) excursion toward /u/ during the "
                     f"consonant",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.3)

    for j in range(len(all_results), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Trough effect: Body parameter variation during /b/, /d/, /g/\n"
        "arXiv:2307.02299 (Berthommier 2023, Fig. 1b) — "
        "T=16, K=10, Kvoy=30, Pexp=1",
        fontsize=12, fontweight="bold")
    fig_path = DEMO4_DIR / "trough_effect.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")

    # ── Figure 2: Body parameter only, overlay for all VCV ──
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for i, (input_text, desc, result, formants, blocks) in enumerate(all_results):
        P = result.Pval
        t_ms = np.arange(P.shape[0]) * T_STEP_MS
        body = P[:, 1]  # Body parameter (Pval column 1)
        ax.plot(t_ms, body, color=colors[i], linewidth=2.0,
                label=f"{input_text} ({desc})")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Body parameter (tongue body)")
    ax.set_title("Trough effect — Body parameter overlay\n"
                 "(during /b/ the Body rises toward the /u/ direction "
                 "[29]; over /a/ the excursion is weak)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig_path2 = DEMO4_DIR / "trough_effect_overlay.png"
    fig.savefig(fig_path2, dpi=120)
    plt.close(fig)
    print(f"  Overlay saved: {fig_path2}")

    # ── Figure 3: Polar trajectory showing /b/ at /u/ position ──
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
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
        ax.plot(x, y, "o", color="#d62728", markersize=5, alpha=0.3)
        ax.text(x, y, k, color="#aa3333", fontsize=8, alpha=0.5,
                ha="center", va="bottom")
    for k, (rho, theta) in consonants.items():
        x, y = rho * np.cos(theta), rho * np.sin(theta)
        ax.plot(x, y, "s", color="#1f77b4", markersize=6, alpha=0.3)
        ax.text(x, y, k, color="#3366aa", fontsize=8, alpha=0.5,
                ha="center", va="bottom")

    # Highlight the /b/ and /u/ overlap
    bx, by = 1.0 * np.cos(np.pi / 3), 1.0 * np.sin(np.pi / 3)   # article /b/
    u_rho, u_theta = vowels["u"]
    ux, uy = u_rho * np.cos(u_theta), u_rho * np.sin(u_theta)
    ax.plot(bx, by, "s", color="#ff7f0e", markersize=15, zorder=10,
            markeredgecolor="red", markeredgewidth=2)
    ax.plot(ux, uy, "o", color="#ff7f0e", markersize=12, zorder=10,
            markeredgecolor="red", markeredgewidth=2)
    ax.annotate("/b/ target\n(same position as /u/)", (bx, by),
                textcoords="offset points", xytext=(50, 35),
                fontsize=10, color="#ff7f0e", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e",
                                connectionstyle="arc3,rad=0.3"))
    ax.annotate("/u/ vowel\n(same position)", (ux, uy),
                textcoords="offset points", xytext=(-90, -35),
                fontsize=10, color="#ff7f0e", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e",
                                connectionstyle="arc3,rad=-0.3"))

    ax.set_title("Why the trough effect occurs:\n"
                 "/b/ is at the same position as /u/ in the complex plane\n"
                 "→ Body follows zc(t) toward /u/ during /b/",
                 fontsize=11, fontweight="bold", color="#003366")
    fig_path3 = DEMO4_DIR / "trough_effect_polar_explanation.png"
    fig.savefig(fig_path3, dpi=120)
    plt.close(fig)
    print(f"  Polar explanation: {fig_path3}")

    # ── Detailed block-based analysis (per VCV) ──
    # The old argmin analysis scanned frames T..3T and mistook the /i/
    # plateau (Body = -2.25) for the trough. The excursion is measured
    # inside the RECORDED cluster blocks against the preceding vowel
    # plateau instead.
    if all_results:
        print("\n  Block-based Body analysis (plateau -> cluster extremum):")
        for input_text, desc, result, formants, blocks in all_results:
            P = result.Pval
            body = P[:, 1]
            t_ms = np.arange(P.shape[0]) * T_STEP_MS
            ranges, f0 = [], 0
            for b in blocks:
                ranges.append((b["kind"], f0, f0 + b["n"]))
                f0 += b["n"]
            plateaus = [(s, e) for k, s, e in ranges if k == "plateau"]
            for k, s, e in ranges:
                if k != "cluster":
                    continue
                prev = [(ps, pe) for ps, pe in plateaus if pe <= s]
                ref = body[prev[-1][0]:prev[-1][1]].mean() if prev else np.nan
                seg = body[s:e]
                i_ext = s + int(np.argmax(np.abs(seg - ref)))
                ext = body[i_ext]
                print(f"    {input_text}: plateau Body = {ref:+.3f} | "
                      f"cluster extremum {ext:+.3f} at "
                      f"t={t_ms[i_ext]:.0f} ms "
                      f"(amplitude {ext - ref:+.3f}, "
                      f"{'PEAK' if ext > ref else 'DIP'})")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("LOCUS EQUATIONS + TROUGH EFFECT DEMOS")
    print(f"(arXiv:2307.02299, Berthommier 2023)")
    print(f"T={T_BASE}, K=10, Kvoy=30, Pexp=1, nu=-1 (article parameters)")
    print(f"Output: {OUT_BASE}")
    print("=" * 72)

    # Demo 2: Locus equations
    run_demo2_locus()

    # Demo 4: Trough effect
    run_demo4_trough()

    print("\n" + "=" * 72)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 72)
    print(f"  Demo 2 (locus equations): {DEMO2_DIR}")
    print(f"  Demo 4 (trough effect):   {DEMO4_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
