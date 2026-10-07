#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
make_article_figures.py — Figures 1, 3 and 4 of arXiv:2307.02299 in the
article's own figure format, plus the article-conditions planning
panels previously produced by workspace-only scripts.

Outputs
-------
docs/figures/article_format/
  figure1_ibia_4panel.png        Fig. 1 — /ibia/, T=100 ms (T=10 steps),
                                 4 stacked panels: (a) polar planning
                                 (plt.polar(angle, |z|), r-grid 0.5/1/1.5),
                                 (b) 7 Maeda parameters with phoneme ticks
                                 /i/ /b/ /i/ /a/, (c) F1-F2-F3, (d)
                                 spectrogram + wave.
  figure3_locus_equations.png    Fig. 3 — locus equations /b,d,g/ on the
  figure3_locus_combined.png     8 vowels (same layout/measurement as
                                 run_locus_trough_demos demo 2, article
                                 titles).
  figure4_bigbi_bigbi.png        Fig. 4 — big.bi | bi.gbi under the
                                 article figure conditions COEFCEN=1
                                 (delta_e), VOYDEB=0.5 (delta_o), T=16:
                                 top polar planning panels (r-grid
                                 0.3..1.8, 30-degree angle ticks, teardrop
                                 z_c, red z_v), middle Maeda parameters
                                 with phoneme ticks, bottom aligned
                                 wave + spectrogram.
docs/figures/                    (raft refresh, same content as the
  fig_bigbi_article_conditions_nu_inv.png   validated workspace figures)
  fig_words_article_conditions_nu_inv.png
  planning_4panels_syntsyl.png

The display branches are rebuilt from the RECORDED engine blocks
(polar_sync) with the original Syllable_Synthesis arcplot anchoring —
teardrop z_c (K=10), arcplot z_v (K=30), display nu_v=-1 / nu_c=+1;
see docs/DISPLAY_VS_ENGINE.md. The engine is not modified.

Usage: python scripts/make_article_figures.py [--only fig1|fig3|fig4|panels]
"""

from __future__ import annotations

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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import polar_sync  # noqa: E402
from synthSYL import panphon_pipeline  # noqa: E402
from synthSYL.phonology import VOWELS_SYNTSYL, CONSONANTS_SYNTSYL  # noqa: E402
import vlam  # noqa: E402

# run_locus_trough_demos: constants + article consonant positions
# (module import is side-effect free: it only creates its output dirs)
import run_locus_trough_demos as ltd  # noqa: E402

T_STEP_MS = ltd.T_STEP_MS          # 10
FS_AUDIO = ltd.FS_AUDIO            # 20000
GUI_LEN_MM = ltd.GUI_LEN_MM        # 195
VALRECT = ltd.VALRECT              # 0.75

OUT_DIR = REPO_ROOT / "docs" / "figures" / "article_format"
FIG_DIR = REPO_ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAEDA_LABELS = ["Jaw", "Body", "Dorsum", "Tip", "LipP", "LipH", "Hy"]
MAEDA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2"]

# Article figure-4 conditions (synthSYL legacy naming: COEFCEN = end
# coefficient -> delta_e, VOYDEB = onset coefficient -> delta_o)
DELTA_O_FIG4 = 0.5
DELTA_E_FIG4 = 1.0
T_FIG4 = 16
PAUSEKW = dict(short_pause_duration_factor=1.0,
               long_pause_duration_factor=1.0)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def run_recorded(text, T, **kwargs):
    """(result, blocks) — pipeline replay with polar_sync recording."""
    blocks: list = []
    result = polar_sync.record_pipeline(
        panphon_pipeline, blocks, text=text, T=T, **kwargs)
    return result, blocks


def branches(blocks, n_steps, sr_display=100.0):
    """(z_v, z_c) with the original arcplot anchoring (see module doc)."""
    z_v, z_c, _ = polar_sync.build_branches(
        blocks, n_steps, t_step_ms=T_STEP_MS, sr_display=sr_display,
        k_v=30.0, k_c=10.0, nu_v=-1, nu_c=1)
    return z_v, z_c


def synth(result, T_base):
    """(signal, formants) via the frozen VLAM synthesizer."""
    state = vlam.VlamState.initial(GUI_LEN_MM)
    config = vlam.SynthConfig(play_audio=False)
    sr = vlam.synthwordfen(
        state=state, articulatory_params=result.Pval,
        word_tokens=["O", "V", "F"],
        f0_scale=1.0, soft_rect_s=VALRECT,
        duration_factor=T_base, envelope=result.envelope, config=config,
    )
    return sr.signal, sr.formants


def label_runs(labels):
    """[(key, i_start, i_end)] over engine steps."""
    runs, cur, s0 = [], labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            runs.append((cur, s0, i - 1))
            cur, s0 = labels[i], i
    runs.append((cur, s0, len(labels) - 1))
    return runs


def phoneme_ticks(labels):
    """[(key, center_step)] for every non-'_' labeled run."""
    out = []
    for key, a, b in label_runs(labels):
        if key != "_":
            out.append((key, 0.5 * (a + b)))
    return out


def plot_zc_segments(ax, z_c, x=None, polar=False, **kw):
    """Plot z_c as separate line segments at the NaN breaks."""
    valid = np.isfinite(z_c.real)
    if not valid.any():
        return
    idx = np.where(valid)[0]
    spans, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1:
            spans.append((s, a))
            s = b
    spans.append((s, idx[-1]))
    for a, e in spans:
        seg = z_c[a:e + 1]
        if polar:
            ax.plot(np.angle(seg), np.abs(seg), **kw)
        elif x is None:
            ax.plot(seg.real, seg.imag, **kw)
        else:
            ax.plot(x[a:e + 1], seg.real, **kw)


# ═══════════════════════════════════════════════════════════════════
# Figure 1 — /ibia/, T=100 ms (T=10), four stacked panels
# ═══════════════════════════════════════════════════════════════════
def figure1():
    print("=" * 72)
    print("ARTICLE FIGURE 1: /ibia/, T=100 ms — 4 stacked panels")
    print("=" * 72)
    T = 10
    result, blocks = run_recorded("ibia", T=T, verbose=False)
    labels, _ = polar_sync.build_labels(blocks, result.seg_map)
    z_v, z_c = branches(blocks, result.Pval.shape[0], sr_display=1000.0)
    sig, formants = synth(result, T)
    P = result.Pval
    n_steps = P.shape[0]
    t_ms = np.arange(n_steps) * T_STEP_MS

    fig = plt.figure(figsize=(7.5, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 1, 0.8, 1])

    # (a) Polar planning (original: plt.polar(np.angle(Tval), np.abs(Tval)))
    ax = fig.add_subplot(gs[0], projection="polar")
    ax.set_facecolor("#ffffff")
    v = np.isfinite(z_v.real)
    ax.plot(np.angle(z_v[v]), np.abs(z_v[v]), "-", color="#d62728",
            linewidth=1.4, label=r"$z_v$ (K=30, $\nu$=−1)")
    plot_zc_segments(ax, z_c, polar=True, color="#1f77b4",
                     linewidth=1.8, label=None)
    ax.plot([], [], "-", color="#1f77b4", linewidth=1.8,
            label=r"$z_c$ (K=10, $\nu$=+1)")
    # phoneme targets of the utterance
    vowels = {k: (vv["rho"], vv["theta"])
              for k, vv in VOWELS_SYNTSYL.items()}
    cons = {k: (vv["rho"], vv["theta"])
            for k, vv in CONSONANTS_SYNTSYL.items()}
    for key in ("i", "a"):
        if key in vowels:
            r0, th0 = vowels[key]
            ax.plot(th0, r0, "o", color="#aa3333", ms=4, alpha=0.6)
            ax.text(th0, r0 + 0.13, key, color="#aa3333", fontsize=9,
                    ha="center", va="center")
    for key in ("b",):
        if key in cons:
            r0, th0 = cons[key]
            ax.plot(th0, r0, "s", color="#3366aa", ms=4, alpha=0.6)
            ax.text(th0, r0 + 0.13, key, color="#3366aa", fontsize=9,
                    ha="center", va="center")
    ax.set_rgrids([0.5, 1.0, 1.5], fontsize=8)
    ax.set_rlim(0, 1.65)
    ax.set_title("(a) Planning trajectories $z_v(t)$, $z_c(t)$ "
                 "(polar, $\\rho$ / $\\theta$)", fontsize=11,
                 fontweight="bold", pad=14)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.12, -0.14), fontsize=8,
              frameon=False)

    # (b) Flow of the 7 Maeda parameters, phoneme ticks
    ax = fig.add_subplot(gs[1])
    for i in range(7):
        lw = 2.0 if i == 1 else 1.1
        alpha = 1.0 if i == 1 else 0.75
        ax.plot(t_ms, P[:, i], color=MAEDA_COLORS[i], linewidth=lw,
                alpha=alpha, label=MAEDA_LABELS[i])
    ticks = phoneme_ticks(labels)
    ax.set_xticks([c * T_STEP_MS for _, c in ticks])
    ax.set_xticklabels([f"/{k}/" for k, _ in ticks], fontsize=10)
    ax.set_xlim(0, t_ms[-1])
    ax.set_ylim(-4, 5)
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Parameter value", fontsize=10)
    ax.set_title("(b) Flow of articulatory parameters "
                 "(trough effect on Body during /b/)", fontsize=11,
                 fontweight="bold")
    ax.legend(loc="upper right", ncol=7, fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3)

    # (c) F1-F2-F3
    ax = fig.add_subplot(gs[2])
    nf = formants.shape[0]
    tf = np.arange(nf) * T_STEP_MS
    ax.plot(tf, formants[:, 0], color="#2ca02c", lw=1.2, label="F1")
    ax.plot(tf, formants[:, 1], color="#d62728", lw=1.8, label="F2 (S-shaped)")
    ax.plot(tf, formants[:, 2], color="#1f77b4", lw=1.2, label="F3")
    ax.set_ylim(0, 4000)
    ax.set_xlim(0, tf[-1])
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Formant frequency (Hz)", fontsize=10)
    ax.set_title("(c) Formant trajectory F1-F2-F3", fontsize=11,
                 fontweight="bold")
    ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)

    # (d) Spectrogram + wave
    ax = fig.add_subplot(gs[3])
    sig_f = np.asarray(sig, dtype=float)
    m = float(np.max(np.abs(sig_f))) or 1.0
    sig_f = sig_f / m
    ax.specgram(sig_f, NFFT=256, Fs=FS_AUDIO, noverlap=192,
                cmap="Greys", scale="dB")
    t_audio = np.arange(len(sig_f)) / FS_AUDIO * 1000.0
    ax.plot(t_audio, 2500 + 1200 * sig_f, color="#333333", linewidth=0.4)
    ax.set_ylim(0, 4000)
    ax.set_xlim(0, t_audio[-1])
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    ax.set_title("(d) Output spectrogram + wave", fontsize=11,
                 fontweight="bold")

    fig.suptitle("Figure 1: four-step synthesis of /ibia/ — T = 100 ms "
                 "(K=10, Kvoy=30, Pexp=1)\narXiv:2307.02299 "
                 "(Berthommier 2023)", fontsize=12, fontweight="bold")
    out = OUT_DIR / "figure1_ibia_4panel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")
    print(f"  phoneme ticks: {[(k, int(c)) for k, c in ticks]}")


# ═══════════════════════════════════════════════════════════════════
# Figure 3 — locus equations /b,d,g/ on the 8 vowels (article titles)
# ═══════════════════════════════════════════════════════════════════
def collect_locus_data():
    """Same measurement as run_locus_trough_demos.run_demo2_locus."""
    state = vlam.VlamState.initial(GUI_LEN_MM)
    config = vlam.SynthConfig(play_audio=False)
    from synthSYL.phonology import SEGMENT_REGISTRY
    original_registry = {k: SEGMENT_REGISTRY[k]
                         for k in ltd.CONSONANTS_TO_TEST}
    sps = int(T_STEP_MS / 1000 * FS_AUDIO)
    locus_data = {}
    try:
        for cons in ltd.CONSONANTS_TO_TEST:
            locus_data[cons] = []
            for vowel in ltd.VOWELS_TO_TEST:
                input_text = cons + vowel
                if cons == 'b':
                    ltd.set_consonant_position('b', 1.0, np.pi / 3)
                elif cons == 'd':
                    ltd.set_consonant_position('d', 1.2, 3 * np.pi / 2)
                elif cons == 'g':
                    if vowel in ltd.G_PALATAL_VOWELS:
                        ltd.set_consonant_position('g', 1.1, 23 * np.pi / 12)
                    else:
                        ltd.set_consonant_position('g', 1.2, np.pi / 3)
                result = panphon_pipeline(input_text, T=ltd.T_BASE,
                                          delta_o=0.5, delta_e=0.5,
                                          verbose=False)
                if result is None:
                    continue
                sr = vlam.synthwordfen(
                    state=state, articulatory_params=result.Pval,
                    word_tokens=["O", "V", "F"], f0_scale=1.0,
                    soft_rect_s=VALRECT, duration_factor=ltd.T_BASE,
                    envelope=result.envelope, config=config)
                formants = sr.formants
                env_steps = result.envelope[::sps]
                release = None
                for f in range(2, len(env_steps)):
                    if env_steps[f] > 1e-3 and env_steps[f - 1] <= 1e-3:
                        release = f - 1
                        break
                on = np.nonzero(env_steps > 0.999)[0]
                if release is None or len(on) == 0:
                    continue
                locus_frame = min(release + ltd.LOCUS_OFFSET_STEPS,
                                  formants.shape[0] - 1)
                f2_locus = formants[locus_frame, 1]
                f2_vowel = float(np.mean(formants[on.min():on.max() + 1, 1]))
                g_type = ""
                if cons == 'g':
                    g_type = (' (anterior)' if vowel in ltd.G_PALATAL_VOWELS
                              else ' (posterior)')
                locus_data[cons].append((vowel, f2_vowel, f2_locus, g_type))
    finally:
        SEGMENT_REGISTRY.update(original_registry)
    return locus_data


def figure3():
    print("=" * 72)
    print("ARTICLE FIGURE 3: locus equations /b, d, g/ — 8 vowels")
    print("=" * 72)
    locus_data = collect_locus_data()
    consonant_colors = {"b": "#1f77b4", "d": "#ff7f0e", "g": "#2ca02c"}
    offset = ltd.LOCUS_OFFSET_STEPS * T_STEP_MS

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for i, cons in enumerate(ltd.CONSONANTS_TO_TEST):
        ax = axes[i]
        data = locus_data[cons]
        if cons == 'g':
            for group, col, lab in (
                    ('(anterior)', '#2ca02c', 'palatal (front + /a/)'),
                    ('(posterior)', '#d62728', 'velar (back)')):
                gdata = [(d[1], d[2]) for d in data if group in d[3]]
                if gdata:
                    fv, fl = zip(*gdata)
                    ax.scatter(fv, fl, c=col, s=90, zorder=5,
                               edgecolors="white", linewidth=1.5,
                               label=f'/g/ {lab}')
                    if len(fv) >= 2:
                        a, b = np.polyfit(fv, fl, 1)
                        x = np.linspace(min(fv) - 100, max(fv) + 100, 100)
                        ax.plot(x, a * x + b, "--", color=col, lw=1.5,
                                label=f'y = {a:.3f}x + {b:.0f}')
        else:
            fv = np.array([d[1] for d in data])
            fl = np.array([d[2] for d in data])
            ax.scatter(fv, fl, c=consonant_colors[cons], s=90, zorder=5,
                       edgecolors="white", linewidth=1.5)
            for (vowel, f_v, f_l, _) in data:
                ax.annotate(f"/{vowel}/", (f_v, f_l),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=9, color="#333333")
            if len(fv) >= 2:
                a, b = np.polyfit(fv, fl, 1)
                x = np.linspace(fv.min() - 100, fv.max() + 100, 100)
                ax.plot(x, a * x + b, "--", color=consonant_colors[cons],
                        lw=1.5, alpha=0.8,
                        label=f'y = {a:.3f}x + {b:.0f}')
        all_fv = np.array([d[1] for d in data])
        all_fl = np.array([d[2] for d in data])
        lim = [min(all_fv.min(), all_fl.min()) - 200,
               max(all_fv.max(), all_fl.max()) + 200]
        ax.plot(lim, lim, ":", color="#999999", lw=1, alpha=0.6,
                label="y = x")
        ax.set_xlabel("F2 vowel (Hz)", fontsize=11)
        ax.set_ylabel(f"F2 locus (Hz, {offset:.0f} ms after release)",
                      fontsize=11)
        ax.set_title(f"/{cons}/", fontsize=15, fontweight="bold",
                     color=consonant_colors[cons])
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    fig.suptitle("Figure 3: Locus equations for /b/, /d/, /g/ "
                 "(F2 locus vs vowel F2, 8 vowels)\n"
                 "T=16, K=10, Kvoy=30, Pexp=1, delta_o=0.5 — "
                 "article consonant positions (/b/ rho=1.0, /d/ 3pi/2, "
                 "/g/ velar-palatal split)\n"
                 "arXiv:2307.02299 (Berthommier 2023)",
                 fontsize=11, fontweight="bold")
    out = OUT_DIR / "figure3_locus_equations.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")

    # Combined panel (article-style single plot)
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    for cons in ltd.CONSONANTS_TO_TEST:
        data = locus_data[cons]
        if cons == 'g':
            for group, col, lab in (
                    ('(anterior)', '#2ca02c', '/g/ palatal'),
                    ('(posterior)', '#d62728', '/g/ velar')):
                gdata = [(d[1], d[2]) for d in data if group in d[3]]
                if gdata:
                    fv, fl = zip(*gdata)
                    ax.scatter(fv, fl, c=col, s=80, zorder=5,
                               edgecolors="white", linewidth=1.2,
                               label=lab)
                    a, b = np.polyfit(fv, fl, 1)
                    x = np.linspace(min(fv) - 100, max(fv) + 100, 100)
                    ax.plot(x, a * x + b, "--", color=col, lw=1.5,
                            alpha=0.8, label=f'{a:.3f}x+{b:.0f}')
        else:
            fv = np.array([d[1] for d in data])
            fl = np.array([d[2] for d in data])
            ax.scatter(fv, fl, c=consonant_colors[cons], s=80, zorder=5,
                       edgecolors="white", linewidth=1.2,
                       label=f"/{cons}/")
            a, b = np.polyfit(fv, fl, 1)
            x = np.linspace(fv.min() - 100, fv.max() + 100, 100)
            ax.plot(x, a * x + b, "--", color=consonant_colors[cons],
                    lw=1.5, alpha=0.8, label=f'{a:.3f}x+{b:.0f}')
    all_fv = np.concatenate([np.array([d[1] for d in locus_data[c]])
                             for c in ltd.CONSONANTS_TO_TEST])
    all_fl = np.concatenate([np.array([d[2] for d in locus_data[c]])
                             for c in ltd.CONSONANTS_TO_TEST])
    lim = [min(all_fv.min(), all_fl.min()) - 200,
           max(all_fv.max(), all_fl.max()) + 200]
    ax.plot(lim, lim, ":", color="#999999", lw=1, alpha=0.6, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("F2 vowel (Hz)", fontsize=12)
    ax.set_ylabel(f"F2 locus (Hz, {offset:.0f} ms after release)",
                  fontsize=12)
    ax.set_title("Figure 3 (combined): locus equations — slope < 1 = "
                 "coarticulation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    out2 = OUT_DIR / "figure3_locus_combined.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"  saved: {out2}")

    # print slopes (gate cross-check vs run_locus_trough_demos)
    print("  slopes:")
    for cons in ltd.CONSONANTS_TO_TEST:
        data = locus_data[cons]
        if cons == 'g':
            for gname, gf in (('palatal', '(anterior)'), ('velar', '(posterior)')):
                gd = [(d[1], d[2]) for d in data if gf in d[3]]
                if len(gd) >= 2:
                    a, b = np.polyfit([x for x, _ in gd], [y for _, y in gd], 1)
                    print(f"    /g/ {gname}: {a:.3f}x + {b:.0f}")
        else:
            fv = [d[1] for d in data]
            fl = [d[2] for d in data]
            if len(fv) >= 2:
                a, b = np.polyfit(fv, fl, 1)
                print(f"    /{cons}/: {a:.3f}x + {b:.0f}")


# ═══════════════════════════════════════════════════════════════════
# Figure 4 — big.bi | bi.gbi under the article conditions
# ═══════════════════════════════════════════════════════════════════
def _fig4_case(text):
    result, blocks = run_recorded(text, T=T_FIG4, delta_o=DELTA_O_FIG4,
                                  delta_e=DELTA_E_FIG4, **PAUSEKW)
    labels, _ = polar_sync.build_labels(blocks, result.seg_map)
    z_v, z_c = branches(blocks, result.Pval.shape[0])
    sig, formants = synth(result, T_FIG4)
    return result, labels, z_v, z_c, sig, formants


def figure4():
    print("=" * 72)
    print("ARTICLE FIGURE 4: big.bi | bi.gbi "
          f"(COEFCEN=1, VOYDEB=0.5, T={T_FIG4})")
    print("=" * 72)
    cases = [("big.bi", "big bi"), ("bi.gbi", "bi.gbi")]
    data = {}
    for title, text in cases:
        data[title] = _fig4_case(text)
        print(f"  {title}: {data[title][0].Pval.shape[0]} steps, "
              f"sig {data[title][4].shape[0]} samples")

    fig = plt.figure(figsize=(13, 15), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.25, 1, 0.45, 1])

    for col, (title, _) in enumerate(cases):
        result, labels, z_v, z_c, sig, formants = data[title]
        P = result.Pval
        n_steps = P.shape[0]
        t_ms = np.arange(n_steps) * T_STEP_MS

        # top: polar planning panel (article style)
        ax = fig.add_subplot(gs[0, col], projection="polar")
        ax.set_facecolor("#ffffff")
        v = np.isfinite(z_v.real)
        ax.plot(np.angle(z_v[v]), np.abs(z_v[v]), "-", color="#d62728",
                linewidth=1.5, label=r"$z_v$ (K=30, $\nu$=−1)")
        plot_zc_segments(ax, z_c, polar=True, color="#1f77b4",
                         linewidth=1.9)
        ax.plot([], [], "-", color="#1f77b4", linewidth=1.9,
                label=r"$z_c$ (K=10, $\nu$=+1)")
        # phoneme targets visited by the utterance
        vows = {k: (vv["rho"], vv["theta"])
                for k, vv in VOWELS_SYNTSYL.items()}
        consd = {k: (vv["rho"], vv["theta"])
                 for k, vv in CONSONANTS_SYNTSYL.items()}
        marks = [("i", vows["i"], "#aa3333", "o"),
                 ("@", (DELTA_O_FIG4 * vows["i"][0], vows["i"][1]),
                  "#aa3333", "o"),
                 ("b", consd["b"], "#3366aa", "s"),
                 ("g", consd["g"], "#3366aa", "s")]
        for key, (r0, th0), col_, mk in marks:
            ax.plot(th0, r0, mk, color=col_, ms=4, alpha=0.65)
            ax.text(th0, r0 + 0.16, key, color=col_, fontsize=9,
                    ha="center", va="center")
        ax.set_rgrids([0.3, 0.6, 0.9, 1.2, 1.5, 1.8], fontsize=7)
        ax.set_thetagrids(np.arange(0, 360, 30), fontsize=7)
        ax.set_rlim(0, 1.95)
        ax.set_title(f"{title}", fontsize=15, fontweight="bold", pad=16)
        ax.legend(loc="lower left", bbox_to_anchor=(-0.08, -0.16),
                  fontsize=8, frameon=False)

        # middle: Maeda parameters with phoneme ticks
        ax = fig.add_subplot(gs[1, col])
        for i in range(7):
            ax.plot(t_ms, P[:, i], color=MAEDA_COLORS[i], linewidth=1.1,
                    alpha=0.85, label=MAEDA_LABELS[i])
        ticks = phoneme_ticks(labels)
        ax.set_xticks([c * T_STEP_MS for _, c in ticks])
        ax.set_xticklabels([f"/{k}/" for k, _ in ticks], fontsize=9)
        ax.set_xlim(0, t_ms[-1])
        ax.set_ylim(-4, 5)
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Parameter value", fontsize=10)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(loc="upper right", ncol=7, fontsize=7, frameon=False)

        # bottom: wave + spectrogram (aligned)
        ax = fig.add_subplot(gs[2, col])
        sig_f = np.asarray(sig, dtype=float)
        mm = float(np.max(np.abs(sig_f))) or 1.0
        sig_f = sig_f / mm
        t_audio = np.arange(len(sig_f)) / FS_AUDIO * 1000.0
        ax.plot(t_audio, sig_f, color="#333333", linewidth=0.25)
        ax.set_xlim(0, t_audio[-1])
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(True, alpha=0.2)
        if col == 0:
            ax.set_title("Wave + spectrogram (aligned)", fontsize=10,
                         fontweight="bold")

        ax = fig.add_subplot(gs[3, col], sharex=ax)
        ax.specgram(sig_f, NFFT=256, Fs=FS_AUDIO, noverlap=192,
                    cmap="Greys", scale="dB")
        ax.set_ylim(0, 4000)
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)

    fig.suptitle("Figure 4: big.bi (δo=0.5, δe=1) | bi.gbi — planning, "
                 "articulatory parameters, wave + spectrogram\n"
                 "COEFCEN = 1, VOYDEB = 0.5, T=16, K=10, Kvoy=30 — "
                 "arXiv:2307.02299 (Berthommier 2023)",
                 fontsize=12, fontweight="bold")
    out = OUT_DIR / "figure4_bigbi_bigbi.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved: {out}")
    for title, _ in cases:
        result, labels, z_v, z_c, _, _ = data[title]
        c = np.isfinite(z_c.real)
        idx = np.where(c)[0]
        spans, s = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 1:
                spans.append((s, a))
                s = b
        spans.append((s, idx[-1]))
        apexes = []
        for a, e in spans:
            seg = z_c[a:e + 1]
            k = int(np.argmax(np.abs(seg)))
            apexes.append((round(float(np.degrees(np.angle(seg[k]))) % 360),
                           round(float(abs(seg[k])), 2)))
        print(f"  {title}: {len(spans)} z_c span(s), apexes={apexes}")


# ═══════════════════════════════════════════════════════════════════
# Raft panels (article-conditions planning, cartesian grid style)
# ═══════════════════════════════════════════════════════════════════
INVENTORY = {**{k: (v["rho"], v["theta"]) for k, v in VOWELS_SYNTSYL.items()},
             **{k: (v["rho"], v["theta"])
                for k, v in CONSONANTS_SYNTSYL.items()}}


def _cartesian_polar_axes(ax):
    ax.set_facecolor("#fafafa")
    for rr in [0.5, 1.0, 1.2]:
        ax.add_patch(Circle((0, 0), rr, fill=False, ls=":", lw=0.7,
                            color="#888", alpha=0.5))
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.axvline(0, color="#ccc", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    for key, (rho, th) in INVENTORY.items():
        x, y = rho * np.cos(th), rho * np.sin(th)
        ax.plot(x, y, "o" if key in VOWELS_SYNTSYL else "s",
                color="#d62728" if key in VOWELS_SYNTSYL else "#1f77b4",
                ms=3.2, alpha=0.25)


def _draw_branches(ax, z_v, z_c):
    v, c = np.isfinite(z_v.real), np.isfinite(z_c.real)
    ax.plot(z_v[v].real, z_v[v].imag, "-", color="#d62728", lw=1.7,
            alpha=0.9, label=r"$z_v$ ($\nu$=−1)")
    if c.any():
        ax.plot(z_c[c].real, z_c[c].imag, "-", color="#1f77b4", lw=2.1,
                alpha=0.95, label=r"$z_c$ ($\nu$=+1)")


def _zc_spans(z_c):
    c = np.isfinite(z_c.real)
    idx = np.where(c)[0]
    spans, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1:
            spans.append((s, a))
            s = b
    spans.append((s, idx[-1]))
    return spans


def panels():
    print("=" * 72)
    print("RAFT PANELS: article-conditions planning figures")
    print("=" * 72)

    # ── 2-panel big.bi / bi.gbi (fig_bigbi_article_conditions_nu_inv) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    for ax, (title, text) in zip(axes, [("big.bi", "big bi"),
                                        ("bi.gbi", "bi.gbi")]):
        result, blocks = run_recorded(text, T=T_FIG4,
                                      delta_o=DELTA_O_FIG4,
                                      delta_e=DELTA_E_FIG4, **PAUSEKW)
        z_v, z_c = branches(blocks, result.Pval.shape[0])
        _cartesian_polar_axes(ax)
        _draw_branches(ax, z_v, z_c)
        v = np.isfinite(z_v.real)
        ax.plot(z_v[v][0].real, z_v[v][0].imag, "o", color="#aa0000",
                ms=8, zorder=5)
        ax.plot(z_v[v][-1].real, z_v[v][-1].imag, "s", color="#aa0000",
                ms=8, zorder=5)
        ax.set_title(f"{title}   (COEFCEN=1, VOYDEB=0.5)", fontsize=12,
                     fontweight="bold", color="#003366")
        ax.legend(loc="upper left", fontsize=9)
        spans = _zc_spans(z_c)
        apexes = []
        for a, e in spans:
            seg = z_c[a:e + 1]
            k = int(np.argmax(np.abs(seg)))
            apexes.append((round(float(np.degrees(np.angle(seg[k]))) % 360),
                           round(float(abs(seg[k])), 2)))
        print(f"  {title}: {len(spans)} z_c span(s), apexes={apexes}")
    fig.suptitle("Syllable planning, article figure conditions — "
                 "COEFCEN = 1 ($\\delta_e$: Ve = V), VOYDEB = 0.5 "
                 "($\\delta_o$: Vo = half radius), $\\nu_v$=−1, "
                 "$\\nu_c$=+1\narXiv:2307.02299 Fig. 4 conditions, T=16 "
                 "(big.bi with inter-word pause; bi.gbi single word)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = FIG_DIR / "fig_bigbi_article_conditions_nu_inv.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  saved: {out}")

    # ── 4 word panels big | .bi | bi | .gbi (fig_words_..._nu_inv) ──
    def slice_word2(blocks_, syllable_dot=False):
        if syllable_dot:
            last = max(k for k, b in enumerate(blocks_)
                       if b["kind"] == "cluster")
            return blocks_[last:]
        seen = False
        for k, b in enumerate(blocks_):
            if b["kind"] == "decay":
                seen = True
                continue
            if seen and b["kind"] == "arc":
                return blocks_[k:]
        raise ValueError("boundary not found")

    r_gbg, bl_gbg = run_recorded("big bi", T=T_FIG4, delta_o=DELTA_O_FIG4,
                                 delta_e=DELTA_E_FIG4, **PAUSEKW)
    w2 = slice_word2(bl_gbg)
    w1 = bl_gbg[:len(bl_gbg) - len(w2)]
    r_bi, bl_bi = run_recorded("bi", T=T_FIG4, delta_o=DELTA_O_FIG4,
                               delta_e=DELTA_E_FIG4, **PAUSEKW)
    r_gbi, bl_gbi = run_recorded("bi.gbi", T=T_FIG4, delta_o=DELTA_O_FIG4,
                                 delta_e=DELTA_E_FIG4, **PAUSEKW)
    w_gbi = slice_word2(bl_gbi, syllable_dot=True)
    w_bi = bl_gbi[:len(bl_gbi) - len(w_gbi)]

    panels_ = [
        ("big", branches(w1, sum(b["n"] for b in w1))),
        (".bi", branches(w2, sum(b["n"] for b in w2))),
        ("bi", branches(bl_bi, r_bi.Pval.shape[0])),
        (".gbi", branches(w_gbi, sum(b["n"] for b in w_gbi))),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    for k, (name, (z_v, z_c)) in enumerate(panels_):
        ax = axes[k]
        _cartesian_polar_axes(ax)
        _draw_branches(ax, z_v, z_c)
        ax.set_title(f'"{name}"', fontsize=13, fontweight="bold",
                     color="#003366")
        if k == 0:
            ax.legend(loc="upper left", fontsize=8)
        spans = _zc_spans(z_c)
        apexes = []
        for a, e in spans:
            seg = z_c[a:e + 1]
            kk = int(np.argmax(np.abs(seg)))
            apexes.append((round(float(np.degrees(np.angle(seg[kk]))) % 360),
                           round(float(abs(seg[kk])), 2)))
        print(f"  {name:>4}: {len(spans)} z_c span(s), apexes={apexes}")
    fig.suptitle("Word-level planning under the article figure conditions — "
                 "COEFCEN = 1 ($\\delta_e$), VOYDEB = 0.5 ($\\delta_o$), "
                 "T=16, $z_v$ K=30 / $z_c$ K=10, display $\\nu_v$=−1 / "
                 "$\\nu_c$=+1\n('.bi' = word 2 of big.bi sliced at the "
                 "pause; '.gbi' = syllable 2 of bi.gbi sliced at the dot)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = FIG_DIR / "fig_words_article_conditions_nu_inv.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  saved: {out}")

    # ── planning_4panels_syntsyl (delta_o = delta_e = 0.5) ──
    KW = dict(T=T_FIG4, verbose=False, delta_o=0.5, delta_e=0.5, **PAUSEKW)
    r_bigbi, bl_bigbi = run_recorded("big bi", **KW)
    r_bi2, bl_bi2 = run_recorded("bi", **KW)
    r_gbi2, bl_gbi2 = run_recorded("gbi", **KW)
    w2b = slice_word2(bl_bigbi)
    w1b = bl_bigbi[:len(bl_bigbi) - len(w2b)]
    four = [
        ("big", branches(w1b, sum(b["n"] for b in w1b))),
        (".bi", branches(w2b, sum(b["n"] for b in w2b))),
        ("bi", branches(bl_bi2, r_bi2.Pval.shape[0])),
        (".gbi", branches(bl_gbi2, r_gbi2.Pval.shape[0])),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for k, (name, (z_v, z_c)) in enumerate(four):
        ax = axes[k]
        _cartesian_polar_axes(ax)
        _draw_branches(ax, z_v, z_c)
        ax.set_title(f'"{name}"', fontsize=12, fontweight="bold",
                     color="#003366")
        if k == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("synthSYL syllable planning — big / .bi / bi / .gbi "
                 "(arXiv:2307.02299 §4; $\\delta_o=\\delta_e$=0.5, T=16, "
                 "$z_v$ K=30, $z_c$ K=10)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIG_DIR / "planning_4panels_syntsyl.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  saved: {out}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    only = sys.argv[sys.argv.index("--only") + 1] \
        if "--only" in sys.argv else "all"
    print(f"ARTICLE-FORMAT FIGURES -> {OUT_DIR}")
    if only in ("all", "fig1"):
        figure1()
    if only in ("all", "fig3"):
        figure3()
    if only in ("all", "fig4"):
        figure4()
    if only in ("all", "panels"):
        panels()
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
