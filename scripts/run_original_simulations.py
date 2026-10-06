#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
run_original_simulations.py — Reproduce the article's simulations
EXACTLY as described in arXiv:2307.02299, using the article's
original parameters.

Three original simulations:

1. Figure 1: /ibia/ with T=100ms (T_base=10)
   - 4-step pipeline: (a) planning, (b) params, (c) formants, (d) spectrogram
   - Article: "The four steps of synthesis of /ibia/ with T = 100ms"

2. Figure 4: big.bi (δo=δe=1) and bi.gbi
   - Article §4: "When δo = δe tends to 1..."
   - Left: big.bi with Vo=Ve=V (δ=1)
   - Right: bi.gbi (fused /gb/ cluster)

3. Supplement: /ib/ → /bi/ classical transformation
   - Article §3: "The classical transformation of /ib/ into /bi/
     is given in the supplement."

Article parameters: T=10 (for /ibia/), T=16 (for big.bi/bi.gbi),
K=10, Kvoy=30, Pexp=1, nu=-1, valrect=0.75.

Output: <repo>/output/article_original_simulations/

The generated .npz are then compared against the reference copies in
<repo>/original_simulations/ (Pval/formants must match to ~1e-12; the
display branches z_v/z_c are checked separately since K_DISPLAY only
affects display, not the acoustic engine).
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

try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from polar_primitives import polar_arc, stationary_point  # noqa: E402
from synthSYL import panphon_pipeline  # noqa: E402
from synthSYL.phonology import VOWELS_SYNTSYL, CONSONANTS_SYNTSYL  # noqa: E402
import vlam  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
T_STEP_MS = 10
SR_GESTURE = 100.0
K_DISPLAY = 30.0        # vocalic display curvature (Kvoy)
K_C_DISPLAY = 10.0       # consonantal display curvature (article K=10)
SR_DISPLAY = 1000.0      # display sample rate (10x engine for smooth curves)
FS_AUDIO = 20_000
GUI_LEN_MM = 195
VALRECT = 0.75  # VLAM soft-rectification

OUT_BASE = REPO_ROOT / "output" / "article_original_simulations"
FIG1_DIR = OUT_BASE / "figure1_ibia"       # /ibia/ with T=100ms
FIG4_DIR = OUT_BASE / "figure4_bigbi_bigbi" # big.bi (δ=1) and bi.gbi
SUPP_DIR = OUT_BASE / "supplement_ib_bi"    # /ib/ → /bi/ transformation

for d in (FIG1_DIR, FIG4_DIR, SUPP_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAEDA_LABELS = ["Jaw", "Body", "Dorsum", "Tip", "LipP", "LipH", "Hy"]
MAEDA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2"]


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def get_inventory():
    vowels = {k: (v["rho"], v["theta"]) for k, v in VOWELS_SYNTSYL.items()}
    consonants = {k: (v["rho"], v["theta"]) for k, v in CONSONANTS_SYNTSYL.items()}
    return vowels, consonants


def synth_and_get_formants(result, T_base, out_wav=None):
    """Synthesize audio and return signal + formants."""
    state = vlam.VlamState.initial(GUI_LEN_MM)
    config = vlam.SynthConfig(play_audio=False)
    sr = vlam.synthwordfen(
        state=state, articulatory_params=result.Pval,
        word_tokens=["O", "V", "F"],
        f0_scale=1.0, soft_rect_s=VALRECT,
        duration_factor=T_base, envelope=result.envelope, config=config,
    )
    sig = sr.signal
    formants = sr.formants
    if out_wav:
        max_val = float(np.max(np.abs(sig))) if np.max(np.abs(sig)) > 1e-12 else 1.0
        sig_int16 = np.int16(32767 * sig / (1.01 * max_val))
        from scipy.io.wavfile import write as wav_write
        wav_write(str(out_wav), FS_AUDIO, sig_int16)
    return sig, formants


def reconstruct_polar(result, T_base, pause_ms=0.0):
    """Reconstruct z_v and z_c branches."""
    anchors = result.anchors
    nodes = result.nodes
    n_steps = result.Pval.shape[0]
    z_v_list, z_c_list = [], []
    prev_pt = None
    T_voy = T_base * T_STEP_MS
    T_cons = T_base * T_STEP_MS
    T_pause = max(pause_ms, T_STEP_MS)

    for idx in range(len(anchors) - 1):
        A, B = anchors[idx], anchors[idx + 1]
        a_term = A.kind in ("pause", "synth")
        b_term = B.kind in ("pause", "synth")
        a_v = A.kind == "V"
        b_v = B.kind == "V"

        if A.hold and a_v and not A.is_vowel_onset and not A.is_word_end:
            z, _ = stationary_point(A.pt[0], A.pt[1], T_voy, SR_DISPLAY)
            z_v_list.append(z)
            z_c_list.append(np.full_like(z, np.nan + 0j))
            prev_pt = A.pt

        ss = A.i + 1 if not A.is_vowel_onset else A.i
        cons = [nodes[k] for k in range(ss, B.i)
                if 0 <= k < len(nodes) and nodes[k].kind == "C"]
        m = len(cons)

        if a_v and b_v:
            if m > 0:
                dur = (m + 1) * T_cons
                z_v, _ = polar_arc(A.pt[0], A.pt[1], B.pt[0], B.pt[1],
                                   dur, SR_DISPLAY, K_DISPLAY, 1, "inverse")
                z_v_list.append(z_v)
                pts = [A.pt] + [[c.rho, c.theta] for c in cons] + [B.pt]
                zc_blocks = []
                for j in range(len(pts) - 1):
                    zc, _ = polar_arc(pts[j][0], pts[j][1], pts[j+1][0], pts[j+1][1],
                                      T_cons, SR_DISPLAY, K_C_DISPLAY, -1, "inverse")
                    zc_blocks.append(zc)
                zc = np.concatenate(zc_blocks)
                if len(zc) < len(z_v):
                    zc = np.concatenate([zc, np.full(len(z_v) - len(zc), np.nan + 0j)])
                z_c_list.append(zc)
            else:
                z_v, _ = polar_arc(A.pt[0], A.pt[1], B.pt[0], B.pt[1],
                                   2 * T_voy, SR_DISPLAY, K_DISPLAY, 1, "inverse")
                z_v_list.append(z_v)
                z_c_list.append(np.full_like(z_v, np.nan + 0j))
            prev_pt = B.pt
        elif a_v and b_term:
            z, _ = stationary_point(A.pt[0], A.pt[1], T_cons, SR_DISPLAY)
            z_v_list.append(z)
            z_c_list.append(np.full_like(z, np.nan + 0j))
            prev_pt = A.pt
        elif a_term and b_v:
            if prev_pt is not None:
                z_v, _ = polar_arc(prev_pt[0], prev_pt[1], B.pt[0], B.pt[1],
                                   T_pause, SR_DISPLAY, K_DISPLAY, 1, "inverse")
                z_v_list.append(z_v)
                z_c_list.append(np.full_like(z_v, np.nan + 0j))
            prev_pt = B.pt

    if z_v_list:
        z_v = np.concatenate([np.asarray(z, dtype=complex) for z in z_v_list])
        z_c = np.concatenate([np.asarray(z, dtype=complex) for z in z_c_list])
    else:
        z_v = np.zeros(n_steps, dtype=complex)
        z_c = np.full(n_steps, np.nan + 0j)
    # Align to the DISPLAY-rate length (SR_DISPLAY = 10 x SR_GESTURE),
    # not the engine step count: the branches are sampled at SR_DISPLAY.
    # (Aligning on n_steps kept only the first 10% of the trajectory and
    # made z_c entirely NaN — see docs/POLAR_K_AUDIT.md.)
    disp_len = int(round(n_steps * SR_DISPLAY / SR_GESTURE))
    if len(z_v) < disp_len:
        last = z_v[-1] if len(z_v) else 0
        z_v = np.concatenate([z_v, np.full(disp_len - len(z_v), last, dtype=complex)])
    else:
        z_v = z_v[:disp_len]
    if len(z_c) < disp_len:
        z_c = np.concatenate([z_c, np.full(disp_len - len(z_c), np.nan + 0j, dtype=complex)])
    else:
        z_c = z_c[:disp_len]
    return z_v, z_c, disp_len


# ═══════════════════════════════════════════════════════════════════
# SIMULATION 1: /ibia/ with T=100ms (Figure 1)
# ═══════════════════════════════════════════════════════════════════
def run_figure1_ibia():
    """Figure 1: Four-step synthesis of /ibia/ with T=100ms."""
    print("=" * 72)
    print("ORIGINAL SIMULATION 1: /ibia/ with T=100ms (Figure 1)")
    print("=" * 72)

    T = 10  # T=10 steps * 10ms = 100ms per period (article: T=100ms)
    print(f"  Input: ibia")
    print(f"  T={T} (={T*T_STEP_MS}ms per period, matching article T=100ms)")
    print(f"  K=10, Kvoy=30, Pexp=1, valrect={VALRECT}")

    result = panphon_pipeline("ibia", T=T, verbose=True)
    if result is None:
        print("  FAILED")
        return

    sig, formants = synth_and_get_formants(result, T, FIG1_DIR / "ibia.wav")
    print(f"  Signal: {sig.shape[0]} samples ({sig.shape[0]/FS_AUDIO*1000:.0f} ms)")
    print(f"  Formants: {formants.shape}")
    print(f"  F2 range: {formants[:,1].min():.0f} - {formants[:,1].max():.0f} Hz")

    z_v, z_c, n_disp = reconstruct_polar(result, T)  # display-rate arrays
    P = result.Pval
    t_ms = np.arange(P.shape[0]) * T_STEP_MS
    t_ms_z = np.arange(n_disp) / (SR_DISPLAY / 1000.0)

    # 4-panel figure (a, b, c, d)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), constrained_layout=True)

    # (a) Planning trajectories
    ax = axes[0]
    valid_v = ~np.isnan(z_v.real)
    valid_c = ~np.isnan(z_c.real)
    ax.plot(t_ms_z[valid_v], z_v[valid_v].real, "-", color="#d62728",
            linewidth=1.5, label=r"$\Re(z_v)$")
    ax.plot(t_ms_z[valid_v], z_v[valid_v].imag, "--", color="#d62728",
            linewidth=1, alpha=0.5, label=r"$\Im(z_v)$")
    if np.any(valid_c):
        seg_s = np.where(valid_c & ~np.roll(valid_c, 1))[0]
        seg_e = np.where(valid_c & ~np.roll(valid_c, -1))[0]
        for s, e in zip(seg_s, seg_e):
            ax.plot(t_ms_z[s:e+1], z_c[s:e+1].real, "-", color="#1f77b4", linewidth=1.8)
            ax.plot(t_ms_z[s:e+1], z_c[s:e+1].imag, "--", color="#1f77b4", linewidth=1, alpha=0.5)
        ax.plot([], [], "-", color="#1f77b4", linewidth=1.8, label=r"$\Re(z_c)$")
    ax.set_ylabel("Complex value")
    ax.set_title("(a) Planning trajectories $z_v(t)$ and $z_c(t)$",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (ms)")

    # (b) Articulatory parameters (trough effect)
    ax = axes[1]
    for i in range(7):
        lw = 2.5 if i == 1 else 1.2
        alpha = 1.0 if i == 1 else 0.7
        ax.plot(t_ms, P[:, i], color=MAEDA_COLORS[i], linewidth=lw,
                alpha=alpha, label=MAEDA_LABELS[i])
    ax.set_ylabel("Maeda parameter")
    ax.set_title("(b) Articulatory parameters (trough effect on Body)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (ms)")

    # (c) Formant trajectory
    ax = axes[2]
    nf = formants.shape[0]
    tf = np.arange(nf) * T_STEP_MS
    ax.plot(tf, formants[:, 0], color="#2ca02c", linewidth=1.5, label="F1")
    ax.plot(tf, formants[:, 1], color="#d62728", linewidth=2.0, label="F2")
    ax.plot(tf, formants[:, 2], color="#1f77b4", linewidth=1.5, label="F3")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("(c) Formant trajectory F1-F2-F3", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (ms)")
    ax.set_ylim(0, 4000)

    # (d) Spectrogram
    ax = axes[3]
    sig_f = sig.astype(float)
    sig_f = sig_f / (np.max(np.abs(sig_f)) if np.max(np.abs(sig_f)) > 0 else 1.0)
    ax.specgram(sig_f, NFFT=256, Fs=FS_AUDIO, noverlap=192, cmap="magma", scale="dB")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("(d) Output spectrogram", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 4000)

    fig.suptitle(
        f"Figure 1 (original): Four-step synthesis of /ibia/ "
        f"(T={T*T_STEP_MS}ms, K=10, Kvoy=30, Pexp=1)\n"
        f"arXiv:2307.02299 (Berthommier 2023)",
        fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(FIG1_DIR / "ibia_4panel_original.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {FIG1_DIR}/ibia_4panel_original.png")
    np.savez(FIG1_DIR / "ibia_data.npz", Pval=P, formants=formants,
             z_v=z_v, z_c=z_c, t_ms=t_ms, t_z_ms=t_ms_z, sig=sig)


# ═══════════════════════════════════════════════════════════════════
# SIMULATION 2: big.bi (δ=1) and bi.gbi (Figure 4)
# ═══════════════════════════════════════════════════════════════════
def run_figure4_bigbi():
    """Figure 4: big.bi (δo=δe=1) and bi.gbi."""
    print("\n" + "=" * 72)
    print("ORIGINAL SIMULATION 2: big.bi (δ=1) and bi.gbi (Figure 4)")
    print("=" * 72)

    T = 16  # didactic mode
    print(f"  T={T}, K=10, Kvoy=30, Pexp=1, valrect={VALRECT}")

    vowels, consonants = get_inventory()

    # Left panel: big.bi with δo=δe=1 (Vo=Ve=V)
    print(f"\n  [Left] big bi with delta=1.0 (Vo=Ve=V)")
    result_bigbi = panphon_pipeline("big bi", T=T, delta_o=1.0, delta_e=1.0, verbose=True)
    sig_bigbi, formants_bigbi = synth_and_get_formants(
        result_bigbi, T, FIG4_DIR / "bigbi_delta1.wav")
    z_v_bigbi, z_c_bigbi, _ = reconstruct_polar(result_bigbi, T)
    print(f"    Signal: {sig_bigbi.shape[0]} samples ({sig_bigbi.shape[0]/FS_AUDIO*1000:.0f} ms)")

    # Right panel: bi.gbi (fused /gb/ cluster)
    print(f"\n  [Right] bi.gbi (fused /gb/ cluster)")
    result_bigbi_fused = panphon_pipeline("bi.gbi", T=T, delta_o=1.0, delta_e=1.0, verbose=True)
    sig_fused, formants_fused = synth_and_get_formants(
        result_bigbi_fused, T, FIG4_DIR / "bigbi_fused.wav")
    z_v_fused, z_c_fused, _ = reconstruct_polar(result_bigbi_fused, T)
    print(f"    Signal: {sig_fused.shape[0]} samples ({sig_fused.shape[0]/FS_AUDIO*1000:.0f} ms)")

    # Combined figure: 4 rows x 2 columns (big.bi left, bi.gbi right)
    fig, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)

    for col, (label, result, sig, formants, z_v, z_c) in enumerate([
        ("big.bi (δ=1)", result_bigbi, sig_bigbi, formants_bigbi, z_v_bigbi, z_c_bigbi),
        ("bi.gbi", result_bigbi_fused, sig_fused, formants_fused, z_v_fused, z_c_fused),
    ]):
        P = result.Pval
        t_ms = np.arange(P.shape[0]) * T_STEP_MS
        t_ms_z = np.arange(len(z_v)) / (SR_DISPLAY / 1000.0)  # display rate

        # Row 1: Planning trajectories
        ax = axes[0, col]
        valid_v = ~np.isnan(z_v.real)
        valid_c = ~np.isnan(z_c.real)
        ax.plot(t_ms_z[valid_v], z_v[valid_v].real, "-", color="#d62728", linewidth=1.5,
                label=r"$z_v$")
        if np.any(valid_c):
            seg_s = np.where(valid_c & ~np.roll(valid_c, 1))[0]
            seg_e = np.where(valid_c & ~np.roll(valid_c, -1))[0]
            for s, e in zip(seg_s, seg_e):
                ax.plot(t_ms_z[s:e+1], z_c[s:e+1].real, "-", color="#1f77b4", linewidth=1.8)
            ax.plot([], [], "-", color="#1f77b4", linewidth=1.8, label=r"$z_c$")
        ax.set_title(f"{label} — Planning trajectories", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Row 2: Articulatory parameters
        ax = axes[1, col]
        for i in range(7):
            ax.plot(t_ms, P[:, i], color=MAEDA_COLORS[i], linewidth=1.2,
                    label=MAEDA_LABELS[i])
        ax.set_title(f"{label} — Articulatory parameters", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, ncol=4)

        # Row 3: Wave + spectrogram
        ax = axes[2, col]
        sig_f = sig.astype(float)
        max_val = np.max(np.abs(sig_f)) if np.max(np.abs(sig_f)) > 0 else 1.0
        sig_f = sig_f / max_val
        t_audio = np.arange(len(sig_f)) / FS_AUDIO
        ax.plot(t_audio, sig_f, color="#333333", linewidth=0.3)
        ax.set_title(f"{label} — Waveform", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.2)

        # Row 4: Spectrogram
        ax = axes[3, col]
        ax.specgram(sig_f, NFFT=256, Fs=FS_AUDIO, noverlap=192, cmap="magma", scale="dB")
        ax.set_title(f"{label} — Spectrogram", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 4000)

    fig.suptitle(
        f"Figure 4 (original): Synthesis of big.bi (δo=δe=1) and bi.gbi\n"
        f"(T={T}, K=10, Kvoy=30, Pexp=1, valrect={VALRECT})\n"
        f"arXiv:2307.02299 (Berthommier 2023)",
        fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(FIG4_DIR / "figure4_original.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {FIG4_DIR}/figure4_original.png")

    # Also save polar comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for col, (label, z_v, z_c) in enumerate([
        ("big.bi (δ=1)", z_v_bigbi, z_c_bigbi),
        ("bi.gbi", z_v_fused, z_c_fused),
    ]):
        ax = axes[col]
        from matplotlib.patches import Circle
        ax.set_facecolor("#fafafa")
        for r in [0.5, 1.0, 1.2]:
            ax.add_patch(Circle((0, 0), r, fill=False, linestyle=":", linewidth=0.6,
                               color="#888888", alpha=0.5))
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.2)
        for k, (rho, theta) in vowels.items():
            x, y = rho * np.cos(theta), rho * np.sin(theta)
            ax.plot(x, y, "o", color="#d62728", markersize=4, alpha=0.25)
            ax.text(x, y, k, color="#aa3333", fontsize=7, alpha=0.4, ha="center", va="bottom")
        for k, (rho, theta) in consonants.items():
            x, y = rho * np.cos(theta), rho * np.sin(theta)
            ax.plot(x, y, "s", color="#1f77b4", markersize=5, alpha=0.25)
            ax.text(x, y, k, color="#3366aa", fontsize=7, alpha=0.4, ha="center", va="bottom")
        valid_v = ~np.isnan(z_v.real)
        ax.plot(z_v[valid_v].real, z_v[valid_v].imag, "-", color="#d62728", linewidth=1.6,
                label=r"$z_v$")
        valid_c = ~np.isnan(z_c.real)
        if np.any(valid_c):
            seg_s = np.where(valid_c & ~np.roll(valid_c, 1))[0]
            seg_e = np.where(valid_c & ~np.roll(valid_c, -1))[0]
            for s, e in zip(seg_s, seg_e):
                ax.plot(z_c[s:e+1].real, z_c[s:e+1].imag, "-", color="#1f77b4", linewidth=1.8)
            ax.plot([], [], "-", color="#1f77b4", linewidth=1.8, label=r"$z_c$")
        ax.set_title(label, fontsize=12, fontweight="bold", color="#003366")
        ax.legend(fontsize=9)

    fig.suptitle("Figure 4 polar (original): big.bi (δ=1) vs bi.gbi",
                 fontsize=12, fontweight="bold")
    fig.savefig(FIG4_DIR / "figure4_polar_original.png", dpi=120)
    plt.close(fig)
    print(f"  Polar saved: {FIG4_DIR}/figure4_polar_original.png")


# ═══════════════════════════════════════════════════════════════════
# SIMULATION 3: /ib/ → /bi/ supplement
# ═══════════════════════════════════════════════════════════════════
def run_supplement_ib_bi():
    """Supplement: /ib/ → /bi/ classical transformation."""
    print("\n" + "=" * 72)
    print("ORIGINAL SIMULATION 3: /ib/ -> /bi/ (supplement)")
    print("=" * 72)

    T = 16
    print(f"  T={T}, K=10, Kvoy=30, Pexp=1, valrect={VALRECT}")

    # Three cases:
    # 1. ib (with schwa, δ=0.5)
    # 2. ib (fusion, δ=1.0, Vo=Ve=V)
    # 3. bi (onset, δ=1.0)
    cases = [
        ("ib", 0.5, "ib (schwa @, δ=0.5)"),
        ("ib", 1.0, "ib (Vo=Ve=V, δ=1.0)"),
        ("bi", 1.0, "bi (onset, δ=1.0)"),
    ]

    all_data = []
    for input_text, delta, label in cases:
        print(f"\n  [{label}] input={input_text}")
        result = panphon_pipeline(input_text, T=T, delta_o=delta, delta_e=delta,
                                  verbose=False)
        if result is None:
            print(f"    FAILED")
            continue
        wav_path = SUPP_DIR / f"{input_text}_delta{int(delta*10):02d}.wav"
        sig, formants = synth_and_get_formants(result, T, wav_path)
        print(f"    Signal: {sig.shape[0]} samples ({sig.shape[0]/FS_AUDIO*1000:.0f} ms)")
        print(f"    Formants: {formants.shape}")
        all_data.append((input_text, delta, label, result, sig, formants))

    # Figure: 3 columns (ib δ=0.5, ib δ=1.0, bi δ=1.0) x 3 rows (params, formants, spectrogram)
    n = len(all_data)
    fig, axes = plt.subplots(3, n, figsize=(6 * n, 12), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(-1, 1)

    for col, (input_text, delta, label, result, sig, formants) in enumerate(all_data):
        P = result.Pval
        t_ms = np.arange(P.shape[0]) * T_STEP_MS

        # Row 1: Articulatory parameters
        ax = axes[0, col]
        for i in range(7):
            ax.plot(t_ms, P[:, i], color=MAEDA_COLORS[i], linewidth=1.2,
                    label=MAEDA_LABELS[i])
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, ncol=4)

        # Row 2: Formants
        ax = axes[1, col]
        nf = formants.shape[0]
        tf = np.arange(nf) * T_STEP_MS
        ax.plot(tf, formants[:, 0], color="#2ca02c", linewidth=1, label="F1")
        ax.plot(tf, formants[:, 1], color="#d62728", linewidth=2, label="F2")
        ax.plot(tf, formants[:, 2], color="#1f77b4", linewidth=1, label="F3")
        ax.set_title(f"Formants — {label}", fontsize=11)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Hz")
        ax.set_ylim(0, 4000)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

        # Row 3: Spectrogram
        ax = axes[2, col]
        sig_f = sig.astype(float)
        max_val = np.max(np.abs(sig_f)) if np.max(np.abs(sig_f)) > 0 else 1.0
        sig_f = sig_f / max_val
        ax.specgram(sig_f, NFFT=256, Fs=FS_AUDIO, noverlap=192, cmap="magma", scale="dB")
        ax.set_title(f"Spectrogram — {label}", fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")
        ax.set_ylim(0, 4000)

    fig.suptitle(
        f"Supplement (original): /ib/ -> /bi/ classical transformation\n"
        f"(T={T}, K=10, Kvoy=30, Pexp=1, valrect={VALRECT})\n"
        f"arXiv:2307.02299 (Berthommier 2023)",
        fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(SUPP_DIR / "supplement_ib_bi_original.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {SUPP_DIR}/supplement_ib_bi_original.png")


# ═══════════════════════════════════════════════════════════════════
# Reference comparison (original_simulations/ shipped with the repo)
# ═══════════════════════════════════════════════════════════════════
def compare_with_reference():
    """Compare freshly generated npz against the repo reference copies.

    Engine arrays (Pval, formants, sig) must match to ~1e-12. Display
    branches (z_v, z_c) are reported but tolerate differences: they
    depend on K_DISPLAY/K_C_DISPLAY, which do not feed the synthesizer.
    """
    ref_base = REPO_ROOT / "original_simulations"
    pairs = [
        ("figure1_ibia/ibia_data.npz", FIG1_DIR / "ibia_data.npz"),
    ]
    print("\n" + "=" * 72)
    print("REFERENCE COMPARISON (original_simulations/)")
    print("=" * 72)
    for rel, new_path in pairs:
        ref_path = ref_base / rel
        if not ref_path.exists() or not new_path.exists():
            print(f"  {rel}: SKIPPED (missing {'reference' if not ref_path.exists() else 'output'})")
            continue
        ref = np.load(ref_path)
        new = np.load(new_path)
        print(f"  {rel}:")
        for key in ref.files:
            if key not in new.files:
                print(f"    {key:10s}: missing in new output")
                continue
            a, b = ref[key], new[key]
            if a.shape != b.shape:
                print(f"    {key:10s}: SHAPE MISMATCH {a.shape} vs {b.shape}")
                continue
            if np.issubdtype(a.dtype, np.number):
                finite = np.isfinite(a) & np.isfinite(b)
                d = np.max(np.abs(a[finite] - b[finite])) if finite.any() else 0.0
                n_nan = int(np.sum(~finite))
                tag = ("engine" if key in ("Pval", "formants", "sig")
                       else "display")
                print(f"    {key:10s} [{tag:7s}]: max|diff| = {d:.3e}"
                      + (f"  ({n_nan} non-finite excluded)" if n_nan else ""))
            else:
                print(f"    {key:10s}: "
                      f"{'identical' if np.array_equal(a, b) else 'DIFFERENT'}")
    print("  (engine arrays must match to ~1e-12; see docs/LOCUS_FIG3_DIAGNOSTIC.md)")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("ORIGINAL SIMULATIONS (exact article parameters)")
    print("arXiv:2307.02299 (Berthommier 2023)")
    print("K=10, Kvoy=30, Pexp=1, nu=-1, valrect=0.75")
    print("=" * 72)

    run_figure1_ibia()
    run_figure4_bigbi()
    run_supplement_ib_bi()

    compare_with_reference()

    print("\n" + "=" * 72)
    print("ALL ORIGINAL SIMULATIONS COMPLETE")
    print("=" * 72)
    print(f"  Figure 1 (/ibia, T=100ms): {FIG1_DIR}")
    print(f"  Figure 4 (big.bi, bi.gbi):  {FIG4_DIR}")
    print(f"  Supplement (/ib/ -> /bi/):  {SUPP_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
