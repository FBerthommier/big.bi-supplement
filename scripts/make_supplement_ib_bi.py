#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
make_supplement_ib_bi.py — Supplement figure: /ib/ -> /bi/ classical
transformation (article §3: "The classical transformation of /ib/
into /bi/ is given in the supplement"), regenerated at the CURRENT
engine state (no Ve plateau — the decay chains directly on the word-end
anchor; dot semantics) and valrect = 1.10.

Three cases, side by side (as in the article supplement):
  1. /ib/ with delta = 0.5  (schwa @ audible, /b/ in coda)
  2. /ib/ with delta = 1.0  (Vo = Ve = V, fusion)
  3. /bi/ with delta = 1.0  (/b/ in onset, the fused form)

Each column: (top) Maeda parameters with Body highlighted, (middle)
formants F1-F2-F3, (bottom) spectrogram + wave.

Gates printed: /ib/ at both deltas must be 112 steps = 1120 ms with the
cluster followed DIRECTLY by the decay (no held @).

Output: docs/figures/article_format/supplement_ib_bi.png
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

try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from synthSYL import panphon_pipeline  # noqa: E402
import vlam  # noqa: E402
from config import T_BASE, T_STEP_MS, FS_AUDIO, GUI_LEN_MM, VALRECT  # noqa: E402

MAEDA_LABELS = ["Jaw", "Body", "Dorsum", "Tip", "LipP", "LipH", "Hy"]
MAEDA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2"]
OUT = REPO_ROOT / "docs" / "figures" / "article_format" / "supplement_ib_bi.png"

CASES = [
    ("ib", 0.5, "/ib/  delta=0.5\n(schwa @ audible, /b/ in coda)"),
    ("ib", 1.0, "/ib/  delta=1.0\n(Vo = Ve = V, fusion)"),
    ("bi", 1.0, "/bi/  delta=1.0\n(/b/ in onset, fused form)"),
]


def main() -> int:
    state = vlam.VlamState.initial(GUI_LEN_MM)
    cfg = vlam.SynthConfig(play_audio=False)

    results = []
    for text, delta, desc in CASES:
        res = panphon_pipeline(text, T=T_BASE, delta_o=delta, delta_e=delta,
                               verbose=False)
        assert res is not None, text
        sr = vlam.synthwordfen(
            state=state, articulatory_params=res.Pval,
            word_tokens=["O", "V", "F"], f0_scale=1.0,
            soft_rect_s=VALRECT, duration_factor=T_BASE,
            envelope=res.envelope, config=cfg)
        n_ms = res.Pval.shape[0] * T_STEP_MS
        print(f"  {text} delta={delta}: {res.Pval.shape[0]} steps "
              f"= {n_ms} ms")
        if text == "ib":
            # no-held-@ gate: 112 steps = 1120 ms (arc+attack+plateau+
            # cluster+decay+arc), the /b/ cluster NOT followed by a held
            # Ve plateau (the revoked addition).
            assert res.Pval.shape[0] == 112 and n_ms == 1120, \
                "/ib/ must stay 112 steps = 1120 ms (no held @)"
        results.append((text, delta, desc, res, sr))

    fig, axes = plt.subplots(3, 3, figsize=(15, 10),
                             constrained_layout=True)
    for col, (text, delta, desc, res, sr) in enumerate(results):
        P = res.Pval
        t_ms = np.arange(P.shape[0]) * T_STEP_MS

        ax = axes[0, col]
        for j in range(7):
            lw = 2.6 if j == 1 else 1.0
            alpha = 1.0 if j == 1 else 0.5
            ax.plot(t_ms, P[:, j], color=MAEDA_COLORS[j], linewidth=lw,
                    alpha=alpha, label=MAEDA_LABELS[j])
        ax.set_title(desc, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Maeda parameter")
            ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (ms)")

        ax = axes[1, col]
        nf = sr.formants.shape[0]
        tf = np.arange(nf) * T_STEP_MS
        ax.plot(tf, sr.formants[:, 0], color="#2ca02c", lw=1.2, label="F1")
        ax.plot(tf, sr.formants[:, 1], color="#d62728", lw=1.8, label="F2")
        ax.plot(tf, sr.formants[:, 2], color="#1f77b4", lw=1.2, label="F3")
        ax.set_ylim(0, 4000)
        if col == 0:
            ax.set_ylabel("Formant frequency (Hz)")
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (ms)")

        ax = axes[2, col]
        sig = np.asarray(sr.signal, dtype=float)
        m = float(np.max(np.abs(sig))) or 1.0
        ax.specgram(sig / m, NFFT=256, Fs=FS_AUDIO, noverlap=192,
                    cmap="magma", scale="dB")
        ax.set_ylim(0, 4000)
        if col == 0:
            ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")

    fig.suptitle(
        "Supplement — /ib/ → /bi/ classical transformation "
        "(arXiv:2307.02299 §3)\n"
        f"current engine (no held @, dot semantics), T={T_BASE}, "
        f"Pexp=1, valrect={VALRECT}",
        fontsize=13, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"  Figure saved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
