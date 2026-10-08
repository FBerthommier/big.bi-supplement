#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
run_trough_ablations.py — Trough-effect measurements and ablations
(arXiv:2307.02299 §3, Lindblom et al. 2002 [29]).

Companion of run_locus_trough_demos.py (Demo 4 / Demo 3 after the
2026-10-08 renumbering). Produces, from RECORDED ENGINE BLOCKS
(scripts/polar_sync.record_pipeline — no engine modification):

A1  Body(t) landmarks for ibi / aba / idi / igi (T=16) and ibia (T=10)
    at the article /b/ position (rho=1, theta=pi/3): vowel-plateau
    value, cluster extremum (sign, amplitude vs the preceding plateau,
    instant in ms), medial closure/release frames, F2 landmarks
    (initial, /i/ plateau, minimum in the cluster window).

A7  Ablations on /ibi/ (T=16):
    (i)   Body removed from the /b/ selection vector (params [0,5]);
    (ii)  rho_b in {0.5, 1.0, 1.2} at theta_b = pi/3;
    (iii) theta_b moved away from theta_u (pi, 3*pi/2) at rho = 1;
    (iv)  /b/ vs /d/ vs /g/ comparison.
    Plus the rho_b = 1.2 registry-default configuration (the one used
    by the article-format Figure 1), which explains the 1.25 vs 1.50
    amplitude difference between the figure families.

The SEGMENT_REGISTRY overrides are local to this script and restored
before exit. Output: console table + output/trough_ablations/report.txt.

Article parameters: T=16, K=10, Kvoy=30, Pexp=1, nu=-1.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthSYL import panphon_pipeline  # noqa: E402
from synthSYL.phonology import SEGMENT_REGISTRY  # noqa: E402
from synthSYL.types import SegmentDescription  # noqa: E402
import vlam  # noqa: E402
import polar_sync  # noqa: E402

try:
    from config import T_BASE, T_STEP_MS, FS_AUDIO, GUI_LEN_MM, VALRECT
except ImportError:  # standalone fallback
    T_BASE, T_STEP_MS, FS_AUDIO, GUI_LEN_MM, VALRECT = 16, 10, 20_000, 195, 1.10
TMS = T_STEP_MS

OUT_DIR = REPO_ROOT / "output" / "trough_ablations"


def set_pos(cons: str, rho: float, theta: float, params=None) -> None:
    """Override SEGMENT_REGISTRY[cons] (script-local, restored at exit)."""
    old = SEGMENT_REGISTRY[cons]
    SEGMENT_REGISTRY[cons] = SegmentDescription(
        key=old.key, IPA=old.IPA, rho=rho, theta=theta, is_vowel=old.is_vowel,
        params=old.params if params is None else params,
        features=old.features)


ORIG = {k: SEGMENT_REGISTRY[k] for k in ("b", "d", "g")}


def analyze(word: str, T: int, label: str, lines: list) -> None:
    blocks: list = []
    res = polar_sync.record_pipeline(
        panphon_pipeline, blocks, text=word, T=T, verbose=False)
    P = res.Pval
    body = P[:, 1]
    t_ms = np.arange(P.shape[0]) * TMS

    ranges, f0 = [], 0
    for b in blocks:
        ranges.append((b["kind"], f0, f0 + b["n"]))
        f0 += b["n"]
    assert f0 == P.shape[0]
    plateaus = [(s, e) for k, s, e in ranges if k == "plateau"]
    clusters = [(s, e) for k, s, e in ranges if k == "cluster"]

    lines.append(f"--- {label}: {word} T={T} "
                 f"({P.shape[0]} steps = {P.shape[0] * TMS} ms) ---")
    for s, e in plateaus:
        lines.append(f"  plateau Body = {body[s:e].mean():+7.3f}  "
                     f"frames {s}..{e - 1}  ({t_ms[s]:.0f}-{t_ms[e - 1]:.0f} ms)")

    # medial closure/release from the envelope
    env = res.envelope[::int(TMS / 1000 * FS_AUDIO)]
    silent = env <= 1e-3
    runs, i = [], 0
    while i < len(silent):
        if silent[i]:
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    closure = next(((s2, e2) for (s2, e2) in runs
                    if len(plateaus) >= 2 and s2 > plateaus[0][1]
                    and e2 < plateaus[-1][1]), None)
    release = next((f for f in range(closure[1] + 1, len(env))
                    if not silent[f]), None) if closure else None
    if closure:
        lines.append(f"  closure frames {closure[0]}..{closure[1]} "
                     f"({closure[0] * TMS:.0f}-{closure[1] * TMS:.0f} ms) | "
                     f"release frame {release} "
                     f"({(release or 0) * TMS:.0f} ms)")

    for s, e in clusters:
        prev = [(ps, pe) for ps, pe in plateaus if pe <= s]
        ref = body[prev[-1][0]:prev[-1][1]].mean() if prev else np.nan
        seg = body[s:e]
        i_ext = s + int(np.argmax(np.abs(seg - ref)))
        ext = body[i_ext]
        lines.append(f"  cluster frames {s}..{e - 1} "
                     f"({t_ms[s]:.0f}-{t_ms[e - 1]:.0f} ms) | "
                     f"extremum Body = {ext:+.3f} at {t_ms[i_ext]:.0f} ms | "
                     f"amplitude vs plateau {ext - ref:+.3f} "
                     f"({'PEAK' if ext > ref else 'DIP'})")

    state = vlam.VlamState.initial(GUI_LEN_MM)
    cfg = vlam.SynthConfig(play_audio=False)
    sr = vlam.synthwordfen(
        state=state, articulatory_params=P, word_tokens=["O", "V", "F"],
        f0_scale=1.0, soft_rect_s=VALRECT, duration_factor=T,
        envelope=res.envelope, config=cfg)
    F2 = sr.formants[:, 1]
    if clusters and plateaus:
        cs, ce = clusters[0]
        w0, w1 = max(0, cs - 2), min(len(F2), ce + T)
        f2min = F2[w0:w1].min()
        f2mint = (w0 + int(np.argmin(F2[w0:w1]))) * TMS
        lines.append(
            f"  F2 initial (frame 0) = {F2[0]:.0f} Hz | F2 plateau = "
            f"{F2[plateaus[0][0]:plateaus[0][1]].mean():.0f} Hz | "
            f"F2 min in cluster window = {f2min:.0f} Hz at {f2mint:.0f} ms")
    lines.append("")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list = []
    lines.append("Trough-effect measurements and ablations "
                 "(arXiv:2307.02299 / Lindblom et al. 2002)")
    lines.append(f"valrect={VALRECT}, article /b/=(1.0, pi/3), "
                 "/d/=(1.2, 3pi/2), /g/ palatal=(1.1, 23pi/12)")
    lines.append("")

    # ── article consonant positions (as in run_locus_trough_demos) ──
    set_pos("b", 1.0, np.pi / 3)
    set_pos("d", 1.2, 3 * np.pi / 2)
    set_pos("g", 1.1, 23 * np.pi / 12)

    lines.append("=" * 72)
    lines.append("A1 — Body(t) per VCV (T=16) + /ibia/ (T=10)")
    lines.append("=" * 72)
    for w, T in [("ibi", 16), ("aba", 16), ("idi", 16), ("igi", 16),
                 ("ibia", 10)]:
        analyze(w, T, "MAIN", lines)

    lines.append("=" * 72)
    lines.append("A7 — ablations on /ibi/ T=16")
    lines.append("=" * 72)
    BP = ORIG["b"].params  # true default params, reused by every case
    analyze("ibi", 16, "BASELINE b=(1.0, pi/3) params=[0,1,5]", lines)

    set_pos("b", 1.0, np.pi / 3, params=[0, 5])
    analyze("ibi", 16, "ABL-(i) Body removed from Sc (params=[0,5])", lines)
    set_pos("b", 1.0, np.pi / 3, params=BP)

    for rb in (0.5, 1.0, 1.2):
        set_pos("b", rb, np.pi / 3, params=BP)
        analyze("ibi", 16, f"ABL-(ii) rho_b={rb}", lines)

    for tb, nm in [(np.pi, "pi (opposite)"),
                   (3 * np.pi / 2, "3pi/2 (/d/-like)")]:
        set_pos("b", 1.0, tb, params=BP)
        analyze("ibi", 16, f"ABL-(iii) theta_b={nm}", lines)

    # registry-default /b/ — the config of the article-format Figure 1
    set_pos("b", ORIG["b"].rho, ORIG["b"].theta, params=BP)
    analyze("ibia", 10, "A3: registry-default b (rho=1.2)", lines)
    analyze("ibi", 16, "A3: registry-default b (rho=1.2)", lines)
    set_pos("b", 1.0, np.pi / 3, params=BP)

    for c in ("b", "d", "g"):
        analyze(f"i{c}i", 16, f"ABL-(iv) /{c}/", lines)

    SEGMENT_REGISTRY.update(ORIG)

    report = "\n".join(lines)
    print(report)
    out = OUT_DIR / "report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"Report written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
