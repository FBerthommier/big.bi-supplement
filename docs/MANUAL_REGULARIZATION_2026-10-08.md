# Manual regularization report — big.bi → bi.gbi v1.0 (2026-10-08)

Regularization of `docs/manual.tex` against the primary sources
(`P:\bigbi-workspace\2307.02299v1.pdf` and
`P:\bigbi-workspace\lindblom_etal_2002-2.pdf`) and against fresh
measurements. Companion tasks: removal of the S-shaped material and of
`original_simulations/` (author request, same day).

All numbers below come from `scripts/run_trough_ablations.py`
(report: `output/trough_ablations/report.txt`), the re-run of
`scripts/run_locus_trough_demos.py`, or the previous fresh-run gates
(2026-10-07, `docs/VALRECT_110.md`). valrect = 1.10 everywhere.

## 1. Verification table

| Item | Manual claim (before) | Measured value | Label | Status |
|---|---|---|---|---|
| A2 /i/ plateau | Body ≈ 0 before /b/ | **−2.250** (ibi, idi, igi); +1.000 (aba) | [VÉRIF-CODE] | corrected |
| A2 excursion sign | "trough/dip at −2.25, t=160 ms" | **PEAK +1.250 at 630 ms** (T=16); +1.250 at 390 ms (ibia T=10); the −2.25/160 ms reading was an argmin artifact over the /i/ plateau | [VÉRIF-CODE] | corrected |
| A3 amplitude 1.25 vs 1.5 | unexplained | ρb=1.0 (trough demo) → **+1.250**; registry default ρb=1.2 (Fig. 1, planning panels) → **+1.500** | [VÉRIF-CODE] | documented |
| A4 Sv/Sc partition | "Sv={3,4,5,7} stay /i/ while Sc={1,2,6} drives Body" | Body belongs to BOTH sets; **sequential overwrite**: plateaus write all 7 params from z_v, cluster blocks overwrite Sc params from z_c (`synthSYL/trajectory.py`, `inject_active_parameters()`, l.156–167) | [VÉRIF-CODE] | documented |
| A5 overlay claim | "most pronounced for /b/ (ibi, aba)" | ibi **+3.50**, aba **+0.298** (weak), idi +0.022, igi +0.305 | [VÉRIF-CODE] | corrected |
| A6 "S-shaped F2" | starts at /i/ ≈1830 Hz, dip 1564 Hz | F2(frame 0) = **1826 Hz** (neutral onset, not /i/); /i/ plateau **2276 Hz**; V-minimum **1630 Hz** at 630 ms (ρb=1.0) / **1570 Hz** (ρb=1.2); single minimum, not an S | [VÉRIF-CODE] | corrected, "S-shaped" removed |
| A7 ablation (i) | — | Body out of Sc: excursion +3.50 → **+0.003**; F2 dip gone | [VÉRIF-CODE] | added (table) |
| A7 ablation (ii) | — | ρb = 0.5/1.0/1.2 → excursion +2.875/+3.500/+3.750; F2 min 1773/1630/1570 Hz | [VÉRIF-CODE] | added |
| A7 ablation (iii) | — | θb=π → excursion unchanged (+3.500; Body responds to ρ) but F2 min differs (1582 Hz); θb=3π/2 → excursion +0.086, no dip | [VÉRIF-CODE] | added |
| A7 ablation (iv) | — | /b/ +3.500, /d/ +0.022, /g/ +0.305 | [VÉRIF-CODE] | added |
| A8 "confirms the trough effect" | confirmation claim | Lindblom et al.: "relatively small" deviations, **movement-related data only, never acoustic** (their Introduction); simulated +3.50 is an order of magnitude larger → "**qualitatively consistent with**", limitation stated | [ÉCART-LITTÉRATURE] | reworded |
| B1 §8.1 table | "Added Ve plateau (full-amplitude) before decay block" | revoked 2026-10-07 (§4.3) | [ÉCART-INTERNE] | corrected |
| B2 conclusion | "3.06T ≈ 490 ms matches 3T" | measured **1T = 160 ms** (dot form, §7.1); 3T reading belonged to the space form incl. pause machinery | [ÉCART-INTERNE] | aligned |
| B3 §5.2 vs §12.2 | "same U-shaped Tp" (nonrev 160..10..0 vs reversible 160..0) + durations | nonrev descent passes **10 ms**, reversible has 0 at bottom; 16.9 s vs 18.6 s, both 10 segments | [ÉCART-INTERNE] | clarified |
| B4 slopes | "reproduced", "agrees" | /b/ 0.703 (art. ≈0.70 ✓); /d/ 0.421 vs ≈0.55 (flatter); palatal /g/ 0.630 vs ≈0.75; velar 2.335 vs ≈2.0 — ordering/trends preserved, quantitative deviations stated; velar r²=1.000 on **3 points** (barely informative); combined caption: slope>1 = opposite regime | [ÉCART-ARTICLE] | reworded |
| B5 numbering | "Demo 1,2,4,5" (no 3); Eq. art. 4 numbered (5) | renumbered **Demo 1/2/3/4** (S-shaped demo removed); blend equation now tagged **(art. 4)**; Fig. 4 subsection notes the δo=0.5/δe=1 article conditions vs the Demonstration-1 δ-sweep | [ÉCART-INTERNE] | corrected |
| B6 0/1-based sets | — | re-checked: Sc={1,2,6}/{0,1,5}; Sv={3,4,5,7}/{2,3,4,6}; clusters {1,2,3,6}/{0,1,2,5} and {1,2,3,4}/{0,1,2,3} — all consistent | [VÉRIF-CODE] | OK |
| B7 numeric values | — | F2 1826/2276/1630/1570 (fresh); durations 24.2/18.6/16.9 s (2026-10-07 gates); F0 119–124 Hz; HF>5kHz 0.15/0.12 % — unchanged | [VÉRIF-CODE] | OK |
| C2 gates | — | demo2 locus figures regenerated **bit-identical** (same F2 table → slopes 0.703/0.421/0.630/2.335); trough_effect_polar_explanation.png bit-identical; article-format figure4 unchanged | [VÉRIF-CODE] | OK |

## 2. Manual changelog (section — old → new — reason)

1. **§10 Demo 1** — "(trough effect visible on Body)" → "Body excursion toward /u/ during /b/" with measured amplitudes (+1.50 registry / +1.25 article ρb) — the excursion is a peak, and the amplitude depends on ρb.
2. **§10 Demo 1(c)** — "S-shaped pattern: starts at the /i/ value (≈1830 Hz), dips to 1564 Hz" → "rises from the neutral onset 1830 Hz to the /i/ plateau 2280 Hz, V-shaped dip (min 1570 Hz at ρb=1.2)" — 1830 Hz is the frame-0 neutral value, not /i/; the dip is a single minimum; measured at valrect 1.10.
3. **§10 Demo 1 caption** — "(trough effect visible on Body)" / "(S-shaped F2)" → excursion / V-shaped dip wording — same reason.
4. **§10 Demo 2 caption + results** — "Slopes < 1 indicate coarticulation" completed with the slope>1 velar regime; "reproduced/agrees" → /b/ matches, others deviate quantitatively (0.42 vs 0.55; 0.63 vs 0.75; 2.34 vs 2.0) while ordering/trends are preserved; velar r²=1.000 on 3 points flagged — B4.
5. **§10 Demo 2 note** — "and the S-shaped F2 trajectory analysis" dropped from the companion-article pointer — task 1.
6. **§10 Demo 4 → Demo 3 (trough), full rewrite** — old: "trough at −2.25, t=160 ms, Body before ≈0", "confirms the trough effect"; new: Perkell/Lindblom definition (primary-source quote), measured peak table (+1.250 @ 630 ms, cluster frames 48–79), sequential-overwrite mechanism with file/line, ρb 1.0-vs-1.2 amplitude explanation, full ablation table, "qualitatively consistent with" + limitation — A2/A4/A5/A7/A8.
7. **§10 Demo 5 → Demo 4 (clusters)** — renumbering only — B5.
8. **§11.6** — "Original article simulations (exact reproduction)" removed (3 subsubsections + 2 figures from `original_simulations/`); replaced by "Figure 4: big.bi | bi.gbi (article format)" keeping the article-format figure + a removal note + a δ-convention clarification (B5) + a note that the /ib/→/bi/ 3-case stored figure belonged to the removed set.
9. **§7.2** — WAV locations: `original_simulations/*/` → `output/article_demos/` and `output/trough_ablations/` — task 2.
10. **§8.1 table** — trajectory.py row: "Added Ve plateau" → "REVOKED addition … decay chains directly on the word-end anchor (no held @)" — B1.
11. **§13 Conclusion** — "3.06T ≈ 490 ms matches 3T" → measured 1T = 160 ms in the dot form; the 3T reading belonged to the space form — B2.
12. **§12.2 (non-reversible HTML)** — "the same U-shaped Tp as the reversible demonstration" → "a U-shaped Tp like the reversible demonstration (descent passes through 10 ms…)", durations 16.9 s vs 18.6 s made explicit — B3.
13. **Equation (art. 4)** — blend equation now manually tagged `(art. 4)` instead of colliding with the manual's own counter — B5.

## 3. Code / config modifications (no engine file touched)

- `scripts/run_locus_trough_demos.py` — demo 4 (trough) rebuilt on RECORDED engine blocks (`polar_sync.record_pipeline`): consonant-region shading now = actual cluster frames (480–790 ms), analysis reports plateau/extremum/instant/sign instead of the biased argmin; docstrings corrected (Body = Pval column 1; peak wording); `polar_sync` import added.
- `scripts/run_article_demos.py` — Demo 3 (S-shaped F2) REMOVED (function, VCV list, dirs, main); Demo 1 label "F2 (S-shaped)" → "F2"; Body highlight comment updated.
- `scripts/make_article_figures.py` — fig1 panel (b) title "(trough effect on Body…)" → "(Body excursion toward /u/ during /b/)"; panel (c) label "F2 (S-shaped)" → "F2".
- `scripts/run_trough_ablations.py` — NEW: A1 measurements + A7 ablations, report to `output/trough_ablations/report.txt`.
- `scripts/run_original_simulations.py` — DELETED (with its directory, see §4).
- No `vlam.py` / `synthSYL/` change.

## 4. Removed / regenerated artifacts

- REMOVED: `original_simulations/` (figure1_ibia, figure4_bigbi_bigbi, supplement_ib_bi — figures, WAVs, npz) and `docs/figures/demo3/` (s_shaped_f2_trajectories.png, f2_overlay.png); `.gitignore` exceptions updated.
- REGENERATED: `docs/figures/demo4/trough_effect.png` + `trough_effect_overlay.png` (block-based shading, corrected titles); `docs/figures/article_format/figure1_ibia_4panel.png` (panel titles/labels); `docs/figures/demo1/ibia_4panel.png` + `ibia.wav` (synced); `docs/figures/demo2/*` and `figure3_*`, `figure4_bigbi_bigbi.png`, planning panels, `trough_effect_polar_explanation.png` — regenerated but **bit-identical** (gates).
- PATCHED: `simulations/trough_effect.html` — aba claim corrected ("most pronounced for (ibi, aba)" → strong for /ibi/, weak for /aba/ +0.30), section title "(trough visible)" → "(Body excursion visible)", shading wording (recorded cluster block), A8 limitation paragraph ("qualitatively consistent with", "relatively small", movement-data-only), 3 figure blobs swapped to the regenerated PNGs.
- REBUILT: `docs/manual.pdf` — 26 pages, 0 unresolved references.

## 5. Author rulings on the decision points (2026-10-08, second pass)

1. **ρb for the article-format Figure 1** → **KEEP the registry default
   ρb = 1.2** (homogeneity with Figure 1's Body peak +1.50). The
   article value is CONFIRMED in the arXiv source, §3 verbatim:
   "we found that /b/ is easy to reach with (ρb = 1, θb = π/3)"; the
   manual now quotes it and documents the 1.2-vs-1.0 deviation.
2. **valrect** → **1.10 confirmed**; deviations flagged where they
   influence a number (Demo 2 slope table; Demo 3 formant note added —
   Body values are Pval-level and valrect-independent).
3. **/ib/→/bi/ supplement figure** → **REGENERATED at the current
   engine** (`scripts/make_supplement_ib_bi.py` →
   `docs/figures/article_format/supplement_ib_bi.png`): 3 cases
   (/ib/ δ=0.5, /ib/ δ=1.0, /bi/ δ=1.0), gate verified — /ib/ = 112
   steps = 1120 ms at BOTH deltas, cluster followed directly by the
   decay (no held @, unlike the removed pre-revocation reproduction).
   Referenced in manual §11.6 with a figure environment.
4. **Slope deviations** → explained in the manual (Demo 2, "Why the
   slopes deviate from the article"): a locus slope is a ratio against
   the F2vowel spread; the article-era synthSYL vowel table (corner
   vowels at ρ=1) was revised, compressing the F2vowel axis
   (/i/ 2290 vs ≈2660 Hz, /u/ 1029 vs ≈880 Hz), and the line with the
   highest intercept (/d/) moves the most. Coordinate-table
   difference, not a change in coarticulation behaviour.
5. **Tip c0 = −2.75** → left as-is (author ruling: "à laisser passer").
