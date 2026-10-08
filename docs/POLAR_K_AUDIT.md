# Polar display-curvature audit — K = 10 for consonant arcs (z_c)

**Date:** 2026-10-05 · **Scope:** pre-push revision of `P:\bigbi-demos`
against arXiv:2307.02299.

The article (§2.1) prescribes: *"When K is large (K = 30 for vowel arcs
and K = 10 for consonant arcs), the trajectories become straight
lines…"* — i.e. **K = 30 ("Kvoy") for the vocalic branch z_v and
K = 10 for the consonantal branch z_c** in every polar *display*.
The acoustic engine already follows this convention
(`synthSYL/constants.py`: `DEFAULT_K = 10`, `DEFAULT_KVOY = 30`); the
engine was **not** modified at any point in this audit.

## 1. Audit table (display calls to `polar_arc`)

| Script | z_v K before → after | z_c K before → after | ν (z_v / z_c) | Import before → after |
|---|---|---|---|---|
| `scripts/run_article_demos.py` | 30 → 30 | 10 → 10 | +1 / −1 | `vtl_synth.core.polar` → `polar_primitives` |
| `scripts/run_original_simulations.py` | 30 → 30 | 10 → 10 | +1 / −1 | `vtl_synth.core.polar` → `polar_primitives` |
| `scripts/run_bigbi_polar_sweep.py` | 30 → 30 | 10 → 10 | +1 / −1 | `vtl_synth.core.polar` → `polar_primitives` |
| `scripts/run_ibbi_ushape_nonreversible_en.py` | 30 → 30 | 10 → 10 | +1 / −1 | `vtl_synth.core.polar` → `polar_primitives` |
| `scripts/run_ibbi_ushape_sweep.py` | 30 → 30 | **30 → 10** | +1 / −1 | `vtl_synth.core.polar` → `polar_primitives` |
| `scripts/make_ibbi_polar_video.py` | 30 → 30 | **30 → 10** | +1 / −1 | `vtl_synth.core.polar` → `polar_primitives` |
| `scripts/make_bigbi_dual_video.py` | — (loads `zvc_*.npz` written by the sweep) | — | — | no polar import |
| `scripts/run_locus_trough_demos.py` | — (no polar-arc display; the K constants are documented conventions only) | — | — | — |

**Two scripts really had the defect** (z_c drawn with K = 30, i.e.
near-straight consonant excursions); four already used K = 10. All now
declare `K_DISPLAY = 30.0` / `K_C_DISPLAY = 10.0` explicitly.
`polar_primitives.polar_arc` keeps the vocalic default K = 30 and every
consonant caller passes K = 10 (article §2.1).

`vtl_synth.core.polar.polar_arc` and `polar_primitives.polar_arc` were
verified to be **numerically identical** implementations, so the import
switch is a no-op on values (confirmed by the 0.000e+00 reference
comparison below); it removes the external covtl-pipeline dependency.

Stale "K=30" captions were also corrected: demo-5 polar suptitle,
`run_bigbi_polar_sweep` figure footnote, `make_ibbi_polar_video`
footnote + startup print, and the `z_c (blue, K=30, ν=−1)` description
in the three embedded-video HTML pages.

## 2. Quantified curvature check (why K = 10, not K = 30)

Sagitta (max perpendicular deviation from the chord) of the /ibia/
consonant sub-arcs, default synthSYL targets
(/i/ = (0.9, 5π/3), /b/ = (1.2, π/3)), T = 160 ms, ν = −1:

| arc | K | chord | sagitta | sagitta/chord |
|---|---|---|---|---|
| i→b | 1 | 1.825 | 0.532 | 29.2 % |
| i→b | 3 | 1.825 | 0.260 | 14.3 % |
| i→b | **10** | 1.825 | 0.086 | **4.7 %** |
| i→b | 30 | 1.825 | 0.030 | 1.6 % |
| b→i | 1 | 1.825 | 0.693 | 38.0 % |
| b→i | 3 | 1.825 | 0.227 | 12.5 % |
| b→i | **10** | 1.825 | 0.064 | **3.5 %** |
| b→i | 30 | 1.825 | 0.021 | 1.1 % |
| i→a (z_v, ν=+1) | 30 | 1.473 | 0.018 | 1.2 % |

At K = 30 the consonant arcs sag no more than the vocalic arcs
(≈ 1–2 % of the chord: effectively straight — the defect reported as
"arcs consonantiques trop rectilignes"). At K = 10 they keep a clearly
visible bow (3.5–4.7 %), matching the article's Figure 4 planning
trajectories. The regenerated comparison figure
`docs/figures/comparison_polar.png` shows our δ = 0.995 / δ = 1.00
renders with the K = 10 curvature; it is to be compared visually with
the top row of the article's Figure 4 (arXiv:2307.02299, viewed on the
arXiv site or a local copy kept outside the repository — no article
image is redistributed in this repo).

## 3. Defect found during regeneration: display arrays truncated to engine length

While regenerating, the freshly rendered videos came out ~10× too short
(3.5 s instead of 26 s) and the regenerated demo-1 panel (a) had **no
z_c curve at all**. Root cause (pre-existing in the repository copies of
the scripts, **not** in the sandbox outputs):

`reconstruct_polar_branches()` samples z_v/z_c at
`SR_DISPLAY = 1000 Hz` (10 × the 100 Hz engine grid) but its final
alignment block trimmed/padded them to `n_steps = Pval.shape[0]`
(the engine grid), keeping only the **first 10 %** of each display
trajectory. Consequences before the fix:

* `original_simulations/figure1_ibia/ibia_data.npz`: z_c was **100 % NaN**
  (90 samples of a 900-sample trajectory), while the article's Figure 1a
  and the sandbox figure (`docs/figures/demo1/ibia_4panel.png`, z_c
  spanning 68.9 % of the time axis) clearly show z_c;
* the embedded big.bi video was 25.98 s in the sandbox version but the
  repo script produced 3.49 s (7 frames/segment);
* video subtitle timing and sagittal-tract state indices mixed the two
  sampling grids.

Fix applied to `run_article_demos.py`, `run_original_simulations.py`,
`run_bigbi_polar_sweep.py`, `run_ibbi_ushape_nonreversible_en.py`
(align to `disp_len = n_steps × SR_DISPLAY / SR_GESTURE`; display→engine
index mapping `// 10` for tract states / phoneme labels; trail length
computed at SR_DISPLAY; time labels at display rate) and to
`make_bigbi_dual_video.py` (loads the display-rate `zvc_*.npz`).
`run_ibbi_ushape_sweep.py` and `make_ibbi_polar_video.py` were already
self-consistent at 100 Hz and are untouched.

Evidence after the fix:

* demo-1 panel (a): z_c spans 68.96 % of the time axis (sandbox figure:
  68.89 %; article Figure 1a shows the same transient structure);
* big.bi dual video: **25.98 s**, matching the previously embedded
  sandbox video (25.98 s / 642 frames); U-shape dual 28.46 s;
  non-reversible dual 19.82 s;
* the three `simulations/*_embedded.html` pages were re-embedded with
  these K = 10, full-length videos and refreshed posters.

## 4. Reference data update (`original_simulations/figure1_ibia/ibia_data.npz`)

> **NOTE (2026-10-08):** `original_simulations/` was removed from the
> repository after this audit (discordant with the current engine); the
> equivalent figure is `docs/figures/article_format/figure1_ibia_4panel.png`
> (generated by `scripts/make_article_figures.py`). The historical
> record below is unchanged.

The stored z_v/z_c arrays were the truncated (degenerate) display
branches described above; z_v/z_c are **display-only** quantities (they
never feed `vlam.synthwordfen`). The reference was regenerated after the
fix and now stores full display-rate arrays (900 samples, plus a
`t_z_ms` key). Engine arrays are unchanged:

| key | role | max |diff| (before fix vs after fix) |
|---|---|---|
| Pval | engine | 4.441e-16 |
| formants | engine | 8.640e-12 |
| sig | engine | 2.026e-13 |
| z_v, z_c | display | shape 90 → 900 (was truncated; z_c was all-NaN) |
| t_ms | display | 0 (unchanged) + new `t_z_ms` |

A fresh run of `scripts/run_original_simulations.py` now reproduces
every array of the updated reference at exactly 0.000e+00.

## 5. Regenerated artefacts

* `output/article_demos/` (demos 1/3/5) → copied to `docs/figures/demo1`,
  `demo3`, `demo5`;
* `docs/figures/demo2`, `demo4` (locus slopes 0.704 / 0.412 / 2.107 /
  0.644 — unchanged, engine untouched; trough depth 2.25 at 160 ms);
* `output/bigbi_polar_sweep/` + `make_bigbi_dual_video.py` →
  `simulations/bigbi_bi_gbi_embedded.html`,
  `docs/figures/polar_bigbi_delta05.png`, `polar_bigbi_fused.png`,
  `comparison_polar.png`;
* `output/ibbi_ushape_en/` → `simulations/ibbi_ushape_embedded.html`;
* `output/ibbi_ushape_nonrev_en/` →
  `simulations/ibbi_ushape_nonreversible_embedded.html`;
* `original_simulations/figure1_ibia|figure4_bigbi_bigbi|supplement_ib_bi`
  figures + npz.

No change to the acoustic engine (`vlam.py`, `synthSYL/*.py`); the S-shaped
F2, locus-equation and trough-effect numbers are bit-identical to the
previously validated values (see `docs/LOCUS_FIG3_DIAGNOSTIC.md`).

## 6. Validation checkpoints (2026-10-05, clean venv)

* Demo 2 locus slopes: /b/ = 0.704, /d/ = 0.412, /g/velar = 2.107,
  /g/palatal = 0.644 — unchanged.
* Demo 4 trough (article /b/ = (1.0, π/3), set by the script): Body
  /ibi/ = −2.25 → **+1.25**, excursion +3.50. (With the synthSYL
  default /b/ = (1.2, π/3) the maximum would be +1.50; the script
  explicitly selects the article position, matching the article's
  Figure 1b and the trough HTML page.)
* `run_original_simulations.py` reference comparison after the update:
  all arrays (Pval, formants, sig, z_v, z_c, t_ms, t_z_ms) reproduced
  at max |diff| = 0.000e+00.
* All five entry scripts ran in a clean Python 3.14 venv with only
  `requirements.txt` + `requirements-vlam.txt` installed.

