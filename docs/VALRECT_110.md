# valrect = 1.10 — author ruling (2026-10-07) and reference re-baseline

## Decision

After comparative listening of the test WAVs
(`P:\bigbi-workspace\valrect_test_{bibi,ibib}_{075,09,11,15}.wav`,
synthesized at valrect 0.75 / 0.90 / 1.10 / 1.50), the author set
**valrect = 1.10** for every script and artifact of this repository
(ruling of 2026-10-07). Motivation: reduce the perceived *fizz* on
/i/. The historical canonical value **0.75** remains the default of
the original `FBerthommier/timit-to-Maeda` package and is kept in the
documentation as a historical/comparative value only.

`valrect` (`soft_rect_s`) is a **synthesis parameter**, not a formula:
it is the smoothness `s` of the soft rectification applied to the
**area function** (`vlam.soft_rect`, prevents zero/negative sections
during closure). No engine formula was modified for this change; only
default values and call sites moved from 0.75 to 1.10
(`config.py`, `vlam.synthwordfen`, `batch_synthesize.py --valrect`,
`VLAMvidmaker.py --valrect`, all `scripts/run_*.py`,
`scripts/make_article_figures.py`, `examples/*.py`).

## Measured effects (fresh runs, 2026-10-07)

Audio quality proxies ("bi bi" / "ib ib" test utterances,
20 kHz, autocorrelation F0, FFT energy ratio above 5 kHz):

| valrect | HF > 5 kHz ("bi bi") | HF > 5 kHz ("ib ib") | F0 median | duration |
|---|---|---|---|---|
| 0.75 | 0.17 % | 0.13 % | ≈ 119–124 Hz | unchanged |
| 1.10 | 0.15 % | 0.12 % | ≈ 119–124 Hz | unchanged |

No digital clipping (< 0.02 % of samples at peak at either value).

Locus-equation slopes (Demo 2 protocol: F2 at release + 30 ms,
delta_o = 0.5, T = 16), measured at both values:

| Consonant | slope @ 0.75 | slope @ 1.10 |
|---|---|---|
| /b/ | 0.704 | 0.703 |
| /d/ | 0.412 | 0.421 |
| /g/ palatal | 0.644 | 0.630 |
| /g/ velar | 2.107 | 2.335 |

The ordering a(/d/) < a(/b/) < 1 < palatal < velar and the agreement
with the article slopes (≈ 0.70 / 0.55 / 0.75 / 2.0) are preserved;
the manual's Demo-2 table now carries the 1.10 values.

## Reference re-baseline (`original_simulations/`)

`valrect` feeds the synthesizer only: the planning/engine arrays do
not depend on it. Verified by regenerating Figure 1 (/ibia/, T=100 ms)
at 1.10 and comparing against the previous (0.75) reference
`original_simulations/figure1_ibia/ibia_data.npz`:

| array | max abs diff (0.75 reference vs 1.10 run) |
|---|---|
| Pval | **0.000e+00** (engine — valrect-independent) |
| z_v, z_c, t_ms, t_z_ms | 0.000e+00 (display) |
| formants | 9.364e+01 (Hz — expected: synthesis-level change) |
| sig | 4.065e-01 (expected) |

Consequently the shipped reference was **re-baselined at 1.10** on
2026-10-07: `original_simulations/figure1_ibia/ibia_data.npz` (and the
shipped WAVs of Figures 1/4 and the supplement) now contain the 1.10
synthesis. The engine gate stays exact: a fresh
`scripts/run_original_simulations.py` run must match the shipped
reference at 0.000e+00 for every array, Pval included.

Other 2026-10-07 gates re-verified at 1.10:

- supplement /ib/ (δ = 0.5 and δ = 1.0): 112 steps = **1120 ms** (not
  1280), block sequence arc → attack → plateau → **cluster → decay**
  (no held @; decay anchor = (δ·ρ_i, θ_i) = (0.45|5.236) resp.
  (0.90|5.236)) — now asserted in `run_original_simulations.py`;
- embedded dual-video durations: big.bi/bi.gbi **24.21 s** (14
  segments), /ibi/ reversible **18.64 s** (10), /ibi/ non-reversible
  **16.91 s** (10) — identical to the 0.75 round (valrect does not
  change timing);
- trough effect (Pval-level, valrect-independent): /ibi/ Body
  −2.25 (/i/) → +1.25 (toward the /u/ direction) during /b/;
  `simulations/trough_effect.html` audio re-synthesized at 1.10 with
  the position setup validated against its documented Body values
  (+1.25 / +1.30 / −2.26 / −1.95).

## Artifacts regenerated at 1.10 (2026-10-07)

- `docs/figures/demo1|demo3|demo5` (article demos),
  `docs/figures/demo2|demo4` (locus + trough),
  `docs/figures/article_format/` (Figures 1/3/4),
  `docs/figures/fig_bigbi_article_conditions_nu_inv.png`,
  `fig_words_article_conditions_nu_inv.png`, `planning_4panels_syntsyl.png`,
  `docs/figures/comparison_polar.png` (12-segment polar sweep);
- `original_simulations/**` (figures, WAVs, reference npz — re-baselined);
- `simulations/bigbi_bi_gbi_embedded.html`,
  `simulations/ibbi_ushape_embedded.html`,
  `simulations/ibbi_ushape_nonreversible_embedded.html`
  (video + poster + rebuilt chapters/WebVTT + duration chips),
  `simulations/trough_effect.html` (4 audio blobs + footer).

See `docs/LOCUS_FIG3_DIAGNOSTIC.md` for the (unchanged) sensitivity
analysis and `docs/manual.tex` §"Audio verification" for the listening
context.
