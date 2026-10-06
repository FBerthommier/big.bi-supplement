# Trajectory matching: reference big/.bi/bi/.gbi panels vs synthSYL

**Date:** 2026-10-06 · **Reference:** author-provided figure with four
polar planning panels (`big`, `.bi`, `bi`, `.gbi`) + AP-model gesture
activations (472×231 px screenshot; high-resolution source not located
— scanned 156 PDFs on the workstation without a hit, including the
Interspeech 2025/2026 supplements and the comparison drafts).

## Method

- Reference side: colour segmentation of the four panels (blue =
  consonantal branch; the red branch is nearly invisible at this
  resolution) + vision-based estimates of the target layout
  (/i/ ≈ 90°, /b/ ≈ 220–240°, /g/ ≈ 300–320°, /@/ ≈ 270°).
- synthSYL side: exact replay via `scripts/polar_sync.py`
  (T = 16, δo = δe = 0.5, z_v K = 30, z_c K = 10). Panels: `big` =
  word 1 of "big bi"; `.bi` = word 2 of "big bi" (sliced at the pause
  arc, i.e. correct word-2 boundary conditions); `bi` = standalone;
  `.gbi` = standalone cluster word.
- Figures: `P:\bigbi-workspace\match_bigbi_4panels.png` (reference row
  + synthSYL row; workspace only — the reference image stays out of the
  repository) and `docs/figures/planning_4panels_syntsyl.png`
  (synthSYL-only version, in the repository).

## Result — gestural topology: 4/4 panels match

| panel | reference (blue excursions) | synthSYL z_c spans | synthSYL apexes (θ, ρ) | red branch |
|---|---|---|---|---|
| `big`   | 2 excursions: /b/ then /g/ | 2 (steps 16–47, 64–95) | (60°, 1.20) = /b/; (345°, 1.10) = palatal /g/ | loop ρ 0.45–0.90 on the /i/ radius, Δθ ≈ 2° |
| `.bi`   | 1 excursion: /b/ | 1 (16–47) | (60°, 1.20) | idem |
| `bi`    | 1 excursion: /b/ | 1 (16–47) | (60°, 1.20) | idem |
| `.gbi`  | 1 fused excursion /g/→/b/ | 1 span of 48 samples (cluster /gb/: 3 sub-arcs) | max (60°, 1.20); visits /g/ (345°, 1.10) then /b/ | idem |

The structural claim of arXiv:2307.02299 §4 is reproduced exactly:
`big` carries two separate consonantal gestures, `.gbi` one fused
cluster gesture (single span, g→b), and the vocalic branch stays on
the /i/ radius (schwa loop to ρ = δ·ρ_i), never through the origin.

## Geometric layout — differs by more than a rotation

synthSYL (article §2.1 table): /i/ 300°, /b/ 60° (ρ 1.2 default,
1.0 article), palatal /g/ 345°, /@/ on the /i/ radius. Separations:
/i/–/b/ = 120°, /b/–/g/ = 75°.
Reference (estimated at low resolution): /i/ ≈ 90°, /b/ ≈ 230°,
/g/ ≈ 310°. Separations: /i/–/b/ ≈ 140°, /b/–/g/ ≈ 80°.

No single rotation maps one layout onto the other (rotating synthSYL
by +150° aligns /i/ exactly but leaves /b/ ≈ 10–30° off and the
palatal /g/ ≈ 150° off): the reference figure uses a different polar
target table (AP model / pantimit-family layout). Angular separations
are nevertheless comparable (within ~20°).

## Article-figure conditions (COEFCEN = 1, VOYDEB = 0.5)

The article's Figure 4 was generated under the legacy parameters
**COEFCEN = 1** and **VOYDEB = 0.5** — i.e. in synthSYL terms
`delta_e = 1.0` (Ve = V: full vowel at the syllable end, no schwa on
the coda side) and `delta_o = 0.5` (Vo at half radius: schwa-like
onset anchor). This differs from the symmetric δo = δe = 0.5 used for
the first comparison figure. Naming check: `types.py` maps
`is_voydeb → is_vowel_onset` (VOYDEB = onset coefficient) and
`constants.py` defines COEFCEN as the end/centre-weight coefficient.

Under these conditions (`docs/figures/fig_bigbi_article_conditions.png`,
T = 16, z_v K = 30 / z_c K = 10):

| utterance | z_c gestures | apexes (θ, ρ) | red branch |
|---|---|---|---|
| `big.bi` | **3** (b → g → b) | (60°, 1.20) ; (345°, 1.10) palatal g ; (60°, 1.20) | ρ 0.45–0.90: schwa at word ONSETS (Vo), full V elsewhere (Ve = V) |
| `bi.gbi` | **2** (b + fused /gb/) | (60°, 1.20) ; gb span visits g (345°, 1.10) then b | idem |

The three-gesture vs two-gesture contrast (coda /g/ as a separate
gesture in big.bi vs the single fused /gb/ cluster gesture in bi.gbi)
is the §4 syllabification demonstration, now reproduced under the
article's exact figure parameters.

### nu inversion (author's reading of the article figure)

Comparing the above figure with the article's Figure 4, the author
reads the arc convexity as MIRRORED: ν flips convexity into
concavity. The display convention is therefore inverted —
`docs/figures/fig_bigbi_article_conditions_nu_inv.png` shows the same
simulation (COEFCEN=1, VOYDEB=0.5, T=16) with **ν_v = −1, ν_c = +1**
(was +1 / −1). Gesture topology is unchanged (same spans, same
apexes: big.bi b→g→b; bi.gbi b + fused /gb); only the bow direction
of every arc is mirrored.

Convention note: the engine drives ALL arcs (vowel backgrounds and
consonant sub-arcs alike) with `DEFAULT_NU = −1` via `arc_B`; the
per-branch ν (+1/−1) is a DISPLAY convention inherited from the
reference display scripts. If the ν_v=−1 / ν_c=+1 convention is
confirmed, it should be propagated to `polar_sync` defaults, the
sweep/video scripts and covtl-pipeline's `polar_video.py`
(NU_VOCALIC/NU_CONSONANTAL) for consistency.

### The 'bi' component of big.bi (why it was invisible) + word panels

Under the article conditions the word-2 onset cluster of "big bi"
retraces EXACTLY the word-1 onset path — blocks (0.45, 300°) →
b (1.20, 60°) → (0.90, 300°) for both words (Vo = 0.45·i at δo = 0.5,
Ve = V at δe = 1) — so the /b/ of "bi" superposes the /b/ of "big"
stroke-for-stroke in the merged panel. It is not missing from the
model; it is hidden by perfect superposition.
`docs/figures/fig_words_article_conditions_nu_inv.png` shows the four
word-level panels under the same conditions (big | .bi | bi | .gbi;
".bi" sliced at the inter-word pause, ".gbi" at the syllable dot),
display ν_v = −1 / ν_c = +1. Metrics: big = 2 gestures (b, palatal g);
.bi = 1 (b); bi = 1 (b); .gbi = 1 fused /gb/ span visiting g then b.

### ν convention — confirmed: no confusion in the engine, only in the display

Verification against Eq. 2 of arXiv:2307.02299
(`z(t) = (1−ρ)ρ₁e^{iθ₁} + ρρ₂e^{i(θ₂ + (ν/K)θ(t))}`, arrival phase with
PLUS sign):

* **synthSYL engine — no confusion.** `synthSYL/polar.py: arc_B` writes
  `cos(Ψ2 − θ2 − (ν/K)·θ(t))`, i.e. an arrival angle θ2 + (ν/K)θ —
  VERBATIM Eq. 2 (same sign, no notational flip). The engine runs the
  single value ν = −1 (`DEFAULT_NU`) on every arc, vowel backgrounds
  and consonant sub-arcs alike, and its outputs reproduce the
  article's validated numbers (locus equations, trough, S-shaped F2).
* **Display — the confusion reappears.** `polar_primitives.polar_arc`
  uses the same Eq.-2 sign, so display ν is directly comparable — but
  the inherited per-branch values (ν_v = +1, ν_c = −1) produced bows
  mirrored versus the article's planning figures on BOTH branches
  (author's reading, 2026-10-06; z_v follows the same rule as z_c).
  Corrected display convention: **ν_v = −1, ν_c = +1**, now the
  `polar_sync.build_branches` defaults and documented in
  `polar_primitives.py`. This is a display convention: it does not
  mirror the engine's driven arcs (engine ν = −1 everywhere).
* **covtl-pipeline — treat SEPARATELY.** covtl's active `syl` engine
  writes the cosine with the OPPOSITE sign, so its ν = −1 is
  numerically Eq.-2 ν = +1 (covtl manual §5: "a notational convention
  only — ν = 1 is recovered when the cosine is written with the
  opposite sign"). Display ν values are therefore NOT transferable
  between the two engines; the convention note has been added to both
  display engines independently.

## Limitations / next step

The 472-px reference allows topology matching and target-layout
estimation, not point-by-point trajectory residuals. For a
quantitative point-wise matching (e.g. mean Fréchet distance after
layout alignment), the high-resolution source of the reference figure
(PDF or generator program) is required.
