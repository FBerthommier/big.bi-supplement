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

Under these conditions
(`docs/figures/fig_bigbi_article_conditions_nu_inv.png`,
T = 16, z_v K = 30 / z_c K = 10; the earlier +1/−1-ν rendering
`fig_bigbi_article_conditions.png` was superseded by the
ν-inversion ruling below and removed):

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

### Dot concatenation (C.C) — the reference semantics, and the pause fix (2026-10-07)

Author ruling: **`Syllable_Synthesis` is NOT the reference code for the
word/syllable-boundary question** (it has no `.`); the reference is
`FBerthommier/Timit-to-Maeda`. Its documented semantics
(`synthSYL/parsing.py::_should_split_at_dot`):

* `C.C → split` (keep the syllable boundary) — big.bi: the g|b dot
  splits big + bi, **no pause block**;
* `C.V, V.C, V.V → merge` (dot ignored) — bi.gbi: the i|g dot merges,
  giving the fused /gb/ onset cluster (this is WHY bi.gbi fuses).

And `synthSYL/gesture.py::_insert_syllable_onset_anchors` weights the
boundary onset anchor with **COEFCEN** (the syllable-chaining
coefficient), not VOYDEB:
`weight = (1 − prev_boolast) + COEFCEN·prev_boolast·curr_booldeb`.
Under the article conditions (COEFCEN = 1) the anchor IS the full
previous vowel: **the Ve of "big" is coarticulated with the /i/ of
"bi" — the very definition of the '.' concatenation between two
consonants.**

Defects fixed accordingly:

1. **Engine** (`synthSYL/gesture.py`): the syllable-onset weight used
   `delta_o` (VOYDEB, the word-INITIAL onset coefficient) instead of
   `delta_e` (COEFCEN) — under COEFCEN=1/VOYDEB=0.5 the anchor sat at
   0.45 instead of the full /i/ (0.9). Fixed to `delta_e` (reference
   formula). All validated reproductions use symmetric deltas and stay
   bit-identical (0.000e+00 re-verified after the change).
2. **Figure 4 inputs**: the left column used the space form
   `"big bi"`, which inserts the engine's inter-word pause machinery
   (decay + 160-ms silent arc at factor 1.0 → a 400-ms acoustic gap in
   `bigbi_delta1.wav`), inherited from `Syllable_Synthesis`'s
   concatenation loop (`sig = concat(sig, zeros(dur*200), sig1)`).
   Both `make_article_figures.py` and `run_original_simulations.py`
   now use the dot form. Measured result: only closure silences remain
   (~80 ms ≈ T/2 at T=16), matching the author's reference realization
   (`bi_gbiT100ms.wav`: 45-ms closures at T=100 ms, no word gap).
   Word/syllable panels: "big"/".bi" are now the two SYLLABLES of
   big.bi sliced at the C.C boundary (topology: big = 2 excursions
   b+g; .bi = 1; .gbi = 1 fused) — matching the article panels.

Still on the space form (deliberately, pending author decision): the
`run_bigbi_polar_sweep.py` δ-sweep videos, whose pause machinery is
part of the Tp/fusion demonstration. NOTE: `specgram` panels plotting
in seconds must not share an x-axis set in ms (content squeezed into
invisibility) — fixed in `make_article_figures.py` figure 4.

### δ-sweep also in the dot form; the joint Tp sweep removed (2026-10-07, author ruling)

"For the demonstration to make sense, the big.bi δ-sweep must also use
the dot form — and in that case the joint Tp sweep no longer makes
sense." Applied to `run_bigbi_polar_sweep.py` (+ `make_bigbi_dual_video.py`,
which re-derives the pipeline):

* input `big bi` → `big.bi` (C.C syllable boundary, no pause, no
  pause factors — the engine change above supplies the COEFCEN-weighted
  boundary anchor);
* the Tp coupling (`Tp = 160·(1−δ)/(1−0.5) ms`) is GONE:
  `factor_for_delta`/`tp_ms_for_delta` now return 0.0 (kept for the
  dual-video import compatibility), npz keys `factor`/`tp_ms` kept at
  0 for schema stability;
* δ now means ONLY the boundary-anchor weight (δ·ρ_i): at 0.5 a
  salient boundary schwa @, at 1.0 the full /i/ (fusion-ready), then
  bi.gbi ×3 (fused /gb/);
* video captions updated ("boundary schwa", polar_sync/arcplot note
  replacing the stale `polar_arc` mention);
* new dual video: 24.21 s (was 25.98 s with the pause); HTML
  re-embedded with the chapter list REBUILT from the measured segment
  durations and all texts updated (dot form, no Tp).

The standalone Tp sweeps (`run_ibbi_ushape_sweep.py`,
`run_ibbi_ushape_nonreversible_en.py`) are UNCHANGED: they demonstrate
the /ibi/ supplement transformation (§3), where the inter-word pause
IS the object of study.

### /ibi/ sweeps: continuous correlated Tp ramp (2026-10-07, author ruling)

"Modifier le principe des deux démonstrations ib ib: ne pas construire
des suites de paires mais faire varier Tp de façon continue et
corrélée avec delta jusqu'à Tp = 0 et delta = 1 — garder la notion de
variation en U." Applied to both sweeps:

* **Reversible** (`run_ibbi_ushape_sweep.py`): the 13-value staircase
  becomes a CONTINUOUS ramp of 33 segments — Tp = 160, 150, …, 10, 0
  (10-ms steps, one engine pause step each: the finest realizable
  resolution), then back up 10 … 160. delta correlated at every point
  (0.5 + 0.5(1−Tp/160)); the bottom of the U (Tp=0) is the fused
  "bi bi". Dual video: 70.96 s (was ~28 s).
* **Non-reversible** (`run_ibbi_ushape_nonreversible_en.py`): the same
  continuous descent (16 "ib ib" segments, Tp 160→10), the fusion
  point (Tp=0, delta=1, "bi bi"), then HELD at fusion (3 segments).
  Dual video: 40.09 s (was 19.82 s).

Both HTMLs re-embedded with chapter lists rebuilt from the measured
segment durations and "continuous ramp" subtitles. Control frame
verified (Seg 17/33 = bottom of U: Tp=0, δ=1.000, bi bi fusion).

### Then: 10 segments + NO held @ (2026-10-07, author rulings)

"Ramener à 10 segments" — both /ibi/ sweeps reduced to 10 segments
(still one continuous correlated flow, no restarts):
reversible Tp = [160, 120, 80, 40, 0, 40, 80, 120, 160, 160]
(U with end baseline; dual video 18.64 s); non-reversible
Tp = [160, 120, 80, 40, 10, 0, 0, 0, 0, 0] (descent + fusion hold;
dual video 15.83 s). HTMLs re-embedded (10-chapter lists).

"No held @: on ne prononce pas 'ibe'" — the "Ve plateau fix"
(a full-amplitude T_voy plateau before the decay at word-end Ve) is
REVOKED. VERIFIED in the reference timit-to-Maeda
(`synthSYL/trajectory.py::_handle_b_terminal`): the decay block
follows the word-end anchor DIRECTLY, with no plateau — the plateau
was a bigbi-demos addition. Our engine now matches it
(`synthSYL/trajectory.py`, same handler): "ib ib" drops from 240 to
208 steps (the 160-ms @ hold after each coda /b/ is gone; the
trajectory reaches the @ and decays). Unaffected (verified by block
probes): the nuclear-vowel holds of CV words ("bi": final /i/ via the
last-anchor hold), "big.bi" (176 steps, unchanged), "ibia" (90 steps,
unchanged → all reference gates re-verified at 0.000e+00). The
supplement /ib/ renditions drop from 1280 to 1120 ms. The trough HTML
contains no plateau text and /ibi/ has no coda @ — unaffected.

### Non-reversible demo: U-shaped Tp + HTML denominators fixed (2026-10-07, author rulings)

"Refaire la démonstration ib ib non réversible avec un Tp U-shaped comme
la démonstration réversible" — `run_ibbi_ushape_nonreversible_en.py`
goes back to the U: Tp = [160, 120, 80, 40, 10, **0**, 40, 80, 120, 160]
(delta correlated), texts ["ib ib"]×5 + ["bi bi"]×5 — the Tp profile
mirrors the reversible demo, but after the fusion point the input STAYS
"bi bi" while Tp rises: the /b/ remains in onset (non-reversible
content). Dual video 16.91 s.

"L'affichage x/13 et x/11 sous HTML ne donne pas le bon nombre
d'éléments au dénominateur (il y en a 10 seulement)" — the stale
denominators lived in the EMBEDDED WebVTT captions ("Seg 1/13 …",
"Seg 1/11 …" — authored with the original 13/11-segment pages and never
regenerated) and in the metadata chips ("13 segments · U-shaped Tp",
"11 segments · NON-REVERSIBLE U-shape", stale Duration chip, Method
description). Both HTMLs re-embedded with: rebuilt 10-chapter lists,
REGENERATED WebVTT (10 cues, x/10, current Tp/δ), and refreshed chips
(10 segments, measured durations 18.64 s / 16.91 s, Method text,
polar_sync mention). Control frame verified: "Seg 6/10 (FUSION POINT)
Tp=0 ms δ=1.000 — bi bi (FUSION—non-reversible!)".

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

### Word-2 onset anchoring under COEFCEN=1 (engine correction)

Author's ruling (2026-10-06): under COEFCEN = 1 the "bi" of big.bi
must NOT return to the half-radius schwa (0.45) at the word boundary —
the article defines the pause as the diphthong between the PREVIOUS
Ve and the next Vo, so a non-initial word onset anchors on the
previous Ve. The engine previously applied `Vo = delta_o·rho` at every
word start; `synthSYL/gesture.py` now uses
`Vo := (delta_e·rho_prev, theta_prev)` for non-initial words.
Verified: pause arc of "big bi" (COEFCEN=1, VOYDEB=0.5) is now the
stationary hold (0.90→0.90), the word-2 /b/ departs from V=(0.90,
300°), and the red branch dips below ρ=0.5 exactly ONCE (the
utterance-initial Vo, t=0–220 ms). All reference reproductions remain
bit-identical (0.000e+00) because every validated gate uses symmetric
delta_o = delta_e, where the new formula gives the same value; the
"big@bi" schwa demonstrations are preserved.

### Teardrop loop SOLVED — ported from the original Syllable_Synthesis display

The author pointed to FBerthommier/Syllable_Synthesis (`synthSYL.py`,
the code behind the article figures). Its display builder
`makelooplot`/`arcplot` differs from our reconstruction in the sub-arc
ANCHORING: each leg is drawn with the departure-side point as polar
departure (pd) and the arrival-side point as phase carrier (pa):
`z(θ) = ρ(θ)·p_a·e^{iνθ/K} + (1−ρ(θ))·p_d`, ρ = cos(θ/2), with the
approach leg (V→C) sweeping θ ∈ [0, π] (pd = C, pa = V) and the
release legs θ ∈ [−π, 0] (first sample dropped, opint = −1). The phase
perturbation therefore sits on the SHORTER radius (ρ_V = 0.9 < ρ_C =
1.2), which produces exactly the article's closed TEARDROP: rounded
belly at the vowel, pointed at the consonant. (Our previous drawing
put the phase on the arrival C — with ρ_C larger, the bow always
bulged consonant-side; a closed V→b→V pair in that anchoring is a
symmetric lens, and the earlier wedge attempt was rejected.)

`polar_sync.build_branches` now draws z_c sub-arcs with this original
anchoring (z_leg_orig). Verified on the 'bi' loop of big.bi
(COEFCEN=1, VOYDEB=0.5): closed loop (|end−start| = 0.0000), width
profile peaks at chord fraction 0.35 from /i/ (0.163) and falls to
0.058→0 at /b/ — belly at the vowel, pointed at the consonant, as in
Figure 4. All reference reproductions remain bit-identical
(0.000e+00). Figures regenerated:
fig_bigbi_article_conditions_nu_inv.png,
fig_words_article_conditions_nu_inv.png. Sweep scripts use
polar_sync, so the next video regeneration carries the teardrops.

## Limitations / next step

The 472-px reference allows topology matching and target-layout
estimation, not point-by-point trajectory residuals. For a
quantitative point-wise matching (e.g. mean Fréchet distance after
layout alignment), the high-resolution source of the reference figure
(PDF or generator program) is required.
