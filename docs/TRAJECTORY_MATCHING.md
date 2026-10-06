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

## Limitations / next step

The 472-px reference allows topology matching and target-layout
estimation, not point-by-point trajectory residuals. For a
quantitative point-wise matching (e.g. mean Fréchet distance after
layout alignment), the high-resolution source of the reference figure
(PDF or generator program) is required.
