# Display vs engine — audit of the original `arc()` anchoring

**Date:** 2026-10-07 · **Scope:** closure of the teardrop-loop
investigation (see `TRAJECTORY_MATCHING.md`, § "Teardrop loop SOLVED")
· **Reference code:** `FBerthommier/Syllable_Synthesis` `synthSYL.py`
(the code behind the figures of arXiv:2307.02299).

The article's Figure-4 teardrop loops (rounded belly at the vowel,
point at the consonant) were traced to the sub-arc ANCHORING of the
original display builder (`arcplot`/`makelooplot`): the departure-side
point is the polar departure (`pd` = the consonant on the approach
leg), the arrival-side point carries the phase (`pa` = the vowel).
This note establishes the corresponding fact for the ENGINE and
concludes: **the engine was already correct; only the display copies
had drifted — moteur inchangé, affichage corrigé.**

## 1. Correspondence table: original `arc()`/`makeloop()` ↔ synthSYL engine

Original engine primitives: `arc(pt, co, params, D, thetabounds, opint,
nu, K, Pexp)` (synthSYL.py, l.1407) and `makeloop(n, sylP1, sylpt, ...)`
(l.1527). Repo engine: `synthSYL/projection.py::arc_B` and
`synthSYL/trajectory.py::build_cluster_pval`. For a V→C₁(→C₂)→V loop
with points `[V1, C1(, C2), V2]`:

| arc (original `makeloop` call) | pd (departure) | pa (phase carrier) | θ range | opint | K | repo engine call (`build_cluster_pval` → `arc_B`) |
|---|---|---|---|---|---|---|
| `arc(sylpt[[1,0]], …, T, [0,π], 0, nu, K)` — approach leg V1→C1 | **C1** | **V1** | [0, π] | 0 | 10 | `k=0`: `pt_d=C1, pt_a=anchor_dep, tb=[0,π], oi=0` (trajectory.py:210-213) |
| `arc(sylpt[[1,2]], …, T, [−π,0], −1, nu, K)` — C1→C2 (cluster) | C1 | C2 | [−π, 0] | −1 | 10 | `k<m`: `pt_d=C_{k−1}, pt_a=C_k, tb=[−π,0], oi=−1` (trajectory.py:215-218) |
| `arc(sylpt[[2,3]] / sylpt[[1,2]], …, T, [−π,0], −1, nu, K)` — release Cₘ→V2 | Cₘ | V2 | [−π, 0] | −1 | 10 | `k=m`: `pt_d=C_m, pt_a=anchor_arr, tb=[−π,0], oi=−1` (trajectory.py:220-223) |
| `arc(sylpt[[0,2]] / sylpt[[0,3]], …, n·T, [−π,0], 0, nu, Kvoy)` — vocalic background (non-selected params) | V1 | V2 | [−π, 0] | 0 | 30 | `build_background(dep, arr, Dtot, nu, Kvoy)` = `arc_B(…, [−π,0], 0, nu, Kvoy)` (trajectory.py:148-153, 191) |
| pause/terminal arcs `arc(…, [−π,0], 0, nu, Kpause)` | previous end | next start | [−π, 0] | 0 | 30 | `_append_arc` = `arc_B(…, [−π,0], 0, nu, K)` (trajectory.py:63-73) |

Same θ grids (opint 0: `linspace(tb0, tb1, D)`; opint −1: first sample
dropped), same blend `rk = cos(θ/2)^Pexp` with the phase `+ (nu/K)·θ`
on the arrival term, in `arc_B` verbatim (projection.py:68-76) versus
`arc()` (synthSYL.py l.1422). **Every pd/pa pair, every θ range and
every opint in the repo engine matches the original engine.** The
repo's `arc_B` additionally unwraps `pa`'s angle (a no-op on the
article's target table) — the only textual difference.

Parameter defaults: the repo engine runs the article values
`DEFAULT_NU = −1`, `K = 10`, `Kvoy = 30`, `DEFAULT_PEXP = 1`
(rho = cos(θ/2), article §2.1: "When K is large (K = 30 for vowel arcs
and K = 10 for consonant arcs)…"). Documented discrepancy of the
distributed original script: its `main()` entry point sets
`Pexp = 2 # Tau model` with `K = Kvoy = Kpause = 1000` for interactive
use (the article values remain in the adjacent comments); the
formulas themselves are identical. The repo keeps Pexp = 1 everywhere,
consistent with the article text and with every validated gate.

## 2. Correspondence table: original `arcplot()`/`makelooplot()` ↔ display

Original display primitives: `arcplot(pt, D, thetabounds, opint, nu,
K, Pexp)` (l.1384) and `makelooplot(n, sylpt, T, nu, K, Kvoy, Pexp)`
(l.1640) — column 0 of `Tval` is z_c, column 1 is z_v. Repo display:
`scripts/polar_sync.py::build_branches` (`z_leg_orig`).

| original display call | pd | pa | θ range | opint | K | repo display |
|---|---|---|---|---|---|---|
| `arcplot(sylpt[[1,0]], T, [0,π], 0, nu, K)` | C1 | V1 | [0, π] | 0 | 10 | `z_leg_orig(pts[1], pts[0], 0, π, k_c, nu_c, T)` (commit 7755fa0) |
| `arcplot(sylpt[[1,2]]/[[2,3]], T, [−π,0], −1, nu, K)` | C₁/Cₘ | C₂/V2 | [−π, 0] | −1 | 10 | `z_leg_orig(pts[j], pts[j+1], −π, 0, …, drop_first=True)` |
| `arcplot(sylpt[[0,2]]/[[0,3]], n·T, [−π,0], 0, nu, Kvoy)` — z_v | V1 | V2 | [−π, 0] | 0 | 30 | `z_leg_orig(dep, arr, −π, 0, k_v, nu_v, n)` — `zv_form="arcplot"` default since 2026-10-07 |

The historical display copies (`reconstruct_polar_branches` in
`run_article_demos.py`, `run_original_simulations.py`,
`make_ibbi_polar_video.py`, and the initial `polar_sync` z_v) drew the
z_c sub-arcs as `polar_arc(dep→arr)` — phase on the ARRIVAL consonant,
which closes a V→C→V pair into a symmetric LENS instead of the article
teardrop; and drew z_v with `polar_primitives.polar_arc`
(orientation="inverse"), which approximates `arcplot` only up to a
Pexp=2 blend with a mirrored phase sweep (max deviation 0.12 rho
units on a K=30 arc; the loop handedness at the vowel is mirrored).
Both are now replaced by the exact original form.

## 3. Conclusion — engine untouched, display corrected

* **Engine (`vlam.py`, `synthSYL/*.py`): NO modification.** The engine
  had replicated the original anchoring all along (table §1); the
  0.000e+00 equivalence with `original_simulations/` held before,
  during and after the display migration. No audio resynthesis.
* **Display: corrected to the original convention.** z_c sub-arcs
  (commit 7755fa0) and, since 2026-10-07, the vocalic branch z_v
  (`zv_form="arcplot"` default) are drawn with the exact
  `arcplot` anchoring — pd on the departure side (the consonant for
  z_c legs), phase on pa, θ ∈ [0,π] (approach) / [−π,0] (release,
  background), ρ = cos(θ/2). This convention is the reference, not an
  option; the legacy forms are kept only as documented options
  (`zv_form="polar_arc"`).
* **Per-branch display ν** (ν_v = −1, ν_c = +1) remains the
  author-validated planning-figure convention
  (`TRAJECTORY_MATCHING.md` § nu inversion); the original code runs
  the single value ν = −1 on every display arc. It is a display
  convention and does not mirror the engine's driven arcs.

Any future deviation observed in regenerations is a display-side
question: document it, do not touch the engine.
