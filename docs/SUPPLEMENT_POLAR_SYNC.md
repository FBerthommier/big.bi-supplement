# Supplement-simulation QA: polar-diagram synchronization, fusion, /g/ label

**Date:** 2026-10-06 · **Scope:** the three embedded-video simulations
(`simulations/bigbi_bi_gbi_embedded.html`, `ibbi_ushape_embedded.html`,
`ibbi_ushape_nonreversible_embedded.html`) reviewed for the article
supplement (arXiv:2307.02299).

Three defects were reported on the previously imported simulations.
All three are **reproducible from this repository's own scripts** (the
scripts ARE the generators of the embedded videos — no external
generator program is needed), and all three are fixed here.

## Issue 1 — "ib ib → bi bi" never converges to the "bi bi" percept

**Diagnosis.** At the bottom of the U (Tp = 0, δ = 1) the reversible
sweep kept feeding **"ib ib"** (coda /b/) to the engine. The engine
faithfully renders the *x.VC.VC.x* graph it is given: it does not
spontaneously resyllabify, and the residual word-boundary blocks
(decay/attack, word-end plateaus) remain — Pval("ib ib", Tp=0, δ=1)
has 195 steps vs 131 for Pval("bi bi", Tp=0, δ=1); max|diff| = 3.75.
The label claimed "bi bi (fusion)" but the audio kept the coda
structure, so the percept never switched.

This is a **script-level** defect, not an engine defect: the article
(§2.2) states that when the time pressure cancels Tp, *x.VC.VC.x* and
*xV.CV.x* are the same utterance — the syllable-graph transition of
Figure 2 A is part of the demonstration. The non-reversible variant of
the demo already applied that transition (its input switches to
"bi bi" at the fusion point); the reversible one did not.

**Fix.** `scripts/run_ibbi_ushape_sweep.py::run_pipeline` now switches
the input to **"bi bi" when Tp = 0** (single bottom segment of the U;
ascending side returns to "ib ib" as the pause reappears). Verified:
the fusion segment's Pval is **bit-identical (0.0e+00)** to a direct
"bi bi" run, and its label track is b×32 → i×32 → b×32 → i×32 (onset
/b/ structure).

## Issue 2 — displayed trajectories look unsynchronized / old trajectories not erased enough

**Diagnosis.** Two independent causes:

1. *Structural desynchronization.* The display branches (z_v/z_c) and
   the phoneme-label timeline were computed by **two separate
   re-implementations** of the engine's block walk
   (`reconstruct_polar_branches` and `compute_phoneme_timeline`), both
   approximating the engine rather than reading it. They disagreed
   with each other and with the engine's real block sequence (labels
   one block late; the blue ball NaN while the /g/ label was showing).
2. *Trail.* the fading trail spanned 750 ms with a weak contrast ramp
   (alpha 0.2→0.9), i.e. ~19 video frames — most of a segment — so
   past trajectories lingered nearly as visibly as the current one.

**Fix.**
- New module `scripts/polar_sync.py`: re-runs the pipeline with the
  `synthSYL.trajectory` helpers **instrumented** (monkeypatched
  wrappers, engine files untouched) and records the engine's ACTUAL
  block sequence (plateau / background / arc / decay / attack /
  cluster, with dep/arr points and the contextual consonant targets).
  `build_branches` and `build_labels` then derive z_v, z_c, the labels
  AND the label positions from that single recording — ball, label,
  audio and sagittal panel are synchronized by construction. The
  recorded sequence was validated against the engine's own
  `block_info` (via an envelope-call spy): identical block count,
  kinds and lengths (224/224 steps, 11 blocks for "big bi" δ=0.5;
  130/130, 7 blocks for "bi.gbi").
- Trail: `TRAIL_MS` 750 → **500 ms** and fade ramp alpha
  0.2–0.9 → **0.05–0.90** in both renderers (polar-only and dual) of
  the three sweep scripts.

## Issue 3 — blue ball does not follow the /g/ trajectory; /g/ label drawn at the /b/ position

**Diagnosis.** Two distinct display bugs:

1. *Truncated coda excursion.* In the old branch reconstruction, a
   coda cluster before a pause/terminal drew only the first sub-arc
   (vowel → consonant) and dropped the release sub-arc (consonant →
   anchoring point): the blue ball vanished mid-course and never
   reached /g/.
2. *Label position.* the engine resolves the **contextual** /g/ target
   (palatal (1.1, 23π/12 ≈ 345°) after front vowels, velar
   (1.2, π/3 = 60°) after back vowels), but the on-video label was
   looked up in the generic inventory, whose /g/ sits at θ = π/3 —
   the same angular sector as /b/ (θ = π/3). Hence "the /g/ label
   appears in front of the /b/ point".

**Fix.** Both come free with `polar_sync`: the recorded cluster blocks
contain every sub-arc (dep → C₁ → … → Cₘ → arr), so the ball traverses
the full excursion, and `build_labels` returns per-step **contextual**
targets — the /g/ label is drawn at the position the trajectory
actually aims at. Measured on "big bi" δ=0.5: /g/ label target =
(1.10, 345°) and the blue ball at the /g/ sub-arc midpoint is at
distance **0.000** from it (previously the label sat next to /b/ at
60° and the ball never completed the excursion).

## Regenerated artefacts

- `output/bigbi_polar_sweep/` (+ `make_bigbi_dual_video.py`) →
  `simulations/bigbi_bi_gbi_embedded.html`
- `output/ibbi_ushape_en/` → `simulations/ibbi_ushape_embedded.html`
  (bottom-of-U segment now "bi bi")
- `output/ibbi_ushape_nonrev_en/` →
  `simulations/ibbi_ushape_nonreversible_embedded.html`
- The three HTML pages re-embedded (video + poster) from these runs.

## Round 2 (2026-10-06, after author review of the regenerated videos)

Three residual defects were reported against the article's Figure 4
panels and fixed:

1. **/g/ label at the /b/ position while the g→@ leg was unlabeled.**
   Root cause: the embedded big.bi video was rendered by
   `make_bigbi_dual_video.py`, which carried its OWN copy of the dual
   renderer — still using the generic-inventory label lookup (velar
   /g/ at θ=π/3, the same angular sector as /b/) and the legacy
   desynchronized timeline. `make_bigbi_dual_video.render_dual_segment`
   now simply delegates to the sweep's synchronized renderer.
   Additionally, `build_labels` is now **approach-based**: each cluster
   sub-arc is labeled with — and its label positioned at — the point it
   travels toward; the coda release leg /g/→Ve is labeled **'@'** at
   the schwa anchor (same angle as the vowel, shorter radius), instead
   of remaining unlabeled.
2. **Blue ball: false and desynchronized trajectory.** The dual-panel
   renderers sampled the branch arrays at the ENGINE-step index
   (`z_c[gesture_idx]`) while the branches are display-rate (10×) —
   the ball only ever saw the first 10 % of each excursion, appearing
   stuck/erratic. Fixed: the balls are plotted at `z[disp_idx]`
   (display-rate index); `gesture_idx` (engine grid) is kept for the
   sagittal panel and the labels.
3. **Red ball returning to the centre (0, 0).** The engine's
   utterance-edge blocks connect the neutral terminal anchors
   (pt = None → [0, 0]) to the first/last vowel, so the reconstructed
   z_v dove to the origin at the start and end of every segment —
   unlike the article's Figure 4 planning trajectories, which stay on
   the periphery. `build_branches` now renders terminal-touching arcs
   as stationary holds at the boundary vowel: min |z_v| over an
   utterance = the schwa radius (0.45 at δ=0.5), never 0.

Verification ("big bi", δ=0.5): |z_v| ∈ [0.450, 0.900] (no origin
dip); label sequence b → i → g (palatal, 345°) → @ (Ve anchor) → _ →
b → i, each label positioned at the point the blue ball is travelling
toward.

## Answer to "can they be restored here, or must the generator programs be requested?"

They were **restored here**. The repository's `scripts/` are the
generators (the sandbox-era external HTML builders were replaced
during the 2026-10-05 revision by repo-relative, reproducible
pipelines), so every embedded simulation is regenerable locally with
`python scripts/run_<demo>.py` + `python scripts/make_bigbi_dual_video.py`.
No external program is needed.

Engine files (`vlam.py`, `synthSYL/*.py`) remain untouched: the block
recording is a display-layer monkeypatch applied only during the
instrumented re-run.
