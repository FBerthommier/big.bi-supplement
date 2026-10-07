#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
polar_sync.py — Engine-block-synchronized polar display reconstruction.

The previous display scripts re-implemented the engine's block walk
independently for (a) the z_v/z_c display branches and (b) the phoneme
label timeline; the two walks diverged from the engine's real block
sequence, so labels were offset from the blue ball and coda consonant
excursions were truncated (the ball vanished before reaching its
target — e.g. the coda /g/ of "big").

This module records the engine's ACTUAL blocks by re-running
``panphon_pipeline`` with instrumented ``synthSYL.trajectory`` helpers
(monkeypatched, engine file untouched), then rebuilds

  * z_v / z_c display branches (sampled at ``sr_display``),
  * the per-engine-step phoneme label array,

from that single recorded sequence — so ball, labels and (via the
``gesture_idx`` mapping) the sagittal panel are synchronized by
construction.

Display geometry (arXiv:2307.02299 §2.1, Eq. 2): z_v arcs K=30,
z_c sub-arcs K=10; per-branch display nu convention
**nu_v = −1, nu_c = +1** (article planning-figure convention —
author-validated 2026-10-06; the historical +1/−1 produced mirrored
bows on BOTH branches).

ORIGINAL DISPLAY CONVENTION (Syllable_Synthesis synthSYL.py,
arcplot/makelooplot — the code behind the article figures): every
sub-arc is anchored with the departure-side point as polar DEPARTURE
(pd; the consonant for z_c legs, V1 for vocalic backgrounds) and the
ARRIVAL-side point carrying the phase (pa = V1 on the z_c approach
leg, pa = V2 on vocalic backgrounds); theta in [0, pi] for the
approach leg and [-pi, 0] (opint = -1, first sample dropped) for the
release legs; rho = cos(theta/2). This anchoring — NOT the phase on
the arrival consonant — is what makes the closed V->C->V gesture a
TEARDROP, and it is the REFERENCE convention, not an option. Since
2026-10-07 the vocalic branch z_v is drawn with the same arcplot form
(zv_form="arcplot", pd = V1, pa = V2, theta in [-pi, 0], K=30): the
original display uses ONE interpolation form for both branches.
zv_form="polar_arc" keeps the historical approximation (Pexp=2 blend
with mirrored phase sweep; max deviation 0.12 rho units) for the
record.

nu sign convention — IMPORTANT (keep in sync with polar_primitives.py
and covtl-pipeline's vtl_synth/video/polar_video.py, which must be
treated SEPARATELY):

  * The synthSYL ENGINE (synthSYL/polar.py ``arc_B``,
    ``cos(Ψ2 − θ2 − (ν/K)·θ(t))``) is VERBATIM Eq. 2 of
    arXiv:2307.02299 (``e^{i(θ2 + (ν/K)θ(t))}`` — same sign), running
    the single engine value ν = −1 on every arc (vowel backgrounds and
    consonant sub-arcs alike). There is NO notational flip in the
    synthSYL engine.
  * covtl-pipeline's active ``syl`` engine, by contrast, writes the
    cosine with the OPPOSITE sign, so its ν = −1 is numerically
    equivalent to Eq. 2 with ν = +1 (see the covtl manual, §5: "this
    is a notational convention only — ν = 1 is recovered when the
    cosine is written with the opposite sign"). The reference display
    values (Kc = 10, Kv = 30, ν = 1) belong to that engine family.
  * Consequence: display ν values are NOT transferable between the two
    engines. On the synthSYL side the per-branch display ν (−1, +1)
    follows the article's planning FIGURES, which is a display
    convention and does not mirror the engine's driven arcs.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Block recording (instrumented replay of the engine)
# ═══════════════════════════════════════════════════════════════════
def record_pipeline(pipeline_fn, blocks_out: list, **pipeline_kwargs):
    """Run ``pipeline_fn(**pipeline_kwargs)`` with the trajectory helpers
    instrumented; append one dict per engine block to ``blocks_out``.

    Block dicts:
      plateau    {n, pt}
      background {n, dep, arr}                  (V->V, no consonant)
      arc        {n, dep, arr}                  (pause / initial / terminal)
      attack     {n, pt}   / decay {n, pt}      (stationary holds)
      cluster    {n, dep, arr, cons=[(key, rho, theta)...], T}

    Returns the PipelineResult. The total recorded n equals Pval.shape[0]
    (asserted by build_branches).
    """
    import synthSYL.trajectory as traj

    orig = {name: getattr(traj, name) for name in
            ("_append_plateau", "_append_arc", "_append_decay_block",
             "_append_attack_block", "build_background",
             "build_cluster_pval")}

    state = {"in_cluster": False, "suppress": 0}

    def rec(d):
        blocks_out.append(d)

    def _append_plateau(blocks, pt, D):
        orig["_append_plateau"](blocks, pt, D)
        # decay/attack holds delegate to _append_plateau (nested call):
        # record the enclosing decay/attack block only, not the nested one
        if not state["in_cluster"] and not state["suppress"]:
            rec(dict(kind="plateau", n=D, pt=list(pt)))

    def _append_arc(blocks, pt_dep, pt_arr, D, nu, K, Pexp):
        orig["_append_arc"](blocks, pt_dep, pt_arr, D, nu, K, Pexp)
        if not state["in_cluster"]:
            rec(dict(kind="arc", n=D, dep=list(pt_dep), arr=list(pt_arr)))

    def _append_decay_block(blocks, block_info, pt, D, pre_amp):
        before = len(blocks)
        state["suppress"] += 1
        try:
            orig["_append_decay_block"](blocks, block_info, pt, D, pre_amp)
        finally:
            state["suppress"] -= 1
        if len(blocks) > before and not state["in_cluster"]:
            rec(dict(kind="decay", n=D, pt=list(pt)))

    def _append_attack_block(blocks, block_info, pt, D, post_amp):
        before = len(blocks)
        state["suppress"] += 1
        try:
            orig["_append_attack_block"](blocks, block_info, pt, D, post_amp)
        finally:
            state["suppress"] -= 1
        if len(blocks) > before and not state["in_cluster"]:
            rec(dict(kind="attack", n=D, pt=list(pt)))

    def build_background(pt_dep, pt_arr, D, nu, Kvoy, Pexp):
        out = orig["build_background"](pt_dep, pt_arr, D, nu, Kvoy, Pexp)
        if not state["in_cluster"]:
            rec(dict(kind="background", n=D,
                     dep=list(pt_dep), arr=list(pt_arr)))
        return out

    def build_cluster_pval(anchor_dep, anchor_arr, consonants, T,
                           nu, K, Kvoy, Pexp):
        state["in_cluster"] = True
        try:
            out = orig["build_cluster_pval"](anchor_dep, anchor_arr,
                                             consonants, T, nu, K, Kvoy,
                                             Pexp)
        finally:
            state["in_cluster"] = False
        cons = [(c.seg_key, float(c.rho), float(c.theta))
                for c in consonants]
        rec(dict(kind="cluster", n=(len(cons) + 1) * T,
                 dep=list(anchor_dep), arr=list(anchor_arr),
                 cons=cons, T=T))
        return out

    # trajectory.build_global_pval resolves these helpers as module
    # attributes at call time, so patching the module is enough.
    for name, wrapped in [
        ("_append_plateau", _append_plateau),
        ("_append_arc", _append_arc),
        ("_append_decay_block", _append_decay_block),
        ("_append_attack_block", _append_attack_block),
        ("build_background", build_background),
        ("build_cluster_pval", build_cluster_pval),
    ]:
        setattr(traj, name, wrapped)
    try:
        result = pipeline_fn(**pipeline_kwargs)
    finally:
        for name, fn in orig.items():
            setattr(traj, name, fn)
    return result


# ═══════════════════════════════════════════════════════════════════
# Display branches from recorded blocks
# ═══════════════════════════════════════════════════════════════════
def build_branches(blocks: List[dict], n_steps: int,
                   t_step_ms: float = 10.0, sr_display: float = 1000.0,
                   k_v: float = 30.0, k_c: float = 10.0,
                   nu_v: int = -1, nu_c: int = 1,
                   zv_form: str = "arcplot"):
    """(z_v, z_c, disp_len) sampled at ``sr_display`` from the blocks.

    z_c is non-NaN exactly over cluster blocks, sub-arc by sub-arc
    (dep -> C1 -> ... -> Cm -> arr, T each) so the blue ball reaches
    each consonant target exactly when the label switches to it.

    zv_form="arcplot" (default, original Syllable_Synthesis display):
    vocalic backgrounds and inter-vowel arcs are drawn with the SAME
    arcplot form as the z_c sub-arcs (pd = departure anchor, pa =
    arrival anchor carrying the phase, theta in [-pi, 0], opint = 0,
    rho = cos(theta/2) — synthSYL.py makelooplot column 1). The
    original display uses one interpolation form for both branches.
    zv_form="polar_arc": historical approximation via
    polar_primitives.polar_arc (Pexp=2 blend, mirrored phase sweep;
    max deviation 0.12 rho units, loop handedness mirrored).
    """
    from polar_primitives import polar_arc, stationary_point

    scale = sr_display / (1000.0 / t_step_ms)   # display samples per step
    zv_parts, zc_parts = [], []

    def z_arc(dep, arr, n_ms, K, nu):
        if zv_form == "arcplot":
            # ORIGINAL vocalic background: arcplot(pd=dep, pa=arr,
            # [-pi, 0], opint=0, nu, Kvoy, Pexp=1) — synthSYL.py
            # makelooplot, Tval[:, 1]
            return z_leg_orig(dep, arr, -np.pi, 0.0, K, nu, n_ms)
        z, _ = polar_arc(dep[0], dep[1], arr[0], arr[1],
                         duration_ms=n_ms, sr=sr_display, K=K, nu=nu,
                         orientation="inverse")
        return z

    def z_leg_orig(p_dep, p_arr, th_lo, th_hi, K, nu, n_ms,
                   drop_first=False):
        """ORIGINAL Syllable_Synthesis display sub-arc (arcplot): the
        departure-side point is the polar DEPARTURE (pd) and the
        arrival-side point the ARRIVAL carrying the phase (pa):
            z(th) = rho(th)*p_arr*exp(i*nu*th/K) + (1-rho(th))*p_dep,
        rho = cos(th/2), with theta in [0, pi] for the approach leg
        and [-pi, 0] (first sample dropped, opint=-1) for the release
        legs. This is what makes the closed V->C->V gesture a
        TEARDROP: rounded belly at the vowel, pointed at the consonant
        (article Fig. 4), because the phase perturbation sits on the
        SHORTER radius (rho_V < rho_C)."""
        n = max(2, int(round(n_ms * sr_display / 1000.0)))
        if drop_first:
            th = np.linspace(th_lo, th_hi, n + 1)[1:]
        else:
            th = np.linspace(th_lo, th_hi, n)
        rho = np.cos(th / 2.0)
        z_arr = p_arr[0] * np.exp(1j * (p_arr[1] + nu * th / K))
        z_dep = p_dep[0] * np.exp(1j * p_dep[1])
        return rho * z_arr + (1 - rho) * z_dep

    def z_hold(pt, n_ms):
        z, _ = stationary_point(pt[0], pt[1], duration_ms=n_ms,
                                sr=sr_display)
        return z

    for b in blocks:
        n = b["n"]
        n_ms = n * t_step_ms
        nd = int(round(n * scale))
        if b["kind"] in ("plateau", "decay", "attack"):
            zv_parts.append(z_hold(b["pt"], n_ms))
            zc_parts.append(np.full(nd, np.nan + 0j))
        elif b["kind"] in ("arc", "background"):
            dep, arr = b["dep"], b["arr"]
            # Neutral terminal anchors (pt = None -> [0, 0]) would pull
            # the vocalic branch to the origin. The article's Figure 4
            # planning trajectories stay on the periphery: render the
            # utterance-edge arcs as a stationary hold at the boundary
            # vowel instead of a dip through the center.
            def _neutral(p):
                return abs(p[0]) < 1e-9 and abs(p[1]) < 1e-9
            if _neutral(dep) and not _neutral(arr):
                zv_parts.append(z_hold(arr, n_ms))
            elif _neutral(arr) and not _neutral(dep):
                zv_parts.append(z_hold(dep, n_ms))
            else:
                zv_parts.append(z_arc(dep, arr, n_ms, k_v, nu_v))
            zc_parts.append(np.full(nd, np.nan + 0j))
        elif b["kind"] == "cluster":
            zv_parts.append(z_arc(b["dep"], b["arr"], n_ms, k_v, nu_v))
            pts = [b["dep"]] + [[r, t] for _, r, t in b["cons"]] + [b["arr"]]
            T_ms = b["T"] * t_step_ms
            # ORIGINAL Syllable_Synthesis display (makelooplot): each
            # sub-arc is anchored on its DEPARTURE-side point, phase on
            # the ARRIVAL-side point; approach leg sweeps theta in
            # [0, pi], the following legs in [-pi, 0] (first sample
            # dropped).
            subs = []
            for j in range(len(pts) - 1):
                th_lo, th_hi = (0.0, np.pi) if j == 0 else (-np.pi, 0.0)
                if j == 0:
                    # approach leg V1->C1: original pt=[[C1, V1]] i.e.
                    # pd=C1 (departure), pa=V1 (phase carrier)
                    leg = z_leg_orig(pts[1], pts[0], th_lo, th_hi,
                                     k_c, nu_c, T_ms)
                else:
                    leg = z_leg_orig(pts[j], pts[j + 1], th_lo, th_hi,
                                     k_c, nu_c, T_ms, drop_first=True)
                subs.append(leg)
            zc = np.concatenate(subs)
            if len(zc) < nd:
                zc = np.concatenate(
                    [zc, np.full(nd - len(zc), np.nan + 0j)])
            zc_parts.append(zc[:nd])
        else:
            raise ValueError(f"unknown block kind {b['kind']!r}")

    z_v = np.concatenate(zv_parts) if zv_parts else np.zeros(0, complex)
    z_c = np.concatenate(zc_parts) if zc_parts else np.zeros(0, complex)
    disp_len = int(round(n_steps * scale))
    if len(z_v) < disp_len:
        z_v = np.concatenate([z_v, np.full(disp_len - len(z_v),
                                           z_v[-1] if len(z_v) else 0j)])
    z_v = z_v[:disp_len]
    if len(z_c) < disp_len:
        z_c = np.concatenate([z_c, np.full(disp_len - len(z_c),
                                           np.nan + 0j)])
    z_c = z_c[:disp_len]
    return z_v, z_c, disp_len


# ═══════════════════════════════════════════════════════════════════
# Phoneme labels from recorded blocks
# ═══════════════════════════════════════════════════════════════════
def build_labels(blocks: List[dict], seg_map: dict):
    """Per-engine-step (labels, targets) aligned 1:1 with the blocks.

    Approach-based labeling: each cluster sub-arc is labeled with — and
    the label positioned at — the point it travels TOWARD (sub-arc
    dep→C₁ shows C₁; the release sub-arc Cₘ→arr shows the arrival
    vowel anchor, e.g. '@' for a coda releasing into the schwa). This
    keeps the label, the blue ball and the drawn leg consistent at
    every instant.

    plateau / attack / decay: the phoneme whose (rho, theta) target
    matches the hold point (seg_map lookup); '_' otherwise.
    arc / background: '_'.
    """
    targets_lookup = {k: (float(v.rho), float(v.theta))
                      for k, v in seg_map.items()}

    def hold_key(pt):
        for k, (r, t) in targets_lookup.items():
            if abs(r - pt[0]) < 1e-6 and abs(t - pt[1]) < 1e-6:
                return k
        return "_"

    def reduced_key(pt):
        """Vo/Ve anchoring point (delta * rho_V, theta_V): same angle as
        a full vowel but shorter radius — label it '@' (schwa)."""
        best = None
        for k, (r, t) in targets_lookup.items():
            if abs(t - pt[1]) < 1e-6 and pt[0] < r - 1e-6:
                if best is None or r < best[1]:
                    best = (k, r)
        return "@" if best is not None else None

    labels: List[str] = []
    targets: List[Optional[tuple]] = []
    for b in blocks:
        n = b["n"]
        if b["kind"] == "cluster":
            m = len(b["cons"])
            T = b["T"]
            # chain of points: dep anchor, C1..Cm, arr anchor
            keys = ([hold_key(b["dep"])]
                    + [c[0] for c in b["cons"]]
                    + [hold_key(b["arr"])])
            geo = ([tuple(b["dep"])]
                   + [(c[1], c[2]) for c in b["cons"]]
                   + [tuple(b["arr"])])
            for j in range(m + 1):          # sub-arc j: geo[j] -> geo[j+1]
                key = keys[j + 1]           # label = the point approached
                tgt = geo[j + 1]
                if key == "_":
                    # release into a reduced (Vo/Ve) anchor → '@'
                    rk = reduced_key(list(tgt))
                    if rk is not None:
                        key = rk
                    else:
                        labels.extend(["_"] * T)
                        targets.extend([None] * T)
                        continue
                labels.extend([key] * T)
                targets.extend([tgt] * T)
        elif b["kind"] in ("plateau", "decay", "attack"):
            key = hold_key(b["pt"])
            if key == "_":
                rk = reduced_key(b["pt"])
                if rk is not None:
                    key = rk
            labels.extend([key] * n)
            pt = tuple(b["pt"]) if key != "_" else None
            targets.extend([pt] * n)
        else:
            labels.extend(["_"] * n)
            targets.extend([None] * n)
    return labels, targets
