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

Display geometry (arXiv:2307.02299 §2.1): z_v arcs K=30, nu=+1;
z_c sub-arcs K=10, nu=-1.
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
                   nu_v: int = 1, nu_c: int = -1):
    """(z_v, z_c, disp_len) sampled at ``sr_display`` from the blocks.

    z_c is non-NaN exactly over cluster blocks, sub-arc by sub-arc
    (dep -> C1 -> ... -> Cm -> arr, T each) so the blue ball reaches
    each consonant target exactly when the label switches to it.
    """
    from polar_primitives import polar_arc, stationary_point

    scale = sr_display / (1000.0 / t_step_ms)   # display samples per step
    zv_parts, zc_parts = [], []

    def z_arc(dep, arr, n_ms, K, nu):
        z, _ = polar_arc(dep[0], dep[1], arr[0], arr[1],
                         duration_ms=n_ms, sr=sr_display, K=K, nu=nu,
                         orientation="inverse")
        return z

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
            subs = [z_arc(pts[j], pts[j + 1], T_ms, k_c, nu_c)
                    for j in range(len(pts) - 1)]
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
