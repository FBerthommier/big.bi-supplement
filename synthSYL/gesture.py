# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
gesture.py — Gestural graph construction for the synthSYL model.

Builds gesture nodes and gesture anchors from the flat segment list
and polar targets.  This is Step 2 of the synthesis pipeline.

All internal structures use the ``GestureNode`` and ``GestureAnchor``
dataclasses from ``types.py`` rather than anonymous dicts, ensuring
type safety and explicit field documentation.

Pipeline for build_gesture_nodes:
    parse_polar_targets → build_initial_nodes → detect_clusters → adjust_consonants

Pipeline for build_gesture_anchors:
    create_base_anchors → insert_word_end_anchors → insert_vowel_onset_anchors
    → insert_syllable_onset_anchors → ensure_terminal_anchors
"""

from __future__ import annotations

import numpy as np

from .constants import COEFCEN, DEFAULT_DELTA_O, DEFAULT_DELTA_E
from .phonology import (
    IPA_TO_SYNTSYL,
    _CC_ALIAS,
    _CC_RHO_OVERRIDE,
    _TABCONS_IDX,
    adjust_consonant_theta,
    is_vowel as is_vowel_seg,
    VOWELS_SYNTSYL,
)
from .types import GestureAnchor, GestureNode, PolarTarget, SyllableBoundary


# ═══════════════════════════════════════════════════════════════════════
# Pre-computed vowel lookup (O(1) instead of O(n) linear scan)
# ═══════════════════════════════════════════════════════════════════════

def _precompute_vowel_indices(nodes: list[GestureNode]) -> tuple[list[int], list[int]]:
    """For each node, compute the index of the nearest vowel.

    Returns
    -------
    next_vowel : list[int]
        next_vowel[i] = index of the next vowel after node i,
        or -1 if none.
    prev_vowel : list[int]
        prev_vowel[i] = index of the previous vowel before node i,
        or -1 if none.
    """
    n = len(nodes)
    next_vowel = [-1] * n
    prev_vowel = [-1] * n

    # Forward pass
    last_v = -1
    for i in range(n):
        if nodes[i].kind == "V":
            last_v = i
        prev_vowel[i] = last_v

    # Backward pass
    last_v = -1
    for i in range(n - 1, -1, -1):
        if nodes[i].kind == "V":
            last_v = i
        next_vowel[i] = last_v

    return next_vowel, prev_vowel


def _nearest_vowel_theta(nodes: list[GestureNode], i: int,
                         next_vowel: list[int],
                         prev_vowel: list[int]) -> float:
    """Return the theta of the nearest vowel to node *i*.

    Uses pre-computed indices for O(1) lookup instead of O(n) scan.
    Falls back to π if no vowel is found.
    """
    # Search forward (skipping pauses)
    nv = next_vowel[i]
    if nv != -1:
        blocked = False
        for j in range(i + 1, nv):
            if nodes[j].kind == "pause":
                blocked = True
                break
        if not blocked:
            return nodes[nv].theta

    # Search backward (skipping pauses)
    pv = prev_vowel[i]
    if pv != -1:
        blocked = False
        for j in range(pv + 1, i):
            if nodes[j].kind == "pause":
                blocked = True
                break
        if not blocked:
            return nodes[pv].theta

    return np.pi


# ═══════════════════════════════════════════════════════════════════════
# Node construction helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_vowel_node(rho: float, theta: float, seg_key: str) -> GestureNode:
    """Create a vowel gesture node."""
    return GestureNode(kind="V", rho=rho, theta=theta, seg_key=seg_key)


def _make_consonant_node(rho: float, theta: float, params: list[int],
                         seg_key: str) -> GestureNode:
    """Create a consonant gesture node."""
    return GestureNode(
        kind="C", rho=rho, theta=theta,
        params=params, seg_key=seg_key, in_cluster=False,
    )


def _make_pause_node(long: bool) -> GestureNode:
    """Create a pause gesture node."""
    return GestureNode(kind="pause", long=long)


# ═══════════════════════════════════════════════════════════════════════
# Anchor construction helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_vowel_anchor(i: int, pt: list[float] | None, hold: bool,
                       **extra) -> GestureAnchor:
    """Create a vowel anchor with optional extra attributes.

    Extra keyword arguments such as ``is_vowel_onset``, ``is_word_end``,
    or ``is_syl_vowel_onset`` are passed through to the dataclass
    constructor.
    """
    return GestureAnchor(i=i, pt=pt, hold=hold, kind="V", **extra)


def _make_pause_anchor(i: int, long: bool) -> GestureAnchor:
    """Create a pause anchor."""
    return GestureAnchor(i=i, pt=None, hold=False, kind="pause", long=long)


def _make_synth_anchor(i: int, pt: list[float]) -> GestureAnchor:
    """Create a synth (terminal) anchor."""
    return GestureAnchor(i=i, pt=pt, hold=False, kind="synth")


# ═══════════════════════════════════════════════════════════════════════
# Search helpers
# ═══════════════════════════════════════════════════════════════════════

def _find_prev_vowel(nodes: list[GestureNode], start: int) -> tuple[float, float] | None:
    """Search backward from *start* for the nearest vowel, stopping at pauses.

    Returns (rho, theta) of the vowel found, or None if no vowel is
    found before a pause or the beginning of the list.
    """
    for j in range(start, -1, -1):
        if nodes[j].kind == "pause":
            break
        if nodes[j].kind == "V":
            return nodes[j].rho, nodes[j].theta
    return None


def _find_next_vowel_theta(nodes: list[GestureNode], start: int) -> float | None:
    """Search forward from *start* for the nearest vowel, stopping at pauses.

    Returns the theta of the vowel found, or None if no vowel is found
    before a pause or the end of the list.
    """
    n = len(nodes)
    for j in range(start, n):
        if nodes[j].kind == "pause":
            break
        if nodes[j].kind == "V":
            return nodes[j].theta
    return None


def _find_next_vowel_rho_theta(nodes: list[GestureNode], start: int) -> tuple[float, float] | None:
    """Search forward from *start* for the nearest vowel, stopping at pauses.

    Returns (rho, theta) of the vowel found, or None if no vowel is found
    before a pause or the end of the list. Used by _insert_vowel_onset_anchors
    to compute Vo = (delta_o * rho_V, theta_V) per Berthommier 2023 §2.2.
    """
    n = len(nodes)
    for j in range(start, n):
        if nodes[j].kind == "pause":
            break
        if nodes[j].kind == "V":
            return nodes[j].rho, nodes[j].theta
    return None


def _find_last_vowel_in_segs(segs: list[str]) -> tuple[float, float]:
    """Search backward through segment strings for the last vowel.

    Returns (rho, theta) found in VOWELS_SYNTSYL, or (0.8, π) as
    defaults if no vowel is found or the vowel key is not in the
    registry.
    """
    for seg in reversed(segs):
        if is_vowel_seg(seg):
            vkey = IPA_TO_SYNTSYL.get(seg)
            if vkey and vkey in VOWELS_SYNTSYL:
                return VOWELS_SYNTSYL[vkey]["rho"], VOWELS_SYNTSYL[vkey]["theta"]
            return 0.8, np.pi
    return 0.8, np.pi


def _find_anchor_insert_pos(anchors: list[GestureAnchor], node_idx: int) -> int:
    """Find the insertion position in *anchors* for a given node index.

    Returns the index at which a new anchor with node index *node_idx*
    should be inserted to maintain sorted order by ``i``.
    """
    pos = 0
    for ai, a in enumerate(anchors):
        if a.i >= node_idx:
            break
        pos = ai + 1
    return pos


# ═══════════════════════════════════════════════════════════════════════
# Cluster detection helper
# ═══════════════════════════════════════════════════════════════════════

def _cc_eligible(seg_key: str) -> bool:
    """Check if a segment is eligible for cluster detection."""
    resolved = _CC_ALIAS.get(seg_key, seg_key)
    synt = IPA_TO_SYNTSYL.get(resolved, "")
    return synt in _TABCONS_IDX


# ═══════════════════════════════════════════════════════════════════════
# Pipeline: build_gesture_nodes
# ═══════════════════════════════════════════════════════════════════════

def _parse_polar_targets(flat_segments: list[str],
                         all_polars: list[PolarTarget]) -> list[PolarTarget | None]:
    """Phase 1: Read polar targets, aligning with flat segments.

    Returns a list indexed by segment position: ``None`` for pauses
    and gaps, the PolarTarget for all other segments.
    """
    pol_iter = iter(all_polars)
    polar_list: list[PolarTarget | None] = []
    for seg in flat_segments:
        if seg == "|" or seg == "GAP":
            polar_list.append(None)
        else:
            polar_list.append(next(pol_iter))
    return polar_list


def _build_initial_nodes(flat_segments: list[str],
                         polar_list: list[PolarTarget | None]) -> list[GestureNode]:
    """Phase 2: Build initial gesture nodes from segments and polar data.

    Consonant nodes are created without cluster information;
    cluster detection is handled in Phase 3.
    """
    nodes: list[GestureNode] = []
    for i, seg in enumerate(flat_segments):
        if seg == "|":
            nodes.append(_make_pause_node(long=True))
        elif seg == "GAP":
            nodes.append(_make_pause_node(long=False))
        else:
            p = polar_list[i]
            if p.is_vowel:
                nodes.append(_make_vowel_node(p.rho, p.theta, seg))
            else:
                nodes.append(_make_consonant_node(
                    p.rho, p.theta, p.active_params, seg))
    return nodes


def _detect_clusters(nodes: list[GestureNode],
                     syl_boundaries: set[int] | None) -> list[bool]:
    """Phase 3: Detect consonant clusters.

    Returns a cluster mask (list[bool]) aligned with *nodes*.
    """
    cluster_mask = [False] * len(nodes)
    syl_bounds = syl_boundaries or set()

    for i in range(1, len(nodes)):
        if (nodes[i].kind == "C" and nodes[i - 1].kind == "C"
                and i not in syl_bounds
                and _cc_eligible(nodes[i - 1].seg_key)
                and _cc_eligible(nodes[i].seg_key)):
            cluster_mask[i] = True
            cluster_mask[i - 1] = True

    return cluster_mask


def _adjust_consonants(nodes: list[GestureNode],
                       polar_list: list[PolarTarget | None],
                       cluster_mask: list[bool]) -> None:
    """Phase 4: Adjust consonant theta/rho using pre-computed vowel indices.

    Modifies *nodes* in place.
    """
    next_vowel, prev_vowel = _precompute_vowel_indices(nodes)
    for i, nd in enumerate(nodes):
        if nd.kind == "C":
            vr = _nearest_vowel_theta(nodes, i, next_vowel, prev_vowel)
            adj = adjust_consonant_theta(polar_list[i], vr, in_cluster=cluster_mask[i])
            nodes[i].rho = adj.rho
            nodes[i].theta = adj.theta
            nodes[i].in_cluster = cluster_mask[i]


def build_gesture_nodes(flat_segments: list[str],
                        all_polars: list[PolarTarget],
                        syl_boundaries: set[int] | None = None) -> list[GestureNode]:
    """Build the gesture node list from flat segments and polar targets.

    Pipeline: parse_polar_targets → build_initial_nodes → detect_clusters → adjust_consonants.
    """
    polar_list = _parse_polar_targets(flat_segments, all_polars)
    nodes = _build_initial_nodes(flat_segments, polar_list)
    cluster_mask = _detect_clusters(nodes, syl_boundaries)
    _adjust_consonants(nodes, polar_list, cluster_mask)
    return nodes


# ═══════════════════════════════════════════════════════════════════════
# Pipeline: build_gesture_anchors
# ═══════════════════════════════════════════════════════════════════════

def _create_base_anchors(nodes: list[GestureNode]) -> list[GestureAnchor]:
    """Create the initial anchor list from V and pause nodes."""
    anchors: list[GestureAnchor] = []
    n = len(nodes)
    for i, nd in enumerate(nodes):
        if nd.kind == "V":
            prev_is_V = i > 0 and nodes[i - 1].kind == "V"
            next_is_V = i < n - 1 and nodes[i + 1].kind == "V"
            hold = not prev_is_V and not next_is_V
            anchors.append(_make_vowel_anchor(i, [nd.rho, nd.theta], hold))
        elif nd.kind == "pause":
            anchors.append(_make_pause_anchor(i, nd.long))
    return anchors


def _insert_word_end_anchors(anchors: list[GestureAnchor],
                              nodes: list[GestureNode],
                              delta_e: float = DEFAULT_DELTA_E) -> list[GestureAnchor]:
    """Insert word-end anchors (Ve) before pauses that follow consonants.

    Ve = (delta_e * rho_vowel, theta_vowel) — Berthommier (2023, §2.2).
    When delta_e = 1, Ve = V (full vowel, no schwa, fusion ready).
    """
    inserts: list[tuple[int, GestureAnchor]] = []
    for ai, a in enumerate(anchors):
        if a.kind == "pause" and ai > 0:
            pause_i = a.i
            if pause_i > 0 and nodes[pause_i - 1].kind == "C":
                result = _find_prev_vowel(nodes, pause_i - 1)
                last_rho, last_theta = result if result is not None else (0.8, np.pi)
                we = _make_vowel_anchor(
                    pause_i, [delta_e * last_rho, last_theta], False,
                    is_word_end=True)
                inserts.append((ai, we))

    for pos, we in sorted(inserts, key=lambda x: x[0], reverse=True):
        anchors.insert(pos, we)
    return anchors


def _insert_vowel_onset_anchors(anchors: list[GestureAnchor],
                                nodes: list[GestureNode],
                                word_starts: list[int] | None,
                                delta_o: float = DEFAULT_DELTA_O,
                                delta_e: float = DEFAULT_DELTA_E) -> list[GestureAnchor]:
    """Insert vowel-onset anchors (Vo) at word starts that begin with a consonant.

    Utterance-initial word: Vo = (delta_o * rho_V, theta_V) — Berthommier
    2023 §2.2 (VOYDEB). The original hardcoded 0.5 (which gave Vo_rho = 0.5
    regardless of the vowel) is replaced by ``delta_o * rho_V`` so that when
    delta_o -> 1, Vo -> V (full vowel, fusion-ready). This matches the
    article's formula Vo = (delta_o * rho_V, theta_V) and is symmetric with
    the Ve path (which already used delta_e * rho_vowel).

    Non-initial word (after a pause): the pause is the diphthong transition
    between the PREVIOUS Ve and the next Vo (article §2.2), so the onset
    anchors on the previous word's Ve point — Vo := (delta_e * rho_prev,
    theta_prev). Consequences: with COEFCEN = 1 (delta_e = 1, Ve = V) the
    word-2 onset departs from the FULL vowel and there is no return to the
    half-radius schwa at the word boundary; with the symmetric
    delta_o = delta_e the value is unchanged (0.45 for /i/), preserving the
    "big@bi" schwa demonstrations. (Author-validated 2026-10-06.)
    """
    if not word_starts:
        return anchors
    n = len(nodes)
    inserts: list[tuple[int, GestureAnchor]] = []
    for w_idx, ws in enumerate(word_starts):
        if ws < n and nodes[ws].kind == "C":
            prev_v = None
            if w_idx > 0:
                for k in range(ws - 1, -1, -1):
                    if nodes[k].kind == "V":
                        prev_v = (nodes[k].rho, nodes[k].theta)
                        break
            if prev_v is not None:
                rho_v, theta_v = delta_e * prev_v[0], prev_v[1]
            else:
                result = _find_next_vowel_rho_theta(nodes, ws)
                if result is not None:
                    rho_v, theta_v = result
                else:
                    rho_v, theta_v = 0.9, np.pi
                rho_v = delta_o * rho_v
            vd = _make_vowel_anchor(ws, [rho_v, theta_v], False,
                                     is_vowel_onset=True)
            insert_pos = _find_anchor_insert_pos(anchors, ws)
            inserts.append((insert_pos, vd))

    for pos, vd in sorted(inserts, key=lambda x: x[0], reverse=True):
        anchors.insert(pos, vd)
    return anchors


def _insert_syllable_onset_anchors(anchors: list[GestureAnchor],
                                   syl_boundary_info: list[SyllableBoundary] | None,
                                   delta_e: float = DEFAULT_DELTA_E) -> list[GestureAnchor]:
    """Insert syllable-onset anchors (Vo) at inter-syllable boundaries.

    Vo = (weight * rho_prev_vowel, theta_prev_vowel) with
    weight = (1 - prev_boolast) + delta_e * prev_boolast * curr_booldeb —
    the reference (Timit-to-Maeda) formula, which runs on COEFCEN: the
    syllable-chaining coefficient, NOT VOYDEB (delta_o, the
    word/utterance-initial onset weight). Legacy naming: VOYDEB applies
    to the début (word-initial Vo); COEFCEN (delta_e) chains syllables —
    it also sets the Ve, so under COEFCEN = 1 the syllable-onset anchor
    IS the full previous vowel: the Ve of the coda syllable is
    coarticulated with the next syllable's vowel (the very definition of
    the '.' C.C concatenation, e.g. big.bi: the Ve of "big" merges with
    the /i/ of "bi" — author ruling 2026-10-07).
    """
    if not syl_boundary_info:
        return anchors
    inserts: list[tuple[int, GestureAnchor]] = []
    for sb in syl_boundary_info:
        curr_segs = sb.curr_segments
        prev_segs = sb.prev_segments
        flat_idx = sb.flat_idx

        curr_booldeb = 1 if not is_vowel_seg(curr_segs[0]) else 0
        if curr_booldeb == 0:
            continue

        prev_boolast = 1 if not is_vowel_seg(prev_segs[-1]) else 0

        prev_last_rho, prev_last_theta = _find_last_vowel_in_segs(prev_segs)

        weight = (1 - prev_boolast) + delta_e * prev_boolast * curr_booldeb
        vd_rho = weight * prev_last_rho
        vd_theta = prev_last_theta

        vd = _make_vowel_anchor(
            flat_idx, [vd_rho, vd_theta], False,
            is_vowel_onset=True, is_syl_vowel_onset=True)
        insert_pos = _find_anchor_insert_pos(anchors, flat_idx)
        inserts.append((insert_pos, vd))

    for pos, vd in sorted(inserts, key=lambda x: x[0], reverse=True):
        anchors.insert(pos, vd)
    return anchors


def _ensure_terminal_anchors(anchors: list[GestureAnchor], n: int) -> list[GestureAnchor]:
    """Ensure the anchor list starts with a synth anchor and ends with one."""
    if not anchors or anchors[0].is_vowel_onset or anchors[0].kind != "synth":
        anchors.insert(0, _make_synth_anchor(-1, [0.0, 0.0]))
    anchors.append(_make_synth_anchor(n, [0.0, 0.0]))
    return anchors


def build_gesture_anchors(nodes: list[GestureNode],
                           word_starts: list[int] | None = None,
                           syl_boundary_info: list[SyllableBoundary] | None = None,
                           delta_o: float = DEFAULT_DELTA_O,
                           delta_e: float = DEFAULT_DELTA_E) -> list[GestureAnchor]:
    """Build the gesture anchor list from nodes.

    Pipeline: create_base_anchors → insert_word_end_anchors → insert_vowel_onset_anchors
    → insert_syllable_onset_anchors → ensure_terminal_anchors.

    Parameters
    ----------
    delta_o : float, default = COEFCEN (0.5)
        Anchoring vowel Vo onset coefficient (Berthommier 2023, §2.2).
        Vo = (delta_o * rho_vowel, theta_vowel). When delta_o = 1,
        Vo = V (full vowel, fusion-ready).
    delta_e : float, default = COEFCEN (0.5)
        Anchoring vowel Ve end coefficient. Ve = (delta_e * rho_vowel,
        theta_vowel). When delta_e = 1, Ve = V.
    """
    anchors = _create_base_anchors(nodes)
    anchors = _insert_word_end_anchors(anchors, nodes, delta_e=delta_e)
    anchors = _insert_vowel_onset_anchors(anchors, nodes, word_starts,
                                          delta_o=delta_o, delta_e=delta_e)
    anchors = _insert_syllable_onset_anchors(anchors, syl_boundary_info, delta_e=delta_e)
    anchors = _ensure_terminal_anchors(anchors, len(nodes))
    return anchors
