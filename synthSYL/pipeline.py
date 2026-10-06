# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
pipeline.py — Main pipeline orchestrator for the synthSYL model.

Implements the full synthesis pipeline:
  Parsing → Phonological analysis → Polar coordinates →
  Gestural planning → Anchors → Trajectories → Envelope → Maeda projection

This module re-exports ``panphon_pipeline`` as the primary entry point.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    FS,
    T_S,
    DEFAULT_CDEC,
    DEFAULT_CEXP,
    DEFAULT_CONSONANT_FACTOR,
    DEFAULT_DELTA_E,
    DEFAULT_DELTA_O,
    DEFAULT_K,
    DEFAULT_KVOY,
    DEFAULT_LONG_PAUSE_FACTOR,
    DEFAULT_NU,
    DEFAULT_PEXP,
    DEFAULT_SHORT_PAUSE_FACTOR,
    DEFAULT_SIGMOID_STEEPNESS,
    DEFAULT_T,
    DEFAULT_VOWEL_FACTOR,
    NEUTRAL_MAEDA,
)
from .envelope import build_envelope_from_blocks, build_envelope_tokens
from .gesture import build_gesture_anchors, build_gesture_nodes
from .parsing import parse_input
from .phonology import PANPHON_DB, get_panphon, panphon_to_polar
from .trajectory import build_global_pval
from .types import BlockInfo, GestureAnchor, GestureNode, PipelineResult, PolarTarget


def panphon_pipeline(
    text: str,
    T: int = DEFAULT_T,
    nu: int = DEFAULT_NU,
    K: int = DEFAULT_K,
    Kvoy: int = DEFAULT_KVOY,
    Pexp: int = DEFAULT_PEXP,
    cexp: float = DEFAULT_CEXP,
    cdec: float = DEFAULT_CDEC,
    verbose: bool = False,
    vowel_duration_factor: float = DEFAULT_VOWEL_FACTOR,
    consonant_duration_factor: float = DEFAULT_CONSONANT_FACTOR,
    short_pause_duration_factor: float = DEFAULT_SHORT_PAUSE_FACTOR,
    long_pause_duration_factor: float = DEFAULT_LONG_PAUSE_FACTOR,
    pause_duration_factor: float | None = None,
    use_sigmoid_pause: bool = False,
    sigmoid_steepness: float = DEFAULT_SIGMOID_STEEPNESS,
    delta_o: float = DEFAULT_DELTA_O,
    delta_e: float = DEFAULT_DELTA_E,
) -> PipelineResult | None:
    """Run the full synthSYL articulatory synthesis pipeline.

    Parameters
    ----------
    text : str
        Input phonetic text (synthSYL notation with '|' for pauses).
    T : int
        Base Maeda time step.
    nu, K, Kvoy, Pexp : int
        Coarticulation and arc parameters.
    cexp, cdec : float
        Envelope rise/decay exponents.
    verbose : bool
        Print diagnostic information.
    vowel_duration_factor, consonant_duration_factor : float
        Duration multipliers for vowels and consonants.
    short_pause_duration_factor, long_pause_duration_factor : float
        Duration multipliers for pauses.
    pause_duration_factor : float | None
        Override for short_pause_duration_factor (backward compat).
    use_sigmoid_pause : bool
        Use sigmoid transitions for pauses (experimental).
    sigmoid_steepness : float
        Steepness of the sigmoid transition.
    delta_o, delta_e : float
        Anchoring vowel coefficients Vo and Ve (Berthommier 2023, §2.2).
        Vo = (delta_o * rho_vowel, theta_vowel), Ve = (delta_e * rho_vowel,
        theta_vowel). Default = COEFCEN (0.5). When delta_o = delta_e = 1,
        Vo = Ve = V (full fusion, see arXiv:2307.02299 §4).

    Returns
    -------
    PipelineResult | None
        The synthesis result, or None if no phonemes are recognised.
    """
    if pause_duration_factor is not None:
        short_pause_duration_factor = pause_duration_factor

    # ── Step 1(a): Text analysis ──
    parse_result = parse_input(text)
    if not parse_result.flat:
        print("  → no phoneme recognised")
        return None

    flat = parse_result.flat
    syl_boundaries = parse_result.syllable_boundaries
    word_starts = parse_result.word_starts
    syl_boundary_info = parse_result.syllable_boundary_info

    # ── Step 1(b): Phonological analysis → polar coordinates ──
    T_cons = max(1, int(round(T * consonant_duration_factor)))
    T_voy = max(1, int(round(T * vowel_duration_factor)))
    T_pause_short = max(1, int(round(T * short_pause_duration_factor)))
    T_pause_long = max(1, int(round(T * long_pause_duration_factor)))

    all_polars: list[PolarTarget] = []
    for seg in flat:
        if seg == "|" or seg == "GAP":
            continue
        pv = get_panphon(seg)
        if pv is None:
            pv = np.array(PANPHON_DB["ə"], dtype=float)
        p = panphon_to_polar(seg, pv)
        # segment_key is already set inside panphon_to_polar
        all_polars.append(p)

    # ── Step 2: Gestural planning ──
    nodes = build_gesture_nodes(flat, all_polars, syl_boundaries=syl_boundaries)
    anchors = build_gesture_anchors(nodes, word_starts=word_starts,
                                    syl_boundary_info=syl_boundary_info,
                                    delta_o=delta_o, delta_e=delta_e)

    # ── Step 3: Global continuous polar trajectory ──
    Pval, last_pt, block_info = build_global_pval(
        nodes, anchors, T_cons, T_voy, T_pause_short, T_pause_long,
        nu, K, Kvoy, Pexp,
        use_sigmoid_pause, sigmoid_steepness)

    # ── Step 4(a): Envelope tokens ──
    tokens = build_envelope_tokens(
        flat, all_polars, nodes,
        syl_boundaries=syl_boundaries,
        long_pause_duration_factor=long_pause_duration_factor,
        T=T)

    durfen = int(T * FS * T_S)
    sps = int(T_S * FS)

    # ── Step 4(b): Envelope ──
    envelope = build_envelope_from_blocks(
        block_info, nodes, anchors, all_polars, flat, syl_boundaries,
        sps, cexp, cdec)

    # ── Pad/trim envelope to match Pval ──
    n_sig = Pval.shape[0] * sps
    if len(envelope) > n_sig:
        envelope = envelope[:n_sig]
    elif len(envelope) < n_sig:
        pad_len = n_sig - len(envelope)
        last_val = envelope[-1] if len(envelope) > 0 else 0.0
        if pad_len > 1 and last_val > 1e-10:
            envelope = np.concatenate([envelope,
                np.linspace(last_val, 0.0, pad_len + 1)[1:]])
        else:
            envelope = np.concatenate([envelope, np.zeros(pad_len)])

    # ── Verbose diagnostics ──
    if verbose:
        dP = (np.max(np.abs(np.diff(Pval, axis=0))) if Pval.shape[0] > 1 else 0.0)
        dE = (np.max(np.abs(np.diff(envelope))) if len(envelope) > 1 else 0.0)
        print(f"Tokens     : {' '.join(tokens)}")
        print(f"Anchors    : {[(a.kind, round(a.pt[0],2) if a.pt else None, round(a.pt[1]/np.pi,2) if a.pt else None) for a in anchors]}")
        print(f"Pval       : {Pval.shape[0]} steps × 7  |  signal : {n_sig} samples")
        print(f"Continuity : max|ΔP|={dP:.5f}  max|Δenvelope|={dE:.5f}")

    # ── Build result ──
    seg_map = {p._seg: p for p in all_polars}

    return PipelineResult(
        Pval=Pval,
        envelope=envelope,
        tokens=tokens,
        durfen=durfen,
        segments=flat,
        polars=all_polars,
        nodes=nodes,
        anchors=anchors,
        seg_map=seg_map,
        syl_boundaries=syl_boundaries,
    )
