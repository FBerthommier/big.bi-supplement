# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
phonology.py — Unified PanPhon database, classification, mapping,
and polar-coordinate computation.

This module centralises every phonetic table and phonological rule that
was previously scattered across the monolithic plugin.  All functions
are pure (no side-effects) and fully type-annotated.

Public functions that previously returned anonymous dicts now return
``PolarTarget`` dataclass instances.  Dict-style access (``result["rho"]``)
is preserved via ``PolarTarget.__getitem__`` for backward compatibility.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .constants import (
    ART_CLUSTER,
    COEFCEN,
    RHO_CLUSTER_VELAR,
    RHO_FRICATIVE,
    RHO_FRONT_RND,
    RHO_HI,
    RHO_LOW_MID_RND,
    RHO_MID,
    RHO_NASAL,
    RHO_NASAL_FACTOR,
    RHO_STOP,
    THETA_CC_D,
    THETA_CC_G,
    THETA_CORONAL,
    THETA_DENTAL,
    THETA_GLAFF,
    THETA_LABIAL,
    THETA_V_LAX,
    THETA_NASAL_FACTOR,
    THETA_V_BACK_HI,
    THETA_V_BACK_MID,
    THETA_V_BACK_MID_RND,
    THETA_V_FRONT_RND,
    THETA_V_HI,
    THETA_V_HI_RND,
    THETA_V_LOW,
    THETA_V_TENSE,
    THETA_VELAR,
    THETA_VELAR_BACK,
)
from .types import PolarTarget, SegmentDescription

# ═══════════════════════════════════════════════════════════════════════
# PanPhon feature names (21 features)
# ═══════════════════════════════════════════════════════════════════════
PANPHON_FEATURES: list[str] = [
    "syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid",
    "voi", "sg", "cg", "ant", "cor", "distr", "lab", "hi", "lo",
    "back", "rnd", "tense", "long",
]

# Pre-computed feature-index lookup (O(1) instead of rebuilding each call)
_FEATURE_IDX: dict[str, int] = {f: i for i, f in enumerate(PANPHON_FEATURES)}


def feature_index(name: str) -> int:
    """Return the PanPhon feature index for *name* (O(1) lookup)."""
    return _FEATURE_IDX[name]


# ═══════════════════════════════════════════════════════════════════════
# Raw PanPhon feature database (IPA → 21-element feature vector)
# ═══════════════════════════════════════════════════════════════════════
PANPHON_DB: dict[str, list[int]] = {
    "u":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,1,1,-1],
    "o":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,-1],
    "ɔ":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1],
    "a":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1],
    "ɛ":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    "e":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1],
    "i":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,-1],
    "y":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,1,-1],
    "ø":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1],
    "ə":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    "ɛ̃": [1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    "ɑ̃": [1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1],
    "ɔ̃": [1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1],
    "b":  [-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,0,1,-1,-1,-1,-1,0,-1],
    "d":  [-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,0,-1],
    "g":  [-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,0,-1,1,-1,1,-1,0,-1],
    "p":  [-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,0,1,-1,-1,-1,-1,0,-1],
    "t":  [-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,0,-1],
    "k":  [-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,1,-1,1,-1,0,-1],
    "m":  [-1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,0,1,-1,-1,-1,-1,0,-1],
    "n":  [-1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,0,-1],
    "ɲ":  [-1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,0,-1],
    "f":  [-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,0,1,-1,-1,-1,-1,0,-1],
    "v":  [-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,0,1,-1,-1,-1,-1,0,-1],
    "s":  [-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,0,-1],
    "z":  [-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,0,-1],
    "ʃ":  [-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,-1,0,-1],
    "ʒ":  [-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1,0,-1],
    "ʁ":  [-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,-1,0,-1,-1,-1,1,-1,0,-1],
    "l":  [-1,1,1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,0,-1],
    "r":  [-1,1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,0,-1],
    "j":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,-1],
    "w":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,1,-1],
    "ɥ":  [1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1],
}

# ═══════════════════════════════════════════════════════════════════════
# Mapping tables
# ═══════════════════════════════════════════════════════════════════════
IPA_TO_SYNTSYL: dict[str, str] = {
    "u": "u", "o": "o", "ɔ": "O", "a": "a", "ɛ": "è", "e": "é",
    "i": "i", "y": "y", "ø": "E", "b": "b", "d": "d", "g": "g",
    "p": "p", "t": "t", "k": "k",
}

SIMPLE_TO_IPA: dict[str, str] = {
    "u": "u", "o": "o", "a": "a", "e": "e", "i": "i", "y": "y",
    "E": "ø", "O": "ɔ", "è": "ɛ", "é": "e",
    "b": "b", "d": "d", "g": "g", "p": "p", "t": "t", "k": "k",
    "m": "m", "n": "n", "f": "f", "v": "v", "s": "s", "z": "z",
    "l": "l", "r": "ʁ", "j": "j",
    "S": "ʃ", "Z": "ʒ", "N": "ɲ", "R": "ʁ", "H": "ɥ",
    ".": ".", "-": ".", " ": " ",
}

# Consonant table index (place of articulation)
_TABCONS_IDX: dict[str, int] = {
    "b": 0, "p": 0, "d": 1, "t": 1, "g": 2, "k": 2,
    "l": 1, "n": 1,
    "m": 0, "f": 0, "v": 0,
    "s": 1, "z": 1,
    "ɲ": 2, "ʁ": 2, "ʃ": 2, "ʒ": 2,
    "w": 0, "ɥ": 0,
    "j": 2,
}

# Cluster alias: maps a consonant to its synthSYL representative
_CC_ALIAS: dict[str, str] = {
    "l": "d", "n": "d",
    "m": "b", "f": "b", "v": "b",
    "s": "d", "z": "d",
    "ɲ": "g", "ʁ": "g", "ʃ": "g", "ʒ": "g",
    "w": "b", "ɥ": "b",
    "j": "g",
}

# Consonants whose rho is overridden in cluster context
_CC_RHO_OVERRIDE: set[str] = {"l", "n"}

# ═══════════════════════════════════════════════════════════════════════
# Unified SegmentDescription registry
# ═══════════════════════════════════════════════════════════════════════
# Vowels: (key, IPA, rho, theta)
_VOWEL_ENTRIES: list[tuple[str, str, float, float]] = [
    ("u", "u", 1.0, np.pi / 3),
    ("o", "o", 0.8, np.pi / 2),
    ("O", "ɔ", 0.8, 2 * np.pi / 3),
    ("a", "a", 0.8, np.pi),
    ("è", "ɛ", 0.8, 4 * np.pi / 3),
    ("é", "e", 0.8, 3 * np.pi / 2),
    ("i", "i", 0.9, 5 * np.pi / 3),
    ("y", "y", 0.7, 5.5 * np.pi / 3),
    ("E", "ø", 0.3, 5.5 * np.pi / 3),
]

# Consonants: (key, IPA, rho, theta, params)
_CONSONANT_ENTRIES: list[tuple[str, str, float, float, list[int]]] = [
    ("b", "b", 1.2, np.pi / 3, [0, 1, 5]),
    ("p", "p", 1.2, np.pi / 3, [0, 1, 5]),
    ("d", "d", 1.2, 23 * np.pi / 16, [0, 1, 2, 3]),
    ("t", "t", 1.2, 23 * np.pi / 16, [0, 1, 2, 3]),
    ("g", "g", 1.2, np.pi / 3, [0, 1, 2, 3]),
    ("k", "k", 1.2, np.pi / 3, [0, 1, 2, 3]),
]

# Build the unified registry: IPA symbol → SegmentDescription
SEGMENT_REGISTRY: dict[str, SegmentDescription] = {}


def _build_registry() -> None:
    """Populate SEGMENT_REGISTRY from the vowel and consonant tables."""
    for key, ipa, rho, theta in _VOWEL_ENTRIES:
        feats = PANPHON_DB.get(ipa, [])
        SEGMENT_REGISTRY[ipa] = SegmentDescription(
            key=key, IPA=ipa, rho=rho, theta=theta,
            is_vowel=True, params=[], features=feats,
        )
    for key, ipa, rho, theta, params in _CONSONANT_ENTRIES:
        feats = PANPHON_DB.get(ipa, [])
        SEGMENT_REGISTRY[ipa] = SegmentDescription(
            key=key, IPA=ipa, rho=rho, theta=theta,
            is_vowel=False, params=params, features=feats,
        )


_build_registry()

# ── Backward-compatible dicts (derived from the registry) ──
VOWELS_SYNTSYL: dict[str, dict[str, object]] = {}
CONSONANTS_SYNTSYL: dict[str, dict[str, object]] = {}
for _key, _ipa, _rho, _theta in _VOWEL_ENTRIES:
    VOWELS_SYNTSYL[_key] = {"rho": _rho, "theta": _theta, "IPA": _ipa}
for _key, _ipa, _rho, _theta, _params in _CONSONANT_ENTRIES:
    CONSONANTS_SYNTSYL[_key] = {"rho": _rho, "theta": _theta, "IPA": _ipa, "params": _params}


# ═══════════════════════════════════════════════════════════════════════
# Classification helpers (eliminate duplicated logic)
# ═══════════════════════════════════════════════════════════════════════

def is_vowel(seg: str) -> bool:
    """Return True if *seg* is a vowel (PanPhon syl=+1)."""
    pv = PANPHON_DB.get(seg)
    if pv is None:
        return False
    return pv[feature_index("syl")] == 1


def is_sonorant(seg: str) -> bool:
    """Return True if *seg* is a sonorant consonant."""
    pv = PANPHON_DB.get(seg)
    if pv is None:
        return False
    return pv[feature_index("son")] == 1 and pv[feature_index("cons")] == 1


def is_nasal(seg: str) -> bool:
    """Return True if *seg* is a nasal consonant."""
    pv = PANPHON_DB.get(seg)
    if pv is None:
        return False
    return pv[feature_index("nas")] == 1


def is_continuant(seg: str) -> bool:
    """Return True if *seg* is a continuant."""
    pv = PANPHON_DB.get(seg)
    if pv is None:
        return False
    return pv[feature_index("cont")] == 1


def is_lateral(seg: str) -> bool:
    """Return True if *seg* is a lateral."""
    pv = PANPHON_DB.get(seg)
    if pv is None:
        return False
    return pv[feature_index("lat")] == 1


def is_voiced(seg: str) -> bool:
    """Return True if *seg* is voiced."""
    pv = PANPHON_DB.get(seg)
    if pv is None:
        return False
    return pv[feature_index("voi")] == 1


def classify_place(seg: str) -> str:
    """Return the place of articulation for *seg*.

    Returns one of: 'labial', 'coronal', 'dorsal', 'glottal'.
    """
    pv = PANPHON_DB.get(seg)
    if pv is None:
        return "glottal"
    lab = pv[feature_index("lab")]
    cor = pv[feature_index("cor")]
    hi_back = pv[feature_index("hi")] == 1 and pv[feature_index("back")] == 1
    if lab == 1:
        return "labial"
    if cor == 1:
        return "coronal"
    if hi_back:
        return "dorsal"
    if pv[feature_index("back")] == 1:
        return "dorsal"
    return "glottal"


def active_parameters(seg: str) -> list[int]:
    """Return the list of active Maeda parameter indices for *seg*."""
    key = IPA_TO_SYNTSYL.get(seg)
    if key is not None:
        entry = SEGMENT_REGISTRY.get(seg)
        if entry is not None:
            return entry.active_params
    pv = PANPHON_DB.get(seg)
    if pv is None:
        return list(range(7))
    syl = pv[feature_index("syl")]
    if syl == 1:
        return list(range(7))
    place = classify_place(seg)
    if place == "labial":
        return [0, 1, 5]
    if place in ("coronal", "dorsal"):
        return [0, 1, 2, 3]
    return [0, 6]


def is_synthsyl_consonant(key: str) -> bool:
    """Return True if *key* is a synthSYL consonant key."""
    return key in _TABCONS_IDX


# ═══════════════════════════════════════════════════════════════════════
# PanPhon feature vector accessor
# ═══════════════════════════════════════════════════════════════════════

def get_panphon(seg: str) -> np.ndarray | None:
    """Return the PanPhon feature vector for *seg* (or None)."""
    v = PANPHON_DB.get(seg)
    return np.array(v, dtype=float) if v is not None else None


# ═══════════════════════════════════════════════════════════════════════
# Cluster articulatory parameter computation
# ═══════════════════════════════════════════════════════════════════════

def cluster_art_params(cons_keys: list[str]) -> list[int] | None:
    """Compute the active articulatory parameters for a consonant cluster.

    Returns *None* if the cluster has fewer than 2 consonants.
    """
    m = len(cons_keys)
    if m < 2:
        return None
    resolved = [_CC_ALIAS.get(k, k) for k in cons_keys]
    synt_keys = [IPA_TO_SYNTSYL.get(k, "") for k in resolved]
    if m == 2:
        idx_c1 = _TABCONS_IDX.get(synt_keys[0], 0)
        idx_c2 = _TABCONS_IDX.get(synt_keys[1], 0)
        art_idx = int(idx_c1 + idx_c2 > 2)
        return ART_CLUSTER[art_idx]
    if m == 3:
        places = set(_TABCONS_IDX.get(k, 0) for k in synt_keys if k)
        has_labial = 0 in places
        has_dorsal = 2 in places
        has_uvular = "ʁ" in cons_keys
        params: set[int] = {0, 1, 2}
        if has_labial or not has_dorsal:
            params.add(5)
        if has_uvular:
            params.add(6)
        return sorted(params)
    return None


# ═══════════════════════════════════════════════════════════════════════
# Polar-coordinate computation
# ═══════════════════════════════════════════════════════════════════════

def _vowel_theta_rho(hi: int, lo: int, back: int, rnd: int,
                     tns: int, nas: int) -> tuple[float, float]:
    """Infer theta and rho for a vowel from PanPhon features.

    This is the exact logic extracted from the original ``panphon_to_polar``
    inner function, with magic numbers replaced by named constants.
    """
    # --- theta ---
    if back == 1 and hi == 1:
        theta = THETA_V_BACK_HI
    elif lo == 1:
        theta = THETA_V_LOW
    elif back == 1 and hi != 1 and rnd == 1:
        theta = THETA_V_BACK_MID_RND
    elif back == 1 and hi != 1:
        theta = THETA_V_BACK_MID
    elif hi != 1 and lo != 1 and back != 1 and rnd == 1:
        theta = THETA_V_FRONT_RND
    elif hi == 1 and rnd == 1:
        theta = THETA_V_HI_RND
    elif hi == 1:
        theta = THETA_V_HI
    elif tns == 1:
        theta = THETA_V_TENSE
    else:
        theta = THETA_V_LAX

    # --- rho ---
    if hi == 1 and rnd == 1:
        rho = RHO_FRONT_RND
    elif hi != 1 and lo != 1 and back != 1 and rnd == 1:
        rho = RHO_LOW_MID_RND
    elif hi == 1:
        rho = RHO_HI
    else:
        rho = RHO_MID

    # --- nasalization adjustment ---
    if nas == 1:
        theta *= THETA_NASAL_FACTOR
        rho *= RHO_NASAL_FACTOR

    return theta, rho


def panphon_to_polar(seg: str, pv: np.ndarray) -> PolarTarget:
    """Convert a segment + its PanPhon vector to a polar target.

    Returns a ``PolarTarget`` dataclass.  Dict-style access
    (``result["rho"]``, ``result.get("syntsyl_key")``) is preserved
    for backward compatibility.
    """
    key = IPA_TO_SYNTSYL.get(seg)

    # --- Exact lookup (vowel) ---
    if key is not None:
        entry = SEGMENT_REGISTRY.get(seg)
        if entry is not None and entry.is_vowel:
            return PolarTarget(
                rho=entry.rho,
                theta=entry.theta,
                active_params=list(range(7)),
                is_vowel=True,
                syntsyl_key=key,
                origin="exact",
                _seg=seg,
            )
        # --- Exact lookup (consonant) ---
        if entry is not None and not entry.is_vowel:
            return PolarTarget(
                rho=entry.rho,
                theta=entry.theta,
                active_params=entry.params,
                is_vowel=False,
                syntsyl_key=key,
                origin="exact",
                _seg=seg,
            )

    # --- Feature-based inference ---
    idx = _FEATURE_IDX
    syl  = pv[idx["syl"]];  hi   = pv[idx["hi"]];   lo   = pv[idx["lo"]]
    back = pv[idx["back"]]; rnd  = pv[idx["rnd"]];  cor  = pv[idx["cor"]]
    lab  = pv[idx["lab"]];  nas  = pv[idx["nas"]];  cont = pv[idx["cont"]]
    tns  = pv[idx["tense"]]

    is_vowel_feature = syl == 1

    # --- Vowel (not a glide) ---
    if is_vowel_feature and seg not in ("j", "w", "ɥ"):
        theta, rho = _vowel_theta_rho(hi, lo, back, rnd, tns, nas)
        return PolarTarget(
            rho=rho,
            theta=theta,
            active_params=list(range(7)),
            is_vowel=True,
            syntsyl_key=None,
            origin="inferred_v14",
            _seg=seg,
        )

    # --- Glide (j, w, ɥ) ---
    if seg in ("j", "w", "ɥ"):
        theta, rho = _vowel_theta_rho(hi, lo, back, rnd, tns, nas)
        active_params = [0, 1, 5] if rnd == 1 else [0, 1, 2, 3]
        return PolarTarget(
            rho=rho,
            theta=theta,
            active_params=active_params,
            is_vowel=False,
            syntsyl_key=None,
            origin="glide_v14_via_vowel_tree",
            _seg=seg,
        )

    # --- Consonant (inferred from features) ---
    if lab == 1:
        theta = THETA_LABIAL
        active_params = [0, 1, 5]
    elif hi == 1 and back == 1:
        theta = THETA_VELAR
        active_params = [0, 1, 2, 3]
    elif hi == 1:
        theta = -np.pi / 12
        active_params = [0, 1, 2, 3]
    elif back == 1:
        theta = THETA_VELAR
        active_params = [0, 1, 2, 6]
    elif cor == 1:
        theta = THETA_CORONAL
        active_params = [0, 1, 2, 3]
    else:
        theta = THETA_GLAFF
        active_params = [0, 6]

    rho = RHO_STOP if (cont != 1 and nas != 1) else (RHO_NASAL if nas == 1 else RHO_FRICATIVE)

    return PolarTarget(
        rho=rho,
        theta=theta,
        active_params=active_params,
        is_vowel=False,
        syntsyl_key=None,
        origin="inferred_v14",
        _seg=seg,
    )


def adjust_consonant_theta(cp: dict | PolarTarget, voy_theta: float,
                           in_cluster: bool = False) -> dict | PolarTarget:
    """Adjust the theta/rho of a consonant polar target.

    Accepts both a dict and a PolarTarget for backward compatibility.
    Returns the same type as the input.
    """
    # Normalize input to PolarTarget
    if isinstance(cp, dict):
        pt = PolarTarget(
            rho=cp["rho"], theta=cp["theta"],
            active_params=cp.get("active_params", cp.get("params_actifs", [])),
            is_vowel=cp.get("is_vowel", False),
            syntsyl_key=cp.get("syntsyl_key"),
            origin=cp.get("origin", "exact"),
            _seg=cp.get("_seg", ""),
        )
        was_dict = True
    else:
        pt = cp
        was_dict = False

    rho = pt.rho
    theta = pt.theta
    key = pt.syntsyl_key
    effective_key = key
    seg_cc = ""
    if in_cluster and key is None:
        seg_cc = pt._seg
        effective_key = _CC_ALIAS.get(seg_cc, None)

    if in_cluster:
        if effective_key in ("d", "t"):
            if seg_cc in _CC_RHO_OVERRIDE:
                rho = RHO_STOP
            theta = THETA_CC_D
        elif effective_key in ("g", "k"):
            if key in ("g", "k"):
                rho = RHO_CLUSTER_VELAR
            theta = THETA_CC_G
    else:
        if key in ("d", "t"):
            theta = THETA_DENTAL
        elif key in ("g", "k"):
            if voy_theta <= np.pi:
                rho = RHO_STOP
                theta = THETA_VELAR
            else:
                rho = RHO_CLUSTER_VELAR
                theta = THETA_VELAR_BACK

    result = replace(pt, rho=rho, theta=theta)
    return result.to_dict() if was_dict else result


# ═══════════════════════════════════════════════════════════════════════
# Envelope token classification
# ═══════════════════════════════════════════════════════════════════════

def phoneme_base_token(seg: str, polar: dict | PolarTarget) -> str:
    """Classify a phoneme into an envelope token type.

    Accepts both a dict and a PolarTarget for backward compatibility.
    Returns one of: 'V', 'C', 'c', 'R', 'r', 'N', 'L'.
    """
    # Normalize to dict-style access (works for both dict and PolarTarget)
    if isinstance(polar, PolarTarget):
        is_vowel_flag = polar.is_vowel
        syntsyl_key = polar.syntsyl_key or ""
    else:
        is_vowel_flag = polar["is_vowel"]
        syntsyl_key = polar.get("syntsyl_key", "")

    if is_vowel_flag:
        return "V"
    if seg in ("j", "w", "ɥ"):
        return "V"
    pv = get_panphon(seg)
    if pv is None:
        if syntsyl_key in ("b", "d", "g", "p", "t", "k"):
            return "C"
        return "R"
    nas = pv[feature_index("nas")]; cont = pv[feature_index("cont")]
    son = pv[feature_index("son")]; lat  = pv[feature_index("lat")]
    voi = pv[feature_index("voi")]
    if nas == 1:
        return "N"
    if son == 1 and (lat == 1 or cont == 1):
        return "L"
    if cont == 1:
        return "r" if voi == 1 else "R"
    if cont == -1:
        return "C"
    return "r"
