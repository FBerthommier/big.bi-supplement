# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
types.py — Data models for the synthSYL articulatory synthesis model.

Every structured dictionary from the original monolithic plugin is replaced
by a typed dataclass for memory efficiency.  All field
names are preserved from the original dict keys so that the semantic mapping
is transparent and the public API is unchanged.

Backward-compatible dict-style access (``obj["key"]`` and ``obj.get("key")``)
is provided on dataclasses that are returned by public functions, ensuring
that existing callers continue to work without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Unified phonetic segment description
# ═══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SegmentDescription:
    """Unified entry merging VOWELS_SYNTSYL / CONSONANTS_SYNTSYL / PANPHON_DB.

    Attributes:
        key:      synthSYL key (e.g. 'u', 'b', 'd')
        IPA:      IPA symbol (e.g. 'u', 'b', 'd')
        rho:      polar radius in the articulatory space
        theta:    polar angle in the articulatory space
        is_vowel: True for vowels and glides, False for consonants
        params:   active Maeda parameter indices (empty list → all 7)
        features: PanPhon feature vector (21 elements), or empty list
    """

    key: str
    IPA: str
    rho: float
    theta: float
    is_vowel: bool
    params: list[int] = field(default_factory=list)
    features: list[int] = field(default_factory=list)

    @property
    def active_params(self) -> list[int]:
        """Return the active Maeda parameters (all 7 for vowels)."""
        return self.params if self.params else list(range(7))


# ═══════════════════════════════════════════════════════════════════════
# Polar coordinate target
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class PolarTarget:
    """Polar coordinate target for articulatory planning.

    Mirrors the dict returned by ``panphon_to_polar()``.  Supports
    dict-style access (``pt["rho"]``, ``pt.get("syntsyl_key")``)
    for backward compatibility with callers that expect a dict.

    Renamed fields (v16.1):
        ``params_actifs`` → ``active_params``
        The old name is accessible via ``__getattr__`` and dict-style
        access with a ``DeprecationWarning``.
    """

    rho: float
    theta: float
    active_params: list[int] = field(default_factory=list)
    is_vowel: bool = False
    syntsyl_key: str | None = None
    origin: str = "exact"
    _seg: str = ""  # original segment key (for cluster resolution)

    # ── Deprecated field name mapping ──
    _DEPRECATED_FIELDS: ClassVar[dict[str, str]] = {
        "params_actifs": "active_params",
    }

    def __getattr__(self, name: str) -> Any:
        """Redirect deprecated field names with a warning."""
        new_name = type(self)._DEPRECATED_FIELDS.get(name)
        if new_name is not None:
            import warnings
            warnings.warn(
                f"PolarTarget.{name} is deprecated; use "
                f"PolarTarget.{new_name} instead.",
                DeprecationWarning, stacklevel=2,
            )
            return getattr(self, new_name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # ── Backward-compatible dict-style access ──

    def __getitem__(self, key: str) -> Any:
        """Allow ``pt["rho"]`` style access.

        Also maps deprecated key names (e.g. ``"params_actifs"``)
        to their current equivalents.
        """
        resolved = type(self)._DEPRECATED_FIELDS.get(key, key)
        return getattr(self, resolved)

    def get(self, key: str, default: Any = None) -> Any:
        """Allow ``pt.get("key", default)`` style access.

        Also maps deprecated key names.
        """
        try:
            resolved = type(self)._DEPRECATED_FIELDS.get(key, key)
            return getattr(self, resolved)
        except AttributeError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backward compatibility with legacy code.

        Includes both the current key (``"active_params"``) and the
        deprecated key (``"params_actifs"``) for backward compatibility.
        """
        return {
            "rho": self.rho,
            "theta": self.theta,
            "active_params": self.active_params,
            "params_actifs": self.active_params,  # deprecated alias
            "is_vowel": self.is_vowel,
            "syntsyl_key": self.syntsyl_key,
            "origin": self.origin,
            "_seg": self._seg,
        }


# ═══════════════════════════════════════════════════════════════════════
# Gesture node
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class GestureNode:
    """Node in the gestural graph (V, C, or pause).

    Supports dict-style access for backward compatibility.
    """

    kind: str  # 'V', 'C', 'pause'
    rho: float = 0.0
    theta: float = 0.0
    params: list[int] = field(default_factory=list)
    seg_key: str = ""
    in_cluster: bool = False
    long: bool = False  # for pause nodes

    def __getitem__(self, key: str) -> Any:
        """Allow ``nd["kind"]`` style access."""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Allow ``nd.get("key", default)`` style access."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backward compatibility."""
        d: dict[str, Any] = {
            "kind": self.kind,
            "rho": self.rho,
            "theta": self.theta,
            "seg_key": self.seg_key,
        }
        if self.kind == "C":
            d["params"] = self.params
            d["in_cluster"] = self.in_cluster
        if self.kind == "pause":
            d["long"] = self.long
        return d


# ═══════════════════════════════════════════════════════════════════════
# Gesture anchor
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class GestureAnchor:
    """Anchor point in gestural planning.

    Supports dict-style access for backward compatibility.

    Renamed fields (v16.1):
        ``is_voydeb``      → ``is_vowel_onset``
        ``is_syl_voydeb``  → ``is_syl_vowel_onset``
        The old names are accessible via ``__getattr__`` and dict-style
        access with a ``DeprecationWarning``.
    """

    i: int  # node index
    pt: list[float] | None  # [rho, theta]
    hold: bool
    kind: str  # 'V', 'pause', 'synth'
    long: bool = False
    is_vowel_onset: bool = False
    is_word_end: bool = False
    is_syl_vowel_onset: bool = False

    # ── Deprecated field name mapping ──
    _DEPRECATED_FIELDS: ClassVar[dict[str, str]] = {
        "is_voydeb": "is_vowel_onset",
        "is_syl_voydeb": "is_syl_vowel_onset",
    }

    def __getattr__(self, name: str) -> Any:
        """Redirect deprecated field names with a warning."""
        new_name = type(self)._DEPRECATED_FIELDS.get(name)
        if new_name is not None:
            import warnings
            warnings.warn(
                f"GestureAnchor.{name} is deprecated; use "
                f"GestureAnchor.{new_name} instead.",
                DeprecationWarning, stacklevel=2,
            )
            return getattr(self, new_name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, key: str) -> Any:
        """Allow ``a["kind"]`` style access.

        Also maps deprecated key names.
        """
        resolved = type(self)._DEPRECATED_FIELDS.get(key, key)
        return getattr(self, resolved)

    def get(self, key: str, default: Any = None) -> Any:
        """Allow ``a.get("key", default)`` style access."""
        try:
            resolved = type(self)._DEPRECATED_FIELDS.get(key, key)
            return getattr(self, resolved)
        except AttributeError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backward compatibility."""
        d: dict[str, Any] = {
            "i": self.i,
            "pt": self.pt,
            "hold": self.hold,
            "kind": self.kind,
        }
        if self.kind == "pause":
            d["long"] = self.long
        if self.is_vowel_onset:
            d["is_vowel_onset"] = True
            d["is_voydeb"] = True  # deprecated alias
        if self.is_word_end:
            d["is_word_end"] = True
        if self.is_syl_vowel_onset:
            d["is_syl_vowel_onset"] = True
            d["is_syl_voydeb"] = True  # deprecated alias
        return d


# ═══════════════════════════════════════════════════════════════════════
# Block information
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class BlockInfo:
    """Information about a trajectory block.

    Supports dict-style access for backward compatibility.
    """

    n_steps: int
    kind: str  # 'plateau', 'background', 'pause', 'cluster', 'initial', 'terminal', 'decay', 'attack'
    long: bool = False
    pre_amp: float = 0.0
    post_amp: float = 0.0
    n_cons: int = 0
    cons_tokens: list[str] = field(default_factory=list)
    pre_token: str = "O"
    post_token: str = "O"
    has_following_plateau: bool = False

    def __getitem__(self, key: str) -> Any:
        """Allow ``bi["kind"]`` style access."""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Allow ``bi.get("key", default)`` style access."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backward compatibility."""
        d: dict[str, Any] = {"n_steps": self.n_steps, "kind": self.kind}
        if self.kind == "pause":
            d["long"] = self.long
        if self.pre_amp != 0.0:
            d["pre_amp"] = self.pre_amp
        if self.post_amp != 0.0:
            d["post_amp"] = self.post_amp
        if self.kind == "cluster":
            d["n_cons"] = self.n_cons
            d["cons_tokens"] = self.cons_tokens
            d["pre_token"] = self.pre_token
            d["post_token"] = self.post_token
            d["has_following_plateau"] = self.has_following_plateau
        return d


# ═══════════════════════════════════════════════════════════════════════
# Syllable boundary information
# ═══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SyllableBoundary:
    """Structured representation of an inter-syllable boundary.

    Replaces the anonymous ``dict[str, Any]`` previously used for
    ``syl_boundary_info`` entries.

    Attributes:
        flat_idx:       Index in the flat segment list where the boundary occurs.
        prev_segments:  Segments of the syllable preceding the boundary.
        curr_segments:  Segments of the syllable following the boundary.
    """

    flat_idx: int
    prev_segments: list[str]
    curr_segments: list[str]


# ═══════════════════════════════════════════════════════════════════════
# Parse result
# ═══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ParseResult:
    """Structured result of the text parsing stage.

    Replaces the anonymous ``tuple`` previously returned by
    ``parse_input()``.  All fields carry the same data; only the
    encapsulation has changed.

    Attributes:
        flat:                  Flat list of IPA segments and markers.
        blocks:                Syllable blocks (list of segment lists).
        syllables_per_word:    Syllables grouped by word.
        word_starts:           Flat-list indices where each word begins.
        syllable_boundaries:   Set of flat-list indices at syllable boundaries.
        syllable_boundary_info: Detailed boundary descriptors.
    """

    flat: list[str]
    blocks: list[list[str]]
    syllables_per_word: list[list[list[str]]]
    word_starts: list[int]
    syllable_boundaries: set[int]
    syllable_boundary_info: list[SyllableBoundary]


# ═══════════════════════════════════════════════════════════════════════
# Pipeline result
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class PipelineResult:
    """Result of the synthesis pipeline.

    Supports dict-style access (``result["Pval"]``) for backward
    compatibility with callers that expect a dict.
    """

    Pval: np.ndarray
    envelope: np.ndarray
    tokens: list[str]
    durfen: int
    segments: list[str]
    polars: list[PolarTarget]
    nodes: list[GestureNode]
    anchors: list[GestureAnchor]
    seg_map: dict[str, PolarTarget]
    syl_boundaries: set[int]

    def __getitem__(self, key: str) -> Any:
        """Backward-compatible dict-style access."""
        return getattr(self, key)
