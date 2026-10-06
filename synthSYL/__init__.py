# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
panphon_timit_plugin — Refactored synthSYL articulatory synthesis model.

This package implements the unified gestural architecture for articulatory
speech synthesis, based on the PanPhon phonological feature system and the
Maeda articulatory model.

Pipeline stages:
  1. Parsing — Text analysis and syllable structure
  2. Phonological analysis — PanPhon features → polar coordinates
  3. Gestural planning — Gesture nodes and anchors
  4. Trajectory computation — Continuous polar trajectory (Pval)
  5. Envelope — Temporal amplitude envelope
  6. Maeda projection — Articulatory parameter trajectories

Usage::

    from panphon_timit_plugin import panphon_pipeline

    result = panphon_pipeline("ba di gu")
    print(result.Pval.shape, result.envelope.shape)
"""

from .pipeline import panphon_pipeline
from .types import (
    BlockInfo,
    GestureAnchor,
    GestureNode,
    ParseResult,
    PipelineResult,
    PolarTarget,
    SegmentDescription,
    SyllableBoundary,
)
from .constants import (
    CO,
    MAEDA_NAMES,
    COEFCEN,
    FS,
    T_S,
    N_MAEDA_PARAMS,
    NEUTRAL_MAEDA,
)

__all__ = [
    "panphon_pipeline",
    "ParseResult",
    "PipelineResult",
    "PolarTarget",
    "GestureNode",
    "GestureAnchor",
    "BlockInfo",
    "SegmentDescription",
    "SyllableBoundary",
    "CO",
    "MAEDA_NAMES",
    "COEFCEN",
    "FS",
    "T_S",
    "N_MAEDA_PARAMS",
    "NEUTRAL_MAEDA",
]

__version__ = "16.0.0"
