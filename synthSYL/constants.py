# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
constants.py — Named constants for the synthSYL articulatory synthesis model.

All magic numbers from the original monolithic plugin are replaced by
descriptively-named constants here.  Every value is taken verbatim from
the original code so that numerical identity is preserved.
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Maeda articulatory model
# ═══════════════════════════════════════════════════════════════════════
N_MAEDA_PARAMS: int = 7
MAEDA_NAMES: list[str] = ["Jaw", "Body", "Dorsum", "Tip", "LipP", "LipH", "Hy"]

# CO matrix — Maeda projection coefficients (7 × 3)
# Each row: [c0, c1, c2] where P_i = c1 + rho * c0 * cos(c2 - theta)
CO: np.ndarray = np.array(
    [
        [-1.5, 0.0, np.pi],       # 0  Jaw
        [-2.5, 0.0, -np.pi / 3],  # 1  Body
        [3.0, 0.0, np.pi / 3],    # 2  Dorsum
        [-2.75, 0.5, np.pi],      # 3  Tip
        [3.0, 0.0, np.pi / 3],    # 4  LipP (protrusion)
        [2.5, 0.5, np.pi],        # 5  LipH (aperture)
        [-2.0, 0.0, np.pi / 3],   # 6  Hy
    ]
)

# ═══════════════════════════════════════════════════════════════════════
# Physical / sampling constants
# ═══════════════════════════════════════════════════════════════════════
COEFCEN: float = 0.5       # Vowel-decay coefficient (centre weight)
FS: int = 20_000            # Sampling frequency (Hz)
T_S: float = 10e-3          # Sample period (s)

# ═══════════════════════════════════════════════════════════════════════
# Polar-angle constants (theta)
# ═══════════════════════════════════════════════════════════════════════
# Standalone consonant thetas
THETA_DENTAL: float = 23 * np.pi / 16       # d, t  (dental/alveolar)
THETA_LABIAL: float = np.pi / 3             # b, p  (labial)
THETA_VELAR: float = np.pi / 3              # g, k  (velar)
THETA_VELAR_BACK: float = 23 * np.pi / 12   # g, k after back vowel (theta > pi)
THETA_CORONAL: float = 23 * np.pi / 16      # coronal (same as dental here)
THETA_GLAFF: float = np.pi / 6              # default/glotto-affricate

# Cluster consonant thetas
THETA_CC_D: float = -2.5 * np.pi / 6        # dental in cluster
THETA_CC_G: float = -np.pi / 12             # velar in cluster

# Vowel thetas (derived from PANPHON features)
THETA_V_BACK_HI: float = np.pi / 3
THETA_V_LOW: float = np.pi
THETA_V_BACK_MID_RND: float = np.pi / 2
THETA_V_BACK_MID: float = 2 * np.pi / 3
THETA_V_FRONT_RND: float = 5.5 * np.pi / 3
THETA_V_HI_RND: float = 5.5 * np.pi / 3
THETA_V_HI: float = 5 * np.pi / 3
THETA_V_TENSE: float = 3 * np.pi / 2
THETA_V_LAX: float = 4 * np.pi / 3

# Nasalization adjustment factor
THETA_NASAL_FACTOR: float = 0.97

# ═══════════════════════════════════════════════════════════════════════
# Polar-radius constants (rho)
# ═══════════════════════════════════════════════════════════════════════
RHO_STOP: float = 1.2          # Stop consonant
RHO_FRICATIVE: float = 1.1     # Fricative consonant
RHO_NASAL: float = 1.05        # Nasal consonant
RHO_HI: float = 0.9            # High vowel
RHO_FRONT_RND: float = 0.7     # Front rounded vowel
RHO_MID: float = 0.8           # Mid vowel
RHO_LOW_MID_RND: float = 0.3   # Low-mid front rounded vowel
RHO_CLUSTER_VELAR: float = 1.1 # Velar in cluster context

# Nasalization adjustment factor
RHO_NASAL_FACTOR: float = 0.95

# ═══════════════════════════════════════════════════════════════════════
# Articulatory cluster parameter sets
# ═══════════════════════════════════════════════════════════════════════
ART_CLUSTER: list[list[int]] = [
    [0, 1, 2, 5],   # labial-dental cluster
    [0, 1, 2],      # dental-dorsal cluster
]

# ═══════════════════════════════════════════════════════════════════════
# Envelope token amplitude minimums
# ═══════════════════════════════════════════════════════════════════════
TOKEN_AMPL_MIN: dict[str, float] = {
    "O": 0.00,
    "F": 1.00,
    "V": 1.00,
    "C": 0.00,
    "c": 0.00,
    "R": 0.35,
    "r": 0.45,
    "N": 0.60,
    "L": 0.70,
    "y": 0.50,
}

_PLOSIVE_TOKENS: set[str] = {"C", "c"}

# ═══════════════════════════════════════════════════════════════════════
# Neutral Maeda state
# ═══════════════════════════════════════════════════════════════════════
NEUTRAL_MAEDA: np.ndarray = np.array(
    [0.5, -0.5, -0.3, 0.0, 0.3, 0.0, -0.2], dtype=float
)

# ═══════════════════════════════════════════════════════════════════════
# Default pipeline parameters
# ═══════════════════════════════════════════════════════════════════════
# Article parameters (arXiv:2307.02299, §2.1):
#   "When K is large (K = 30 for vowel arcs and K = 10 for consonant
#    arcs), the trajectories become straight lines"
#   rho(t) = cos(theta(t)/2)  →  Pexp = 1
#   T = 16 (didactic mode per README)
DEFAULT_T: int = 16
DEFAULT_NU: int = -1
DEFAULT_K: int = 10        # consonant arc curvature (article: K=10)
DEFAULT_KVOY: int = 30     # vowel arc curvature (article: K=30)
DEFAULT_PEXP: int = 1      # velocity profile exponent (article: cos(th/2), Pexp=1)
DEFAULT_CEXP: float = 1.0
DEFAULT_CDEC: float = 3.0
DEFAULT_LONG_PAUSE_FACTOR: float = 25.0
DEFAULT_SHORT_PAUSE_FACTOR: float = 1.0
DEFAULT_VOWEL_FACTOR: float = 1.0
DEFAULT_CONSONANT_FACTOR: float = 1.0
DEFAULT_SIGMOID_STEEPNESS: float = 4.0

# ═══════════════════════════════════════════════════════════════════════
# Anchoring vowel coefficients (Berthommier 2023, arXiv:2307.02299)
# delta_o (δo): Vo onset – anticipation of vowel before the consonant
# delta_e (δe): Ve end – centralization of the coda vowel (schwa-like)
# When δo = δe → 1, Vo = Ve = V (full fusion into a single cluster)
# Default = COEFCEN (0.5) for backward compatibility.
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_DELTA_O: float = COEFCEN
DEFAULT_DELTA_E: float = COEFCEN
