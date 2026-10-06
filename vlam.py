# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
vlam — VLAM/Maeda articulatory speech synthesis + LPC (single file).

This single-file module gathers all the responsibilities of the VLAM
vocal synthesizer: configuration, vocal-tract geometry, acoustics,
glottal source, envelopes, frame-by-frame synthesis, and optional
visualization. It stems from a refactor of ``syntSYL.py`` but provides
**only** the modern typed API (``VlamState``, ``SynthConfig``,
``synthwordfen``, ``evaluate_frame``). The legacy wrappers
(``initVLAMLength``, ``vlam2009NN``, ``freqevalNN``, the previous
signature of ``synthwordfen``) were removed in version 3.0; if you
have old code to migrate, see the README for the old/new API mapping.

Dependencies
------------
- **Required**: numpy, scipy.
- **Optional** (only for visualization / audio playback): matplotlib,
  sounddevice, librosa. A visualization function called without its
  dependency raises an explicit ``ImportError`` instead of crashing
  abruptly.

Public API
----------

.. code-block:: python

    from vlam import VlamState, SynthConfig, synthwordfen

    state = VlamState.initial(vocal_tract_length_mm=195)
    result = synthwordfen(
        state=state,
        articulatory_params=Pval,    # (T, 7)
        word_tokens=["O", "V", "F"],
        envelope=envelope_externe,   # optional
    )
    # result.signal   : np.ndarray (N,), float64
    # result.formants : np.ndarray (T, 3)
    # result.spectra  : np.ndarray (T, n_freq)

Version : 3.2.0
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks, lfilter, medfilt
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Optional dependencies for visualization / audio playback.
# These flags are consulted by the visualization functions (at the end of
# the file) to raise a clear error message when the dependency is missing,
# instead of a raw ImportError at call time.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    import matplotlib.pyplot as plt  # noqa: F401  (re-exported for the functions)
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import librosa
    import librosa.display  # noqa: F401
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    import sounddevice as sd  # noqa: F401
    _HAS_SD = True
except ImportError:
    _HAS_SD = False

warnings.simplefilter("once", DeprecationWarning)


# ============================================================
# Start of content concatenated from the vlam/* submodules
# ============================================================

# ============================================================
# >>> config.py
# ============================================================


import os
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

# =============================================================================
# Physical constants of the air in the vocal tract
# (extracted from freqevalNN in the original code, lines 907-915)
# =============================================================================

#: Speed of sound in air (cm/s). Original value: 35100.
SOUND_SPEED_CM_S: float = 35100.0

#: Air density (g/cm^3). Original value: 1.14e-3.
AIR_DENSITY_G_CM3: float = 1.14e-3

#: Thermal conductivity of air (cal/(cm·s·K)). Original value: 5.5e-5.
THERMAL_CONDUCTIVITY: float = 5.5e-5

#: Ratio of specific heats of air (cp/cv). Original value: 1.4.
HEAT_CAPACITY_RATIO: float = 1.4

#: Dynamic viscosity of air (poise = g/(cm·s)). Original value: 1.86e-4.
AIR_VISCOSITY: float = 1.86e-4

#: Specific heat of air at constant pressure (cal/(g·K)). Value: 0.24.
SPECIFIC_HEAT_CP: float = 0.24

#: Wall-loss coefficient (resistance). Original value: 1600.
WALL_LOSS_BP: float = 1600.0

#: Wall-loss coefficient (mass). Original value: 1.4.
WALL_LOSS_MP: float = 1.4

#: Ordered array of physical constants, as expected by the
#: function :func:`vlam.acoustics.spectrelec`. The order MUST be preserved
#: for compatibility with the original code.
CONST_DAT: np.ndarray = np.array([
    SOUND_SPEED_CM_S,
    AIR_DENSITY_G_CM3,
    THERMAL_CONDUCTIVITY,
    HEAT_CAPACITY_RATIO,
    AIR_VISCOSITY,
    SPECIFIC_HEAT_CP,
    WALL_LOSS_BP,
    WALL_LOSS_MP,
])


# =============================================================================
# Default vocal-tract geometry
# (extracted from initVLAMLength and vlam2009NN)
# =============================================================================

#: Number of vocal-tract sections (Maeda mesh).
N_SECTIONS: int = 29

#: Number of total sagittal contour points (front + back).
N_SAGITTAL_POINTS: int = 58

#: Default vocal-tract length (mm). Original value: 195.
DEFAULT_VT_LENGTH_MM: float = 195.0

#: Number of Maeda articulatory parameters (jaw, tongue body, tongue
#: dorsum, tongue tip, lip protrusion, lip height, larynx height).
#: Canonical Maeda order: idx 4 = LP, idx 5 = LH.
N_ARTICULATORY_PARAMS: int = 7

#: Labels of the 7 articulatory parameters in the canonical Maeda order
#: (idx 4 = LP, idx 5 = LH). This is the order of the Pval .npz files
#:: and of the public API. The swap to the internal vlam2009NN order
#: (idx 4 = LH, idx 5 = LP) is performed in ``compute_vlam_geometry``.
ARTICULATORY_LABELS: Tuple[str, ...] = (
    "Jaw",           # J  — jaw opening
    "TongueBody",    # B  — tongue-body position (front/back)
    "TongueDorsum",  # D  — tongue-dorsum height
    "TongueTip",     # T  — tongue-tip position
    "LipProtrusion", # LP — lip protrusion/rounding (idx 4)
    "LipHeight",     # LH — vertical lip opening (idx 5)
    "LarynxHeight",  # Hy — larynx height
)


# =============================================================================
# Synthesizer configuration
# =============================================================================

@dataclass
class SynthConfig:
    """Configurable parameters of the vocal synthesizer.

    Attributes
    ----------
    sample_rate : int
        Sampling rate of the output signal, in Hz.
        Original value: 20000.
    frame_step : float
        Time step between two successive articulatory frames, in
        seconds. Corresponds to ``T_s`` in the original code (2 × 5e-3 = 10 ms).
    lpc_order : int
        Order of the LPC filter (number of reflection coefficients,
        excluding the trivial term a[0]=1). Original value: 30.
    f0_baseline : tuple of float
        Quadruplet (f0_1, f0_2, f0_3, f0_4) in Hz giving the fundamental
        melody at the temporal positions 1/3, 1/3, 2/3, 3/3 of the signal.
        Original value: (100, 130, 110, 90).
    f0_jitter_std : float
        Standard deviation of the Gaussian noise added to F0, expressed
        as a fraction of mean F0. Original value: 0.5 / 100 = 0.005.
    f0_jitter_seed : int
        Seed of the random generator for reproducibility of F0 jitter.
        Original value: 0.
    transition_step : float
        Transition duration (in seconds) between two successive sets of
        LPC coefficients in the lattice filter. Original value: 1e-3.
    radiation_pole : float
        Coefficient of the lip-radiation filter (high-pass single-pole).
        The applied filter is ``[1, -radiation_pole]``.
        Original value: 0.9375.
    normalize_peak : bool
        If True, normalizes the output signal by ``1.01 × max(|sig|)``.
        Disabled for debugging use cases.
    play_audio : bool
        If True, plays the signal via sounddevice at the end of synthesis.
        Disabled by default — use :func:`vlam.visualization.play_signal`
        for explicit control.
    median_filter_envelope : int
        Window size of the median filter applied to the external envelope
        (in samples). Set to 0 or 1 to disable.
        Original value: 7.
    """
    sample_rate: int = 20000
    frame_step: float = 2 * 5e-3
    lpc_order: int = 30
    f0_baseline: Tuple[float, ...] = (100.0, 130.0, 110.0, 90.0)
    f0_jitter_std: float = 0.5 / 100.0
    f0_jitter_seed: int = 0
    transition_step: float = 1e-3
    radiation_pole: float = 0.9375
    normalize_peak: bool = True
    play_audio: bool = False
    median_filter_envelope: int = 7

    def __post_init__(self) -> None:
        # Allows disabling audio via the environment variable
        # VLAM_NO_PLAY=1 — useful for batch runs and tests.
        if os.environ.get("VLAM_NO_PLAY", "") == "1":
            self.play_audio = False
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.frame_step <= 0:
            raise ValueError(f"frame_step must be > 0, got {self.frame_step}")
        if self.lpc_order < 1:
            raise ValueError(f"lpc_order must be >= 1, got {self.lpc_order}")
        if len(self.f0_baseline) != 4:
            raise ValueError(
                f"f0_baseline must have 4 elements (pattern at 1/3, 2/3, 3/3), "
                f"got {len(self.f0_baseline)}"
            )


# =============================================================================
# Articulatory state (equivalent of the `gui` dict in the original code)
# =============================================================================

@dataclass
class VlamState:
    """Articulatory state of the vocal tract, typed equivalent of the ``gui`` dict.

    The :attr:`params` field is an array of 11 values organized as follows:

    ===== ========== ==========================================
    Index Name        Description
    ===== ========== ==========================================
    0     Jaw        Jaw opening
    1     TongueBody Tongue-body front/back position
    2     TongueDorsum Tongue-dorsum height
    3     TongueTip  Tongue-tip position
    4     LipProt    Lip protrusion/rounding (LP)
    5     LipHeight  Vertical lip opening (LH)
    6     LarynxH    Larynx height
    7     k_age      Tract length (mm) — derives the age scale
    8     Flat_P     Palate flattening (anti-prototype)
    9     phi        Jaw rotation (degrees)
    10    Flat_T     Tongue flattening (anti-prototype)
    ===== ========== ==========================================

    .. note::
        Canonical Maeda order (v3.2): idx 4 = LP, idx 5 = LH. This is
        the order of the Pval .npz files and of the public API. The swap
        to the internal vlam2009NN order (idx 4 = LH, idx 5 = LP) is
        done in ``compute_vlam_geometry`` — this is the ONLY swap in
        the chain. No swap should be performed by the caller.

    Attributes
    ----------
    sagittal : np.ndarray
        Full sagittal contour, shape (58, 2). Updated by
        :func:`vlam.geometry.compute_vlam_geometry`.
    inci : np.ndarray
        Position of the upper incisor (labial reference point),
        shape (2,).
    area : np.ndarray
        Area function, shape (29, 2). Column 0: half-width of the
        tube (cm), column 1: cross-sectional area (cm²).
    params : np.ndarray
        Vector of 11 extended articulatory parameters (see table above).
    protrusion_distance : float
        Lip protrusion distance (cm). Formerly ``gui['PD']``.
    lip_height_cm : float
        Lip-opening height (cm). Formerly ``gui['LH']``.
    lip_height_index : float
        LH/PD ratio. Formerly ``gui['LHI']``.
    lip_opening_h : float
        Half-height of the lip opening. Formerly ``gui['B']``.
    lip_opening_w : float
        Half-width of the lip opening. Formerly ``gui['A']``.
    """
    sagittal: np.ndarray = field(
        default_factory=lambda: np.zeros((N_SAGITTAL_POINTS, 2))
    )
    inci: np.ndarray = field(default_factory=lambda: np.zeros((1, 2)))
    area: np.ndarray = field(default_factory=lambda: np.zeros((N_SECTIONS, 2)))
    params: np.ndarray = field(default_factory=lambda: np.zeros(11))
    protrusion_distance: float = 0.0
    lip_height_cm: float = 0.0
    lip_height_index: float = 0.0
    lip_opening_h: float = 0.0
    lip_opening_w: float = 0.0

    # --- Aliases for backward compatibility with the old dict API ---
    @classmethod
    def initial(cls, vocal_tract_length_mm: float = DEFAULT_VT_LENGTH_MM) -> "VlamState":
        """Creates a neutral articulatory state for a given tract length.

        Typed equivalent of ``initVLAMLength(np.zeros(7), L)`` in the original code.

        Parameters
        ----------
        vocal_tract_length_mm : float
            Vocal-tract length in millimeters. The value 195 corresponds
            to an average adult male speaker.

        Returns
        -------
        VlamState
            Neutral initial state (all articulatory parameters at 0).
        """
        # Computation of k_age (age scale) — see line 484 of the original code.
        k_age = 5.4749 * vocal_tract_length_mm - 407.1374
        params = np.zeros(11)
        params[7] = k_age
        return cls(params=params)

    def to_dict(self) -> dict:
        """Converts the state into a dict, for backward compatibility with the old ``gui``."""
        return {
            "sagittal": self.sagittal,
            "inci": self.inci,
            "area": self.area,
            "PD": self.protrusion_distance,
            "LH": self.lip_height_cm,
            "LHI": self.lip_height_index,
            "B": self.lip_opening_h,
            "A": self.lip_opening_w,
            "prm": self.params,
        }

    @classmethod
    def from_dict(cls, gui: dict) -> "VlamState":
        """Builds a VlamState from an old ``gui`` dict."""
        return cls(
            sagittal=np.asarray(gui.get("sagittal", np.zeros((N_SAGITTAL_POINTS, 2)))),
            inci=np.asarray(gui.get("inci", np.zeros((1, 2)))),
            area=np.asarray(gui.get("area", np.zeros((N_SECTIONS, 2)))),
            params=np.asarray(gui.get("prm", np.zeros(11))),
            protrusion_distance=float(gui.get("PD", 0.0)),
            lip_height_cm=float(gui.get("LH", 0.0)),
            lip_height_index=float(gui.get("LHI", 0.0)),
            lip_opening_h=float(gui.get("B", 0.0)),
            lip_opening_w=float(gui.get("A", 0.0)),
        )


# =============================================================================
# Constants of the acoustic analysis (vtn2frm_ftr_oral)
# =============================================================================

#: Number of frequency points for the evaluation of H(f).
N_FREQ_POINTS: int = 500

#: Maximum analyzed frequency (Hz).
F_MAX_HZ: float = 10000.0

#: Minimum analyzed frequency (Hz), derived from F_MAX and N_FREQ_POINTS.
F_MIN_HZ: float = F_MAX_HZ / (N_FREQ_POINTS - 1)

#: Initial frequency estimate for the formant search (Hz).
FORMANT_SEARCH_INIT_FE: float = 150.0

#: Initial bandwidth for the search (Hz).
FORMANT_SEARCH_INIT_BNP: float = 50.0

#: Search increment between two formants (Hz).
FORMANT_SEARCH_INCREMENT: float = 100.0

#: Maximum number of Newton-Raphson iterations.
FORMANT_NEWTON_MAX_ITER: int = 100

#: Convergence threshold for Newton-Raphson.
FORMANT_NEWTON_THRESHOLD: float = 0.3

#: Minimum spacing threshold between two formants (Hz).
FORMANT_MIN_SPACING_HZ: float = 10.0

#: Maximum number of formants searched (safety bound).
FORMANT_MAX_COUNT: int = 100

#: Number of formants effectively returned by freqevalNN.
N_OUTPUT_FORMANTS: int = 3


__all__ = [
    # Physical constants
    "SOUND_SPEED_CM_S", "AIR_DENSITY_G_CM3", "THERMAL_CONDUCTIVITY",
    "HEAT_CAPACITY_RATIO", "AIR_VISCOSITY", "SPECIFIC_HEAT_CP",
    "WALL_LOSS_BP", "WALL_LOSS_MP", "CONST_DAT",
    # Geometry
    "N_SECTIONS", "N_SAGITTAL_POINTS", "DEFAULT_VT_LENGTH_MM",
    "N_ARTICULATORY_PARAMS", "ARTICULATORY_LABELS",
    # Acoustic analysis
    "N_FREQ_POINTS", "F_MAX_HZ", "F_MIN_HZ",
    "FORMANT_SEARCH_INIT_FE", "FORMANT_SEARCH_INIT_BNP",
    "FORMANT_SEARCH_INCREMENT", "FORMANT_NEWTON_MAX_ITER",
    "FORMANT_NEWTON_THRESHOLD", "FORMANT_MIN_SPACING_HZ",
    "FORMANT_MAX_COUNT", "N_OUTPUT_FORMANTS",
    # Dataclasses
    "SynthConfig", "VlamState",
]

# ============================================================
# >>> geometry.py
# ============================================================


from typing import Tuple

import numpy as np

# =============================================================================
# Articulatory deformation tables (constants from vlam2009NN)
# =============================================================================
# The matrices below come from the GIPSA-Lab implementation of VLAM.
# They MUST NOT be modified: they encode the sagittal deformation of the
# tract as a function of the normalized articulatory parameters.

# Tongue deformation matrix (26 rows × 4 columns).
# Rows = sections 6..31 of the sagittal contour.
# Columns = combinations of the parameters (jaw, tongue body, tongue dorsum, tongue tip).
_A_TNG: np.ndarray = np.array([
    [1.000000,  0.000000,  0.000000,  0.000000],
    [-0.464047, 0.098776, -0.251690, 0.000000],
    [-0.328015, 0.337579, -0.283667, 0.000000],
    [-0.213039, 0.485565, -0.283533, 0.000000],
    [-0.302565, 0.705432, -0.379044, 0.000000],
    [-0.327806, 0.786897, -0.388116, 0.000000],
    [-0.325065, 0.852409, -0.285125, 0.000000],
    [-0.325739, 0.904725, -0.142602, 0.000000],
    [-0.313741, 0.926339,  0.021042, 0.000000],
    [-0.288138, 0.924019,  0.131949, 0.000000],
    [-0.249008, 0.909585,  0.250320, 0.000000],
    [-0.196936, 0.882236,  0.369083, 0.000000],
    [-0.128884, 0.830243,  0.499894, 0.000000],
    [-0.040825, 0.730520,  0.651662, 0.112048],
    [0.073420,  0.543080,  0.807947, 0.126204],
    [0.202726,  0.230555,  0.919065, 0.163735],
    [0.298853, -0.162541, 0.899074, 0.213884],
    [0.332785, -0.491647, 0.748869, 0.243163],
    [0.349955, -0.681313, 0.567615, 0.245295],
    [0.377277, -0.771200, 0.410502, 0.249425],
    [0.422713, -0.804874, 0.270513, 0.274015],
    [0.474635, -0.797704, 0.129324, 0.314454],
    [0.526087, -0.746938, -0.026201, 0.366149],
    [0.549466, -0.643572, -0.190005, 0.422848],
    [0.494200, -0.504012, -0.350434, 0.488056],
    [0.448797, -0.417352, -0.445410, 0.500909],
])

# Sensitivity (slope) of the tongue deformation per section.
_S_TNG: np.ndarray = np.array([
    27.674635, 29.947931, 44.694466, 99.310226, 96.871323,
    84.140404, 78.357513, 73.387718, 72.926758, 71.453232,
    69.288765, 66.615509, 63.603722, 59.964859, 56.695446,
    56.415058, 62.016468, 73.235176, 84.008438, 91.488312,
    94.124176, 95.246323, 93.516365, 93.000343, 100.934669,
    106.512482,
])

# Rest position of the tongue per section.
_U_TNG: np.ndarray = np.array([
    104.271675, 443.988434, 450.481689, 399.942200, 348.603088,
    351.181122, 365.404633, 370.290955, 356.202301, 341.890167,
    332.117523, 326.826599, 326.512512, 331.631989, 343.175323,
    361.265900, 385.231201, 411.826599, 435.691711, 455.040466,
    462.736023, 453.025055, 432.250488, 407.358368, 384.551056,
    363.836212,
])

# Lip deformation matrix (4 rows × 3 columns).
_A_LIP: np.ndarray = np.array([
    [1.000000,  0.000000, 0.000000],
    [0.178244, -0.395733, 0.888897],
    [-0.154638, 0.987971, 0.000000],
    [-0.217332, 0.825187, -0.303429],
])

_S_LIP: np.ndarray = np.array([27.674635, 33.068081, 99.392258, 213.996170])
_U_LIP: np.ndarray = np.array([104.271675, 122.812141, 135.938339, 460.440857])

# Larynx deformation matrix (5 rows × 2 columns).
_A_LRX: np.ndarray = np.array([
    [1.000000,  0.000000],
    [-0.208338, 0.262446],
    [0.127814,  0.991798],
    [-0.131840, 0.300784],
    [0.097688,  0.934267],
])

_S_LRX: np.ndarray = np.array([27.674635, 41.593315, 65.562340, 44.372742, 66.147499])
_U_LRX: np.ndarray = np.array([104.271675, 143.138733, -948.229309, 404.678223, -962.936401])

# Rest position of the wall (24 values, sections 7..30).
_U_WAL: np.ndarray = np.array([
    550.196533, 604.878601, 674.127197, 678.776489, 665.905579,
    653.312134, 643.223511, 633.836243, 636.994202, 668.834290,
    703.098267, 600, 610, 610, 605, 600, 600, 600, 600, 600,
    600, 600, 600, 600, 600,
])

# =============================================================================
# Constant geometric parameters (from vlam2009NN, lines 566-595)
# =============================================================================

# Origin of the sagittal frame (palate pivot point).
_IX0: float = 3000.0
_IY0: float = 1850.0

# Conversion from VP mesh (visu-phon) -> cm.
_VP_MAP: float = 1.0 / 29.5

# Palatal arc length (in VP units, before normalization).
_TEK_VT: float = 188.679245 * _VP_MAP

# Position of the incisor (VP frame).
_INCI_X: float = (2212.354492 - _IX0) / _TEK_VT
_INCI_Y: float = (1999.574219 - _IY0) / _TEK_VT
_INCI_LIP: float = 0.8
_INCI_LIP_VP: float = _INCI_LIP / _VP_MAP

# Origin of the working frame (different from IX0/IY0!).
_OX: float = 2200.0
_OY: float = 2000.0

# Palatal radius and half-section length.
_R: float = 5.0
_DL: float = 0.5

# Number of sections per tract zone.
_M1: int = 14  # pharyngeal zone (straight segments)
_M2: int = 11  # palatal zone (arc)
_M3: int = 6   # dental zone (straight segments)

# Rotation angles of the pharyngeal and dental zones (degrees).
_OMEGA_DEG: float = -11.25
_THETA_DEG: float = 11.25

# Correction coefficients of the area function (alpha, beta per section).
_ALPHA: np.ndarray = np.array([
    1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8,
    1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7,
    1.8, 1.8, 1.9, 2.0, 2.6,
])
_BETA: np.ndarray = np.array([
    1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
    1.3, 1.4, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
])

# Anti-prototype correction constant (historical VLAM 2009 values).
_PHARYNX_COEF_A: float = 0.8
_MOUTH_COEF_A: float = 0.35

# Scaling constant of the mid palatal zone (applied to zone 33:41).
_PALATAL_SMOOTHING_K: float = 3.0 / 5.0


# =============================================================================
# Precomputed tables (to avoid recomputing at each call)
# =============================================================================

def _precompute_static_geometry() -> np.ndarray:
    """Precomputes the mesh unit vectors (vtos).

    Note: the `igd` and `egd` points CANNOT be precomputed because they
    depend on ``pharynx_scale`` and ``mouth_scale`` which vary with age.
    They are computed dynamically in :func:`_build_sagittal_contour`.

    Returns
    -------
    vtos : np.ndarray
        Unit vectors pointing from the inner to the outer side, shape (31, 2).
    """
    r_vp = _R / _VP_MAP
    ome = np.pi * _OMEGA_DEG / 180.0
    the = np.pi * _THETA_DEG / 180.0

    # Pharyngeal zone: 14 segments
    dx_i = (_DL * np.cos(ome - np.pi / 2.0))
    dy_i = (_DL * np.sin(ome - np.pi / 2.0))
    dx_e = r_vp * np.cos(ome)
    dy_e = r_vp * np.sin(ome)
    igd1 = np.array([
        (dx_i * np.arange(_M1 - 1, -1, -1)) + _OX,
        (dy_i * np.arange(_M1 - 1, -1, -1)) + _OY,
    ]).T
    egd1 = np.array([dx_e + igd1[:, 0], dy_e + igd1[:, 1]]).T

    # Palatal zone: 11 segments along an arc
    gam = the * np.arange(1, _M2 + 1) + ome
    igd2 = np.array([_OX * np.ones(11), _OY * np.ones(11)]).T
    egd2 = np.array([r_vp * np.cos(gam) + _OX, r_vp * np.sin(gam) + _OY]).T

    # Dental zone: 6 segments
    dx_i = (_DL * np.cos(gam[-1] + np.pi / 2.0))
    dy_i = (_DL * np.sin(gam[-1] + np.pi / 2.0))
    dx_e = r_vp * np.cos(gam[-1])
    dy_e = r_vp * np.sin(gam[-1])
    igd3 = np.array([
        (dx_i * np.arange(1, _M3 + 1)) + _OX,
        (dy_i * np.arange(1, _M3 + 1)) + _OY,
    ]).T
    egd3 = np.array([dx_e + igd3[:, 0], dy_e + igd3[:, 1]]).T

    igd = np.vstack((igd1, igd2, igd3))
    egd = np.vstack((egd1, egd2, egd3))

    # Unit vectors inner -> outer (normals to the tract)
    p = egd[:, 0] - igd[:, 0]
    q = egd[:, 1] - igd[:, 1]
    s = np.sqrt(p * p + q * q)
    vtos = np.vstack((p / s, q / s)).T

    # We also keep gam (used for the dynamic computation of igd/egd)
    return vtos, gam


# Single computation at module import
_VTOS, _GAM = _precompute_static_geometry()


def _compute_mesh_points(
    pharynx_scale: float, mouth_scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the inner (igd) and outer (egd) mesh points.

    These points depend on pharynx_scale and mouth_scale, which vary with
    the speaker's age (via k_age). They must therefore be recomputed at
    each call of :func:`compute_vlam_geometry`.

    Parameters
    ----------
    pharynx_scale : float
        Pharyngeal scale (typically between 0.3 and 1.5).
    mouth_scale : float
        Oral scale (typically between 0.65 and 1.5).

    Returns
    -------
    igd : np.ndarray
        Inner mesh points, shape (31, 2).
    egd : np.ndarray
        Outer mesh points, shape (31, 2).
    """
    r_vp = _R / _VP_MAP
    ome = np.pi * _OMEGA_DEG / 180.0
    dl_pharynx_vp = pharynx_scale * _DL / _VP_MAP
    dl_palatal_vp = mouth_scale * _DL / _VP_MAP

    # Pharyngeal zone: uses dl_pharynx_vp
    dx_i = dl_pharynx_vp * np.cos(ome - np.pi / 2.0)
    dy_i = dl_pharynx_vp * np.sin(ome - np.pi / 2.0)
    dx_e = r_vp * np.cos(ome)
    dy_e = r_vp * np.sin(ome)
    igd1 = np.array([
        (dx_i * np.arange(_M1 - 1, -1, -1)) + _OX,
        (dy_i * np.arange(_M1 - 1, -1, -1)) + _OY,
    ]).T
    egd1 = np.array([dx_e + igd1[:, 0], dy_e + igd1[:, 1]]).T

    # Palatal zone: circular arc centered on (OX, OY)
    gam = _GAM  # precomputed angles
    igd2 = np.array([_OX * np.ones(11), _OY * np.ones(11)]).T
    egd2 = np.array([r_vp * np.cos(gam) + _OX, r_vp * np.sin(gam) + _OY]).T

    # Dental zone: uses dl_palatal_vp
    dx_i = dl_palatal_vp * np.cos(gam[-1] + np.pi / 2.0)
    dy_i = dl_palatal_vp * np.sin(gam[-1] + np.pi / 2.0)
    dx_e = r_vp * np.cos(gam[-1])
    dy_e = r_vp * np.sin(gam[-1])
    igd3 = np.array([
        (dx_i * np.arange(1, _M3 + 1)) + _OX,
        (dy_i * np.arange(1, _M3 + 1)) + _OY,
    ]).T
    egd3 = np.array([dx_e + igd3[:, 0], dy_e + igd3[:, 1]]).T

    return np.vstack((igd1, igd2, igd3)), np.vstack((egd1, egd2, egd3))

# Precomputed normalization of the deformation tables
_S_TNG_NORM = _S_TNG / _TEK_VT
_U_TNG_NORM = _U_TNG / _TEK_VT
_S_LIP_NORM = _S_LIP / _TEK_VT
_U_LIP_NORM = _U_LIP / _TEK_VT
_S_LRX_NORM = _S_LRX / _TEK_VT
_U_LRX_NORM = _U_LRX / _TEK_VT
_U_WAL_NORM = _U_WAL / _TEK_VT


# =============================================================================
# Main functions
# =============================================================================

def compute_vlam_geometry(state: VlamState) -> VlamState:
    """Computes the sagittal contour and area function from the state.

    This function is the refactored, typed equivalent of ``vlam2009NN`` in
    the original code. It does not modify ``state``: it returns a new
    :class:`VlamState` object whose ``sagittal``, ``inci``, ``area``
    fields and labial descriptors are updated.

    Parameters
    ----------
    state : VlamState
        Current articulatory state. The ``params`` field must contain 11
        values (see :class:`VlamState`).

    Returns
    -------
    VlamState
        New state with computed sagittal contour and area function.

    Notes
    -----
    The algorithm proceeds in 4 steps:

    1. **Articular deformation**: from the jaw, tongue, lips, larynx
       parameters, the positions of the tongue, lips, and larynx are
       computed in the sagittal frame, using the matrices ``_A_TNG``,
       ``_A_LIP``, ``_A_LRX`` and the sensitivity vectors.
    2. **Sagittal contour construction**: assembly of the inner and outer
       points (58 points in total), then optional application of the
       anti-prototypes ``Flat_P`` (palate) and ``Flat_T`` (tongue) and
       of the ``phi`` rotation.
    3. **Area function computation**: for each of the 27 useful sections,
       the area is computed by the Heinz (1974) formula:
       ``A = 1.4 * alpha * (w ** beta)`` where ``w`` is the effective
       half-width of the tube, and ``alpha``/``beta`` are corrective
       coefficients per section.
    4. **Labial section**: sections 27 and 28 (lip opening) are computed
       separately as an ellipse of area ``pi * lip_h * lip_w``.
    """
    # Defensive copy — we do not mutate the input state
    new_state = VlamState(
        sagittal=state.sagittal.copy(),
        inci=state.inci.copy(),
        area=state.area.copy(),
        params=state.params.copy(),
        protrusion_distance=state.protrusion_distance,
        lip_height_cm=state.lip_height_cm,
        lip_height_index=state.lip_height_index,
        lip_opening_h=state.lip_opening_h,
        lip_opening_w=state.lip_opening_w,
    )

    prm = new_state.params
    k_age = prm[7]
    k_age_max = 650.0

    # Reordering of articulatory parameters for VLAM2009.
    #
    # CONVENTION v3.2: the Pval .npz files follow the canonical Maeda
    # order [J, B, D, T, LP, LH, Hy] (idx 4 = LP, idx 5 = LH).
    # But vlam2009NN internally uses the order [J, B, D, T, LH, LP, Hy]
    # (params[4] = LH, params[5] = LP), because the _A_LIP matrices
    # were designed with col 1 = LH, col 2 = LP.
    #
    # The swap below (p[4] = prm[5]; p[5] = prm[4]) converts the canonical
    # order to the internal vlam2009NN order. This is the ONLY swap in the
    # chain — there must NOT be a mirror swap in batch_synthesize (the
    # batch swap that existed in v2/v3.0 was a bug that canceled this one
    # and caused the matrices to receive [jaw, LP, LH] instead of
    # [jaw, LH, LP]).
    p = np.zeros(10)
    p[0] = prm[0]  # Jaw
    p[1] = prm[1]  # Tongue body
    p[2] = prm[2]  # Tongue dorsum
    p[3] = prm[3]  # Tongue tip
    p[4] = prm[5]  # Lip height (LH) — swapped from canonical idx 5
    p[5] = prm[4]  # Lip protrusion (LP) — swapped from canonical idx 4
    p[6] = prm[6]  # Larynx height
    p[7] = prm[7]  # k_age
    p[8] = _PHARYNX_COEF_A
    p[9] = _MOUTH_COEF_A

    flat_p = -prm[8]
    flat_t = -prm[10]
    phi = prm[9]

    # Pharyngeal and oral scales as a function of age
    pharynx_scale = k_age * p[8] / k_age_max + 0.3
    mouth_scale = k_age * p[9] / k_age_max + 0.65

    # ---- Step 1: articular deformations ----
    v_tng = _S_TNG_NORM * (_A_TNG @ p[:4]) + _U_TNG_NORM
    v_lip = _S_LIP_NORM * (_A_LIP @ np.array([p[0], p[4], p[5]])) + _U_LIP_NORM
    v_lip = np.maximum(v_lip, 0.0)  # no negative coordinates for the lips
    v_lrx = _S_LRX_NORM * (_A_LRX @ np.array([p[0], p[6]])) + _U_LRX_NORM

    # ---- Step 2: assembly of the sagittal contour ----
    sagittal = _build_sagittal_contour(
        v_lrx=v_lrx, v_tng=v_tng, pharynx_scale=pharynx_scale,
        mouth_scale=mouth_scale, v_lip=v_lip,
    )

    # Optional application of the anti-prototypes (flattened palate / tongue)
    if flat_p != 0 or flat_t != 0:
        sagittal = _apply_anti_prototypes(sagittal, flat_p, flat_t)

    # Optional jaw rotation (parameter phi)
    if phi != 0:
        sagittal = _apply_jaw_rotation(sagittal, phi, sagittal[41, :])

    new_state.sagittal = sagittal
    new_state.inci = sagittal[55, :]

    # ---- Step 3: area function computation ----
    af, lip_h, lip_w = _compute_area_function(
        sagittal, pharynx_scale, mouth_scale, v_lip
    )

    new_state.area = af
    new_state.protrusion_distance = float(
        np.sqrt((sagittal[41, 0] - sagittal[55, 0]) ** 2 +
                (sagittal[41, 1] - sagittal[55, 1]) ** 2)
    )
    # LH: distance between section 29/30 (junction) and section 42 (lips)
    x29_30 = (sagittal[28, 0] + sagittal[29, 0]) / 2.0
    y29_30 = (sagittal[28, 1] + sagittal[29, 1]) / 2.0
    new_state.lip_height_cm = float(
        np.sqrt((x29_30 - sagittal[41, 0]) ** 2 +
                (y29_30 - sagittal[41, 1]) ** 2)
    )
    new_state.lip_height_index = (
        new_state.lip_height_cm / new_state.protrusion_distance
        if new_state.protrusion_distance > 0 else 0.0
    )
    new_state.lip_opening_h = lip_h * _VP_MAP
    new_state.lip_opening_w = lip_w * _VP_MAP

    return new_state


def _build_sagittal_contour(
    v_lrx: np.ndarray,
    v_tng: np.ndarray,
    pharynx_scale: float,
    mouth_scale: float,
    v_lip: np.ndarray,
) -> np.ndarray:
    """Builds the full sagittal contour (58 points) from the computed
    articular deformations.

    Refactored equivalent of lines 637-704 of the original code.
    """
    omg = np.pi * _OMEGA_DEG / 180.0
    r_vp = _R / _VP_MAP

    # Laryngeal pivot (inner + outer) with pharyngeal scaling.
    # IMPORTANT: ivt1 uses v_lrx[1:3] (larynx insertion point),
    # evt1 uses v_lrx[3:5] (larynx emergence point) — they are
    # different in the original code.
    ivt1 = _scale_larynx_point(v_lrx[1], v_lrx[2], pharynx_scale, omg)
    evt1 = _scale_larynx_point(v_lrx[3], v_lrx[4], pharynx_scale, omg)

    # Dynamic mesh: igd and evt depend on pharynx_scale and mouth_scale
    igd, egd = _compute_mesh_points(pharynx_scale, mouth_scale)

    # Construction of the per-section scales (uniform pharyngeal zone,
    # graded palatal zone, uniform dental zone)
    sf1 = pharynx_scale * np.ones((8, 1))
    sf2 = ((mouth_scale - pharynx_scale) *
           np.arange(1, _M2 + 1).reshape(-1, 1) / _M2 + pharynx_scale)
    sf3 = mouth_scale * np.ones((6, 1))
    scale_factor = np.vstack((sf1, sf2, sf3)).flatten()

    # Tongue contour: min(v_tng, u_wal) then per-section scaling
    v = scale_factor * np.minimum(v_tng[1:26], _U_WAL_NORM)
    xy1 = np.column_stack((_VTOS[6:31, 0] * v, _VTOS[6:31, 1] * v)) + igd[6:31, :]

    # Reference wall contour (without deformation)
    vtos_wal = np.column_stack((_VTOS[6:31, 0] * _U_WAL_NORM,
                                _VTOS[6:31, 1] * _U_WAL_NORM))
    xy2 = np.column_stack((vtos_wal[:, 0] * scale_factor,
                           vtos_wal[:, 1] * scale_factor)) + igd[6:31, :]

    ivt3 = xy1
    evt3 = xy2
    ivt4 = np.vstack((ivt1, (ivt1 + xy1[0, :]) / 2.0, ivt3))
    evt4 = np.vstack((evt1, (evt1 + xy2[0, :]) / 2.0, evt3))

    # Labial points (incisor + opening)
    omg = np.pi * _THETA_DEG / 180.0
    x1, y1 = _INCI_X, _INCI_Y + _INCI_LIP_VP
    x0 = (y1 - np.tan(omg + np.pi / 2.0) * x1) / (
        np.tan(omg) - np.tan(omg + np.pi / 2.0))
    y0 = np.tan(omg) * x0
    x1 = mouth_scale * (x1 - x0) + x0
    y1 = mouth_scale * (y1 - y0) + y0
    b = y1 - np.tan(omg) * x1
    x0 = b / (np.tan(omg + np.pi / 2.0) - np.tan(omg))
    y0 = np.tan(omg + np.pi / 2.0) * x0

    evtn = np.zeros((2, 2))
    evtn[0, 0] = mouth_scale * (x1 - x0) + x0 + _OX
    evtn[0, 1] = mouth_scale * (y1 - y0) + y0 + _OY
    ivtn = np.zeros((2, 2))
    ivtn[0, 0] = evtn[0, 0]
    ivtn[0, 1] = evtn[0, 1] - mouth_scale * v_lip[2]
    evtn[1, 0] = evtn[0, 0] - mouth_scale * v_lip[1]
    evtn[1, 1] = evtn[0, 1]
    ivtn[1, 0] = evtn[1, 0]
    ivtn[1, 1] = ivtn[0, 1]

    ivt = np.vstack((ivt4, ivtn))
    evt = np.vstack((evt4, evtn))
    sagittal = np.vstack((np.flipud(ivt), evt))

    # Smoothing of the mid palatal zone (sections 33..40)
    sagittal = _smooth_palatal_zone(sagittal)

    return sagittal


def _scale_larynx_point(
    x1: float, y1: float, scale: float, omg: float
) -> np.ndarray:
    """Applies the pharyngeal scaling to a larynx point.

    Computes the intersection pivot between the line passing through (x1, y1)
    at angle ``omg`` and its perpendicular, then applies the homothety of
    center the pivot and ratio ``scale``.

    Returns
    -------
    np.ndarray
        Scaled point, shape (1, 2).
    """
    b = y1 - np.tan(omg + np.pi / 2.0) * x1
    x0 = b / (np.tan(omg) - np.tan(omg + np.pi / 2.0))
    y0 = np.tan(omg) * x0
    x1 = scale * (x1 - x0) + x0
    y1 = scale * (y1 - y0) + y0
    b = y1 - np.tan(omg) * x1
    x0 = b / (np.tan(omg + np.pi / 2.0) - np.tan(omg))
    y0 = np.tan(omg + np.pi / 2.0) * x0
    return np.array([[scale * (x1 - x0) + x0 + _OX,
                      scale * (y1 - y0) + y0 + _OY]])


def _smooth_palatal_zone(sagittal: np.ndarray) -> np.ndarray:
    """Applies the smoothing of the mid palatal zone (sections 33..40).

    This zone corresponds to the transition between the hard palate and the
    soft palate. The original code applies a rotation + projection to
    flatten geometric irregularities.
    """
    s = sagittal
    x42, y42 = s[41, 0], s[41, 1]
    x34, y34 = s[33, 0], s[33, 1]

    # Computation of the line passing through (x34, y34) and (x42, y42)
    ab = np.linalg.solve(np.array([[x42, 1], [x34, 1]]),
                         np.array([y42, y34]))

    sagmiddle = s[33:41, :]
    alpha = -np.arctan((y42 - y34) / (x42 - x34))
    centre = np.array([0, ab[1]])

    # Rotation of the mid palate toward the smoothing frame
    sag_p0 = sagmiddle - centre
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    sag_p = R.dot(sag_p0.T).T + centre

    # Flatten y -> ab[1]
    sag_p = np.hstack((sag_p[:, 0].reshape(-1, 1),
                       ab[1] * np.ones((8, 1))))

    # Inverse rotation
    alpha = -alpha
    sag_p0 = sag_p - centre
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    sag_p = R.dot(sag_p0.T).T + centre

    # Application of the correction vector with coefficient 3/5
    mat_vect = np.column_stack((sag_p[:, 0] - sagmiddle[:, 0],
                                sag_p[:, 1] - sagmiddle[:, 1]))
    sagmiddle = sagmiddle + _PALATAL_SMOOTHING_K * mat_vect

    return np.vstack((s[:33, :], sagmiddle, s[41:, :]))


def _apply_anti_prototypes(
    sagittal: np.ndarray, flat_p: float, flat_t: float
) -> np.ndarray:
    """Applies the anti-prototypes (palate / tongue flattening).

    Refactored equivalent of lines 742-771 of the original code.
    """
    x50, y50 = sagittal[49, 0], sagittal[49, 1]
    x9, y9 = sagittal[8, 0], sagittal[8, 1]
    x44, y44 = sagittal[43, 0], sagittal[43, 1]
    x15, y15 = sagittal[14, 0], sagittal[14, 1]

    # Intersection of the two lines (palate + tongue)
    Y1 = np.array([y44, y15])
    X1 = np.array([[x44, 1], [x15, 1]])
    A1 = np.linalg.lstsq(X1, Y1, rcond=None)[0]
    Y2 = np.array([y50, y9])
    X2 = np.array([[x50, 1], [x9, 1]])
    A2 = np.linalg.lstsq(X2, Y2, rcond=None)[0]
    A = np.array([[-A1[0], 1], [-A2[0], 1]])
    B = np.array([[A1[1]], [A2[1]]])
    intersec = np.linalg.lstsq(A, B, rcond=None)[0]

    if flat_p != 0:
        eps = np.arange(0, np.pi, np.pi / 14)
        sagpalais = sagittal[41:56, :].copy()
        vect_AI = np.column_stack((intersec[0] - sagpalais[:, 0],
                                   intersec[1] - sagpalais[:, 1]))
        sagpalais = sagpalais + flat_p * np.column_stack(
            (np.sin(eps) * vect_AI[:, 0], np.sin(eps) * vect_AI[:, 1])
        )
        sagittal = np.vstack((sagittal[:41, :], sagpalais, sagittal[56:, :]))

    if flat_t != 0:
        eps = np.arange(0, np.pi, np.pi / 15)
        saglangue = sagittal[2:18, :].copy()
        vect_AI = np.column_stack((intersec[0] - saglangue[:, 0],
                                   intersec[1] - saglangue[:, 1]))
        saglangue = saglangue + flat_t * np.column_stack(
            (np.sin(eps) * vect_AI[:, 0], np.sin(eps) * vect_AI[:, 1])
        )
        sagittal = np.vstack((sagittal[:2, :], saglangue, sagittal[18:, :]))

    return sagittal


def _apply_jaw_rotation(
    sagittal: np.ndarray, phi_deg: float, centre: np.ndarray
) -> np.ndarray:
    """Applies the jaw rotation (parameter phi) to the pharyngeal and
    dental zones, leaving the palatal zone untouched.

    Refactored equivalent of lines 772-785 of the original code.
    """
    phi = phi_deg * np.pi / 180.0
    supp = np.array([50.0, 50.0])
    cphi = centre + supp

    sagintrot = sagittal[:17, :].copy()
    sagextrot = sagittal[41:, :].copy()

    cos_p, sin_p = np.cos(phi), np.sin(phi)
    R = np.array([[cos_p, -sin_p], [sin_p, cos_p]])

    sagintrot = (R.dot((sagintrot - cphi).T)).T + cphi
    sagextrot = (R.dot((sagextrot - cphi).T)).T + cphi

    return np.vstack((sagintrot, sagittal[17:41, :], sagextrot))


def _compute_area_function(
    sagittal: np.ndarray,
    pharynx_scale: float,
    mouth_scale: float,
    v_lip: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Computes the area function (29 sections) from the sagittal contour.

    Refactored equivalent of lines 792-826 of the original code.

    Parameters
    ----------
    sagittal : np.ndarray
        Sagittal contour, shape (58, 2).
    pharynx_scale : float
        Pharyngeal scale (not used here but kept for API consistency).
    mouth_scale : float
        Oral scale, used for the labial sections 27 and 28.
    v_lip : np.ndarray
        Labial deformation vector, shape (4,). ``v_lip[2]`` is the
        half-height, ``v_lip[3]`` is the half-width.

    Returns
    -------
    area : np.ndarray
        Array of shape (29, 2) — column 0: half-width of the tube (cm),
        column 1: cross-sectional area (cm²).
    lip_h : float
        Half-height of the lip opening (VP units, unnormalized).
    lip_w : float
        Half-width of the lip opening (VP units, unnormalized).
    """
    # Computation of the labial parameters (exact formula from the original code, L699-700)
    lip_h = mouth_scale * v_lip[2] / 2.0
    lip_w = mouth_scale * v_lip[3] / 2.0

    # Area computation for the first 27 sections (pharyngeal + palatal zones)
    NP = N_SECTIONS
    ivt = np.flipud(sagittal[0:29, :])
    evt = sagittal[29:58, :]

    # Roll by 28 = -1: shifts each point to form quadrilaterals
    nb_permut = len(ivt) - 1
    ivt2 = np.roll(ivt, nb_permut, axis=0)
    evt2 = np.roll(evt, nb_permut, axis=0)

    # Distances between points for area computation by quadrilateral
    p = np.sqrt((ivt[:, 0] - ivt2[:, 0])**2 + (ivt[:, 1] - ivt2[:, 1])**2)
    q = np.sqrt((evt[:, 0] - evt2[:, 0])**2 + (evt[:, 1] - evt2[:, 1])**2)
    r = np.sqrt((evt[:, 0] - ivt[:, 0])**2 + (evt[:, 1] - ivt[:, 1])**2)
    s = np.sqrt((ivt2[:, 0] - evt2[:, 0])**2 + (ivt2[:, 1] - evt2[:, 1])**2)
    t = np.sqrt((ivt[:, 0] - evt2[:, 0])**2 + (ivt[:, 1] - evt2[:, 1])**2)

    # We keep only the first 27 sections (sections 27 and 28
    # are computed separately, see below).
    p, q, r, s, t = p[:27], q[:27], r[:27], s[:27], t[:27]

    # Area via Heron's formula for two triangles
    a1 = 0.5 * (p + s + t)
    a2 = 0.5 * (q + r + t)
    s1 = np.sqrt(np.maximum(a1 * (a1 - p) * (a1 - s) * (a1 - t), 0.0))
    s2 = np.sqrt(np.maximum(a2 * (a2 - q) * (a2 - r) * (a2 - t), 0.0))

    x1 = ivt2[:27, 0] + evt2[:27, 0] - ivt[:27, 0] - evt[:27, 0]
    y1 = ivt2[:27, 1] + evt2[:27, 1] - ivt[:27, 1] - evt[:27, 1]
    d = 0.5 * np.sqrt(x1**2 + y1**2)

    # Area correction coefficient (historical VLAM value)
    c = _VP_MAP
    cc = c * c
    w = c * (s1 + s2) / np.maximum(d, 1e-12)  # protection against d=0

    af = np.zeros((N_SECTIONS, 2))
    af[:27, 0] = c * d
    # Heinz formula: A = 1.4 * alpha * (w ** beta)
    af[:27, 1] = 1.4 * _ALPHA * (w ** _BETA)

    # Labial section (27, 28): elliptical opening
    af[27, 0] = (ivt[NP - 2, 0] - ivt[NP - 1, 0]) * c
    af[27, 1] = np.pi * lip_h * lip_w * cc
    af[28, 0] = af[27, 0]
    af[28, 1] = af[27, 1]

    # Protection against negative lengths (can occur in case of
    # self-intersection of the contour)
    tmp = np.where(af[:, 0] <= 0)[0]
    af[tmp, 0] = 0.01

    return af, lip_h, lip_w


__all__ = ["compute_vlam_geometry"]

# ============================================================
# >>> acoustics.py
# ============================================================


from typing import Tuple

import numpy as np
from scipy.signal import find_peaks

# =============================================================================
# Transmission line (T-network)
# =============================================================================

def spectrelec(
    w: np.ndarray,
    area: np.ndarray,
    zr: np.ndarray,
    lengths: np.ndarray,
    no_vibration: int,
    const_dat: np.ndarray,
) -> np.ndarray:
    """Computes the transfer function H(f) of a segmented tract.

    Discrete transmission-line model with visco-thermal losses and
    yielding walls. Each section is modeled by a symmetrical T two-port.

    Parameters
    ----------
    w : np.ndarray
        Angular frequencies (rad/s), shape (n_freq,).
    area : np.ndarray
        Cross-sectional areas per section (cm²), shape (n_sections, 1).
    zr : np.ndarray
        Radiation impedance at the lips, shape (n_freq,).
    lengths : np.ndarray
        Lengths of each section (cm), shape (n_sections, 1).
    no_vibration : int
        If 1, disables wall vibration (YP = 0).
    const_dat : np.ndarray
        Array of the 8 physical constants (see :data:`vlam.config.CONST_DAT`).

    Returns
    -------
    np.ndarray
        H(f), complex array of shape (n_freq,).
    """
    c, ro = const_dat[0], const_dat[1]
    lambda_, eta = const_dat[2], const_dat[3]
    mu, cp, bp, mp = const_dat[4], const_dat[5], const_dat[6], const_dat[7]

    # Geometry of each section
    S = 2.0 * np.sqrt(area * np.pi)           # equivalent perimeter
    L = ro / area * lengths                    # acoustic inductance
    C = area * lengths / (ro * c * c)          # acoustic capacitance

    # Visco-thermal losses (skin effect)
    R_coef = np.sqrt(ro * mu / (2.0 * w))
    G_coef = (eta - 1) / (ro * c**2) * np.sqrt(lambda_ * w / (2.0 * cp * ro))
    R = S * lengths / (area**2) * R_coef
    G = S * lengths * G_coef

    # Vibrating wall admittance
    YP_coef = 1.0 / (bp**2 + mp**2 * w**2)
    YP = S * lengths * ((bp - 1j * mp * w) * YP_coef)
    if no_vibration == 1:
        YP = np.zeros_like(YP)

    # Impedance Z and admittance Y per section (shape (n_sections, n_freq))
    Z = R + 1j * L * w
    Y = G + 1j * C * w + YP

    # ABCD transfer matrices per section (symmetrical T two-port)
    aa = 1.0 + (Z * Y / 2.0)
    bb = -(Z + Z**2 * Y / 4.0)
    cc = -Y
    dd = aa  # symmetrical

    # Matrix product of the two-ports
    aaa = aa[0, :].copy()
    bbb = bb[0, :].copy()
    ccc = cc[0, :].copy()
    ddd = dd[0, :].copy()

    H = np.ones_like(aaa) / (aaa - ccc * zr)  # init: section 0 only
    for ind in range(len(area) - 1):
        proda = aa[ind + 1, :] * aaa + bb[ind + 1, :] * ccc
        prodb = aa[ind + 1, :] * bbb + bb[ind + 1, :] * ddd
        prodc = cc[ind + 1, :] * aaa + dd[ind + 1, :] * ccc
        prodd = cc[ind + 1, :] * bbb + dd[ind + 1, :] * ddd
        aaa, bbb, ccc, ddd = proda, prodb, prodc, prodd
        H = np.ones_like(aaa) / (aaa - ccc * zr)

    return H


# =============================================================================
# Lip radiation impedance
# =============================================================================

def lip_radiation_impedance(
    lip_area: float,
    freqs: np.ndarray,
    const_dat: np.ndarray = CONST_DAT,
) -> np.ndarray:
    """Computes the radiation impedance at the lips.

    Monopole radiation model with a resistive term proportional to w²
    and a reactive term proportional to w.

    Parameters
    ----------
    lip_area : float
        Lip-opening cross-section (cm²). Must be > 0.
    freqs : np.ndarray
        Frequencies (Hz), shape (n_freq,).
    const_dat : np.ndarray, optional
        Physical constants, defaults to :data:`vlam.config.CONST_DAT`.

    Returns
    -------
    np.ndarray
        Complex impedance Zr, shape (n_freq,).

    Raises
    ------
    ValueError
        If ``lip_area <= 0`` (would cause sqrt of a negative number).
    """
    if lip_area <= 0:
        raise ValueError(
            f"lip_area must be > 0 (got {lip_area}). "
            "Check the area function at the lips."
        )
    c, ro = const_dat[0], const_dat[1]
    w = 2.0 * np.pi * freqs
    # Resistive term (baffled piston) + reactive term (radiation mass)
    Zr = (ro / (2.0 * np.pi * c)) * (w**2) \
        + 1j * (8.0 * ro / (3.0 * np.pi * np.sqrt(np.pi * lip_area))) * w
    return Zr


# =============================================================================
# Transfer function of the oral tract (without formant correction)
# =============================================================================

def area_to_transfer_function(
    area: np.ndarray,
    n_freq: int = N_FREQ_POINTS,
    f_max: float = F_MAX_HZ,
    f_min: float = F_MIN_HZ,
    no_vibration: int = 0,
    const_dat: np.ndarray = CONST_DAT,
) -> np.ndarray:
    """Computes the transfer function H(f) of the oral tract.

    Refactored equivalent of ``aire2spectre_oral``.

    Parameters
    ----------
    area : np.ndarray
        Area function, shape (29, 2) — column 0: lengths (cm),
        column 1: cross-sections (cm²).
    n_freq : int
        Number of frequency points, default 500.
    f_max : float
        Maximum frequency (Hz), default 10000.
    f_min : float
        Minimum frequency (Hz), default derived from f_max and n_freq.
    no_vibration : int
        If 1, rigid walls (no vibration).
    const_dat : np.ndarray
        Physical constants.

    Returns
    -------
    np.ndarray
        H(f), complex, shape (n_freq,).
    """
    freqs = np.linspace(f_min, f_max, n_freq)
    w = np.squeeze(2.0 * np.pi * freqs)

    Zr_oral = lip_radiation_impedance(float(area[-1, 1]), freqs, const_dat)
    H_oral = spectrelec(
        w=w,
        area=area[:, 1].reshape(-1, 1),
        zr=Zr_oral,
        lengths=area[:, 0].reshape(-1, 1),
        no_vibration=no_vibration,
        const_dat=const_dat,
    )
    return H_oral


# =============================================================================
# Transfer function corrected for the resonances already found
# =============================================================================

def area_to_corrected_transfer_function(
    area: np.ndarray,
    n_freq: int,
    f_max: float,
    f_min: float,
    formants_found: np.ndarray,
    n_formants: int,
    no_vibration: int = 0,
    const_dat: np.ndarray = CONST_DAT,
) -> np.ndarray:
    """Computes H(f) after division by the already identified resonances.

    Refactored equivalent of ``aire2spectre_cor_oral``.

    This function is used by the iterative formant search: at each
    iteration, H(f) is divided by the contribution of the formants
    already found in order to isolate the next formant.

    Parameters
    ----------
    area : np.ndarray
        Area function, shape (29, 2).
    n_freq, f_max, f_min : int, float, float
        Parameters of the frequency axis.
    formants_found : np.ndarray
        Array of the formants already found, shape (n_max, 3).
        Columns: (0, frequency, bandwidth).
    n_formants : int
        Number of formants actually present in ``formants_found``.
    no_vibration : int
        If 1, rigid walls.
    const_dat : np.ndarray
        Physical constants.

    Returns
    -------
    np.ndarray
        H_eq(f), complex, shape (n_freq,).
    """
    H_eq = area_to_transfer_function(area, n_freq, f_max, f_min,
                                     no_vibration, const_dat)
    freqs = np.linspace(f_min, f_max, n_freq)
    SI = 2.0j * np.pi * freqs

    for I in range(n_formants):
        # Complex pole: s = -pi*BP + i*2*pi*F
        SI1 = complex(formants_found[I, 2] * np.pi,
                      formants_found[I, 1] * 2.0 * np.pi)
        # Division by (s - s1)(s - s1*) / (s1 * s1*)
        H_eq = H_eq * np.squeeze(
            ((SI - SI1) * (SI - np.conj(SI1))) / (SI1 * np.conj(SI1))
        )
    return H_eq


# =============================================================================
# Formant search by Newton-Raphson
# =============================================================================

def _find_one_formant_newton(
    area: np.ndarray,
    fe_init: float,
    bnp_init: float,
    iter_max: int,
    f_max: float,
    formants_found: np.ndarray,
    n_formants: int,
    no_vibration: int,
    const_dat: np.ndarray,
) -> Tuple[float, float]:
    """Formant search by Newton-Raphson iteration.

    Refactored equivalent of ``nraph_oral``.

    Parameters
    ----------
    area : np.ndarray
        Area function.
    fe_init : float
        Initial frequency estimate (Hz).
    bnp_init : float
        Initial bandwidth estimate (Hz).
    iter_max : int
        Maximum number of iterations.
    f_max : float
        Maximum search frequency (Hz).
    formants_found : np.ndarray
        Formants already found.
    n_formants : int
        Number of formants already found.
    no_vibration : int
        If 1, rigid walls.
    const_dat : np.ndarray
        Physical constants.

    Returns
    -------
    F : float
        Formant frequency (Hz). ``nan`` if not converged.
    BP : float
        Bandwidth (Hz). ``nan`` if not converged.
    """
    deltas = complex(30.0, 30.0)
    seuil = FORMANT_NEWTON_THRESHOLD

    SI = complex(-bnp_init * np.pi, fe_init * 2.0 * np.pi)
    F = np.nan
    BP = np.nan

    for _ in range(iter_max):
        FIcx = -1j * SI / (2.0 * np.pi)
        Q = 1.0 / area_to_corrected_transfer_function(
            area, 1, FIcx, FIcx, formants_found, n_formants,
            no_vibration, const_dat,
        )
        # Extract the scalar (Q can be a 0-d array)
        Q = np.asarray(Q).flat[0]
        SIP = SI + deltas
        FIPcx = -1j * SIP / (2.0 * np.pi)
        QP = 1.0 / area_to_corrected_transfer_function(
            area, 1, FIPcx, FIPcx, formants_found, n_formants,
            no_vibration, const_dat,
        )
        QP = np.asarray(QP).flat[0]
        Q1D = (QP - Q) / deltas
        if abs(Q1D) < 1e-30:
            break  # avoid division by zero
        SISU = complex(SI - (Q / Q1D))  # force a Python scalar
        if abs(SISU - SI) < seuil:
            F = SISU.imag / (2.0 * np.pi)
            BP = -SISU.real / np.pi
            # Filtering: the formant is valid if F > 0 and BP > 0
            if F > 0 and BP > 0:
                return F, BP
            else:
                return np.nan, np.nan
        SI = SISU

    return float(F), float(BP)


# =============================================================================
# Iterative search of all formants
# =============================================================================

def find_formants(
    area: np.ndarray,
    n_freq: int = N_FREQ_POINTS,
    f_max: float = F_MAX_HZ,
    f_min: float = F_MIN_HZ,
    no_vibration: int = 0,
    const_dat: np.ndarray = CONST_DAT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterative search of the vocal-tract formants.

    Refactored equivalent of ``vtn2frm_ftr_oral``.

    The algorithm proceeds by peak search on |H(f)|, followed by
    Newton-Raphson refinement. At each iteration, the formant found is
    divided out from H(f) to allow detection of the next one.

    Parameters
    ----------
    area : np.ndarray
        Area function, shape (29, 2).
    n_freq : int
        Number of frequency points.
    f_max : float
        Maximum frequency (Hz).
    f_min : float
        Minimum frequency (Hz).
    no_vibration : int
        If 1, rigid walls.
    const_dat : np.ndarray
        Physical constants.

    Returns
    -------
    H_eq : np.ndarray
        Final transfer function (complex), shape (n_freq,).
    F_form1 : np.ndarray
        Frequencies of the first 3 formants (Hz), shape (3, 1).
    """
    freqs = np.linspace(f_min, f_max, n_freq)
    fe = FORMANT_SEARCH_INIT_FE
    bnp = FORMANT_SEARCH_INIT_BNP
    finc = FORMANT_SEARCH_INCREMENT

    # Storage array for formants (up to FORMANT_MAX_COUNT)
    F_form = np.zeros((FORMANT_MAX_COUNT, 3))
    NF = 0
    F = 0.0

    while F <= f_max and NF < FORMANT_MAX_COUNT:
        if NF > 0:
            fe = F + finc

        H_eq = area_to_corrected_transfer_function(
            area, n_freq, f_max, f_min, F_form, NF,
            no_vibration, const_dat,
        )

        # Protection against log(0): replace zeros with a very small number
        H_mag_db = 20.0 * np.log10(np.maximum(np.abs(H_eq), 1e-30))

        ind_maxi, _ = find_peaks(np.asarray(H_mag_db, dtype=np.float64).flatten())
        ind_mini, _ = find_peaks(np.asarray(-H_mag_db, dtype=np.float64).flatten())

        if len(ind_maxi) == 0:
            break  # no more peak: we stop

        frq_max = freqs[ind_maxi]
        fe = float(frq_max[0]) if len(frq_max) > 0 else fe

        F, BP = _find_one_formant_newton(
            area, fe, bnp, FORMANT_NEWTON_MAX_ITER, f_max,
            F_form, NF, no_vibration, const_dat,
        )

        # We store F as is, even if it is NaN (identical behavior to
        # the original syntSYL.py, line 465: `F_form[NF,:] = [0, F, BP]`).
        #
        # This is CRITICAL for loop termination: the condition
        # `while F <= f_max` becomes automatically False when F is NaN
        # (since any comparison with NaN returns False in NumPy/Python),
        # which causes the natural exit of the loop.
        #
        # The old "recovery" code that did `F = fe + finc` on Newton
        # failure CREATED an infinite loop: `fe` being reinitialized at
        # each iteration from the first peak of H_eq, and `NF` never
        # being incremented, the algorithm stayed stuck on the same
        # peak indefinitely. See bug on params_0001.npz
        # (frame 321: 'éb.bga kHi b.da').
        F_form[NF, :] = [0, F, BP]
        NF += 1

    # Return the first N_OUTPUT_FORMANTS formants
    H_eq = area_to_transfer_function(area, n_freq, f_max, f_min,
                                     no_vibration, const_dat)
    F_form1 = np.zeros((N_OUTPUT_FORMANTS, 1))
    for k in range(min(N_OUTPUT_FORMANTS, NF)):
        F_form1[k] = F_form[k, 1]
    return H_eq, F_form1


# =============================================================================
# Public API: area -> (spectrum, formants)
# =============================================================================

def area_to_spectrum_and_formants(
    area: np.ndarray,
    n_freq: int = N_FREQ_POINTS,
    f_max: float = F_MAX_HZ,
    f_min: float = F_MIN_HZ,
    no_vibration: int = 0,
    const_dat: np.ndarray = CONST_DAT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the spectrum |H(f)| and the first 3 formants F1, F2, F3.

    Unified API replacing the former ``freqevalNN`` function that returned
    an ambiguous tuple (F1, F2, F3, gui, spec).

    Parameters
    ----------
    area : np.ndarray
        Area function, shape (29, 2).
    n_freq : int
        Number of frequency points (default: 500).
    f_max : float
        Maximum frequency (Hz).
    f_min : float
        Minimum frequency (Hz).
    no_vibration : int
        If 1, rigid walls.
    const_dat : np.ndarray
        Physical constants.

    Returns
    -------
    spectrum : np.ndarray
        |H(f)|, shape (n_freq,). Real positive.
    F1 : float
        Frequency of the first formant (Hz).
    F2 : float
        Frequency of the second formant (Hz).
    F3 : float
        Frequency of the third formant (Hz).

    Notes
    -----
    If fewer than 3 formants are found (tract too short or closed),
    the missing formants are returned as 0.
    """
    H_eq, F_form = find_formants(area, n_freq, f_max, f_min,
                                  no_vibration, const_dat)
    spectrum = np.abs(H_eq)
    F1 = float(F_form[0, 0]) if F_form.shape[0] > 0 else 0.0
    F2 = float(F_form[1, 0]) if F_form.shape[0] > 1 else 0.0
    F3 = float(F_form[2, 0]) if F_form.shape[0] > 2 else 0.0
    return spectrum, F1, F2, F3


__all__ = [
    "spectrelec",
    "lip_radiation_impedance",
    "area_to_transfer_function",
    "area_to_corrected_transfer_function",
    "find_formants",
    "area_to_spectrum_and_formants",
]

# ============================================================
# >>> source.py
# ============================================================


from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d


# =============================================================================
# Conversion poly → reflection coefficients
# =============================================================================

def poly2rc(a: np.ndarray) -> np.ndarray:
    """Converts LPC coefficients into reflection coefficients (k).

    Refactored equivalent of ``poly2rc`` in ``syntSYL.py`` (line 41).

    Parameters
    ----------
    a : np.ndarray
        LPC coefficients, shape (p+1,). Must have ``a[0] == 1``.

    Returns
    -------
    np.ndarray
        Reflection coefficients k, shape (p,), float64.

    Raises
    ------
    ValueError
        If ``a[0] != 1`` (LPC coefficients must be normalized).
    """
    if a[0] != 1:
        raise ValueError(
            f"The first coefficient of a must be 1 (got {a[0]}). "
            "Normalize by a[0] before the call."
        )
    p = len(a) - 1
    k = np.zeros(p, dtype=np.float64)
    a_current = np.copy(a[1:]).astype(np.float64)
    for i in range(p - 1, -1, -1):
        k[i] = a_current[i]
        if i > 0:
            flipped_a = np.flip(a_current[:i])
            denom = 1.0 - k[i] ** 2
            if abs(denom) < 1e-30:
                # Safety: if k[i] = ±1, the filter is marginal
                a_current[:i] = 0.0
            else:
                a_current[:i] = (a_current[:i] - k[i] * flipped_a) / denom
    return np.float64(k)


# =============================================================================
# Levinson-Durbin algorithm
# =============================================================================

def levinson_durbin(r: np.ndarray, p: int) -> Tuple[np.ndarray, float]:
    """Solves the Yule-Walker system with the Levinson-Durbin algorithm.

    Refactored equivalent of ``levinson_durbin`` in ``syntSYL.py`` (line 119).

    Parameters
    ----------
    r : np.ndarray
        Autocorrelation function, shape (p+1,). ``r[0]`` must be >= 0.
    p : int
        Order of the LPC filter.

    Returns
    -------
    a : np.ndarray
        LPC coefficients, shape (p+1,), float64. ``a[0] = 1``, ``a[1:]``
        are the prediction coefficients (negative sign included).
    e : float
        Final prediction error (residual variance).
    """
    a = np.zeros(p + 1, dtype=np.float64)
    e = float(r[0])
    if e == 0:
        a[0] = 1.0
        return a, 0.0
    a[0] = 1.0
    for i in range(1, p + 1):
        if i == 1:
            k = r[1] / e
        else:
            k = (r[i] - np.dot(a[1:i], r[i - 1:0:-1])) / e
        a_new = a[1:i] - k * np.flip(a[1:i])
        a[i] = k
        a[1:i] = a_new
        e *= (1.0 - k ** 2)
        if e < 0:
            e = 0.0
    a[1:] = -a[1:]
    return np.float64(a), float(e)


# =============================================================================
# Conversion |H(f)| → (gain, LPC)
# =============================================================================

def hfreq_to_lpc(H: np.ndarray, p: int) -> Tuple[float, np.ndarray]:
    """Converts an amplitude spectrum |H(f)| into LPC coefficients.

    Refactored equivalent of ``Hfreq2lpc`` in ``syntSYL.py`` (line 140).

    The algorithm proceeds in 3 steps:

    1. Construction of the full symmetric spectrum (mirrored around f_Nyq).
    2. IFFT to obtain the autocorrelation function R(τ).
    3. Yule-Walker resolution by Levinson-Durbin.

    Parameters
    ----------
    H : np.ndarray
        Amplitude spectrum |H(f)|, shape (N,). Must be real positive.
    p : int
        Order of the LPC filter.

    Returns
    -------
    g : float
        Gain (square root of the prediction error).
    a : np.ndarray
        LPC coefficients, shape (p+1,), float64.
    """
    H = np.asarray(H).flatten()
    N = len(H)
    # Construction of the symmetric spectrum: [H, H[N-1], flip(H[1:N-1])]
    Hr = np.concatenate([H, [H[N - 1]], np.flipud(H[1:N - 1])])
    # Autocorrelation = IFFT of |H|² (Wiener-Khinchin theorem)
    R = np.float64(np.real(np.fft.ifft(np.abs(Hr) ** 2)))
    R = R[:p + 1]
    a, e = levinson_durbin(R, p)
    g = float(np.sqrt(max(e, 0.0)))
    return g, np.float64(a)


# =============================================================================
# Generation of the glottal source
# =============================================================================

def generate_glottal_source(
    n_samples: int,
    sample_rate: int,
    f0: np.ndarray,
    stretch: bool = False,
) -> np.ndarray:
    """Generates a glottal source signal (simplified LF impulse train).

    Refactored equivalent of ``gen_src_3`` in ``syntSYL.py`` (line 83).

    Parameters
    ----------
    n_samples : int
        Length of the signal to generate (in samples).
    sample_rate : int
        Sampling rate (Hz).
    f0 : np.ndarray
        Fundamental frequency profile F0(t), shape (n_samples,) or (M,)
        with M < n_samples (in which case F0 is interpolated).
    stretch : bool
        If True, the pulse profile is recomputed at each period ("stretch"
        mode). If False, a single profile is convolved with the impulse
        train.

    Returns
    -------
    np.ndarray
        Source signal, shape (n_samples,), float64.

    Notes
    -----
    The pulse profile used is the two-phase Rosenberg model: opening
    (rising cubic phase) + closing (falling quadratic phase). This is an
    approximation of the Liljencrants-Fant model.
    """
    M = len(f0)
    if M != n_samples:
        f0 = np.interp(np.linspace(0, 1, n_samples),
                       np.linspace(0, 1, M), f0)

    # Safety: replace F0 <= 0 to avoid an infinite loop
    f0_safe = np.where(f0 <= 0, 1.0, f0)

    s = np.zeros(n_samples, dtype=np.float64)

    if not stretch:
        # Normal mode: single profile, convolution with impulse train
        f0_max = float(np.max(f0_safe))
        T = max(int(sample_rate / f0_max / 0.8), 1)
        Tp = max(int(T / 3), 1)
        Tn = max(int(T / 4), 1)
        t1 = np.arange(1, Tp + 1)
        t2 = np.arange(Tp + 1, Tp + Tn + 1)
        # Opening phase: 3·(t/Tp)² - 2·(t/Tp)³ (cubic polynomial)
        # Closing phase: 1 - ((t-Tp)/Tn)² (parabola)
        pulse = np.concatenate([
            3.0 * (t1 / Tp) ** 2 - 2.0 * (t1 / Tp) ** 3,
            1.0 - ((t2 - Tp) / Tn) ** 2,
        ])
        train = np.zeros(n_samples)
        ind = 0
        while ind < n_samples:
            train[ind] = 1.0
            period = max(int(sample_rate / f0_safe[ind]), 1)
            ind += period
        s = np.convolve(train, pulse, mode="full")
    else:
        # Stretch mode: profile recomputed at each period (variable duration)
        ind = 0
        while ind < n_samples:
            T = max(int(sample_rate / f0_safe[ind]), 1)
            Tp = max(int(T / 3), 1)
            Tn = max(int(T / 6), 1)
            t1 = np.arange(1, Tp + 1)
            t2 = np.arange(Tp + 1, Tp + Tn + 1)
            pulse = np.concatenate([
                3.0 * (t1 / Tp) ** 2 - 2.0 * (t1 / Tp) ** 3,
                1.0 - ((t2 - Tp) / Tn) ** 2,
            ])
            end_idx = min(ind + Tp + Tn, n_samples)
            s[ind:end_idx] = pulse[:end_idx - ind]
            period = max(int(sample_rate / f0_safe[ind]), 1)
            ind += period

    return s[:n_samples]


# =============================================================================
# Lattice filtering with temporal interpolation of coefficients
# =============================================================================

def lpc_filter_lattice(
    excitation: np.ndarray,
    frame_step: float,
    transition_step: float,
    sample_rate: int,
    lpc_coeffs: np.ndarray,
) -> np.ndarray:
    """Applies an LPC lattice filter to an excitation signal.

    Refactored equivalent of ``f_lpc_exc2sig`` in ``syntSYL.py`` (line 54).

    The lattice filtering uses the reflection coefficients k (computed
    from the LPC coefficients by :func:`poly2rc`). Between two
    successive frames, the k are linearly interpolated over a duration
    ``transition_step`` to avoid clicks.

    Parameters
    ----------
    excitation : np.ndarray
        Excitation signal (glottal source), shape (L,).
    frame_step : float
        Duration of an articulatory frame (seconds).
    transition_step : float
        Transition duration between two coefficient sets (seconds).
    sample_rate : int
        Sampling rate (Hz).
    lpc_coeffs : np.ndarray
        Per-frame LPC coefficients, shape (M, p+1). ``lpc_coeffs[m, 0]``
        must be 1 for every frame m.

    Returns
    -------
    np.ndarray
        Synthesized signal, shape (L,), float64.

    Raises
    ------
    ValueError
        If the number of frames ``M`` does not match ``L / (frame_step *
        sample_rate)``.
    """
    N_step = int(frame_step * sample_rate)
    N_trans = int(transition_step * sample_rate)
    L = len(excitation)
    M = L // N_step
    if M != lpc_coeffs.shape[0]:
        raise ValueError(
            f"Dimension mismatch: signal = {M} frames "
            f"({L} samples / {N_step}), but lpc_coeffs has "
            f"{lpc_coeffs.shape[0]} frames."
        )

    p = lpc_coeffs.shape[1] - 1
    sig = np.zeros(L, dtype=np.float64)
    e_f = np.zeros(p + 1, dtype=np.float64)
    e_b = np.zeros(p + 1, dtype=np.float64)

    # Initialization: reflection coefficients of the first frame
    k_P = poly2rc(lpc_coeffs[0, :])

    for m in range(M):
        k_C = poly2rc(lpc_coeffs[m, :])
        if N_trans > 0:
            dk = (k_C - k_P) / N_trans
        else:
            dk = np.zeros_like(k_C)

        for ind in range(N_step):
            if ind < N_trans:
                k_P = k_P + dk
            e_f[p] = excitation[ind + m * N_step]
            for ind_p in range(p):
                e_f[p - ind_p - 1] = (
                    e_f[p - ind_p] - k_P[p - ind_p - 1] * e_b[p - ind_p - 1]
                )
                e_b[p - ind_p] = (
                    e_b[p - ind_p - 1] + k_P[p - ind_p - 1] * e_f[p - ind_p - 1]
                )
            e_b[0] = e_f[0]
            sig[ind + m * N_step] = e_f[0]

    return sig


# =============================================================================
# Generation of the F0 contour by cubic interpolation
# =============================================================================

def generate_f0_contour(
    n_samples: int,
    sample_rate: int,
    f0_baseline: Tuple[float, float, float, float] = (100.0, 130.0, 110.0, 90.0),
    jitter_std: float = 0.005,
    jitter_seed: int = 0,
) -> np.ndarray:
    """Generates an F0 contour by cubic interpolation of a 4-point pattern.

    The original pattern corresponds to a neutral declarative intonation:
    low start (100 Hz), rise at one third (130 Hz), fall at two thirds
    (110 Hz), final drop (90 Hz).

    Parameters
    ----------
    n_samples : int
        Total number of samples of the signal to synthesize.
    sample_rate : int
        Sampling rate (Hz).
    f0_baseline : tuple of float
        Quadruplet (f0_1, f0_2, f0_3, f0_4) in Hz at the temporal positions
        1/3, 1/3, 2/3, 3/3 of the signal.
    jitter_std : float
        Standard deviation of the Gaussian noise added to F0, expressed
        as a fraction of mean F0.
    jitter_seed : int
        Seed of the random generator for reproducibility.

    Returns
    -------
    np.ndarray
        F0(t), shape (n_samples,), float64.
    """
    # Anchor points at 1, N/3, 2N/3, N (1-based indexing like MATLAB)
    anchors_x = np.array([
        1,
        int(np.fix(n_samples / 3)),
        int(2 * np.fix(n_samples / 3)),
        n_samples,
    ])
    anchors_y = np.array(f0_baseline, dtype=np.float64)

    f0 = interp1d(anchors_x, anchors_y, kind="cubic")(
        np.arange(1, n_samples + 1)
    )

    # Gaussian jitter (reproducible)
    if jitter_std > 0:
        rng = np.random.RandomState(jitter_seed)
        f0 = f0 + rng.randn(*f0.shape) * np.mean(f0) * jitter_std

    return f0


__all__ = [
    "poly2rc",
    "levinson_durbin",
    "hfreq_to_lpc",
    "generate_glottal_source",
    "lpc_filter_lattice",
    "generate_f0_contour",
]

# ============================================================
# >>> envelope.py
# ============================================================


from typing import Optional, Sequence

import numpy as np
from scipy.signal import medfilt


# =============================================================================
# Global envelope (fade-in/fade-out)
# =============================================================================

def compute_global_envelope(n_samples: int) -> np.ndarray:
    """Computes the global fade-in/fade-out envelope.

    Applies a Hanning window over the first and last quarter of the
    signal, with a plateau at 1 in the middle. The last quarter is squared
    for a softer fade-out.

    Parameters
    ----------
    n_samples : int
        Total length of the signal (in samples).

    Returns
    -------
    np.ndarray
        Global envelope, shape (n_samples,), values in [0, 1].
    """
    lg = n_samples
    lg2 = max(round(lg / 4), 1)
    han = np.hanning(lg2)
    handeb = han[: round(lg2 / 2)]
    hanfin = han[round(lg2 / 2):]
    # Central plateau + squared final fade (softens the ending)
    return np.concatenate([handeb, np.ones(lg - lg2), hanfin ** 2])


# =============================================================================
# Phonological envelope (computed)
# =============================================================================

def compute_phonological_envelope(
    word_tokens: Sequence[str],
    attack_exp: float,
    decay_exp: float,
    frame_samples: int,
) -> np.ndarray:
    """Computes the phonological envelope of a word from its pattern.

    Refactored equivalent of ``synthfen`` in ``syntSYL.py`` (line 327).

    Parameters
    ----------
    word_tokens : sequence of str
        Sequence of phonological tokens. Convention:

        ====== =========================================
        Token  Meaning
        ====== =========================================
        ``O``  Onset (word beginning, initial silence)
        ``V``  Vowel (maximum amplitude)
        ``C``  Consonant (variable amplitude)
        ``c``  Weak consonant (reduced amplitude)
        ``F``  End (word end, terminal silence)
        ====== =========================================

        Transitions between successive tokens determine the envelope.
        Example: ``['O', 'V', 'C', 'V', 'F']`` for a VCV word.
    attack_exp : float
        Exponent applied to attack phases (rise). A value of 1 gives a
        linear rise; > 1 gives a faster rise.
    decay_exp : float
        Exponent applied to decay phases.
    frame_samples : int
        Number of samples per phonological frame. Corresponds to the
        duration of one token multiplied by the sampling rate.

    Returns
    -------
    np.ndarray
        Phonological envelope, shape (n_tokens - 1) * frame_samples,).

    Notes
    -----
    The algorithm walks through the transitions between successive tokens
    and stacks segments of ``frame_samples`` samples:

    - ``OC`` (onset → consonant): silence
    - ``OV`` (onset → vowel): maximum amplitude
    - ``CF`` (consonant → end): rise according to ``attack_exp``
    - ``CV`` (consonant → vowel): rise + plateau
    - ``VC`` (vowel → consonant): decay according to ``decay_exp``
    - ``VV`` (vowel → vowel): maximum amplitude
    - ``CC`` (consonant → consonant): reduced rise + decay (0.5×)
    """
    word = "".join(word_tokens)
    n_transitions = len(word) - 1
    fen3 = np.hanning(2 * frame_samples)
    fen2 = np.array([])
    for k in range(n_transitions):
        syl = word[k:k + 2]
        if syl == "OC":
            fen2 = np.concatenate([fen2, np.zeros(frame_samples)])
        elif syl == "OV":
            fen2 = np.concatenate([fen2, np.ones(frame_samples)])
        elif syl == "CF":
            fen2 = np.concatenate([fen2, fen3[:frame_samples] ** attack_exp])
        elif syl == "VF":
            pass  # VF transition: no segment (fen2 unchanged)
        elif syl == "VV":
            fen2 = np.concatenate([fen2, np.ones(frame_samples)])
        elif syl in ("CV", "cV"):
            fen2 = np.concatenate([
                fen2,
                fen3[:frame_samples] ** attack_exp,
                np.ones(frame_samples),
            ])
        elif syl in ("VC", "Vc"):
            fen2 = np.concatenate([fen2, fen3[frame_samples:] ** decay_exp])
        elif syl == "CC":
            fen2 = np.concatenate([
                fen2,
                0.5 * fen3[:frame_samples] ** attack_exp,
                0.5 * fen3[frame_samples:] ** (3 * decay_exp),
            ])
        elif syl in ("Cc", "cC"):
            product = (fen3[:frame_samples] ** attack_exp) * \
                      (fen3[frame_samples:] ** (3 * decay_exp))
            fen2 = np.concatenate([fen2, product])
    return fen2


# =============================================================================
# Application of the envelopes to the signal
# =============================================================================

def apply_envelopes(
    signal: np.ndarray,
    sample_rate: int,
    word_tokens: Optional[Sequence[str]] = None,
    attack_exp: float = 1.0,
    decay_exp: float = 2.0,
    duration_factor: float = 16.0,
    frame_step: float = 10e-3,
    external_envelope: Optional[np.ndarray] = None,
    median_filter_size: int = 7,
) -> np.ndarray:
    """Applies the global + phonological envelopes to the signal.

    Refactored equivalent of the envelope part of
    ``synthsimpleWORDfen`` (lines 183-204 of the original code).

    Parameters
    ----------
    signal : np.ndarray
        Raw LPC signal, shape (L,).
    sample_rate : int
        Sampling rate (Hz).
    word_tokens : sequence of str, optional
        Phonological tokens (e.g. ``['O', 'V', 'F']``). Required if
        ``external_envelope`` is None.
    attack_exp : float
        Attack exponent for the computed phonological envelope.
    decay_exp : float
        Decay exponent.
    duration_factor : float
        Duration factor for the computed envelope. In the original code,
        the duration of each phonological frame was ``dur * fs * T_s``.
    frame_step : float
        Time step of an articulatory frame (seconds).
    external_envelope : np.ndarray, optional
        Pre-computed external envelope, shape (M,). If provided, it is
        resampled to the signal length and then median-filtered.
        Takes precedence over the computed phonological envelope.
    median_filter_size : int
        Size of the median filter applied to the external envelope. Set
        to 0 or 1 to disable.

    Returns
    -------
    np.ndarray
        Signal multiplied by both envelopes, shape (L,).
    """
    n_samples = len(signal)
    fen = compute_global_envelope(n_samples)

    if external_envelope is not None:
        fen2 = external_envelope
        if len(fen2) != n_samples:
            fen2 = np.interp(
                np.linspace(0, 1, n_samples),
                np.linspace(0, 1, len(fen2)),
                fen2,
            )
        if median_filter_size and median_filter_size > 1:
            fen2 = medfilt(fen2, kernel_size=median_filter_size)
    else:
        if word_tokens is None:
            raise ValueError(
                "word_tokens is required when external_envelope is not provided."
            )
        frame_samples = int(duration_factor * sample_rate * frame_step)
        fen2 = compute_phonological_envelope(
            word_tokens, attack_exp, decay_exp, frame_samples,
        )
        if len(fen2) != n_samples:
            fen2 = np.interp(
                np.linspace(0, 1, n_samples),
                np.linspace(0, 1, len(fen2)),
                fen2,
            )

    return fen * fen2 * signal


__all__ = [
    "compute_global_envelope",
    "compute_phonological_envelope",
    "apply_envelopes",
]

# ============================================================
# >>> synthesis.py
# ============================================================


from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

# =============================================================================
# Utility function: soft rectification of sections
# =============================================================================

def soft_rect(x: np.ndarray, s: float) -> np.ndarray:
    """Applies a soft rectification to an array.

    Refactored equivalent of ``softrect`` in ``syntSYL.py`` (line 38).

    The soft rectification is a differentiable approximation of the
    ``max(x, 0)`` function: it smooths the zero crossing over a width ~s.

    .. math::
        \\text{softrect}(x, s) = \\frac{\\sqrt{x^2 + s} + x}{2}

    Parameters
    ----------
    x : np.ndarray
        Input array.
    s : float
        Smoothness parameter. ``s = 0`` corresponds to ``max(x, 0)``.
        Typical value: 0.75.

    Returns
    -------
    np.ndarray
        Rectified array, same shape as ``x``.
    """
    return (np.sqrt(x ** 2 + s) + x) / 2.0


# =============================================================================
# Synthesis result
# =============================================================================

@dataclass
class SynthResult:
    """Result of a vocal synthesis.

    Attributes
    ----------
    signal : np.ndarray
        Synthesized audio signal, shape (N,), float64. Normalized to
        ``1 / 1.01`` if ``SynthConfig.normalize_peak`` is True.
    formants : np.ndarray
        Trajectories of the first 3 formants (F1, F2, F3) per
        articulatory frame, shape (T, 3), float64, in Hz.
    spectra : np.ndarray
        Per-frame |H(f)| spectra, shape (T, n_freq), float64.
    final_state : VlamState
        Final articulatory state (after the last frame). Allows chaining
        successive syntheses.
    sample_rate : int
        Sampling rate of the signal (Hz).
    n_frames : int
        Number of articulatory frames processed.
    """
    signal: np.ndarray
    formants: np.ndarray
    spectra: np.ndarray
    final_state: VlamState
    sample_rate: int
    n_frames: int


# =============================================================================
# Evaluation of a single articulatory frame
# =============================================================================

def evaluate_frame(
    state: VlamState,
    articulatory_params: np.ndarray,
    soft_rect_s: float,
    config: SynthConfig,
) -> Tuple[VlamState, np.ndarray, float, float, float]:
    """Evaluates a single articulatory frame.

    Refactored equivalent of ``freqevalNN`` in ``syntSYL.py`` (line 906).

    Computes, from a vector of 7 Maeda articulatory parameters:

    1. The tract geometry (sagittal contour + area function) via VLAM.
    2. The |H(f)| spectrum of the vocal tract.
    3. The frequencies of the first 3 formants F1, F2, F3.

    Parameters
    ----------
    state : VlamState
        Current articulatory state (will be copied, not mutated).
    articulatory_params : np.ndarray
        Vector of 7 Maeda parameters in canonical order
        (jaw, tongue body, tongue dorsum, tongue tip, lip protrusion,
        lip height, larynx height). Index 4 corresponds to lip
        protrusion (LP) and index 5 to lip height (LH).
        The swap to the internal vlam2009NN order is handled
        automatically in ``compute_vlam_geometry``.
    soft_rect_s : float
        Smoothness parameter for the rectification of sections.
        Typical value: 0.75.
    config : SynthConfig
        Synthesizer configuration (uses ``sample_rate``,
        ``frame_step``, etc.).

    Returns
    -------
    new_state : VlamState
        New state after parameter update and geometric computation.
    spectrum : np.ndarray
        |H(f)| spectrum, shape (n_freq,).
    F1, F2, F3 : float
        Frequencies of the first 3 formants (Hz).
    """
    if len(articulatory_params) != 7:
        raise ValueError(
            f"articulatory_params must have 7 elements (got {len(articulatory_params)})."
        )
    # Defensive copy + update of the first 7 parameters
    new_state = VlamState(
        sagittal=state.sagittal.copy(),
        inci=state.inci.copy(),
        area=state.area.copy(),
        params=state.params.copy(),
        protrusion_distance=state.protrusion_distance,
        lip_height_cm=state.lip_height_cm,
        lip_height_index=state.lip_height_index,
        lip_opening_h=state.lip_opening_h,
        lip_opening_w=state.lip_opening_w,
    )
    new_state.params[:7] = articulatory_params

    # VLAM geometric computation (sagittal contour + area function)
    new_state = compute_vlam_geometry(new_state)

    # Soft rectification of the sections (avoids zero sections)
    area = np.column_stack((
        new_state.area[:, 0],
        soft_rect(new_state.area[:, 1], soft_rect_s),
    ))

    # Spectrum + formants computation
    spectrum, F1, F2, F3 = area_to_spectrum_and_formants(
        area=area,
        n_freq=N_FREQ_POINTS,
        f_max=F_MAX_HZ,
        f_min=F_MIN_HZ,
        no_vibration=0,
        const_dat=CONST_DAT,
    )
    return new_state, spectrum, F1, F2, F3


# =============================================================================
# Main API: synthwordfen
# =============================================================================

def synthwordfen(
    state: VlamState,
    articulatory_params: np.ndarray,
    word_tokens: Sequence[str],
    f0_scale: float = 1.0,
    soft_rect_s: float = 0.75,
    duration_factor: float = 16.0,
    envelope: Optional[np.ndarray] = None,
    config: Optional[SynthConfig] = None,
) -> SynthResult:
    """Synthesizes a word from Maeda articulatory parameters.

    This is the main function of the VLAM vocal synthesizer. It chains:

    1. **Articulatory evaluation**: for each frame of
       ``articulatory_params``, computes the |H(f)| spectrum and the
       formants.
    2. **LPC conversion**: converts each spectrum into LPC coefficients
       (by IFFT + Levinson-Durbin).
    3. **Glottal source**: generates an impulse train modulated by a
       cubically interpolated F0 contour + Gaussian jitter.
    4. **Lattice filtering**: applies the LPC coefficients to the source,
       with temporal interpolation between frames.
    5. **Envelopes**: applies the global envelope (fade) and the
       phonological envelope (computed or external).
    6. **Radiation**: high-pass single-pole filtering at the lips.
    7. **Normalization** (optional).

    Parameters
    ----------
    state : VlamState
        Initial articulatory state. Typically obtained via
        ``VlamState.initial(vocal_tract_length_mm=195)``.
    articulatory_params : np.ndarray
        Matrix of Maeda parameters, shape (T, 7). T = number of frames.
    word_tokens : sequence of str
        Phonological tokens for the computation of the phonological
        envelope (ignored if ``envelope`` is provided). E.g.
        ``['O', 'V', 'F']``.
    f0_scale : float
        Scale factor applied to the F0 contour. ``1.0`` = no
        modification. ``2.0`` = doubled F0 (female/child voice).
    soft_rect_s : float
        Soft-rectification parameter of the sections (default: 0.75).
    duration_factor : float
        Duration factor for the computed phonological envelope. In the
        original code, duration of a phonological frame = ``dur * fs * T_s``.
    envelope : np.ndarray, optional
        Pre-computed external envelope, shape (M,). If provided, takes
        precedence over the computed phonological envelope.
    config : SynthConfig, optional
        Synthesizer configuration. If None, uses default values
        (fs=20000, frame_step=10ms, lpc_order=30, etc.).

    Returns
    -------
    SynthResult
        Synthesis result (signal, formants, spectra, final state).

    Notes
    -----
    The old ``synthwordfen`` function played the signal at the end of
    the synthesis (side effect). This version no longer does so. To play
    the signal, use :func:`vlam.visualization.play_signal` or pass
    ``SynthConfig(play_audio=True)``.

    Examples
    --------
    >>> import numpy as np
    >>> from vlam import VlamState, SynthConfig, synthwordfen
    >>> Pval = np.load("params_0012.npz")["Pval"]  # (880, 7)
    >>> envelope = np.load("params_0012.npz")["envelope"]
    >>> state = VlamState.initial(vocal_tract_length_mm=195)
    >>> result = synthwordfen(
    ...     state=state,
    ...     articulatory_params=Pval,
    ...     word_tokens=["O", "V", "F"],
    ...     envelope=envelope,
    ... )
    >>> result.signal.shape
    (176000,)
    >>> result.formants.shape
    (880, 3)
    """
    if config is None:
        config = SynthConfig()

    fs = config.sample_rate
    T_s = config.frame_step
    p = config.lpc_order

    Pval = np.atleast_2d(articulatory_params)
    n_frames = Pval.shape[0]

    # ---- Step 1: frame-by-frame articulatory evaluation ----
    spectra = np.zeros((n_frames, N_FREQ_POINTS))
    formants = np.zeros((n_frames, 3))
    current_state = state
    for k in range(n_frames):
        current_state, spec, F1, F2, F3 = evaluate_frame(
            state=current_state,
            articulatory_params=Pval[k, :],
            soft_rect_s=soft_rect_s,
            config=config,
        )
        spectra[k, :] = spec
        formants[k, :] = [F1, F2, F3]

    # ---- Step 2: spectrum → LPC conversion ----
    lpc_coeffs = np.zeros((n_frames, p + 1))
    for k in range(n_frames):
        _, a = hfreq_to_lpc(spectra[k, :], p)
        lpc_coeffs[k, :] = a

    # ---- Step 3: temporal parameters ----
    L_t = n_frames * T_s  # total duration in seconds
    N_t = int(np.fix(L_t * fs))  # total number of samples

    # ---- Step 4: generation of the F0 contour ----
    f0 = generate_f0_contour(
        n_samples=N_t,
        sample_rate=fs,
        f0_baseline=config.f0_baseline,
        jitter_std=config.f0_jitter_std,
        jitter_seed=config.f0_jitter_seed,
    )
    f0 = f0 * f0_scale  # apply the scale factor

    # ---- Step 5: generation of the glottal source ----
    excitation = generate_glottal_source(N_t, fs, f0, stretch=False)

    # ---- Step 6: lattice filtering ----
    sig = lpc_filter_lattice(
        excitation=excitation,
        frame_step=T_s,
        transition_step=config.transition_step,
        sample_rate=fs,
        lpc_coeffs=lpc_coeffs,
    )

    # ---- Step 7: application of envelopes ----
    sig = apply_envelopes(
        signal=sig,
        sample_rate=fs,
        word_tokens=word_tokens,
        attack_exp=1.0,
        decay_exp=2.0,
        duration_factor=duration_factor,
        frame_step=T_s,
        external_envelope=envelope,
        median_filter_size=config.median_filter_envelope,
    )

    # ---- Step 8: lip-radiation filtering ----
    sig = lfilter([1.0, -config.radiation_pole], 1.0, sig)

    # ---- Step 9: normalization ----
    if config.normalize_peak:
        peak = float(np.max(np.abs(sig)))
        if peak < 1e-12:
            # Null signal (closed tract): do not divide by 0
            sig = np.zeros_like(sig)
        else:
            sig = sig / (1.01 * peak)

    # ---- Step 10: optional audio playback ----
    if config.play_audio:
        play_signal(sig, fs)

    return SynthResult(
        signal=sig,
        formants=formants,
        spectra=spectra,
        final_state=current_state,
        sample_rate=fs,
        n_frames=n_frames,
    )


__all__ = [
    "soft_rect",
    "SynthResult",
    "evaluate_frame",
    "synthwordfen",
]

# ============================================================
# >>> visualization.py
# ============================================================


from typing import Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# Vocal-tract display
# =============================================================================

def plot_vocal_tract(state: VlamState, ax=None):
    """Displays the sagittal contour of the vocal tract.

    Refactored equivalent of ``showgui`` in ``syntSYL.py`` (line 833).

    Parameters
    ----------
    state : VlamState
        Articulatory state with ``state.sagittal`` populated.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw. If None, creates a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for plot_vocal_tract. "
            "Install it via: pip install matplotlib"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    mx = -0.0153
    my = 138.4566
    sag = state.sagittal / 29.5
    ax.plot(
        (sag[:, 0] - sag[55, 0] + mx),
        (sag[:, 1] + sag[55, 1]) - my,
        '-g', linewidth=2,
    )
    ax.text(-0.5, -7.5, 'VLAM Display/GIPSA-LAB', fontsize=18)
    ax.set_xlim(-1.5, 8.5)
    ax.set_ylim(-7.5, 2.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    return fig


# =============================================================================
# Audio playback
# =============================================================================

def play_signal(signal: np.ndarray, sample_rate: int = 20000) -> None:
    """Plays an audio signal via sounddevice.

    Refactored equivalent of ``play`` in ``syntSYL.py`` (line 35).

    Parameters
    ----------
    signal : np.ndarray
        Audio signal, shape (N,), float64. Will be converted to float32.
    sample_rate : int
        Sampling rate (Hz).
    """
    if not _HAS_SD:
        raise ImportError(
            "sounddevice is required for play_signal. "
            "Install it via: pip install sounddevice"
        )
    sd.play(signal.astype(np.float32), sample_rate)


# =============================================================================
# Plot of articulatory parameters
# =============================================================================

def plot_articulatory_params(
    Pv: np.ndarray,
    labels: Sequence[str] = ARTICULATORY_LABELS,
    x_label: str = "Frame index",
    y_label: str = "Parameter value",
    title: str = "Articulatory parameter trajectories",
    y_range: Optional[Tuple[float, float]] = (-4, 5),
    grid: bool = True,
    ax=None,
):
    """Plots the 7 articulatory parameters as a function of time.

    Refactored equivalent of ``plot_Pv`` in ``syntSYL.py`` (line 996).

    Parameters
    ----------
    Pv : np.ndarray
        Parameter matrix, shape (T, 7).
    labels : sequence of str
        Labels of the 7 parameters.
    x_label, y_label, title : str
        Axis labels and title.
    y_range : tuple, optional
        Bounds of the y-axis. None for automatic adjustment.
    grid : bool
        If True, displays a grid.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for plot_articulatory_params. "
            "Install it via: pip install matplotlib"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    T, N = Pv.shape
    t = np.arange(T)
    ax.plot(t, Pv)
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.set_xlim(0, T - 1)
    ax.legend(labels, loc='best', ncol=N, frameon=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =============================================================================
# Plot of formants
# =============================================================================

def plot_formants(
    formants: np.ndarray,
    labels: Sequence[str] = ("F1", "F2", "F3"),
    x_label: str = "Frame index",
    y_label: str = "Frequency (Hz)",
    title: str = "Formant trajectories",
    y_min: float = 0,
    top_margin: float = 0.2,
    grid: bool = True,
    ax=None,
):
    """Plots the 3 formants as a function of time.

    Refactored equivalent of ``plot_formants`` in ``syntSYL.py`` (line 1018).

    Parameters
    ----------
    formants : np.ndarray
        Formant matrix, shape (T, 3) or (3, T).
    labels : sequence of str
        Labels of the 3 formants.
    x_label, y_label, title : str
        Labels and title.
    y_min : float
        Lower bound of the y-axis.
    top_margin : float
        Relative top margin to the max.
    grid : bool
        If True, displays a grid.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for plot_formants. "
            "Install it via: pip install matplotlib"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    arr = np.squeeze(formants)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"formants must be of shape (T,3) or (3,T); got {arr.shape}"
        )
    T = arr.shape[0]
    t = np.arange(T)
    ax.plot(t, arr)
    ax.set_xlim(0, T - 1)
    ymax_data = float(np.nanmax(arr)) if arr.size > 0 else 1000.0
    ax.set_ylim(y_min, ymax_data + (ymax_data - y_min) * top_margin)
    ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
              ncol=3, frameon=True, borderpad=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =============================================================================
# Spectrogram
# =============================================================================

def plot_spectrogram(
    signal: np.ndarray,
    sample_rate: int = 20000,
    title: str = "Spectrogram",
    use_mel: bool = False,
    n_mels: int = 64,
    frame_length: int = 400,
    hop_length: int = 200,
    freq_max: float = 4500.0,
    ax=None,
):
    """Displays the spectrogram of an audio signal.

    Refactored equivalent of ``spectreplot`` in ``syntSYL.py`` (line 1048).

    Parameters
    ----------
    signal : np.ndarray
        Audio signal, shape (N,).
    sample_rate : int
        Sampling rate (Hz).
    title : str
        Plot title.
    use_mel : bool
        If True, uses a mel scale.
    n_mels : int
        Number of mel bands (if ``use_mel``).
    frame_length : int
        FFT window size.
    hop_length : int
        Window advance step.
    freq_max : float
        Maximum displayed frequency (Hz).
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for plot_spectrogram. "
            "Install it via: pip install matplotlib"
        )
    if not _HAS_LIBROSA:
        raise ImportError(
            "librosa is required for plot_spectrogram. "
            "Install it via: pip install librosa"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    if use_mel:
        S = librosa.feature.melspectrogram(
            y=signal, sr=sample_rate, n_fft=frame_length,
            hop_length=hop_length, n_mels=n_mels,
        )
    else:
        S = np.abs(librosa.stft(signal, n_fft=frame_length,
                                 hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    if use_mel:
        librosa.display.specshow(S_db, sr=sample_rate,
                                  hop_length=hop_length,
                                  x_axis='time', y_axis='mel', ax=ax)
    else:
        librosa.display.specshow(S_db, sr=sample_rate,
                                  hop_length=hop_length,
                                  x_axis='time', y_axis='linear', ax=ax)
    fig.colorbar(ax.images[-1], ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(0, freq_max)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_vocal_tract",
    "play_signal",
    "plot_articulatory_params",
    "plot_formants",
    "plot_spectrogram",
]





# ============================================================
# Public API (version 3.0 — without legacy wrappers)
# ============================================================

__version__ = "3.2.0"

__all__ = [
    # Configuration
    "SynthConfig", "VlamState", "CONST_DAT",
    "ARTICULATORY_LABELS", "DEFAULT_VT_LENGTH_MM",
    "N_ARTICULATORY_PARAMS", "N_SECTIONS", "N_SAGITTAL_POINTS",
    "N_FREQ_POINTS", "F_MAX_HZ", "F_MIN_HZ",
    # Geometry
    "compute_vlam_geometry",
    # Acoustics
    "spectrelec", "lip_radiation_impedance",
    "area_to_transfer_function", "area_to_corrected_transfer_function",
    "area_to_spectrum_and_formants",
    "find_formants",
    "poly2rc", "levinson_durbin", "hfreq_to_lpc",
    # Source
    "generate_glottal_source", "lpc_filter_lattice", "generate_f0_contour",
    # Envelopes
    "compute_global_envelope", "compute_phonological_envelope",
    "apply_envelopes",
    # Synthesis
    "soft_rect", "evaluate_frame", "synthwordfen", "SynthResult",
    # Visualization
    "plot_vocal_tract", "play_signal",
    "plot_articulatory_params", "plot_formants", "plot_spectrogram",
    # Version
    "__version__",
]
