# SPDX-License-Identifier: MIT
"""
config.py — Configuration for bigbi-bi.gbi simulation scripts.

This module sets up the paths for the simulation scripts so they
work both in the development environment and when cloned from GitHub.

Usage in scripts:
    from config import REPO_ROOT, OUTPUT_BASE, T_BASE, FS_AUDIO
"""
from pathlib import Path
import sys

# ─────────────────────────────────────────────────────────────────────
# Repository root (this file's parent directory)
# ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()

# synthSYL package and vlam.py are in the repo root
SYNTHSYL_REPO = REPO_ROOT

# Make the repo root importable (for `import synthSYL`, `import vlam`)
if str(SYNTHSYL_REPO) not in sys.path:
    sys.path.insert(0, str(SYNTHSYL_REPO))

# ─────────────────────────────────────────────────────────────────────
# Output base directory (all simulation outputs go here)
# ─────────────────────────────────────────────────────────────────────
OUTPUT_BASE = REPO_ROOT / "output"

# ─────────────────────────────────────────────────────────────────────
# Model parameters (VLAM / Maeda, 7 parameters)
# ─────────────────────────────────────────────────────────────────────
T_BASE = 16           # Maeda time step (didactic mode)
T_STEP_MS = 10        # 1 step = 10 ms
SR_GESTURE = 100.0    # Gesture sampling rate (Hz)
K_DISPLAY = 30.0      # Display curvature, vowel arcs z_v (article §2.1: Kvoy=30)
K_C_DISPLAY = 10.0    # Display curvature, consonant arcs z_c (article §2.1: K=10)
FPS_OUT = 25.0        # Output video frame rate
FS_AUDIO = 20_000     # Audio sample rate (Hz)
TRAIL_MS = 750        # Polar trail length (ms)
GUI_LEN_MM = 195      # Vocal tract length (mm, adult male)
VALRECT = 0.75        # VLAM soft-rectification (canonical value from
                      # timit-to-Maeda: default of vlam.synthwordfen and
                      # of batch_synthesize.py --valrect; applied to the
                      # area function, not to articulatory parameters)
LOCUS_OFFSET_STEPS = 3  # Locus-equation measurement (article Fig. 3):
                        # F2 sampled 30 ms (3 steps of 10 ms) after the
                        # consonant release (start of the C->V block)

# ─────────────────────────────────────────────────────────────────────
# Article reference
# ─────────────────────────────────────────────────────────────────────
ARTICLE = "arXiv:2307.02299"
ARTICLE_URL = "https://arxiv.org/abs/2307.02299"
ARTICLE_AUTHOR = "Berthommier, 2023"
ARTICLE_TITLE = (
    "Why can big.bi be changed to bi.gbi? "
    "A mathematical model of syllabification and articulatory synthesis"
)


def get_output_dir(simulation_name: str) -> Path:
    """Get the output directory for a given simulation name."""
    d = OUTPUT_BASE / simulation_name
    d.mkdir(parents=True, exist_ok=True)
    return d
