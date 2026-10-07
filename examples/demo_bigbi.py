#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
demo_bigbi.py — Minimal demo: big.bi → bi.gbi transformation.

Shows the verbal transformation from big.bi (salient schwa @) to
bi.gbi (fused /gb/ cluster) by varying delta from 0.50 to 1.00.

Usage:
    python examples/demo_bigbi.py
"""
import sys
import os
import numpy as np
from pathlib import Path

# Make synthSYL and vlam importable
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from synthSYL import panphon_pipeline
import vlam
from scipy.io.wavfile import write

# ── Configuration ──
T = 16  # Maeda time step (didactic mode)
FS = 20000  # Sample rate (Hz)
OUT_DIR = REPO / "output" / "demo_bigbi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sweep delta from 0.50 to 1.00 ──
DELTA_VALUES = [0.50, 0.70, 0.85, 0.95, 1.00]
INPUT_TEXTS = ["big bi", "big bi", "big bi", "big bi", "bi.gbi"]

# ── Synthesizer setup ──
state = vlam.VlamState.initial(195)
config = vlam.SynthConfig(play_audio=False)


def run_one(delta: float, input_text: str, idx: int):
    """Run pipeline + synth for one delta value."""
    # Tp coupled to delta: Tp = 160 * (1 - delta) / 0.5
    if delta >= 1.0:
        factor = 0.0
    else:
        factor = (160.0 / (T * 10)) * (1.0 - delta) / (1.0 - 0.50)

    result = panphon_pipeline(
        input_text,
        T=T,
        short_pause_duration_factor=factor,
        long_pause_duration_factor=max(factor, 1.0),
        delta_o=delta, delta_e=delta,
        verbose=False,
    )
    if result is None:
        print(f"  [delta={delta}] Pipeline failed")
        return

    # Synthesize audio
    sr = vlam.synthwordfen(
        state=state,
        articulatory_params=result.Pval,
        word_tokens=["O", "V", "F"],
        f0_scale=1.0, soft_rect_s=1.10,
        duration_factor=T, envelope=result.envelope, config=config,
    )
    sig = sr.signal
    max_val = float(np.max(np.abs(sig))) if np.max(np.abs(sig)) > 1e-12 else 1.0
    sig_int16 = np.int16(32767 * sig / (1.01 * max_val))

    # Save WAV
    tag = f"d{int(round(delta * 100)):03d}"
    wav_path = OUT_DIR / f"synth_{tag}.wav"
    write(str(wav_path), FS, sig_int16)

    # Ve rho
    ve_rho = delta * 0.9  # rho_V = 0.9 for /i/
    form = "bi.gbi (fused /gb/)" if delta >= 1.0 else f"big.bi (Ve rho={ve_rho:.3f})"
    print(f"  [delta={delta:.2f}] {input_text:8s}  "
          f"Pval={result.Pval.shape}  "
          f"signal={sig.shape[0]/FS*1000:.0f} ms  "
          f"Ve_rho={ve_rho:.3f}  -> {wav_path.name}  ({form})")


def main():
    print("=" * 60)
    print("DEMO: big.bi -> bi.gbi (T=16, delta 0.50 -> 1.00)")
    print("=" * 60)
    print(f"Output: {OUT_DIR}")
    print()

    for i, (delta, input_text) in enumerate(zip(DELTA_VALUES, INPUT_TEXTS)):
        run_one(delta, input_text, i)

    print()
    print("Done. Listen to the WAV files to hear the transformation:")
    print("  - d050: big@bi (salient schwa @)")
    print("  - d070: big.bi (audible schwa)")
    print("  - d085: big.bi (residual schwa)")
    print("  - d095: big.bi (Vo approx Ve approx V)")
    print("  - d100: bi.gbi (fused /gb/ cluster)")


if __name__ == "__main__":
    main()
