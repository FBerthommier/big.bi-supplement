#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
demo_ibbi.py — Minimal demo: ib ib → bi bi classical transformation.

Shows the U-shaped Tp variation: ib ib (start) → bi bi (fusion at
bottom) → ib ib (end).

Usage:
    python examples/demo_ibbi.py
"""
import sys
import numpy as np
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from synthSYL import panphon_pipeline
import vlam
from scipy.io.wavfile import write

T = 16
FS = 20000
OUT_DIR = REPO / "output" / "demo_ibbi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# U-shaped Tp values (ms): 160, 80, 0, 80, 160
TP_MS_VALUES = [160, 80, 0, 80, 160]
LABELS = ["start (ib ib)", "descending", "bottom (bi bi)", "ascending", "end (ib ib)"]

state = vlam.VlamState.initial(195)
config = vlam.SynthConfig(play_audio=False)


def tp_factor(tp_ms):
    return tp_ms / (T * 10)


def delta_for_tp(tp_ms):
    return 0.5 + 0.5 * (1.0 - tp_ms / 160.0)


def run_one(tp_ms, label, idx):
    factor = tp_factor(tp_ms)
    delta = delta_for_tp(tp_ms)
    result = panphon_pipeline(
        "ib ib",
        T=T,
        short_pause_duration_factor=factor,
        long_pause_duration_factor=max(factor, 1.0),
        delta_o=delta, delta_e=delta,
        verbose=False,
    )
    if result is None:
        return

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

    tag = f"tp{int(tp_ms):03d}"
    wav_path = OUT_DIR / f"synth_{tag}.wav"
    write(str(wav_path), FS, sig_int16)

    ve_rho = delta * 0.9
    print(f"  [{label}] Tp={tp_ms:3d}ms delta={delta:.3f}  "
          f"Pval={result.Pval.shape}  "
          f"signal={sig.shape[0]/FS*1000:.0f}ms  "
          f"Ve_rho={ve_rho:.3f}  -> {wav_path.name}")


def main():
    print("=" * 60)
    print("DEMO: ib ib -> bi bi -> ib bi (U-shaped Tp, T=16)")
    print("=" * 60)
    print(f"Output: {OUT_DIR}")
    print()

    for i, (tp_ms, label) in enumerate(zip(TP_MS_VALUES, LABELS)):
        run_one(tp_ms, label, i)

    print()
    print("Listen to hear: ib ib -> bi bi -> ib ib")
    print("  tp160: ib ib (clear, schwa @)")
    print("  tp080: ib ib (transition)")
    print("  tp000: bi bi (fusion Vo=Ve=V)")
    print("  tp080: ib ib (transition)")
    print("  tp160: ib ib (clear, return)")


if __name__ == "__main__":
    main()
