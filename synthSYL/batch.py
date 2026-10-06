# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
batch.py — Batch processing for TIMIT parameter generation.

Processes an input file (format: 'index: text with |') and generates
Maeda parameter files (.npz) for each block.

Refactored from the original ``batch_generate_timit_params.py`` with
proper package imports and optional smoothing/plotting.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from .pipeline import panphon_pipeline

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

MAEDA_LABELS = ("J", "B", "D", "T", "LH", "LP", "Hy")


def plot_Pv(Pv: np.ndarray, output_dir: str, idx: int,
            labels: tuple[str, ...] = MAEDA_LABELS) -> None:
    """Plot Maeda parameters and save as PNG."""
    if not HAS_MPL:
        return
    T, N = Pv.shape
    t = np.arange(T)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(N):
        ax.plot(t, Pv[:, i], label=labels[i], linewidth=1.5)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Parameter value")
    ax.set_title(f"Maeda Parameters – Block {idx:04d}")
    ax.legend(loc="upper right", ncol=4)
    ax.grid(True, alpha=0.3)
    out_file = os.path.join(output_dir, f"params_{idx:04d}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"✓ Plot saved: {out_file}")


# ═══════════════════════════════════════════════════════════════════════
# Smoothing
# ═══════════════════════════════════════════════════════════════════════

def smooth_pval(Pval: np.ndarray, fc: float = 12.0,
               fs: float = 100.0, order: int = 4) -> np.ndarray:
    """Low-pass filter Pval (requires scipy)."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for smoothing")
    nyquist = 0.5 * fs
    cutoff_freq = fc / nyquist
    b, a = butter(order, cutoff_freq, btype="low")
    Pval_smooth = np.zeros_like(Pval)
    for i in range(Pval.shape[1]):
        Pval_smooth[:, i] = filtfilt(b, a, Pval[:, i])
    return Pval_smooth


# Backward-compatible alias (deprecated)
lisser_pval = smooth_pval


def smooth_envelope(envelope: np.ndarray, fc: float = 50.0,
                    fs: float = 20000.0, order: int = 4) -> np.ndarray:
    """Low-pass filter the envelope (requires scipy)."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for smoothing")
    nyquist = 0.5 * fs
    wc = fc / nyquist
    b, a = butter(order, wc, btype="low")
    return filtfilt(b, a, envelope)


# Backward-compatible alias (deprecated)
lisser_envelope = smooth_envelope


# ═══════════════════════════════════════════════════════════════════════
# Batch processing
# ═══════════════════════════════════════════════════════════════════════

def generate_params_from_file(
    input_file: str,
    output_dir: str = "maeda_params",
    T: int = 16,
    pause_duration_factor: float = 25.0,
    verbose: bool = True,
    plot: bool = False,
) -> None:
    """Process an input file and generate Maeda parameter files.

    Parameters
    ----------
    input_file : str
        Path to the input file (format: 'index: text with |').
    output_dir : str
        Output directory for .npz files.
    T : int
        Base Maeda time step.
    pause_duration_factor : float
        Duration factor for pauses (T_pause = T * factor).
    verbose : bool
        Print progress information.
    plot : bool
        Generate parameter plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        if ":" not in line:
            continue
        idx_str, text = line.split(":", 1)
        idx = int(idx_str.strip())
        text = text.strip()
        if not text:
            print(f"⚠️ Line {idx}: empty string, skipped.")
            continue

        print(f"Processing block {idx}: {text[:50]}...")

        result = panphon_pipeline(
            text,
            T=T,
            long_pause_duration_factor=pause_duration_factor,
            verbose=verbose,
        )

        if result is None:
            print(f"⚠️ Block {idx}: pipeline failed.")
            continue

        Pval = result.Pval
        envelope = result.envelope

        out_file = os.path.join(output_dir, f"params_{idx:04d}.npz")
        np.savez(out_file, Pval=Pval, envelope=envelope)
        print(f"✓ Block {idx} saved to {out_file}")

        if plot:
            plot_Pv(Pval, output_dir, idx)


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Command-line entry point for batch parameter generation."""
    parser = argparse.ArgumentParser(
        description="Generate Maeda parameters from TIMIT text files."
    )
    parser.add_argument("input_file", help="Input file (format: 'index: text with |')")
    parser.add_argument("--output_dir", default="maeda_params", help="Output directory")
    parser.add_argument("--T", type=int, default=16, help="Base Maeda time step")
    parser.add_argument("--pause_duration_factor", type=float, default=25.0,
                        help="Pause duration factor (T_pause = T * factor)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Generate parameter plots")
    args = parser.parse_args()

    generate_params_from_file(
        input_file=args.input_file,
        output_dir=args.output_dir,
        T=args.T,
        pause_duration_factor=args.pause_duration_factor,
        verbose=args.verbose,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
