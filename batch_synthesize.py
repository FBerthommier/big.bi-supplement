# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
batch_synthesize -- Batch WAV synthesis from Maeda parameter files
==================================================================

Migrated version of ``batch_synthesize.py`` -- uses the typed API of the
single-file :mod:`vlam` module while preserving the exact behavior of the
original script.

Key differences from the original
---------------------------------
- ``import syntSYL``  ->  ``import vlam``
- ``syntSYL.initVLAMLength(np.zeros(7), 195)``  ->
  ``vlam.VlamState.initial(195)``
  (The 7 articulatory parameters default to zero, as in the original;
  no need to pass them explicitly.)
- ``syntSYL.synthwordfen(gui, Pval, ...)``  ->
  ``vlam.synthwordfen(state, Pval, ...)``
  **Warning**: the new signature uses keyword arguments and returns a
  :class:`vlam.SynthResult` instead of a ``(sig, fval)`` tuple.
- ``syntSYL.play = lambda x: None``  ->
  ``vlam.SynthConfig(play_audio=False)``
  (No need to neutralize ``play`` -- it is disabled by default.)

Incremental migration procedure
--------------------------------
1. Keep the original ``batch_synthesize.py`` (step 1) until you are
   confident in the new API.
2. To test the migration, copy this file under ``batch_synthesize_v2.py``
   and run it in parallel.
3. Compare the WAVs produced by the two versions: they must be
   sample-identical (same random seeds, same constants).

Usage
-----
.. code-block:: bash

    python batch_synthesize.py maeda_params/ --output_dir synth_wavs
"""

from __future__ import annotations

import os
import sys
import glob
import argparse

import numpy as np
from scipy.io.wavfile import write

# =============================================================================
# Import vlam (single-file module) -- no config/acoustics/etc. submodules.
# =============================================================================
import vlam


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Batch synthesis from .npz files (Maeda parameters) -- '
                    'typed API version'
    )
    parser.add_argument('input_dir',
                        help='Directory containing the params_*.npz files')
    parser.add_argument('--output_dir', default='wav_output',
                        help='Output directory for the .wav files')
    parser.add_argument('--dur', type=int, default=16,
                        help='Duration factor T (Maeda) (default: 16)')
    parser.add_argument('--valrect', type=float, default=1.10,
                        help='VLAM soft-rectification factor (default: 1.10)')
    parser.add_argument('--cf0', type=float, default=1,
                        help='F0 scaling factor (default: 1)')
    parser.add_argument('--word', type=str, default='O V F',
                        help='Dummy tokens for synthfen (default: "O V F")')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Synthesizer configuration ----
    # SynthConfig replaces the scattered constants of the old syntSYL.
    # play_audio=False by default -> no need to neutralize play().
    config = vlam.SynthConfig(play_audio=False)

    # ---- Initial articulatory state ----
    # NEW SIGNATURE: VlamState.initial(vocal_tract_length_mm)
    # The 7 articulatory parameters are zero by default (as in the
    # original np.zeros(7)). No need to pass them here.
    # For a non-neutral initial state, you would do:
    #   state = vlam.VlamState.initial(195)
    #   state.params[:7] = my_ap_vector
    state = vlam.VlamState.initial(195)

    # List the .npz files sorted by index for deterministic ordering.
    files = sorted(glob.glob(os.path.join(args.input_dir, 'params_*.npz')))
    if not files:
        print(f"Error: no 'params_*.npz' file found in {args.input_dir}")
        sys.exit(1)

    print(f"{len(files)} files found.")

    word_tokens = args.word.split()

    for fname in files:
        basename = os.path.basename(fname)
        idx = int(basename.split('_')[1].split('.')[0])

        # Load the data.
        data = np.load(fname)
        # Pval is stored in the canonical Maeda order
        # [J, B, D, T, LP, LH, Hy] (idx 4 = LP, idx 5 = LH).
        # NO swap is applied here: the swap to the internal vlam2009NN
        # order (idx 4 = LH, idx 5 = LP) is performed automatically in
        # ``vlam.compute_vlam_geometry``.
        #
        # IMPORTANT: the old batch swap `Pval[:, [4, 5]] = Pval[:, [5, 4]]`
        # was REMOVED. It was a bug that canceled the internal swap of
        # vlam2009NN, causing the _A_LIP matrices to receive
        # [jaw, LP, LH] instead of [jaw, LH, LP] -- thereby confusing
        # LipP and LipH at the synthesis level (see vlam.py v3.2.0).
        Pval = data['Pval']
        envelope = data['envelope']

        # ---- New typed API: synthwordfen ----
        # Returns a SynthResult (dataclass) with .signal, .formants, .spectra
        result = vlam.synthwordfen(
            state=state,
            articulatory_params=Pval,
            word_tokens=word_tokens,
            f0_scale=args.cf0,
            soft_rect_s=args.valrect,
            duration_factor=args.dur,
            envelope=envelope,
            config=config,
        )
        sig = result.signal
        # fval in legacy layout (3, T) for backward compatibility.
        fval = result.formants.T

        # Save the signal as a 16-bit WAV at 20 kHz.
        wavfile = os.path.join(args.output_dir, f'synth_{idx:04d}.wav')
        max_val = np.max(np.abs(sig))
        if max_val < 1e-9:
            max_val = 1.0
        signal_int16 = np.int16(32767 * sig / (1.01 * max_val))
        write(wavfile, 20000, signal_int16)

        print(f'Sentence {idx:04d} -> {wavfile}  '
              f'(F1={fval[0, -1]:.0f} Hz, F2={fval[1, -1]:.0f} Hz, '
              f'F3={fval[2, -1]:.0f} Hz)')


if __name__ == '__main__':
    main()
