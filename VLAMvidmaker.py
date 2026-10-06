# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
VLAMvidmaker -- Generate VLAM videos from Maeda parameter files
================================================================

Reads every ``params_*.npz`` file from a directory (each containing
``Pval`` and ``envelope``), generates a sagittal-contour video per file
using :mod:`vlam`, then downsamples to 25 fps and dubs the corresponding
audio track (``synth_XXXX.wav`` from the audio directory).

This script is the video companion of :mod:`vlam` and
:mod:`batch_synthesize`. The three modules compose as follows::

    params_XXXX.npz  ──►  batch_synthesize.py  ──►  synth_XXXX.wav
          │
          └──────────►  VLAMvidmaker.py  ──►  vlc_XXXX.avi
                                                │
                                                ▼
                                  (25 fps + audio dubbing)
                                                │
                                                ▼
                                    vlc_XXXX_25fps_dubbed.mp4

Integration notes
-----------------
- Uses the **typed** API of :mod:`vlam` (``VlamState``, ``SynthConfig``,
  ``compute_vlam_geometry``, ``plot_vocal_tract``). The legacy API
  (``initVLAMLength``, ``vlam2009NN``, ``showgui``) is no longer exposed
  by ``vlam`` v3.x; this script was migrated accordingly.
- For video we only need the **geometry** of the vocal tract (sagittal
  contour), not the spectrum or formants. We therefore call
  :func:`vlam.compute_vlam_geometry` directly, skipping the expensive
  Newton-Raphson formant search. This is roughly 10x faster than calling
  :func:`vlam.evaluate_frame` per frame.
- The ``--ffmpeg`` option defaults to ``"ffmpeg"`` (looked up via
  ``$PATH``); pass a full path on Windows, e.g.
  ``--ffmpeg P:/ffmpeg/ffmpeg.exe``.

Usage
-----

.. code-block:: bash

    python VLAMvidmaker.py \\
        --params-dir maeda_params \\
        --audio-dir synth_wavs \\
        --out-dir videos

    # Generate only the raw .avi at 100 fps (no audio, no 25 fps conversion):
    python VLAMvidmaker.py --params-dir maeda_params --out-dir videos --no-dub

    # Limit to the first 3 files (smoke test):
    python VLAMvidmaker.py --params-dir maeda_params --out-dir videos --limit 3

Prerequisites
-------------
- ``vlam.py`` (companion module, in the same directory)
- ``numpy``, ``matplotlib``, ``opencv-python`` (cv2)
- ``ffmpeg`` (optional, only for 25 fps conversion + dubbing)
"""

from __future__ import annotations

import os
import sys
import glob
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np

# -----------------------------------------------------------------------------
# Optional dependencies: matplotlib and cv2 are required for video rendering.
# Fail loudly with an actionable message instead of an opaque ImportError.
# -----------------------------------------------------------------------------
try:
    import matplotlib
    # Force a non-interactive backend for headless video generation.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for VLAMvidmaker. "
        "Install it via: pip install matplotlib"
    ) from exc

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python (cv2) is required for VLAMvidmaker. "
        "Install it via: pip install opencv-python"
    ) from exc

# -----------------------------------------------------------------------------
# Import vlam (single-file module) -- must be alongside this script.
# -----------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vlam
from vlam import VlamState, compute_vlam_geometry, plot_vocal_tract


# =============================================================================
# Constants
# =============================================================================
FS_AUDIO_HZ: int = 20000      # Audio sampling rate (Hz) -- must match vlam.SynthConfig
FPS_DEFAULT: float = 100.0    # Output video frame rate (matches the original)
FPS_OUTPUT: float = 25.0      # Final frame rate after downsampling


# =============================================================================
# Light-weight per-frame evaluation (geometry only)
# =============================================================================

def freqevalNN_visual(prm_vector: np.ndarray,
                      state: VlamState,
                      valrect: float) -> VlamState:
    """Light articulatory evaluation for video rendering.

    Optimized variant of :func:`vlam.evaluate_frame` that computes **only**
    the vocal-tract geometry (sagittal contour + area function). The
    spectrum and formants are not computed -- they are not needed for
    visualization and account for >90% of the runtime of the full
    evaluation.

    Parameters
    ----------
    prm_vector : np.ndarray
        Vector of 7 Maeda parameters for a single frame.
    state : VlamState
        Current articulatory state (will be copied, not mutated).
    valrect : float
        Soft-rectification parameter. Unused here (the area function is
        not post-processed for video), but kept for API compatibility
        with :func:`vlam.evaluate_frame`.

    Returns
    -------
    VlamState
        New articulatory state with ``sagittal`` and ``area`` populated.
    """
    del valrect  # unused: kept for API symmetry with vlam.evaluate_frame
    prm_vector = np.asarray(prm_vector, dtype=float).flatten()
    if prm_vector.shape[0] != 7:
        raise ValueError(
            f"prm_vector must have 7 elements (got {prm_vector.shape[0]})."
        )
    # Defensive copy + update of the first 7 parameters.
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
    new_state.params[:7] = prm_vector
    # VLAM geometric computation (sagittal contour + area function).
    # The LipP/LipH swap to the internal vlam2009NN order is handled
    # automatically inside this function -- no caller-side swap needed.
    return compute_vlam_geometry(new_state)


# =============================================================================
# Video generation
# =============================================================================

# Default figure size in inches at 100 dpi -> 600x400 pixels.
# Override via the ``figsize`` argument of :func:`playVLAMvid_fast` or the
# ``--fig-size`` CLI flag.
DEFAULT_FIG_SIZE: tuple = (6.0, 4.0)


def playVLAMvid_fast(output_video: Union[str, Path],
                     fps: float,
                     gui: VlamState,
                     Pval: np.ndarray,
                     valrect: float,
                     show_progress: bool = True,
                     progress_every: int = 50,
                     figsize: tuple = DEFAULT_FIG_SIZE) -> None:
    """Render a sagittal-contour ``.avi`` video from an articulatory trajectory.

    For each frame of ``Pval`` this function:
      1. Computes the vocal-tract geometry via :func:`freqevalNN_visual`.
      2. Draws the sagittal contour with :func:`vlam.plot_vocal_tract`,
         using a freshly created figure of size ``figsize`` inches.
      3. Captures the matplotlib figure as a numpy array (RGBA -> BGR).
      4. Writes the image into an OpenCV video file (XVID codec).

    Parameters
    ----------
    output_video : str or Path
        Path of the output ``.avi`` file.
    fps : float
        Frame rate of the generated video.
    gui : VlamState
        Initial articulatory state (typed equivalent of the legacy ``gui``
        dict).
    Pval : np.ndarray
        Matrix of Maeda parameters, shape (T, 7).
    valrect : float
        VLAM soft-rectification parameter (forwarded to
        :func:`freqevalNN_visual`; unused for geometry-only rendering but
        kept for API symmetry).
    show_progress : bool
        If True, prints progress every ``progress_every`` frames.
    progress_every : int
        Frame interval between progress prints.
    figsize : tuple of float
        Figure size in inches ``(width, height)``. Combined with the
        fixed 100 dpi rendering, this gives the video resolution in
        pixels: ``width_px = int(figsize[0] * 100)``,
        ``height_px = int(figsize[1] * 100)``. Default: ``(6.0, 4.0)``
        -> 600x400 pixels.

    Raises
    ------
    ValueError
        If ``Pval`` is empty.
    """
    if Pval is None or len(Pval) == 0:
        raise ValueError("Pval is empty. Nothing to render.")

    print(f"Pre-computing states for {len(Pval)} frames...")
    states: list = []
    current_state = gui
    for k in range(len(Pval)):
        current_state = freqevalNN_visual(Pval[k, :], current_state, valrect)
        states.append(current_state)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer: Optional[cv2.VideoWriter] = None

    print("Generating frames...")
    for idx, state in enumerate(states):
        # Create the figure explicitly with the requested size so the
        # video resolution is deterministic. ``plot_vocal_tract`` will
        # reuse the provided ``ax`` instead of allocating its own
        # 10x4 figure (which would yield 1000x400 pixels).
        fig, ax = plt.subplots(figsize=figsize)
        try:
            plot_vocal_tract(state, ax=ax)
            fig.set_dpi(100)
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            img_array = np.array(buf, dtype=np.uint8)
            height, width = img_array.shape[:2]
            img_bgr = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)

            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    str(output_video), fourcc, float(fps), (width, height)
                )
                print(f"Video initialized: {width}x{height} @ {fps} fps")

            video_writer.write(img_bgr)
        finally:
            plt.close(fig)

        if show_progress and (idx + 1) % progress_every == 0:
            print(f"Progress: {idx + 1}/{len(states)}")

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved: {output_video}")


# =============================================================================
# Utility functions
# =============================================================================

def read_params_npz(path: Union[str, Path]) -> np.ndarray:
    """Read a ``.npz`` file containing a ``Pval`` key of shape (N, 7).

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file.

    Returns
    -------
    np.ndarray
        Maeda parameter matrix, shape (N, 7), float64.

    Raises
    ------
    ValueError
        If ``Pval`` is not of shape (N, 7).
    """
    data = np.load(path)
    Pval = data['Pval']
    if Pval.ndim != 2 or Pval.shape[1] != 7:
        raise ValueError(
            f"File {path} must contain an array of shape (N, 7). "
            f"Got {Pval.shape}."
        )
    return Pval.astype(float)


def resample_to_fs(data: np.ndarray,
                   fs_in: float,
                   fs_out: float) -> np.ndarray:
    """Resample a parameter trajectory to a new frame rate.

    Uses column-wise linear interpolation.

    Parameters
    ----------
    data : np.ndarray
        Input matrix, shape (N_in, M).
    fs_in : float
        Input frame rate (Hz).
    fs_out : float
        Output frame rate (Hz).

    Returns
    -------
    np.ndarray
        Resampled matrix, shape (N_out, M).

    Raises
    ------
    ValueError
        If ``N_out < 1`` after resampling.
    """
    if fs_in == fs_out:
        return data.copy()
    N_in = data.shape[0]
    N_out = int(round(N_in * fs_out / fs_in))
    if N_out < 1:
        raise ValueError("N_out < 1 after resampling")
    t_in = np.linspace(0.0, 1.0, N_in)
    t_out = np.linspace(0.0, 1.0, N_out)
    data_out = np.empty((N_out, data.shape[1]), dtype=data.dtype)
    for c in range(data.shape[1]):
        data_out[:, c] = np.interp(t_out, t_in, data[:, c])
    return data_out


# =============================================================================
# 25 fps downsampling + audio dubbing (ffmpeg)
# =============================================================================

def convert_to_25fps_and_dub(video_path: Union[str, Path],
                             audio_path: Union[str, Path],
                             output_dir: Union[str, Path],
                             ffmpeg_path: str = 'ffmpeg',
                             fps_out: float = FPS_OUTPUT) -> Optional[Path]:
    """Downsample a video to ``fps_out`` fps and dub in the audio track.

    Steps:
      1. Convert the source video to ``fps_out`` fps (``-filter:v fps=...``).
      2. Mux the audio track (``-map 0:v:0 -map 1:a:0 -shortest``).

    Parameters
    ----------
    video_path : str or Path
        Path to the source video (typically a 100 fps ``.avi``).
    audio_path : str or Path
        Path to the audio file (``.wav``) to synchronize.
    output_dir : str or Path
        Output directory for converted files.
    ffmpeg_path : str
        Path to the ffmpeg executable. Default: ``'ffmpeg'`` (looked up
        via ``$PATH``). On Windows, pass the full path, e.g.
        ``'P:/ffmpeg/ffmpeg.exe'``.
    fps_out : float
        Output frame rate, default 25 fps.

    Returns
    -------
    Path or None
        Path to the final file (video + audio), or None on failure.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return None

    # Locate the ffmpeg executable: either an explicit path that exists,
    # or a name resolved through ``$PATH`` via shutil.which().
    if os.path.exists(ffmpeg_path):
        ffmpeg_exe = ffmpeg_path
    else:
        ffmpeg_exe = shutil.which(ffmpeg_path)
        if ffmpeg_exe is None:
            print(
                f"ffmpeg not found at '{ffmpeg_path}'. "
                "Please install ffmpeg or specify a valid path."
            )
            return None

    base = video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Frame-rate conversion to fps_out
    video_conv = output_dir / f"{base}_{int(fps_out)}fps.mp4"
    cmd_fps = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-filter:v", f"fps={fps_out}",
        "-c:a", "copy",
        str(video_conv),
    ]
    try:
        subprocess.run(cmd_fps, check=True, capture_output=True)
        print(f"{int(fps_out)} fps video created: {video_conv}")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        print(f"ffmpeg error (frame-rate conversion): {stderr}")
        return None

    # 2. Dub the audio track
    dubbed_video = output_dir / f"{base}_{int(fps_out)}fps_dubbed.mp4"
    cmd_dub = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_conv),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(dubbed_video),
    ]
    try:
        subprocess.run(cmd_dub, check=True, capture_output=True)
        print(f"Dubbed video created: {dubbed_video}")
        return dubbed_video
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        print(f"ffmpeg error (dubbing): {stderr}")
        return None


# =============================================================================
# CLI entry point
# =============================================================================

def main() -> None:
    """CLI entry point of the VLAM video maker."""
    parser = argparse.ArgumentParser(
        description="VLAM video maker from Maeda parameter files"
    )
    parser.add_argument("--params-dir", type=str, default="maeda_params",
                        help="Directory containing params_XXXX.npz files")
    parser.add_argument("--audio-dir", type=str, default="synth_wavs",
                        help="Directory containing synth_XXXX.wav files")
    parser.add_argument("--out-dir", type=str, default="videos",
                        help="Output directory for the videos")
    parser.add_argument("--fs-in", type=float, default=100.0,
                        help="Input frame rate of the parameters (Hz)")
    parser.add_argument("--fs-out", type=float, default=100.0,
                        help="Output frame rate for the video (fps)")
    parser.add_argument("--valrect", type=float, default=0.75,
                        help="VLAM soft-rectification parameter")
    parser.add_argument("--gui-len", type=int, default=195,
                        help="Vocal-tract length (mm) for VlamState.initial")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg",
                        help="Path to the ffmpeg executable "
                             "(default: 'ffmpeg' resolved via $PATH; "
                             "on Windows use e.g. 'P:/ffmpeg/ffmpeg.exe')")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of files to process (for testing)")
    parser.add_argument("--no-dub", action="store_true",
                        help="Generate only the .avi at --fs-out fps, "
                             "skip 25 fps conversion and audio dubbing")
    parser.add_argument("--fig-size", type=float, nargs=2,
                        default=list(DEFAULT_FIG_SIZE),
                        metavar=("WIDTH", "HEIGHT"),
                        help="Figure size in inches at 100 dpi -> video "
                             "resolution in pixels. "
                             f"Default: {DEFAULT_FIG_SIZE[0]} {DEFAULT_FIG_SIZE[1]} "
                             f"({int(DEFAULT_FIG_SIZE[0]*100)}x"
                             f"{int(DEFAULT_FIG_SIZE[1]*100)} px). "
                             "Use e.g. --fig-size 10 4 for the legacy "
                             "1000x400 resolution.")
    args = parser.parse_args()

    figsize = tuple(args.fig_size)
    if figsize[0] <= 0 or figsize[1] <= 0:
        parser.error(f"--fig-size values must be > 0, got {figsize}")

    params_dir = Path(args.params_dir)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # List the .npz files in params_dir, sorted for deterministic ordering.
    npz_files = sorted(glob.glob(str(params_dir / "params_*.npz")))
    if not npz_files:
        print(f"No .npz file found in {params_dir}")
        return

    print(f"Found {len(npz_files)} parameter files.")

    if args.limit is not None:
        npz_files = npz_files[:args.limit]

    # Initialize the base VLAM state once (typed equivalent of
    # ``initVLAMLength(np.zeros(7), L)`` in the legacy API).
    state_base = VlamState.initial(vocal_tract_length_mm=args.gui_len)

    for npz_path in npz_files:
        basename = Path(npz_path).stem
        idx_str = basename.split('_')[1]
        idx = int(idx_str)

        print(f"\n=== Processing sentence {idx:04d} ===")

        try:
            Pval = read_params_npz(npz_path)
        except Exception as e:
            print(f"Error reading {npz_path}: {e}")
            continue

        if args.fs_in != args.fs_out:
            Pval = resample_to_fs(Pval, args.fs_in, args.fs_out)

        if np.any(np.isnan(Pval)) or np.any(np.isinf(Pval)):
            print("Pval contains NaN/inf values. Cleaning up...")
            Pval = np.nan_to_num(Pval, nan=0.0, posinf=0.0, neginf=0.0)

        # Generate the video at args.fs_out fps.
        video_filename = f"vlc_{idx:04d}.avi"
        video_path = out_dir / video_filename

        try:
            playVLAMvid_fast(
                str(video_path), args.fs_out,
                state_base, Pval, args.valrect,
                figsize=figsize,
            )
        except Exception as e:
            print(f"Error during video generation: {e}")
            continue

        if args.no_dub:
            print(f"Video generated (no dubbing): {video_path}")
            continue

        # Locate the matching audio file produced by batch_synthesize.py.
        audio_path = audio_dir / f"synth_{idx:04d}.wav"
        if not audio_path.exists():
            print(f"Warning: audio file {audio_path} missing.")
            continue

        # 25 fps downsampling + audio dubbing.
        final_video = convert_to_25fps_and_dub(
            video_path, audio_path, out_dir, ffmpeg_path=args.ffmpeg
        )
        if final_video is not None:
            print(f"OK - Final file: {final_video}")
        else:
            print(f"Conversion/dubbing failed for sentence {idx:04d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
