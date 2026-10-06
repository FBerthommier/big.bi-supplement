# SPDX-License-Identifier: MIT
"""
polar_primitives.py — Polar arc and stationary point primitives.

Standalone complex-plane interpolation functions used to reconstruct
the polar display branches z_v (vocalic) and z_c (consonantal).

Curvature convention (arXiv:2307.02299 §2.1): K = 30 for vowel arcs
(z_v, "Kvoy") and K = 10 for consonant arcs (z_c). The default below
is the vocalic value; consonant callers pass K=10 explicitly.

Reference:
  - arXiv:2307.02299 (Berthommier 2023, Eq. 2)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def polar_arc(
    rho1: float, theta1: float,
    rho2: float, theta2: float,
    duration_ms: float, sr: float = 100.0,
    K: float = 30.0, nu: int = 1,
    orientation: str = "direct",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a polar arc trajectory in the complex plane.

    Interpolates from (rho1, theta1) to (rho2, theta2) along a curved
    path: the blend factor follows cos^2(th/2) over a half-turn sweep,
    and the end point carries an extra phase nu/K (Berthommier 2023,
    Eq. 2; covtl-pipeline reference model).

    Parameters
    ----------
    rho1, theta1 : float
        Polar coordinates of the arc start point.
    rho2, theta2 : float
        Polar coordinates of the arc end point; theta2 is unwrapped
        so that the arc takes the shortest angular path.
    duration_ms : float
        Total duration of the arc (ms).
    sr : float
        Sampling rate of the output trajectory (Hz). Default: 100.
    K : float
        Curvature constant (rad): the end-point phase offset is nu/K.
        Larger K = straighter path. Default: 30 (vocalic display,
        "Kvoy"); consonant arcs use K=10 (arXiv:2307.02299 §2.1).
    nu : int
        Number of turns of the end-point phase offset (offset = nu/K).
        Default: 1.
    orientation : str
        Sweep direction: 'direct' (0 -> pi) or 'inverse' (pi -> 0).
        Default: 'direct'.

    Returns
    -------
    z : np.ndarray of complex
        Complex trajectory z(t) = rho(t) * exp(i * theta(t)).
    rho_theta : np.ndarray (n, 2)
        The same trajectory as (rho, theta) columns.
    """
    n_samples = max(1, int(round(duration_ms * sr / 1000.0)))
    t = np.linspace(0, 1, n_samples, endpoint=True)

    if orientation == "inverse":
        th = np.linspace(np.pi, 0, n_samples, endpoint=True)
    else:
        th = np.linspace(0, np.pi, n_samples, endpoint=True)

    # Unwrap theta to handle wrapping
    th_uw = np.unwrap([theta1, theta2])
    theta2_uw = th_uw[1]
    if abs(theta2_uw - theta1) < abs(theta2 - theta1):
        theta2_eff = theta2_uw
    else:
        theta2_eff = theta2

    # Blend in complex plane
    rk = np.cos(th / 2) ** 2
    phase2 = theta2_eff + (nu / K) * th
    z1 = rho1 * np.exp(1j * theta1)
    z2 = rho2 * np.exp(1j * phase2)
    z = (1 - rk) * z1 + rk * z2

    rho_theta = np.column_stack([np.abs(z), np.angle(z)])
    return z, rho_theta


def stationary_point(
    rho: float, theta: float,
    duration_ms: float, sr: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a stationary point trajectory (constant rho, theta).

    Parameters
    ----------
    rho, theta : float
        Polar coordinates of the stationary point.
    duration_ms : float
        Duration of the stationary segment (ms).
    sr : float
        Sampling rate (Hz). Default: 100.

    Returns
    -------
    z : np.ndarray of complex
        Constant complex trajectory.
    rho_theta : np.ndarray (n, 2)
        Constant (rho, theta) array.
    """
    n_samples = max(1, int(round(duration_ms * sr / 1000.0)))
    z = np.full(n_samples, rho * np.exp(1j * theta), dtype=complex)
    rho_theta = np.tile([rho, theta], (n_samples, 1))
    return z, rho_theta
