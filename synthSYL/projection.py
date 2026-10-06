# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
projection.py — Maeda articulatory model projection primitives.

Pure mathematical functions: compute_P, arc_B, sigmoid_transition.
No side-effects, no global state.
"""

from __future__ import annotations

import numpy as np

from .constants import CO, N_MAEDA_PARAMS


def compute_P(rho: float, theta: float) -> np.ndarray:
    """Project a polar target (rho, theta) to the 7 Maeda parameters.

    P_i = CO[i,1] + rho * CO[i,0] * cos(CO[i,2] - theta)
    """
    return CO[:, 1] + rho * CO[:, 0] * np.cos(CO[:, 2] - theta)


def arc_B(
    pt_dep: list[float] | np.ndarray,
    pt_arr: list[float] | np.ndarray,
    params: list[int] | range,
    D: int,
    thetabounds: list[float],
    opint: int,
    nu: int,
    K: int,
    Pexp: int = 2,
) -> np.ndarray:
    """Compute an arc trajectory between two polar points.

    Parameters
    ----------
    pt_dep : [rho, theta] of the departure point
    pt_arr : [rho, theta] of the arrival point
    params : list of active Maeda parameter indices
    D      : number of time steps
    thetabounds : [theta_start, theta_end] for the arc parameter
    opint  : interpolation mode (0 = inclusive, -1 = exclusive first)
    nu     : coarticulation index
    K      : coarticulation scaling factor
    Pexp   : exponent for the radial weighting function
    """
    if opint == 0:
        th = np.linspace(thetabounds[0], thetabounds[1], D)
    elif opint == -1:
        th1 = np.linspace(thetabounds[0], thetabounds[1], D + 1)
        th = th1[1:D + 1]
    else:
        th = np.linspace(thetabounds[0], thetabounds[1], D)

    pa_theta_uw = np.unwrap([pt_dep[1], pt_arr[1]])[1]
    if abs(pa_theta_uw - pt_dep[1]) < abs(pt_arr[1] - pt_dep[1]):
        pa_theta = pa_theta_uw
    else:
        pa_theta = pt_arr[1]

    pd = np.array(pt_dep, dtype=float)
    pa = np.array([pt_arr[0], pa_theta], dtype=float)
    params_list = list(params)
    Pt = np.zeros((D, len(params_list)))
    for k in range(D):
        rk = np.cos(th[k] / 2) ** Pexp
        for j, p in enumerate(params_list):
            Pt[k, j] = (
                CO[p, 1]
                + rk * pa[0] * CO[p, 0] * np.cos(CO[p, 2] - pa[1] - (nu / K) * th[k])
                + (1 - rk) * pd[0] * CO[p, 0] * np.cos(CO[p, 2] - pd[1])
            )
    return Pt


def sigmoid_transition(
    pt_dep: list[float] | np.ndarray,
    pt_arr: list[float] | np.ndarray,
    D: int,
    steepness: float = 1.0,
) -> np.ndarray:
    """Compute a sigmoid transition between two Maeda parameter vectors.

    Parameters
    ----------
    pt_dep    : departure Maeda parameter vector (7 elements)
    pt_arr    : arrival Maeda parameter vector (7 elements)
    D         : number of time steps
    steepness : steepness of the sigmoid (0 = linear)
    """
    t = np.linspace(0, 1, D)
    if steepness == 0:
        f = t
    else:
        u = 2 * t - 1
        denom = np.tanh(steepness)
        if abs(denom) < 1e-12:
            f = t
        else:
            f = 0.5 + 0.5 * np.tanh(steepness * u) / denom
            f[0] = 0.0
            f[-1] = 1.0
    dep = np.array(pt_dep)
    arr = np.array(pt_arr)
    traj = np.zeros((D, len(dep)))
    for i in range(len(dep)):
        traj[:, i] = dep[i] + (arr[i] - dep[i]) * f
    return traj
