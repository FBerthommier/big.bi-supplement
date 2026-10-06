# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
polar.py — Polar coordinate projection primitives.

Contains the Maeda model projection functions (compute_P, arc_B,
sigmoid_transition) and the background trajectory builder.

No scientific change: all equations and numerical values identical to v15g.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from .constants import CO


def compute_P(rho: float, theta: float) -> np.ndarray:
    """Project polar target (rho, theta) into Maeda parameter space.

    P(p) = CO[p, 1] + rho * CO[p, 0] * cos(CO[p, 2] - theta)
    Applied to all 7 parameters.
    """
    return CO[:, 1] + rho * CO[:, 0] * np.cos(CO[:, 2] - theta)


def arc_B(pt_dep: List[float], pt_arr: List[float],
          params: List[int], D: int,
          thetabounds: List[float], opint: int,
          nu: int, K: int, Pexp: int = 2) -> np.ndarray:
    """Compute an arc trajectory in Maeda parameter space.

    v14: returns (D, len(params)) like v11, unwrap restored (FIX-7).

    Parameters:
        pt_dep: [rho, theta] departure polar point
        pt_arr: [rho, theta] arrival polar point
        params: list of active Maeda parameter indices
        D: number of trajectory steps
        thetabounds: [theta_start, theta_end] for the interpolation
        opint: interpolation mode (0=linspace, -1=skip first, else=linspace)
        nu, K, Pexp: arc model parameters

    Returns:
        array (D, len(params)) of Maeda parameter values
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
            Pt[k, j] = (CO[p, 1]
                        + rk * pa[0] * CO[p, 0] * np.cos(CO[p, 2] - pa[1] - (nu / K) * th[k])
                        + (1 - rk) * pd[0] * CO[p, 0] * np.cos(CO[p, 2] - pd[1]))
    return Pt


def sigmoid_transition(pt_dep: List[float], pt_arr: List[float],
                       D: int, steepness: float = 1.0) -> np.ndarray:
    """Compute a sigmoid transition between two polar points.

    Parameters:
        pt_dep: [rho, theta] departure point
        pt_arr: [rho, theta] arrival point
        D: number of steps
        steepness: sigmoid steepness (0=linear, large=sharp)

    Returns:
        array (D, 2) of [rho, theta] values
    """
    t = np.linspace(0, 1, D)
    if steepness == 0:
        f = t
    else:
        u = 2*t - 1
        denom = np.tanh(steepness)
        if abs(denom) < 1e-12:
            f = t
        else:
            f = 0.5 + 0.5 * np.tanh(steepness * u) / denom
            f[0] = 0.0; f[-1] = 1.0
    dep = np.array(pt_dep); arr = np.array(pt_arr)
    traj = np.zeros((D, len(dep)))
    for i in range(len(dep)):
        traj[:, i] = dep[i] + (arr[i] - dep[i]) * f
    return traj


def build_background(pt_dep: List[float], pt_arr: List[float],
                     D: int, nu: int, Kvoy: int, Pexp: int) -> np.ndarray:
    """Build background trajectory (all 7 parameters, full arc)."""
    return arc_B(pt_dep, pt_arr, list(range(7)), D,
                 [-np.pi, 0], 0, nu, Kvoy, Pexp)


def inject_active_parameters(Pval: np.ndarray, block_slice: slice,
                             pt_dep: List[float], pt_arr: List[float],
                             active_params: List[int], T: int,
                             nu: int, K: int, Pexp: int,
                             thetabounds: List[float], opint: int) -> None:
    """Inject active parameter trajectory into a Pval block.

    Modifies Pval in-place.
    """
    if not active_params:
        return
    seg = arc_B(pt_dep, pt_arr, active_params, T,
                thetabounds, opint, nu, K, Pexp)
    Pval[block_slice, active_params] = seg
