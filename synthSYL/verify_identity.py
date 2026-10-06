# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
verify_identity.py — Standalone numerical identity verification.

Compares the refactored package output against the original monolithic
plugin for all test cases.  Run with:

    python verify_identity.py

This script must be run from the /home/z/my-project/download/ directory.
"""

import sys
import os
import importlib.util
import numpy as np

# ── Load the original monolithic plugin ──
ORIG_PATH = os.path.join(os.path.dirname(__file__), "..", "upload", "panphon_timit_plugin.py")
if not os.path.exists(ORIG_PATH):
    print(f"ERROR: Original plugin not found at {ORIG_PATH}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("orig_plugin", ORIG_PATH)
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)

# ── Load the refactored package ──
sys.path.insert(0, os.path.dirname(__file__))
from panphon_timit_plugin import panphon_pipeline

# ── Test cases ──
TEST_CASES = [
    ("a i u", "simple_vowels"),
    ("ba di gu", "CV"),
    ("aba idi ugu", "VCV"),
    ("pra kli", "CCV"),
    ("adra ikli", "VCCV"),
    ("pstra", "ternary_cluster"),
    ("ba | di", "pauses"),
    ("bab did guk", "word_end"),
    ("ba.di gu", "syllable_dots"),
    ("ba | di | gu", "long_pauses"),
    ("ma na la", "nasals_liquids"),
    ("fa sa sha", "fricatives"),
    ("va za zha", "voiced_fricatives"),
    ("ja wa", "glides"),
    ("badi gu | pra kli", "mixed"),
]

# ── Polar coordinate test cases ──
POLAR_SEGMENTS = list(orig.PANPHON_DB.keys())

def verify_pipeline_identity():
    """Verify that the refactored pipeline produces identical results."""
    all_pass = True
    for text, name in TEST_CASES:
        orig_result = orig.panphon_pipeline(text, T=5)
        new_result = panphon_pipeline(text, T=5)

        if orig_result is None and new_result is None:
            print(f"  ✓ {name}: both returned None (skipped)")
            continue
        if orig_result is None or new_result is None:
            print(f"  ✗ {name}: one returned None")
            all_pass = False
            continue

        # Compare Pval
        pval_match = np.allclose(new_result.Pval, orig_result["Pval"],
                                  atol=1e-14, rtol=1e-14)
        pval_maxdiff = np.max(np.abs(new_result.Pval - orig_result["Pval"]))

        # Compare envelope
        env_match = np.allclose(new_result.envelope, orig_result["envelope"],
                                 atol=1e-14, rtol=1e-14)
        env_maxdiff = np.max(np.abs(new_result.envelope - orig_result["envelope"]))

        # Compare tokens
        tokens_match = new_result.tokens == orig_result["tokens"]

        # Compare segments
        segs_match = new_result.segments == orig_result["segments"]

        if pval_match and env_match and tokens_match and segs_match:
            print(f"  ✓ {name}: Pval max|Δ|={pval_maxdiff:.2e}, Env max|Δ|={env_maxdiff:.2e}")
        else:
            print(f"  ✗ {name}: Pval match={pval_match} (max|Δ|={pval_maxdiff:.2e}), "
                  f"Env match={env_match} (max|Δ|={env_maxdiff:.2e}), "
                  f"Tokens match={tokens_match}, Segs match={segs_match}")
            all_pass = False

    return all_pass


def verify_polar_identity():
    """Verify that polar coordinate computation is identical for all segments."""
    all_pass = True
    for seg in POLAR_SEGMENTS:
        pv = orig.get_panphon(seg)
        if pv is None:
            continue
        orig_polar = orig.panphon_to_polar(seg, pv)
        from panphon_timit_plugin.phonology import panphon_to_polar
        new_polar = panphon_to_polar(seg, pv)

        # new_polar is a PolarTarget; supports dict-style access
        rho_match = abs(orig_polar["rho"] - new_polar["rho"]) < 1e-14
        theta_match = abs(orig_polar["theta"] - new_polar["theta"]) < 1e-14
        vowel_match = orig_polar["is_vowel"] == new_polar["is_vowel"]
        params_match = orig_polar["params_actifs"] == new_polar["params_actifs"]

        if rho_match and theta_match and vowel_match and params_match:
            pass  # OK
        else:
            print(f"  ✗ Polar '{seg}': rho_match={rho_match}, theta_match={theta_match}, "
                  f"vowel_match={vowel_match}, params_match={params_match}")
            all_pass = False

    if all_pass:
        print(f"  ✓ All {len(POLAR_SEGMENTS)} segments: polar coordinates identical")
    return all_pass


def verify_adjust_consonant_theta():
    """Verify that consonant theta adjustment is identical."""
    from panphon_timit_plugin.phonology import adjust_consonant_theta
    all_pass = True
    test_cases = [
        ({"rho": 1.2, "theta": 0.0, "syntsyl_key": "d", "params_actifs": [0,1,2,3]}, np.pi, False),
        ({"rho": 1.2, "theta": 0.0, "syntsyl_key": "d", "params_actifs": [0,1,2,3]}, np.pi, True),
        ({"rho": 1.2, "theta": 0.0, "syntsyl_key": "g", "params_actifs": [0,1,2,3]}, np.pi, False),
        ({"rho": 1.2, "theta": 0.0, "syntsyl_key": "g", "params_actifs": [0,1,2,3]}, 2*np.pi, False),
        ({"rho": 1.2, "theta": 0.0, "syntsyl_key": "g", "params_actifs": [0,1,2,3]}, np.pi, True),
    ]
    for cp, voy_theta, in_cluster in test_cases:
        orig_result = orig.adjust_consonant_theta(dict(cp), voy_theta, in_cluster=in_cluster)
        new_result = adjust_consonant_theta(dict(cp), voy_theta, in_cluster=in_cluster)
        rho_match = abs(orig_result["rho"] - new_result["rho"]) < 1e-14
        theta_match = abs(orig_result["theta"] - new_result["theta"]) < 1e-14
        if not (rho_match and theta_match):
            print(f"  ✗ adjust_consonant_theta: rho_match={rho_match}, theta_match={theta_match}")
            all_pass = False
    if all_pass:
        print(f"  ✓ All consonant theta adjustment tests: identical")
    return all_pass


if __name__ == "__main__":
    print("=" * 70)
    print("NUMERICAL IDENTITY VERIFICATION")
    print("=" * 70)

    print("\n[1] Pipeline identity (Pval, envelope, tokens, segments):")
    pipeline_ok = verify_pipeline_identity()

    print("\n[2] Polar coordinate identity:")
    polar_ok = verify_polar_identity()

    print("\n[3] Consonant theta adjustment identity:")
    theta_ok = verify_adjust_consonant_theta()

    print("\n" + "=" * 70)
    if pipeline_ok and polar_ok and theta_ok:
        print("✓ ALL IDENTITY CHECKS PASSED — Refactored code is numerically identical")
    else:
        print("✗ SOME IDENTITY CHECKS FAILED — See details above")
        sys.exit(1)
