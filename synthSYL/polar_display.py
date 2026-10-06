# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
polar_display.py — Polar coordinate display of all phoneme targets.

Displays all phonemes in the synthSYL articulatory space as polar
coordinates (rho, theta).  Categories shown:

  V  (blue)       = oral vowels
  Vn (light blue) = nasal vowels
  C  (green)      = isolated consonants
  G  (orange)     = glides (via vowel tree)
  CC (red)        = consonant cluster targets

v14 changes:
  - /j/ -> theta=5pi/3 (like /i/), /ɥ/ -> theta=5.5pi/3 (like /y/)
  - /ʃ,ʒ,ɲ/ moved coronal->dorsal (theta=-pi/12), consistent with CC
  - /ʁ/ dorsal theta=pi/3, params=[0,1,2,6]
  - /w/ -> theta=pi/3 (like /u/), rho=0.7 (dead code: w->u at input)

v14b CC extension for glides:
  - /j/ added to CC system: dorsal class (alias 'g'), theta->consvalG2 in cluster
  - /w/,/ɥ/ added to CC system: labial class (alias 'b'), theta unchanged in cluster
  - Glides participate in art1 computation (e.g. /kw/ -> art1=[0,1,2,4])

Data sources:
  - V  : VOWELS_SYNTSYL (exact lookup via panphon_to_polar)
  - Vn : panphon_to_polar (feature-based inference + nasalization)
  - C  : panphon_to_polar (exact lookup or feature-based inference)
  - G  : panphon_to_polar (glide branch via vowel tree)
  - CC : THETA_CC_D / THETA_CC_G from constants, grouped by _CC_ALIAS

Usage::

    python -m panphon_timit_plugin.polar_display
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

# ── Package imports ──────────────────────────────────────────────────
from .constants import (
    RHO_STOP,
    RHO_FRICATIVE,
    RHO_NASAL,
    RHO_HI,
    RHO_FRONT_RND,
    RHO_MID,
    RHO_LOW_MID_RND,
    RHO_NASAL_FACTOR,
    THETA_CC_D,
    THETA_CC_G,
    THETA_LABIAL,
    THETA_CORONAL,
    THETA_NASAL_FACTOR,
    THETA_V_BACK_HI,
    THETA_V_LOW,
    THETA_V_BACK_MID_RND,
    THETA_V_BACK_MID,
    THETA_V_FRONT_RND,
    THETA_V_HI_RND,
    THETA_V_HI,
    THETA_V_TENSE,
    THETA_V_LAX,
)
from .phonology import (
    VOWELS_SYNTSYL,
    CONSONANTS_SYNTSYL,
    PANPHON_DB,
    _CC_ALIAS,
    _TABCONS_IDX,
    feature_index,
    get_panphon,
    panphon_to_polar,
)


# ═══════════════════════════════════════════════════════════════════════
#  FONTS (multi-platform: Linux / Windows / macOS)
# ═══════════════════════════════════════════════════════════════════════
_FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf',
    os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', 'dejavu.ttf'),
    os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', 'DejaVuSans.ttf'),
    os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', 'arial.ttf'),
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/Library/Fonts/Arial Unicode.ttf',
]
for _fp in _FONT_CANDIDATES:
    if os.path.isfile(_fp):
        try:
            fm.fontManager.addfont(_fp)
        except Exception:
            pass

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'DejaVu Sans Mono', 'Arial', 'Noto Sans SC'],
    'axes.unicode_minus': False,
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FAFBFC',
    'figure.dpi': 200,
    'savefig.dpi': 200,
})


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════
def _normalize_angle(theta_rad: float) -> float:
    """Normalize theta to [0, 2*pi)."""
    return theta_rad % (2 * np.pi)


# ═══════════════════════════════════════════════════════════════════════
#  DATA TABLES — derived from the package's phonological database
# ═══════════════════════════════════════════════════════════════════════
#
# Format: IPA phoneme -> (rho, theta_in_radians, degree_label)
#
# Values are derived from:
#   VOWELS_SYNTSYL          (exact lookup via panphon_to_polar)
#   panphon_to_polar() v14  (feature-based inference: FIX-1 to FIX-8)
#   THETA_CC_D / THETA_CC_G (cluster consonant thetas from constants)

# ──────────────────────────────────────────────────────────────────────
#  V  — Oral vowels (source: VOWELS_SYNTSYL, exact lookup)
# ──────────────────────────────────────────────────────────────────────
V: dict[str, tuple[float, float, str]] = {}
for _key, _data in VOWELS_SYNTSYL.items():
    _ipa = _data["IPA"]
    _rho = _data["rho"]
    _theta = _data["theta"]
    _deg = f"{np.degrees(_theta):.1f}\u00b0"
    V[_ipa] = (_rho, _theta, _deg)

# ──────────────────────────────────────────────────────────────────────
#  Vn — Nasal vowels (v14: FIX-4 + FIX-5, via panphon_to_polar)
#        ɑ̃ : lo==1 -> theta=pi (FIX-4), then x0.97 nasal
#        ɔ̃ : back=1,rnd=1 -> theta=pi/2, rho=0.8 (FIX-5: back!=1 in rho=0.3 rule)
# ──────────────────────────────────────────────────────────────────────
_NASAL_IPA = ['ɛ̃', 'ɑ̃', 'ɔ̃']
Vn: dict[str, tuple[float, float, str]] = {}
for _ipa in _NASAL_IPA:
    _pv = get_panphon(_ipa)
    if _pv is not None:
        _pt = panphon_to_polar(_ipa, _pv)
        Vn[_ipa] = (_pt.rho, _pt.theta, f"{np.degrees(_pt.theta):.1f}\u00b0")

# ──────────────────────────────────────────────────────────────────────
#  C  — Isolated consonants (v14: FIX-3 reorders dorsal/coronal)
#
#        v14: priority labial > dorsal(hi/back) > coronal
#        /ʃ,ʒ,ɲ/ (hi=1) -> dorsal theta=-pi/12 (consistent with _CC_ALIAS->'g')
#        /ʁ/ (back=1) -> dorsal theta=pi/3, params=[0,1,2,6]
# ──────────────────────────────────────────────────────────────────────
_CONSONANT_IPA = [
    # Labials (theta = THETA_LABIAL = pi/3 = 60 deg)
    'b', 'p', 'm', 'f', 'v',
    # Dorsal velar (theta = THETA_LABIAL = pi/3, from CONSONANTS_SYNTSYL)
    'g', 'k',
    # Dorsal /ʁ/ (theta = THETA_LABIAL = pi/3, back==1, FIX-6)
    'ʁ',
    # Dorsal hi=1 (theta = -pi/12 ~ 345 deg, FIX-3)
    'ʃ', 'ʒ', 'ɲ',
    # Coronals (theta = THETA_CORONAL = 23pi/16 ~ 258.75 deg)
    'd', 't', 'n', 'l', 's', 'z', 'r',
]
C: dict[str, tuple[float, float, str]] = {}
for _ipa in _CONSONANT_IPA:
    _pv = get_panphon(_ipa)
    if _pv is not None:
        _pt = panphon_to_polar(_ipa, _pv)
        C[_ipa] = (_pt.rho, _pt.theta, f"{np.degrees(_pt.theta):.1f}\u00b0")

# ──────────────────────────────────────────────────────────────────────
#  G  — Glides (v14 FIX-2: via vowel tree)
#        /j/ -> coordinates of /i/ (hi=1, unrounded)
#        /ɥ/ -> coordinates of /y/ (hi=1, rounded)
#        /w/ -> coordinates of /u/ (hi=1, back=1, rounded)
#        is_vowel=False (no plateau), params according to rnd
# ──────────────────────────────────────────────────────────────────────
_GLIDE_IPA = ['j', 'ɥ', 'w']
G: dict[str, tuple[float, float, str]] = {}
for _ipa in _GLIDE_IPA:
    _pv = get_panphon(_ipa)
    if _pv is not None:
        _pt = panphon_to_polar(_ipa, _pv)
        G[_ipa] = (_pt.rho, _pt.theta, f"{np.degrees(_pt.theta):.1f}\u00b0")

# ──────────────────────────────────────────────────────────────────────
#  CC — Consonant cluster targets
#        v14: consistency isolation/cluster (FIX-3)
#        dorsal CC = g,k,ʁ,ʃ,ʒ,ɲ (all dorsal in isolation too)
# ──────────────────────────────────────────────────────────────────────
CC: dict[tuple[str, ...], tuple[float, float, str]] = {
    # Coronal CC (theta = THETA_CC_D ~ 285 deg)
    ('d', 't', 'l', 'n'):  (RHO_STOP,      THETA_CC_D, f"{np.degrees(THETA_CC_D):.1f}\u00b0"),
    ('s', 'z', 'r'):       (RHO_FRICATIVE,  THETA_CC_D, f"{np.degrees(THETA_CC_D):.1f}\u00b0"),
    # Dorsal CC (theta = THETA_CC_G ~ 345 deg)
    ('g', 'k'):            (RHO_STOP,       THETA_CC_G, f"{np.degrees(THETA_CC_G):.1f}\u00b0"),
    ('ʁ', 'ʃ', 'ʒ', 'j'):  (RHO_FRICATIVE,  THETA_CC_G, f"{np.degrees(THETA_CC_G):.1f}\u00b0"),  # j via alias 'g' (v14b)
    ('ɲ',):                (RHO_NASAL,       THETA_CC_G, f"{np.degrees(THETA_CC_G):.1f}\u00b0"),
    # Labial CC (theta = THETA_LABIAL = pi/3 = 60 deg, unchanged)
    #    b,p,m,f,v keep their isolated position
}


# ═══════════════════════════════════════════════════════════════════════
#  PRINTED SUMMARY
# ═══════════════════════════════════════════════════════════════════════
def print_summary() -> None:
    """Print a summary table of all polar targets."""
    print("═══ PanPhon v14 Polar Targets ═══\n")
    print(f"{'Cat.':<4} {'Phon.':<6} {'ρ':>5}  {'θ (rad)':>12}  {'θ (deg)':>8}  Note")
    print("─" * 70)
    for ph, (rho, th, deg) in V.items():
        print(f"{'V':<4} {ph:<6} {rho:>5.2f}  {th:>12.4f}  {deg:>8}")
    for ph, (rho, th, deg) in Vn.items():
        print(f"{'Vn':<4} {ph:<6} {rho:>5.2f}  {th:>12.4f}  {deg:>8}  (nasal)")
    for ph, (rho, th, deg) in C.items():
        print(f"{'C':<4} {ph:<6} {rho:>5.2f}  {th:>12.4f}  {deg:>8}")
    for ph, (rho, th, deg) in G.items():
        print(f"{'G':<4} {ph:<6} {rho:>5.2f}  {th:>12.4f}  {deg:>8}  (glide→homologue vowel)")
    for members, (rho, th, deg) in CC.items():
        label = ' '.join(members)
        cat = 'cor' if abs(th - THETA_CC_D) < 1e-6 else ('dor' if abs(th - THETA_CC_G) < 1e-6 else 'lab')
        print(f"{'CC':<4} {label:<12} {rho:>5.2f}  {th:>12.4f}  {deg:>8}  ({cat})")
    print("─" * 70)
    n_cc = sum(len(m) for m in CC)
    print(f"  Total: {len(V)}V + {len(Vn)}Vn + {len(C)}C + {len(G)}G "
          f"+ {n_cc} phon. in {len(CC)} CC groups\n")


# ═══════════════════════════════════════════════════════════════════════
#  POLAR PLOT
# ═══════════════════════════════════════════════════════════════════════

# ── Category colours ──
COL_V    = '#2563EB'   # bright blue
COL_VN   = '#93C5FD'   # light blue
COL_C    = '#059669'   # emerald green
COL_G    = '#EA580C'   # orange (glides)
COL_CC   = '#DC2626'   # red (coronal CC)
COL_CCD  = '#7C3AED'   # purple (dorsal CC)
COL_CCL  = '#6B7280'   # grey (labial CC — note)


def generate_plot(output_path: str = 'polar_targets_panphon_v14.png') -> str:
    """Generate the polar coordinate plot and save to *output_path*.

    Returns the output file path.
    """
    # ── Create figure ──
    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(polar=True),
                           constrained_layout=False)
    ax.set_facecolor('#FAFBFC')
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('#D1D5DB')
    ax.spines['polar'].set_linewidth(0.8)
    ax.grid(color='#E5E7EB', linewidth=0.4, alpha=0.6)
    ax.set_rgrids([0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                  labels=['0.3', '0.5', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2'],
                  fontsize=7, color='#9CA3AF', angle=25)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    # ── Angle labels ──
    deg_labels = []
    for a in range(0, 360, 30):
        rad_map = {
            0: '0°', 60: '60°\n(π/3)', 90: '90°\n(π/2)', 120: '120°\n(2π/3)',
            180: '180°\n(π)', 270: '270°\n(3π/2)', 300: '300°\n(5π/3)',
            330: '330°\n(11π/6)',
        }
        deg_labels.append(rad_map.get(a, f'{a}°'))
    ax.set_thetagrids(range(0, 360, 30), labels=deg_labels, fontsize=7, color='#6B7280')
    ax.tick_params(pad=14)

    # ══════════════════════════════════════════════════════════════════
    #  PLOT POINTS
    # ══════════════════════════════════════════════════════════════════

    # ── V: oral vowels ──
    for ph, (rho, theta, _deg) in V.items():
        ax.scatter(_normalize_angle(theta), rho, s=90, c=COL_V,
                   edgecolors='white', linewidth=1.2, zorder=5, marker='o')

    # ── Vn: nasal vowels ──
    for ph, (rho, theta, _deg) in Vn.items():
        ax.scatter(_normalize_angle(theta), rho, s=70, c=COL_VN,
                   edgecolors='white', linewidth=1, zorder=5, marker='o', alpha=0.85)

    # ── C: isolated consonants ──
    for ph, (rho, theta, _deg) in C.items():
        ax.scatter(_normalize_angle(theta), rho, s=80, c=COL_C,
                   edgecolors='white', linewidth=1, zorder=5, marker='s')

    # ── G: glides — border coloured by CC group ──
    #    j -> dorsal (purple),  w/ɥ -> labial (soft blue)
    G_CC_EDGE = {
        'j':  COL_CCD,   # dorsal CC
        'ɥ':  '#3B82F6', # labial CC (soft blue)
        'w':  '#3B82F6', # labial CC (soft blue)
    }
    for ph, (rho, theta, _deg) in G.items():
        ax.scatter(_normalize_angle(theta), rho, s=65, c=COL_G,
                   edgecolors=G_CC_EDGE.get(ph, 'white'), linewidth=2.0,
                   zorder=6, marker='D', alpha=0.9)

    # ── CC: cluster targets ──
    for members, (rho, theta, _deg) in CC.items():
        th_cc = _normalize_angle(theta)
        if abs(theta - THETA_CC_D) < 1e-6:
            col = COL_CC
        elif abs(theta - THETA_CC_G) < 1e-6:
            col = COL_CCD
        else:
            col = COL_CCL
        ax.scatter(th_cc, rho, s=200, c=col,
                   edgecolors='white', linewidth=1.8, zorder=8, marker='*', alpha=0.95)

    # ── CC labial: explicit markers for w, ɥ (theta unchanged = pi/3) ──
    #    Small open triangles at slightly higher rho to show that w and ɥ
    #    participate in the labial CC without displacing the target.
    ax.scatter(_normalize_angle(THETA_LABIAL), 0.75, s=50, c='none',
               edgecolors=COL_CCL, linewidth=1.5, zorder=7, marker='^', alpha=0.7)
    ax.scatter(_normalize_angle(THETA_LABIAL), 0.82, s=50, c='none',
               edgecolors=COL_CCL, linewidth=1.5, zorder=7, marker='^', alpha=0.7)

    # ══════════════════════════════════════════════════════════════════
    #  ANNOTATIONS
    # ══════════════════════════════════════════════════════════════════

    def ann(text, theta, rho, fs=11, col='black', ha='left', va='center',
            dth=0.02, drho=0.04, bold=True, bbox_col=None, bbox_alpha=0.85):
        th = _normalize_angle(theta)
        kw = dict(fontsize=fs, color=col, ha=ha, va=va, zorder=10,
                  fontweight='bold' if bold else 'normal')
        if bbox_col:
            kw['bbox'] = dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor=bbox_col, alpha=bbox_alpha, linewidth=0.8)
        return ax.text(th + dth, rho + drho, text, **kw)

    # ── Oral vowels ──
    ann('u',   THETA_V_BACK_HI,     1.0,  13, COL_V, dth=-0.05, drho=0.05, ha='right')
    ann('o',   THETA_V_BACK_MID_RND, 0.8,  12, COL_V, dth=0.04,  drho=0.04)
    ann('ɔ',   THETA_V_BACK_MID,     0.8,  12, COL_V, dth=0.04,  drho=0.04)
    ann('a',   THETA_V_LOW,          0.8,  13, COL_V, drho=-0.06, ha='center', va='top')
    ann('ɛ',   THETA_V_LAX,          0.8,  12, COL_V, dth=-0.04, drho=0.04, ha='right')
    ann('e',   THETA_V_TENSE,        0.8,  13, COL_V, dth=-0.04, drho=0.04, ha='right')
    ann('i',   THETA_V_HI,           0.9,  13, COL_V, dth=-0.04, drho=0.04, ha='right')
    ann('y',   THETA_V_HI_RND,       0.7,  11, COL_V, dth=-0.04, drho=0.04, ha='right')
    ann('ø',   THETA_V_FRONT_RND,    0.3,  10, COL_V, dth=-0.04, drho=-0.04,
        ha='right', va='top', bold=False)

    # ── Nasal vowels ──
    _nasal_base = {
        'ɛ̃': THETA_V_LAX,
        'ɑ̃': THETA_V_LOW,
        'ɔ̃': THETA_V_BACK_MID_RND,
    }
    for _ipa, _base_theta in _nasal_base.items():
        _nasal_theta = _base_theta * THETA_NASAL_FACTOR
        _nasal_rho = RHO_MID * RHO_NASAL_FACTOR
        if _ipa == 'ɛ̃':
            ann('ɛ̃', _nasal_theta, _nasal_rho, 10, COL_VN,
                dth=-0.06, drho=-0.04, ha='right', bold=False)
        elif _ipa == 'ɑ̃':
            ann('ɑ̃', _nasal_theta, _nasal_rho, 10, COL_VN,
                dth=-0.04, drho=-0.05, ha='right', bold=False)
        elif _ipa == 'ɔ̃':
            ann('ɔ̃', _nasal_theta, _nasal_rho, 10, COL_VN,
                dth=0.06, drho=-0.04, bold=False)

    # ── Isolated consonants: theta = pi/3 (labials + dorsals) ──
    ann('b p g k ʁ',      THETA_LABIAL,  1.2,  11, COL_C, dth=0.08, drho=0.04)
    ann('f v',             THETA_LABIAL,  1.1,  10, COL_C, dth=0.10, drho=0.02)
    ann('m',               THETA_LABIAL,  1.05, 10, COL_C, dth=0.10, drho=-0.03, va='top')

    # ── Isolated consonants: theta = -pi/12 (dorsal hi=1, FIX-3) ──
    ann('ʃ ʒ',            -np.pi/12, 1.1,  11, COL_C, dth=-0.06, drho=0.04)
    ann('ɲ',              -np.pi/12, 1.05, 10, COL_C, dth=-0.06, drho=-0.04, va='top')

    # ── Isolated consonants: theta = THETA_CORONAL (coronals) ──
    ann('d t',            THETA_CORONAL, 1.2,  11, COL_C, dth=0.08, drho=0.04)
    ann('l s z r',        THETA_CORONAL, 1.1,  10, COL_C, dth=0.08, drho=0.02)
    ann('n',              THETA_CORONAL, 1.05, 10, COL_C, dth=0.08, drho=0.0)

    # ── CONNECTORS: glides -> CC target (dashed lines) ──
    #    Show the theta "migration" when the glide enters a cluster.
    connectors = [
        # j (300°, rho=0.9) -> CC dorsal (345°, rho=1.1)
        (THETA_V_HI, 0.9, THETA_CC_G, 1.1, COL_CCD),
        # w (60°, rho=0.7) -> CC labial (60°, rho=0.82) — same theta, short line
        (THETA_LABIAL, 0.7, THETA_LABIAL, 0.82, '#3B82F6'),
        # ɥ (330°, rho=0.7) -> CC labial (60°, rho=0.75) — large theta jump
        (THETA_V_HI_RND, 0.7, THETA_LABIAL, 0.75, '#3B82F6'),
    ]
    for (g_th, g_rho, cc_th, cc_rho, col) in connectors:
        ax.annotate('',
            xy=(_normalize_angle(cc_th), cc_rho), xytext=(_normalize_angle(g_th), g_rho),
            arrowprops=dict(arrowstyle='->', color=col, lw=1.4,
                            linestyle='dashed', alpha=0.55,
                            connectionstyle='arc3,rad=0.15'),
            zorder=4)

    # ── Glides (v14: coincide with homologue vowel) ──
    #    Enriched annotation: isolated position + CC membership
    ann('j ≡ i',  THETA_V_HI,     0.9,  10, COL_G, dth=0.03, drho=-0.06, va='top', bold=False,
        bbox_col=G_CC_EDGE['j'], bbox_alpha=0.5)
    ann('ɥ ≡ y',  THETA_V_HI_RND,  0.7,  10, COL_G, dth=0.03, drho=0.03, va='top', bold=False,
        bbox_col=G_CC_EDGE['ɥ'], bbox_alpha=0.5)
    ann('w ≡ u',  THETA_V_BACK_HI,  0.7,  10, COL_G, dth=-0.02, drho=-0.03, ha='right', va='top', bold=False,
        bbox_col=G_CC_EDGE['w'], bbox_alpha=0.5)

    # ── CC targets (pure consonants, without glides) ──
    ann('CC cor. (d t l n)',    THETA_CC_D, 1.2,  10, COL_CC,  dth=0.06, drho=0.05, bbox_col=COL_CC)
    ann('CC cor. (s z r)',      THETA_CC_D, 1.1,   9, COL_CC,  dth=0.06, drho=-0.04, va='top', bbox_col=COL_CC)
    ann('CC dor. (g k)',        THETA_CC_G, 1.2,  10, COL_CCD, dth=0.08, drho=0.05, bbox_col=COL_CCD)
    ann('CC dor. (ʁ ʃ ʒ)',     THETA_CC_G, 1.1,   9, COL_CCD, dth=0.06, drho=-0.05, va='top', bbox_col=COL_CCD)
    ann('CC dor. (ɲ)',          THETA_CC_G, 1.05,  9, COL_CCD, dth=0.1, drho=-0.06, va='top', bbox_col=COL_CCD)

    # ── CC glides — dedicated annotations, separate from pure consonants ──
    #    j -> dorsal (same star as ʁ ʃ ʒ, but separate label)
    ann('j → dor. CC',  THETA_CC_G, 1.1,  8, COL_CCD,
        dth=-0.13, drho=-0.02, va='top', bold=False, bbox_col=COL_CCD, bbox_alpha=0.6)
    #    w, ɥ -> labial (theta unchanged, open triangles)
    ann('w → lab. CC',  THETA_LABIAL, 0.82,  8, '#3B82F6',
        dth=0.06, drho=0.03, bold=False, bbox_col='#3B82F6', bbox_alpha=0.5)
    ann('ɥ → lab. CC',  THETA_LABIAL, 0.75,  8, '#3B82F6',
        dth=-0.06, drho=0.03, va='top', bold=False, bbox_col='#3B82F6', bbox_alpha=0.5)

    # ── CC labial note (pure consonants) ──
    ann('CC lab. (b p m f v)\nθ = π/3, unchanged', THETA_LABIAL, 1.2,  9, COL_CCL,
        dth=0.06, drho=0.0, ha='right', bbox_col=COL_CCL, bold=False)

    # ══════════════════════════════════════════════════════════════════
    #  LEGEND
    # ══════════════════════════════════════════════════════════════════
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_V,
               markersize=10, label='V — oral vowels', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_VN,
               markersize=8, label='Vn — nasal vowels', markeredgecolor='white'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COL_C,
               markersize=9, label='C — isolated consonants', markeredgecolor='white'),
        # Glides with CC border
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COL_G,
               markeredgecolor=COL_CCD, markersize=7,
               label='G — glides (border = CC group)'),
        # CC consonants
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COL_CC,
               markersize=16, label='CC coronal (D2 ≈ 285°)',
               markeredgecolor='white'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COL_CCD,
               markersize=16, label='CC dorsal (G2 ≈ 345°)',
               markeredgecolor='white'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COL_CCL,
               markersize=16, label='CC labial (θ = π/3)',
               markeredgecolor='white'),
        # Glides in CC
        Line2D([0], [0], marker='^', color='w', markerfacecolor='none',
               markeredgecolor='#3B82F6', markersize=8,
               label='Glide CC (labial, θ unchanged)'),
        Line2D([0], [0], linestyle='--', color=COL_CCD, lw=1.4, alpha=0.6,
               label='Glide → CC target connector'),
    ]

    leg = ax.legend(handles=legend_elements, loc='upper left',
                    bbox_to_anchor=(-0.18, 1.18), fontsize=9,
                    frameon=True, facecolor='white', edgecolor='#E5E7EB',
                    title='Target categories (v14)', title_fontsize=10,
                    borderpad=0.8, labelspacing=0.7)
    leg.get_title().set_fontweight('bold')
    leg.set_zorder(20)

    # ── Title ──
    ax.set_title(
        'PanPhon Plugin v14b Polar Targets\n'
        'Vowels (V) · Consonants (C) · CC-eligible Glides (G) · Clusters (CC)',
        fontsize=15, fontweight='bold', pad=30, color='#1F2937', loc='center')

    # ── Info box: glide → CC correspondence ──
    info_lines = (
        'CC-eligible Glides\n'
        '────────────────────\n'
        'j  → dorsal CC (alias g)\n'
        '     θ: 300° → 345°\n'
        'w  → labial CC (alias b)\n'
        '     θ: 60° (unchanged)\n'
        'ɥ → labial CC (alias b)\n'
        '     θ: 330° → 60°'
    )
    fig.text(0.02, 0.38, info_lines,
             fontsize=8.5, fontfamily='monospace', color='#374151',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F9FAFB',
                       edgecolor='#D1D5DB', linewidth=1.0, alpha=0.95))

    # ── Footer ──
    fig.text(0.5, 0.01,
             'ρ = gestural amplitude  |  θ = articulatory direction  |  '
             '17 consonants, 12 vowels, 3 CC-eligible glides  |  '
             '★ = CC target  |  ◆ = isolated glide  |  '
             '▲ = glide in labial CC  |  - - - = θ migration toward CC',
             ha='center', fontsize=8, color='#9CA3AF', style='italic')

    # ══════════════════════════════════════════════════════════════════
    #  SAVE
    # ══════════════════════════════════════════════════════════════════
    fig.savefig(output_path, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)

    return output_path


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    """Generate the polar coordinate display and print summary."""
    print_summary()

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'polar_targets_panphon_v14.png',
    )
    result = generate_plot(output_path)

    size_kb = os.path.getsize(result) / 1024
    print(f'File: {result} ({size_kb:.0f} KB)')


if __name__ == "__main__":
    main()
