#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
compare_article_polar.py — Compare notre figure polaire à celle de
l'article arXiv:2307.02299 Figure 4 (Berthommier 2023).

Crée une figure de comparaison côte-à-côte :
  - Article Figure 4 (top row: planning trajectories for big.bi & bi.gbi)
  - Notre polar_d995.png (big bi, δ=0.995 ≈ article's big.bi δ=1)
  - Notre polar_d1000.png (bi.gbi, δ=1.0 = article's bi.gbi)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
except Exception:
    pass
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Paths — the article figure is extracted locally from arXiv:2307.02299
# (kept outside the repository); our figures come from
# scripts/run_bigbi_polar_sweep.py under <repo>/output/.
# Usage: compare_article_polar.py [article_fig.png [ours_bigbi.png [ours_bigbi_fused.png]]]
REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_STATIC = REPO_ROOT / "output" / "bigbi_polar_sweep" / "polar" / "static"
ARTICLE_FIG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("article_figures/fig-006.png")
OUR_BIGBI = Path(sys.argv[2]) if len(sys.argv) > 2 else SWEEP_STATIC / "polar_d995.png"
OUR_BIGBI_FUSED = Path(sys.argv[3]) if len(sys.argv) > 3 else SWEEP_STATIC / "polar_d1000.png"
OUT_PATH = SWEEP_STATIC / "comparison_article.png"


def load_image(path):
    """Load an image as a numpy array."""
    img = Image.open(path)
    return np.array(img)


def crop_top_row(img_array, fraction=0.30):
    """Crop the top portion of an image (the planning trajectories row)."""
    h = img_array.shape[0]
    return img_array[:int(h * fraction), :, :3]


def main():
    # Load images
    article_img = load_image(ARTICLE_FIG)
    our_bigbi = load_image(OUR_BIGBI)
    our_fused = load_image(OUR_BIGBI_FUSED)

    print(f"Article fig: {article_img.shape}")
    print(f"Our bigbi (δ=0.995): {our_bigbi.shape}")
    print(f"Our bi.gbi (δ=1.0): {our_fused.shape}")

    # The article Figure 4 has 4 rows × 2 columns (big.bi left, bi.gbi right)
    # Top row = planning trajectories
    # We want to extract the top row and split left/right
    h, w = article_img.shape[:2]
    top_row = article_img[:int(h * 0.28), :, :3]  # top ~28% = planning row
    # Split left/right
    mid_x = w // 2
    article_bigbi = top_row[:, :mid_x, :]
    article_bigbi_fused = top_row[:, mid_x:, :]

    print(f"Article big.bi (top-left): {article_bigbi.shape}")
    print(f"Article bi.gbi (top-right): {article_bigbi_fused.shape}")

    # Create comparison figure: 2 rows × 3 columns
    # Row 1: Article Figure 4 top row (big.bi left, bi.gbi right)
    # Row 2: Our figures (big bi δ=0.995 left, bi.gbi δ=1.0 right)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                              gridspec_kw={"height_ratios": [1, 1]})

    # Row 1: Article
    axes[0, 0].imshow(article_bigbi)
    axes[0, 0].set_title(
        "Article Figure 4 (gauche)\n"
        "big.bi avec δo = δe = 1\n"
        "(Berthommier 2023, arXiv:2307.02299 §4)",
        fontsize=11, color="#003366", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(article_bigbi_fused)
    axes[0, 1].set_title(
        "Article Figure 4 (droite)\n"
        "bi.gbi (cluster /gb/ fusionné)\n"
        "(Berthommier 2023, arXiv:2307.02299 §4)",
        fontsize=11, color="#003366", fontweight="bold")
    axes[0, 1].axis("off")

    # Row 2: Our figures
    axes[1, 0].imshow(our_bigbi)
    axes[1, 0].set_title(
        "Notre figure (big bi, δ = 0,995)\n"
        "Vo ≈ Ve ≈ V = (0,896, 5π/3) → ≈ /i/\n"
        "z_c : 2 excursions séparées vers /b/ et /g/",
        fontsize=11, color="#0066cc", fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(our_fused)
    axes[1, 1].set_title(
        "Notre figure (bi.gbi, δ = 1,00)\n"
        "Vo = Ve = V = (0,9, 5π/3) = /i/ (fusion)\n"
        "z_c : 1 parcours /g/ → /b/ (cluster /gb/)",
        fontsize=11, color="#0066cc", fontweight="bold")
    axes[1, 1].axis("off")

    # Add row labels
    fig.text(0.02, 0.75, "ARTICLE", fontsize=14, fontweight="bold",
             color="#003366", rotation=90, va="center")
    fig.text(0.02, 0.25, "NOTRE SIMULATION", fontsize=14, fontweight="bold",
             color="#0066cc", rotation=90, va="center")

    fig.suptitle(
        "Comparaison : figure polaire de l'article vs notre simulation\n"
        "Transition big.bi → bi.gbi (arXiv:2307.02299 §4, Berthommier 2023)\n"
        "Les deux utilisent VLAM = modèle de Maeda à 7 paramètres",
        fontsize=13, fontweight="bold", y=0.98)

    # Add comparison annotation at the bottom
    fig.text(0.5, 0.02,
             "Similitudes : (1) branche vocalique z_v (rouge) quasi-ponctuelle "
             "car Vo ≈ Ve ≈ V ; (2) branche consonantique z_c (bleue) fait des "
             "excursions vers /b/ et /g/ ; (3) à gauche (big.bi), 2 gestes "
             "séparés ; à droite (bi.gbi), 1 geste /gb/ fusionné.\n"
             "Différences mineures : (1) l'article utilise une grille polaire "
             "angulaire, nous une grille cartésienne (Re, Im) ; (2) calibration "
             "légèrement différente (Tip : c₀=-2,75 dans synthSYL vs c₀=-3 dans "
             "l'article) ; (3) inventaire de fond : l'article montre /b, d, g/, "
             "nous montrons /b, d, g, k, p, t/. Le modèle est identique (VLAM/Maeda).",
             ha="center", fontsize=9, color="#444444",
             wrap=True)

    plt.tight_layout(rect=[0.04, 0.06, 1, 0.95])
    plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Comparison figure saved: {OUT_PATH}")
    print(f"  Size: {OUT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
