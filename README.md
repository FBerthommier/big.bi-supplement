# bigbi-bi.gbi: Articulatory Simulation of the Verbal Transformation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2307.02299-red.svg)](https://arxiv.org/abs/2307.02299)

A Python implementation of the articulatory speech synthesis model described in:

> **Frédéric Berthommier**. *"Why can big.bi be changed to bi.gbi? A mathematical model of syllabification and articulatory synthesis."* arXiv:2307.02299, July 2023.

This repository provides the code, scripts, and didactic documentation to reproduce the demonstrations described in the article, including:

1. The **big.bi → bi.gbi** verbal transformation (§4 of the article)
2. The **ib ib → bi bi** classical transformation (§3)
3. The **U-shaped Tp variation** for perceptual demonstration
4. The **polar trajectory visualization** (two-branch reconstruction)

## Quick Start

Requires **Python ≥ 3.10**.

```bash
# Minimal install (simulation + synthesis + figures)
pip install -r requirements.txt

# Complete install (adds video generation; ffmpeg must be on PATH
# for audio dubbing / concatenation)
pip install -r requirements.txt -r requirements-vlam.txt

# Reproduce the article figures (Demo 1 /ibia/, Demo 3 S-shaped F2,
# Demo 5 consonant clusters)  -> output/article_demos/
python scripts/run_article_demos.py

# Locus equations (Figure 3) + trough effect (Figure 1b)
# -> docs/figures/demo2/, docs/figures/demo4/
python scripts/run_locus_trough_demos.py

# Exact article reproductions (original_simulations/ references)
python scripts/run_original_simulations.py

# Main sweep: big.bi -> bi.gbi with polar coords (14 segments)
python scripts/run_bigbi_polar_sweep.py

# Dual-panel video (sagittal | polar) from the sweep outputs
python scripts/make_bigbi_dual_video.py

# U-shaped Tp variation: ib ib -> bi bi -> ib ib
python scripts/run_ibbi_ushape_sweep.py
```

Polar display conventions (arXiv:2307.02299 §2.1): vowel arcs z_v use
K = 30 ("Kvoy"), consonant arcs z_c use K = 10 — see
`polar_primitives.py` and `docs/POLAR_K_AUDIT.md`. These constants only
affect display; the acoustic engine (K = 10, Kvoy = 30 in
`synthSYL/constants.py`) is untouched.

## Repository Structure

```
bigbi-demos/
├── LICENSE                    # MIT license
├── README.md                  # This file
├── config.py                  # Shared constants (T_BASE, K_DISPLAY, VALRECT, ...)
├── polar_primitives.py        # Standalone polar arc primitives (display branches)
├── requirements.txt           # Core dependencies (numpy/scipy/matplotlib)
├── requirements-vlam.txt      # Optional video-generation dependencies
├── synthSYL/                  # Modified synthSYL package (with delta_o, delta_e, Ve plateau fix)
│   ├── constants.py            # COEFCEN, DEFAULT_K=10, DEFAULT_KVOY=30, DEFAULT_DELTA_O/E
│   ├── gesture.py              # Vo/Ve anchor construction with delta_o, delta_e
│   ├── trajectory.py           # Ve plateau fix (full-amplitude before decay)
│   ├── pipeline.py             # panphon_pipeline() with delta_o, delta_e kwargs
│   └── ...                     # Other modules from timit-to-Maeda
├── vlam.py                    # VLAM (Maeda) articulatory synthesizer
├── VLAMvidmaker.py            # Video generation from Maeda parameters
├── batch_synthesize.py        # Batch WAV synthesis
├── scripts/                   # Simulation scripts (US English)
│   ├── run_article_demos.py          # Demos 1/3/5 (ibia, S-shaped F2, clusters)
│   ├── run_locus_trough_demos.py     # Demos 2/4 (locus equations, trough effect)
│   ├── run_original_simulations.py   # Exact article reproductions + reference check
│   ├── run_bigbi_polar_sweep.py      # big.bi → bi.gbi polar sweep
│   ├── run_ibbi_ushape_sweep.py      # U-shaped Tp variation (reversible)
│   ├── run_ibbi_ushape_nonreversible_en.py  # Non-reversible variant
│   ├── make_bigbi_dual_video.py      # Dual-panel video (sagittal | polar)
│   ├── make_ibbi_polar_video.py      # Polar-only video (T=5 sweep)
│   ├── embed_video_in_html.py        # Self-contained HTML with base64 video
│   └── compare_article_polar.py      # Compare with article Figure 4
├── simulations/               # Self-contained HTML simulations (base64 video embedded)
│   ├── bigbi_bi_gbi_embedded.html           # big.bi → bi.gbi (T=16, δ=0.50→1.00)
│   ├── ibbi_ushape_nonreversible_embedded.html # NON-REVERSIBLE ib ib → bi bi → bi bi
│   ├── ibbi_ushape_embedded.html            # Reversible U-shape ib ib → bi bi → ib ib
│   └── trough_effect.html                   # Trough effect explainer (audio + figures)
├── original_simulations/      # Exact article reproductions (original parameters)
│   ├── figure1_ibia/          # /ibia/ with T=100ms (Figure 1, 4-panel + WAV + npz)
│   ├── figure4_bigbi_bigbi/   # big.bi (δ=1) + bi.gbi (Figure 4; compare with arXiv:2307.02299 Fig. 4)
│   └── supplement_ib_bi/      # /ib/ → /bi/ classical transformation (supplement)
├── docs/                      # Didactic documentation
│   ├── manual.tex             # LaTeX source (US English)
│   ├── manual.pdf             # Compiled PDF (25 pages)
│   ├── LOCUS_FIG3_DIAGNOSTIC.md  # Demo 2 measurement-frame bug analysis
│   ├── POLAR_K_AUDIT.md       # K=10/K=30 display-curvature audit
│   └── figures/               # Simulation figures (Demos 1–5 + comparisons)
└── examples/                  # Quick demos
    ├── demo_bigbi.py          # Minimal big.bi → bi.gbi demo
    └── demo_ibbi.py           # Minimal ib ib → bi bi demo
```

## Key Modifications to the Original synthSYL

This repository extends the original [timit-to-Maeda](https://github.com/FBerthommier/timit-to-Maeda) package with:

1. **`delta_o` and `delta_e` parameters** (Berthommier 2023, §2.2):
   - `Vo = (delta_o * rho_V, theta_V)` — onset anchoring vowel
   - `Ve = (delta_e * rho_V, theta_V)` — end anchoring vowel
   - Added as optional kwargs to `panphon_pipeline()`

2. **Ve plateau fix** (Berthommier 2023, §2.2: "Ve is a NODE"):
   - Added a full-amplitude plateau (T_voy duration) before the decay block
   - Makes the schwa-like vocoid (Ve) perceptually salient
   - Without this fix, the @ → I transformation when delta → 1 is imperceptible

3. **Vo formula correction**:
   - Original: `Vo = [delta_o, theta_V]` (rho = delta_o directly)
   - Fixed: `Vo = [delta_o * rho_V, theta_V]` (matches article formula)
   - At delta_o = 1.0, Vo = V (full vowel, fusion-ready)

4. **Non-initial word onsets anchor on the previous Ve** (article §2.2:
   the pause is the diphthong transition between the previous Ve and
   the next Vo):
   - Original: every word-start onset used `Vo = delta_o * rho_V`,
     so a second schwa dip appeared at each word boundary even when
     COEFCEN (delta_e) = 1.
   - Fixed: for non-initial words, `Vo := (delta_e * rho_prev,
     theta_prev)` — with COEFCEN = 1 the word-2 onset departs from the
     full vowel and there is no return to the half-radius schwa at the
     boundary; with the symmetric delta_o = delta_e the value is
     unchanged (the "big@bi" schwa demonstrations are preserved, and
     all reference reproductions remain bit-identical).

## The Model

The model uses **VLAM** (a Maeda model) with **7 articulatory parameters**:

| Idx | Symbol | Description |
|-----|--------|-------------|
| 0 | J | Jaw opening |
| 1 | B | Tongue body position |
| 2 | D | Tongue dorsum height |
| 3 | T | Tongue tip position |
| 4 | LP | Lip protrusion/rounding |
| 5 | LH | Vertical lip opening |
| 6 | Hy | Larynx height |

The **coordination function** (article Eq. 1):

```
P_i = Omega_i + rho * Psi1_i * cos(Psi2_i - theta)
```

maps a polar target `(rho, theta)` to the 7 Maeda parameters.

## The Verbal Transformation

The article describes how `big.bi` (CVC.CV with separate /g/ and /b/) transforms into `bi.gbi` (CV.CCV with fused /gb/ cluster) when `delta_o = delta_e → 1`:

- At `delta = 0.5`: `Vo = Ve = (0.45, 5π/3)` — salient schwa @
- At `delta = 0.7`: `Vo = Ve = (0.63, 5π/3)` — audible schwa
- At `delta = 1.0`: `Vo = Ve = V = (0.9, 5π/3)` — fusion, /g/+/b/ → /gb/

## Documentation

The didactic documentation is in `docs/manual.tex` (LaTeX source) and `docs/manual.pdf` (compiled PDF). It covers:

1. The mathematical model (coordination function, arcs, superposition)
2. The syllable graph architecture (CV, CVC, CCV)
3. The verbal transformation mechanism
4. Step-by-step reproduction guide
5. Comparison with the article's figures

## Citation

If you use this code in your research, please cite:

```bibtex
@article{berthommier2023bigbi,
  title={Why can big.bi be changed to bi.gbi? A mathematical model of syllabification and articulatory synthesis},
  author={Berthommier, Fr{\'e}d{\'e}ric},
  journal={arXiv preprint arXiv:2307.02299},
  year={2023}
}
```

## Acknowledgments

- Original synthSYL/VLAM code: [FBerthommier/timit-to-Maeda](https://github.com/FBerthommier/timit-to-Maeda)
- The polar trajectory primitives in `polar_primitives.py` and the demo
  scripts were written by the repository author; an equivalent
  implementation also exists in [FBerthommier/covtl-pipeline](https://github.com/FBerthommier/covtl-pipeline)
- VLAM (Vocal tract Learner Articulatory Model): GIPSA-Lab, Grenoble

## License

MIT License — see [LICENSE](LICENSE) for details. All Python files carry
an `SPDX-License-Identifier: MIT` header; `synthSYL/LICENSE` applies the
same MIT terms to the packaged engine.
