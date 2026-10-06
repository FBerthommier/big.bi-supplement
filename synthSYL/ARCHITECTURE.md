# Architecture & Refactoring Report — panphon_timit_plugin

## 1. New Architecture

```
panphon_timit_plugin/
├── __init__.py          # Public API: panphon_pipeline
├── constants.py         # Named constants, magic number elimination
├── types.py             # Dataclasses (PolarTarget, GestureNode, etc.)
├── phonology.py         # PanPhon DB, classification, polar mapping
├── parsing.py           # Input text analysis
├── polar.py             # Maeda projection primitives
├── gesture.py           # Gestural nodes + anchors
├── trajectory.py        # Global continuous polar trajectory (Pval)
├── envelope.py          # Amplitude envelope construction
├── pipeline.py          # Main pipeline orchestration
├── CHANGELOG.md         # Version history (moved from code)
└── tests/
    ├── __init__.py
    └── test_regression.py  # Full regression + unit tests
```

## 2. New Modules

| Module        | Lines | Role                                                    |
|---------------|-------|---------------------------------------------------------|
| constants.py  |  165  | All physical/model constants + named magic numbers      |
| types.py      |  164  | 9 dataclasses replacing anonymous dicts                 |
| phonology.py  |  517  | PanPhon DB, IPA tables, classify_*(), panphon_to_polar |
| parsing.py    |   88  | parse_input → flat + boundaries + word info             |
| polar.py      |  105  | compute_P, arc_B, sigmoid, build_background, inject     |
| gesture.py    |  331  | build_gesture_nodes, build_gesture_anchors (decomposed) |
| trajectory.py |  345  | build_cluster_pval, build_global_pval                   |
| envelope.py   |  415  | _env_*(), _build_envelope_tokens, build_envelope*,      |
|               |       | _build_cluster_env, build_envelope_from_blocks          |
| pipeline.py   |  127  | panphon_pipeline orchestration                           |

Total refactored: ~2057 lines across 9 modules (original: 2008 lines monolithic)

## 3. New Classes (dataclasses)

| Class                  | Replaces              | Fields                           |
|------------------------|-----------------------|----------------------------------|
| SegmentDescription     | PANPHON_DB entries    | ipa, syntsyl_key, rho, theta...  |
| PolarTarget            | polar dict {'rho':...}| rho, theta, params_actifs, seg   |
| GestureNode            | node dict {'kind':...}| kind, rho, theta, seg_key...     |
| GestureAnchor          | anchor dict           | i, pt, hold, kind, is_voydeb... |
| ClusterConsonant       | cluster dict          | pt, params, seg_key              |
| TrajectoryBlock        | block_info dict       | n_steps, kind, cons_tokens...    |
| SyllableBoundaryInfo   | boundary dict         | flat_idx, prev_segs, curr_segs   |
| PipelineResult         | pipeline return dict  | Pval, envelope, tokens...        |

## 4. Modifications per Function

### 4.1 Eliminated Duplications

- **classify_vowel()**: Centralized vowel theta/rho tree, called by:
  - panphon_to_polar (Tier 3a: true vowels)
  - panphon_to_polar (Tier 2: glides j, w, ɥ)

- **classify_consonant()**: Centralized consonant theta/rho/params tree, called by:
  - panphon_to_polar (Tier 3b: consonants)
  - active_parameters() (unknown segments)
  - classify_place() (place inference fallback)

- **classify_place()**: Centralized place of articulation index, called by:
  - cluster_art_params (ternary clusters)

- **active_parameters()**: Centralized active parameter determination, called by:
  - trajectory.py (uniform_params fallback)

### 4.2 Magic Number Elimination

All 50+ hardcoded values replaced by named constants:
- Angles: THETA_D, THETA_G, THETA_CC_D, THETA_CC_G, THETA_CORONAL, THETA_LABIAL...
- Rho values: RHO_STOP, RHO_NASAL, RHO_FRICATIVE, RHO_HIGH, RHO_FRONT_RND...
- Envelope: TOKEN_AMPL_O/C/V/R/r/N/L/F, PLOSIVE_TOKENS
- Physical: CO, COEFCEN, FS, T_S, NEUTRAL_MAEDA
- Pipeline defaults: DEFAULT_T, DEFAULT_NU, DEFAULT_K...

### 4.3 Function Decomposition

build_gesture_anchors() (150→5 functions):
- _insert_base_anchors()     — V/pause anchor creation
- _insert_word_end()         — word-end before pause after C
- _insert_word_voydeb()      — word-initial voydeb [0.5, θ]
- _insert_syllable_voydeb()  — inter-syllable voydeb
- _insert_synth_start_end()  — initial/final synth anchors

### 4.4 Dict → Dataclass Migration

Every anonymous dict replaced by typed dataclass:
- node["theta"] → node.theta
- anchor["is_voydeb"] → anchor.is_voydeb
- polar["params_actifs"] → polar.params_actifs
- block_info["kind"] → block_info.kind

### 4.5 Type Annotations

All public functions annotated:
- panphon_pipeline(text: str, ...) → Optional[PipelineResult]
- panphon_to_polar(seg: str, pv: np.ndarray) → PolarTarget
- classify_vowel(hi, lo, back, rnd, tns, nas) → PolarTarget
- classify_consonant(lab, hi, back, cor, cont, nas) → PolarTarget
- build_gesture_nodes(...) → List[GestureNode]
- build_gesture_anchors(...) → List[GestureAnchor]
- build_global_pval(...) → Tuple[np.ndarray, List[float], List[TrajectoryBlock]]

## 5. Reasons for Each Modification

| Modification          | Reason                                      |
|-----------------------|---------------------------------------------|
| Module split          | Separation of concerns, independent testing |
| Dataclasses           | IDE autocompletion, type safety, readability |
| Named constants       | Self-documenting code, single source of truth|
| Central classification| Eliminate 3× duplicated vowel/consonant trees|
| Function decomposition| Reduce 150-line functions to ≤40 lines each |
| CHANGELOG.md          | Remove 120-line version history from .py     |
| Pipeline result type  | Typed return instead of anonymous dict       |
| Type annotations      | Static analysis, documentation, IDE support  |

## 6. Demonstration of Identical Results

Regression test script verified on 30 test cases (16 basic + 14 edge cases):

**Pval**: max|Δ| = 0.0 for all test cases
**Envelope**: max|Δ| = 0.0 for all test cases
**Tokens**: identical for all test cases
**Polar coordinates**: identical for all 30 phonemes (vowels + consonants + glides + nasals)

Test cases covered:
- Vowels: a, ɛ, i, u, ø, ɛ̃, ɔ̃, ɑ̃
- CV: ba, ma, sa, la, ra, ja
- VCV: aba
- CCV: gda, bda, pra, kla, bra
- Ternary: stra
- VCCV: adba
- VV: ai (diphthong)
- Syllable boundary: bi.gbi, bi.gi, a.di.o
- Pauses: ba | da (long), implicit GAP
- Word-end C: ad
- Multi-word: ba da gi, badagidabu, badagi | ka du
- Nasals: ɛ̃, ɔ̃, ɑ̃
- Glides: ja
