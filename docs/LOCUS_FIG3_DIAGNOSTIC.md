# Diagnostic — Inversion /b/–/d/ des équations du locus (Figure 3, arXiv:2307.02299)

**Date** : 5 octobre 2026 · **Dépôt** : `P:\bigbi-demos` · **Script** : `scripts/run_locus_trough_demos.py` (Demo 2)

## 1. Symptôme

La simulation de la figure 3 de l'article donnait /d/ plus « raide » que
/b/, contrairement à la figure publiée :

| Consonne | Avant (défectueux) | Article (extraction pixel, cf. §3) | Littérature |
|---|---|---|---|
| /b/ | a = 0,974 | a ≈ 0,70 | labiale : pente la plus forte |
| /d/ | **a = 1,633** (b = −1316, r² = 0,991) | **a ≈ 0,55** | alvéolaire : pente la plus plate, locus ≈ 1700–1800 Hz |
| /g/ vélaire | (mélangé) a = 0,779 | a ≈ 2,0 (3 points) | vélaire : bimodale |
| /g/ palatale | (mélangé) | a ≈ 0,75 | |

## 2. Cause racine (prouvée)

**Bug de protocole de mesure, pas un défaut du modèle ni du synthétiseur.**

L'ancien code calculait l'instant de mesure par `v_plateau_start =
anchor.i * T_BASE`, où `anchor.i` est l'**index du nœud** dans la liste
des segments — pas un indice de temps. Pour un mot CV (« bi »), l'ancre V
a i = 1 → frame 16. Or la chronologie réelle (T = 16 frames de 10 ms) est :

| Frames | Bloc | Enveloppe |
|---|---|---|
| 0 – 15 | arc initial neutre → Vo | 0 |
| 16 – 31 | fermeture Vo → C | 0 (silence) |
| **32 – 47** | **transition C → V (release à 32)** | montée 0 → 1 |
| 48 – 63 | plateau vocalique | 1 |

F2 était donc échantillonné aux frames 19 (« locus ») et 24 (« voyelle »)
— **toutes deux pendant la fermeture silencieuse**, jusqu'à 13 frames
avant le release. La régression F2-fermeture × F2-fermeture a produit la
droite /d/ spurieusement raide.

**Preuve de reproduction** : en ré-exécutant la mesure ancienne
(frames 19/24) on retrouve exactement les valeurs défectueuses
documentées : /d/ a = 1,633, b = −1316, r² = 0,991 ; /g/ a = 0,779,
b = 353, r² = 0,972.

## 3. Référence extraite de l'article (figure 3 du PDF arXiv)

Extraction pixel des marqueurs (fichier source arXiv `Figure2.png`,
1500 × 656, calibration sur les graduations 500–3000 Hz, confirmation
visuelle de la légende : bleu = /b/, vert = /d/, rouge = /gv/ + /gp/) :

- /b/ : (879→966) (1029→1056) (1220→1240) (1783→1758) (2412→2187)
  (2534→2124) (2584→2217) (2657→2192) ⇒ **a ≈ 0,70**
- /d/ : (879→1631) (1029→1699) (1220→1832) (1783→2236) (2412→2573)
  (2534→2456) (2584→2637) (2657→2588) ⇒ **a ≈ 0,55**
- /gv/ : (879→844) (1029→1172) (1220→1528) ⇒ **a ≈ 2,0**
- /gp/ : (1783→2080) (2412→2710) (2534→2519) (2584→2773) (2657→2725)
  ⇒ **a ≈ 0,75** — noter que **/a/ est sur la branche palatale**.

L'article (§3) : « The Figure 3 is constructed by taking the F2 values
30 ms after the onset for the 8 vowels and δo = 0.5 » — « onset » =
début de la voyelle au **release** (tradition F2-onset des équations du
locus), donc frame 2T + 3. L'ordre canonique dans la littérature est
labiale > vélaire > alvéolaire (Sussman et al. 1991 ; Iskarous, Fowler
& Whalen 2010, JASA 128(4):2021–2032) : la pente de /d/ est la plus
PLATE, avec intercept haut (locus ≈ 1700–1800 Hz) — l'attente
« /d/ ≈ 1 » mentionnée initialement était erronée ; la figure de
l'article elle-même montre a(/d/) ≈ 0,55 < a(/b/) ≈ 0,70.

## 4. Balayage de sensibilité (preuve)

Paramètres balayés : `soft_rect_s` ∈ {0 (hard), 0,5, 0,75, 1,0, 2,0} ×
δo ∈ {0,5, 0,7, 1,0} × offset de mesure 0–6 frames après le release,
T ∈ {5, 10, 16} ; pentes a(/b/), a(/d/) :

- **`soft_rect_s` (valrect) est disculpé** : sur s ∈ [0,5 ; 2,0],
  a(/d/) varie de ±0,01 seulement (0,555→0,584 à release+3 ; 0,562 à
  s=0,75, release+50 ms). `valrect = 0,75` est la valeur canonique des
  packages d'origine : `vlam.py` local est **identique octet par octet**
  à `FBerthommier/timit-to-Maeda`, dont `batch_synthesize.py` a
  `--valrect 0.75` par défaut. La rectification douce s'applique à la
  **fonction d'aire** (`soft_rect(area)`, évite les sections nulles
  pendant la fermeture), pas aux paramètres articulatoires. Seul le
  hard-clipping (s → 0) détruit la synthèse (/b/ ≈ 0,03–0,06,
  /d/ ≈ 1,34 : la souplesse est nécessaire, sa valeur exacte non).
- **δo est disculpé** : 0,5 → 1,0 déplace a(/d/) de < 0,02.
- **L'instant de mesure est LE paramètre décisif** : à release+0 → +6
  frames, a(/d/) passe de 0,25 à 0,79 et a(/b/) de 0,50 à 0,92 ; la
  valeur de l'article est atteinte vers release+3 (30 ms).
- T = 10 (« T = 100 ms » de la figure 1 de l'article) donne
  (b, d, gv, gp) = (0,826 ; 0,594 ; 1,786 ; 0,827) ; T = 16 (défaut du
  dépôt) donne (0,704 ; 0,412 ; 2,107 ; 0,644). T = 16 est retenu
  (cohérence avec le reste des démos) ; les deux restituent l'ordre.

## 5. Correctif appliqué

1. `scripts/run_locus_trough_demos.py` — mesure corrigée : release
   détecté sur l'enveloppe (dernière frame silencieuse avant la
   montée = début du bloc C→V, frame 2T), **F2_locus = F2(release+3)**
   (30 ms), **F2_voyelle = moyenne du plateau** (enveloppe = 1) ;
   /a/ regroupé avec les voyelles palatales pour /g/ (conforme à la
   figure de l'article) ; chemins rendus portables (plus de
   `/home/z/...`) ; import `vtl_synth` vestigial supprimé ; demo 4
   fixe explicitement /b/ à (ρ=1, π/3) (position article, cf. son
   propre docstring) au lieu de dépendre de l'état résiduel du
   registry.
2. `config.py` — nouveau `LOCUS_OFFSET_STEPS = 3` (30 ms), commentaire
   documentant `VALRECT = 0.75` (valeur canonique d'origine).
3. `docs/manual.tex` — section Demo 2 réécrite : protocole exact,
   tableau comparatif (ci-dessous), sous-section « Root cause of the
   previous /b/–/d/ inversion », références bibliographiques
   (Sussman 1991, Iskarous et al. 2010). PDF reconstruit (23 pages).

**Aucun fichier moteur modifié** (`vlam.py` et `synthSYL/*.py`
identiques au zip de référence — vérifié par diff).

## 6. Résultats finaux

| Consonne | Pente a | Intercept b | r² | Article |
|---|---|---|---|---|
| /b/ | **0,704** | 351 Hz | 0,982 | ≈ 0,70 ✓ |
| /d/ | **0,412** | 1265 Hz | 0,964 | ≈ 0,55 (même tendance) |
| /g/ palatal (front + /a/) | 0,644 | 903 Hz | 0,809 | ≈ 0,75 |
| /g/ vélaire (back) | 2,107 | −1207 Hz | 0,999 | ≈ 2,0 ✓ |

L'ordre **a(/d/) < a(/b/) < 1** est rétabli ; le locus de /d/ aux
voyelles arrières (1689–1743 Hz) est dans la plage alvéolaire classique
(≈ 1700–1800 Hz) ; /g/+u (1029 → 966 Hz) tombe sous la diagonale
(« downward shift of the velar /g/ » de l'article).

L'axe F2_voyelle est légèrement compressé par rapport à la figure de
l'article (/i/ : 2290 Hz ici contre ≈ 2660 ; /u/ : 1029 contre ≈ 880) :
l'article utilisait une table vocalique antérieure (voyelles de coin à
ρ = 1, §2.1), révisée depuis dans timit-to-Maeda. Cela décale légèrement
les pentes individuelles mais préserve toutes les tendances.

## 7. Validation des autres démos

- `scripts/run_original_simulations.py` : les 3 simulations de l'article
  passent (Figure 1 /ibia/ T=100 ms : plage F2 = 1564–2290 Hz,
  identique à la référence `original_simulations/figure1_ibia/ibia_data.npz`).
- Équivalence numérique moteur : /ibia/ resynthétisé = référence à
  8,6 × 10⁻¹² Hz près sur les formants, 4,4 × 10⁻¹⁶ sur les paramètres.
- Demo 4 (trough) régénérée avec /b/ à la position article (ρ=1) :
  creux du Body = 2,250 à t = 160 ms dans /ibi/.

## 8. Fichiers livrés

- `docs/figures/demo2/locus_equations.png`, `locus_equations_combined.png` (régénérés)
- `docs/figures/demo4/*.png` (régénérés)
- `docs/manual.pdf` (rebuilt), `docs/manual.tex`
- `final_diff_report.txt` (diff complet contre le zip de référence)
