# Mesures conservées et validation

`results.json` contient les résultats des **16 cas J0/J1**, importés de la
campagne `joris_controls_j0_j1_20260926T092000Z`. Ce sont des mesures existantes,
**pas une nouvelle campagne exécutée lors du rangement de ce dépôt**.

## Provenance

- Les objets numériques `result` ont été conservés exactement, sans changement
  de précision, arrondi ni renommage de métrique.
- L'export retire seulement les commandes contenant des chemins propres à la
  machine et les codes de retour enfant redondants. Il conserve les statuts,
  l'enveloppe, les critères et les résultats.
- `provenance.json` contient l'identifiant de l'archive source et les SHA-256
  du manifeste de campagne, des sources exécutées et des 16 JSON individuels.
- `sha256sums.txt` permet de vérifier les fichiers de données partagés ici.
  Les logs bruts complets restent dans l'archive d'origine ; ils ne sont pas
  nécessaires pour exécuter le reproducer.
- Le pilote `run_joris_controls.py` est identique octet pour octet au pilote
  mesuré. Le module `bench_joris_controls.py` diffère uniquement par un lien
  documentaire corrigé. Les tests vérifient aussi cette correspondance.
- La source et le JSON reçus de Joris sont dans `../reference/`, inchangés.

```bash
cd validation
sha256sum -c sha256sums.txt
```

## Critères fixés avant la campagne

- Statut `Solve_Succeeded` pour chaque cas.
- Erreur de la paire `LK−I` ≤ 1e-10 à trois designs admissibles.
- Violation primale initiale ≤ 1e-8.
- Résidus finaux non adimensionnés : primal ≤ 1e-6, stationnarité ≤ 1e-4,
  complémentarité ≤ 1e-4. Ce sont des portes externes, pas les options IPOPT.
- Accord des objectifs à rang fixé : `rtol=1e-8`, `atol=1e-7`.
- À J0 : accord avec l'objectif, les dimensions et les nnz J/H archivés par
  Joris. Les temps et nombres de nœuds de graphe n'étaient pas des portes.

Les 16 cas ont passé ces critères. Les résidus primaux finaux atteignent environ
2e-7 et la complémentarité environ 4,2e-5 : ils reflètent notamment la relaxation
des bornes IPOPT. Ne pas les présenter comme inférieurs à `tol=1e-8`.
Un même objectif ne démontre ni l'unicité du design ni l'optimalité globale.

## Lire les métriques

- `solve_seconds` : temps Python de `opti.solve()`, préparation comprise ;
  `model_build_seconds` mesure séparément la construction du modèle.
- `hessian_triangle_nnz` : triangle effectivement fourni à IPOPT, pas le
  Hessien symétrique complet.
- `hessian_mean_seconds` : moyenne des appels H durant le solve, avec les
  points/multiplicateurs propres à chaque trajectoire.
- `hessian_initial_seconds` : trois appels après échauffement, au point initial,
  facteur objectif 1 et multiplicateurs tous égaux à 1. Coût d'appel externe
  Python/CasADi inclus ; ce n'est pas la même mesure que la moyenne IPOPT.
- `peak_rss_kib` : pic RSS Linux du processus entier, contrôles et mesures après
  solve inclus. Ce n'est ni le pic d'adressage virtuel ni la taille des facteurs.
- `solution` : objectif, paramètres, commandes aux stations et résidus KKT
  calculés avec les fonctions effectives du solveur.

Les fonctions de mesure ne chronomètrent pas séparément la factorisation KKT.
Les différences entre temps total et temps H ne lui sont donc pas attribuées.

## Revalidation sans nouvelle campagne

Depuis `03-control-rank/` :

```bash
python -m pytest tests/test_reference_results.py -q
```

Ces tests relisent les données, appliquent les portes d'origine, vérifient que
les 16 cas attendus sont présents et contrôlent les empreintes des références.
Pour un nouveau benchmark, utiliser un nouveau dossier `runs/` ; ne pas
remplacer les mesures publiées par une nouvelle exécution.
