# Moins de commandes : que reste-t-il du gain de la levée ?

Reprise du modèle de **Joris Gillis**, puis modification du seul nombre de
commandes indépendantes. Le cas est autonome et sans contenu métier.

**Résultat : les 16 cas convergent et passent les critères fixés.** Le gain de
l'opérateur creux se reproduit ; réduire le nombre de commandes réduit surtout
le gain supplémentaire apporté par la levée. Cela ne démontre aucune limite
générale de la méthode.

Pour l'échange : [synthèse et questions à Joris](DISCUSSION.md).
Pour les preuves : [mesures et provenance](validation/README.md).

## 1. Ce qui est conservé, ce qui change

Le noyau exponentiel de Joris est exactement l'inverse d'un opérateur
tridiagonal : `K(p)=L(p)⁻¹`. Quatre paramètres de design `p`, un état scalaire
`v`, des commandes spatiales `u`, une réponse `y` et sa lecture `z` :

```text
D = diag(ω w(p) / v)
b = ω w(p) ⊙ (u + 0.1 s)
(I + D K) y = b
z = K y
```

L'objectif non linéaire local, les bornes et les valeurs initiales sont ceux de
la [source originale](reference/sparse_operator.py), accompagnée de sa
[note](reference/sparse_operator.pdf). Seule la commande devient `u=B_r a` :

- `r=N` : une commande par station, `B_N=I` ;
- `r=4` : les quatre premiers polynômes de Legendre aux stations ;
- `r=1` : une commande constante sur toutes les stations.

Les bornes `−1 ≤ u_i ≤ 1` restent imposées aux stations. Réduire le rang change
l'espace admissible ; ce n'est pas un simple changement de coordonnées.
Les réponses levées sont initialisées par résolution au point initial commun.

| Écriture | Réponse ajoutée au NLP | Équation |
|---|---|---|
| Dense éliminée | aucune | `y=solve(I+DK,b)`, puis `z=Ky` |
| Dense levée | `y` | `(I+DK)y=b`, puis `z=Ky` |
| Sparse éliminée | aucune | `z=solve(L+D,b)` |
| Sparse levée | `z` | `(L+D)z=b` |

Les quatre écritures sont algébriquement équivalentes **à rang fixé**.

## 2. J0 — reproduction à 192 stations et 192 commandes

| Écriture | Variables | nnz H (triangle) | H/appel moyen | Temps `opti.solve()` |
|---|---:|---:|---:|---:|
| Dense éliminée | 197 | 19 503 | 1 386,22 ms | 28,759 s |
| Dense levée | 389 | 57 517 | 1 295,13 ms | 29,676 s |
| Sparse éliminée | 197 | 19 503 | 20,83 ms | 1,079 s |
| **Sparse levée** | **389** | **2 119** | **0,142 ms** | **0,151 s** |

Objectif commun : **−4213,93302773**, en accord avec la référence de Joris.
Dimensions et nnz J/H identiques à sa table. Les nombres de nœuds du graphe
ne sont pas tous identiques à ceux de son JSON historique ; cette identité
n'était pas un critère de validation. J0 utilise le harnais adapté, pas
l'exécution non instrumentée du script original.

## 3. J1 — 42 stations, puis 42, 4 et 1 commande

Temps `opti.solve()`, préparation du solveur comprise :

| Écriture | 42 commandes | 4 commandes | 1 commande |
|---|---:|---:|---:|
| Dense éliminée | 0,231 s | 0,165 s | 0,257 s |
| Dense levée | 0,257 s | 0,218 s | 0,318 s |
| Sparse éliminée | 0,141 s | 0,140 s | 0,123 s |
| Sparse levée | 0,130 s | 0,108 s | 0,118 s |

Pour isoler l'effet de la **levée**, comparer sparse-éliminé et sparse-levé :

| Commandes | nnz H éliminé / levé | H/appel moyen éliminé / levé | Gain moyen H |
|---|---:|---:|---:|
| 42 | 1 128 / 469 | 0,798 / 0,035 ms | ×22,7 |
| 4 | 45 / 449 | 0,143 / 0,060 ms | ×2,4 |
| 1 | 21 / 305 | 0,107 / 0,054 ms | ×2,0 |

**Lecture simple :** éliminer la réponse couple les commandes entre elles.
Lorsqu'il n'y a plus qu'une commande, le Hessien réduit n'a que 21 coefficients
triangulaires : il reste moins à gagner en exposant 42 inconnues supplémentaires.

Cela ne supprime pas l'intérêt de l'opérateur creux. Avec une commande,
dense-éliminé coûte encore 0,257 s contre 0,123 s pour sparse-éliminé.
Cela ne rend pas non plus les temps monotones : dense-éliminé demande 12
itérations avec 42 commandes, contre 30 avec une seule.

## 4. Portée des mesures

- **Fait :** un run par cas, quatre écritures convergées à chaque rang ;
  mêmes objectifs à rang fixé, pas forcément les mêmes paramètres optimaux.
- **Fait :** les moyennes H/appel sont mesurées sur les trajectoires du solveur.
  Au point initial commun, les ratios sparse-éliminé/levé valent plutôt
  ×8,0, ×1,4 et ×1,0 (médiane de trois appels après échauffement, multiplicateurs
  fixés). Ces appels externes incluent le coût Python/CasADi.
- **Limite :** des écarts de quelques millisecondes sur les petits solves ne
  sont pas des avantages statistiques établis. Le temps `opti.solve()` inclut
  la préparation du solveur ; le coût KKT seul n'est pas instrumenté.
- **Limite :** un état de champ par station, un seul scénario ; aucune mesure
  d'un champ auxiliaire 2D plus grand ni de scénarios multiples.
- **Hypothèse étayée :** réduire le rang des commandes réduit l'intérêt propre
  de la levée dans cet exemple. **Non démontré :** un critère général de choix
  de formulation, ou un seuil universel en taille.

## 5. Reproduire depuis ce dépôt

Depuis `03-control-rank/`, dans un environnement Python ≥ 3.11 :

```bash
python -m pip install -r requirements.txt
```

Versions mesurées : Python 3.11.15, NumPy 2.4.6, CasADi 3.7.2,
IPOPT 3.14.11 / MUMPS 5.4.1. Les versions du backend peuvent dépendre du paquet
installé ; elles sont imprimées dans chaque log.

Enveloppe Linux de référence, puis campagne complète :

```bash
ulimit -v 2097152
ulimit -s 65536
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
nice -n 10 python run_joris_controls.py \
  --reference-source reference/sparse_operator.py \
  --reference-results reference/results.json \
  --output runs/reproduction_NEW
```

Un cas seulement, sous la même enveloppe :

```bash
nice -n 10 timeout 300s python bench_joris_controls.py \
  --stations 42 --controls 4 --kernel sparse --lifted \
  --output runs/single_NEW.json
```

Chaque sortie doit être nouvelle. Le pilote crée un processus neuf par cas,
timeout 300 s, sauvegarde les sources/logs/résultats et s'arrête au premier
échec sans retry. Ne pas exécuter de campagne dans `reference/` ou `validation/` :
ce sont les références conservées. Les nouvelles sorties `runs/` sont ignorées
par Git. Aucun chemin vers un autre dépôt n'est nécessaire.

## 6. Vérifications rapides

Installer les outils de développement si nécessaire :

```bash
python -m pip install pytest ruff
# Sous la même enveloppe ; les smoke tests lancent quatre petits solves.
nice -n 10 timeout 180s python -m pytest -x -q
python -m ruff check .
```

**Validation du transfert : 30 tests passent**, à la fois dans ce dépôt et dans
une copie isolée sans autre checkout ni `PYTHONPATH`. Ruff et vérification des
empreintes passent également. Aucune nouvelle campagne J0/J1 n'a été lancée.

Les tests vérifient les bases, la parité avec la source originale, les dérivées
réduites et différences finies, les quatre CLI en dehors de ce dossier,
l'arrêt au premier échec et la cohérence des 16 résultats importés.
Pour éviter tout solve : `python -m pytest -m 'not slow' -q`.

Le harnais conserve les options et critères de la campagne mesurée ; la seule
modification du benchmark lors de son import est un lien dans son docstring.
Les détails de provenance et les critères KKT figurent dans
[validation/README.md](validation/README.md).
