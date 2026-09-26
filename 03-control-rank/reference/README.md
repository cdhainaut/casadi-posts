# Référence reçue de Joris Gillis

Fichiers extraits **sans modification** de l'archive reçue
`dhainaut-20260914T072903Z-1-001.zip` :

- `sparse_operator.py` : exemple original ;
- `results.json` : ses résultats de référence ;
- `sparse_operator.pdf` : « Lifting the operator, not the solve ».

Le script et les résultats sont identiques octet pour octet aux références
utilisées par notre campagne J0/J1. Les empreintes de ces trois fichiers sont
conservées dans `sha256sums.txt`.

```bash
cd reference
sha256sum -c sha256sums.txt
```

Ces documents sont attribués à Joris Gillis et conservés tels que reçus.
Aucune licence supplémentaire n'est présumée ou ajoutée aux documents tiers.
L'archive reçue complète est conservée localement hors Git ; elle n'est pas une
dépendance du reproducer partagé.

Le script original écrit `results.json` dans son répertoire courant : **ne pas
le lancer depuis ce dossier**, sous peine d'écraser les valeurs de référence.
Pour reproduire les expériences, utiliser le pilote au niveau supérieur, qui
crée un dossier neuf et s'arrête au premier échec.
