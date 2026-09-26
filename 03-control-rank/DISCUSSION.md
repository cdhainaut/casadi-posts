# Base d'échange avec Joris

Merci pour la note « Lifting the operator, not the solve ». Nous avons repris
votre exemple pour comprendre ce qui change lorsque les commandes sont moins
nombreuses que les stations, avant d'ajouter des scénarios multiples ou un
champ auxiliaire plus grand.

## Ce qui est reproduit

À N=192, les quatre écritures convergent au même objectif que votre référence,
avec les mêmes dimensions et nnz J/H. Le temps `opti.solve()` passe de
28,76 s en dense-éliminé à 0,15 s en sparse-levé. Les timings absolus diffèrent
de votre machine ; le mécanisme et la structure se reproduisent.

À N=42, nous conservons votre noyau, objectif, quatre paramètres de design et
état scalaire, puis écrivons `u=B_r a`, avec 42, 4 et 1 commande indépendante.
Les bornes restent imposées à `u` aux stations. Les bases réduites sont des
polynômes de Legendre fixes, incluant la commande constante.

Le ratio des coûts moyens du Hessien sparse-éliminé / sparse-levé passe de
×22,7 à ×2,4 puis ×2,0. Les petits temps de solve, de l'ordre de 0,1 s préparation
comprise, sont proches : un seul run ne permet pas d'en établir finement le
classement. Le noyau creux reste avantageux même avec une seule commande.

[Équations, tableaux et reproduction](README.md) —
[mesures et critères](validation/README.md).

## Notre lecture, à discuter

Avec beaucoup de commandes indépendantes, éliminer la réponse donne un Hessien
réduit dense de grande taille. Avec quelques commandes seulement, ce Hessien
est petit : exposer un champ peut encore accélérer ses dérivées, mais il reste
moins à gagner. Cette lecture est cohérente avec les mesures ; ce n'est pas
un critère universel de choix.

Nous souhaitons ensuite séparer deux effets qui ne sont **pas encore mesurés** :

1. plusieurs scénarios, avec paramètres de design partagés ;
2. un champ local de taille M>N, lu aux N stations par une application locale.

## Questions

1. Cette évolution avec le rang des commandes correspond-elle à votre analyse ?
   Les comparaisons et le paramétrage des commandes vous semblent-ils pertinents ?
2. Pour un champ plus grand, la paire algébrique exacte suivante est-elle une
   bonne extension de votre exemple, avec injection `S` et lecture locale `R` ?

   ```text
   L ψ = S y,   z = R ψ,   y + D z = b
   K = R L⁻¹ S
   (L + S D R) ψ = S b
   ```

3. Si la fermeture n'a plus de terme identité en `y`, faut-il garder le système
   par blocs `[L, −S ; D R, 0]`, plutôt que chercher la même élimination locale ?
4. Quelles mesures de factorisation et de mémoire permettraient de distinguer
   un coût intrinsèque du champ d'un défaut de formulation ou d'implémentation ?

Le dossier contient un reproducer sans dépendance métier. L'objectif de cet
échange est de départager ces effets, pas de conclure d'avance à une limite de
la levée d'opérateur. Aucun résultat multi-scénarios ou 2D n'est revendiqué ici.
