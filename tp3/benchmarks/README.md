# TP3 — Benchmarks PPM et cartes cumulées

Mesures sur `jessica` : Intel Core Ultra 9 285, quatre threads OpenMP,
NVIDIA RTX 4000 Ada Generation, compilation Release.

| Taille | Éléments | Nœuds | Image |
|---|---:|---:|---:|
| S | 10³ | 9 261 | 21 × 21 |
| M | 15³ | 29 791 | 31 × 31 |
| L | 20³ | 68 921 | 41 × 41 |

Domaine de 2 000 m par direction, ordre 2, pas de 0,001 s, 1 499 pas exécutés.
Une image XY tous les 50 pas donne 30 images. Les pixels sont uniformément
espacés et prennent la pression du nœud le plus proche.
Une chauffe exclue, puis cinq mesures : moyenne et écart-type sur les courbes.
Les temps des étapes sont cumulés ; le temps total comprend initialisation,
simulation et sorties. La référence sans traitement ne produit aucune donnée.

## `ppm_formats.png` — P3 ou P6

Les courbes comparent les mêmes images générées sur GPU, en texte P3 et en
binaire P6. En haut : temps total et volume des fichiers. En bas : encodage
et écriture des images.

En L, P3 occupe 0,574 Mio contre 0,145 Mio pour P6, à pixels identiques.
L'encodage passe de 0,850 ms à 0,0059 ms. Le volume est environ quatre fois
plus faible ; le gain sur le temps total est moins net.

## `ppm_cpu_gpu.png` — Calcul de l'image

La simulation reste sur GPU ; seul le calcul de l'image change de processeur.
En haut : temps total et génération des images. En bas : volume logique des
pressions ou du RGB, puis calcul de propagation. Le volume affiché ne mesure
pas les copies réelles entre GPU et CPU.

En L, la génération des 30 images passe de 3,091 ms sur CPU à 1,647 ms sur GPU.
Les dispersions du temps total se recouvrent : le gain global reste incertain.

## `normalisation.png` — Échelle des couleurs

Bleu pour les pressions négatives, blanc à zéro, rouge pour les positives.
Trois bornes sont comparées : maximum de l'image courante, valeur fixe de 2,5
et maximum des huit dernières images.
À gauche : évolution de la borne pour L. À droite : coût des images.

La méthode locale rend les petites ondes visibles. La borne fixe permet de
comparer les amplitudes entre images. La méthode glissante limite les changements
brusques. Aucun avantage de temps net ne se dégage.

## `cartes_cout.png` — Coût des cartes

Chaque nœud conserve son amplitude maximale, sa première arrivée au seuil
absolu 0,01 et sa somme de p² : 16 octets par nœud pour les trois cartes.

En haut : temps total sans traitement et avec les cartes, puis mises à jour
individuelles ou groupées. En bas : mises à jour, copies, écritures et images
finales ; puis RAM des cartes et volume des snapshots sur disque.
En L : 0,324 s sans traitement contre 0,417 s avec les cartes ; les mises à jour
prennent 10,6 ms et les écritures 52,8 ms. Le dernier panneau compare RAM et
disque, sans donner un taux de compression.

## `cartes_precision.png` — Tous les pas ou snapshots espacés

La référence utilise tous les pas ; l'autre méthode utilise les snapshots espacés
de 50 ms. Pour la somme de p², la dernière valeur est maintenue entre deux sorties.
En haut : erreurs d'amplitude et arrivées manquées. En bas : retards des arrivées
et erreurs sur la somme de p². La RMSE résume l'erreur sur tous les nœuds ; les
erreurs d'amplitude et de somme de p² sont divisées par le maximum de leur carte.

En L : erreur maximale de 11,02 % sur l'amplitude et 2,54 % sur la somme de p².
Aucun nœud manqué, mais 27,8 ms de retard moyen, jusqu'à 333 ms. Un dépassement
bref du seuil peut être manqué puis une oscillation ultérieure détectée.

## `ppm_fsync.png` — Sans et avec fsync

`buffered` utilise le cache système ; `fsync` attend la synchronisation des
fichiers. Deux courbes par panneau, sans et avec fsync.
En haut : écriture et synchronisation des P3, puis P6. En bas : temps total
avec images P6, puis avec les trois cartes. Sans mesures de cartes, le dernier
panneau détaille le fsync des P6 et du CSV. Le CSV est lui aussi synchronisé.

Les résultats actuels sont uniquement en mode `buffered`. Cette nouvelle figure
attend les mesures avec fsync ; aucun gain chiffré n'est encore établi.
Les cinq premières figures restent basées sur les résultats sans fsync.

## Données et complément fsync

Les campagnes [CPU](results/tp3_20261112T141952.476279Z_cpu.csv) et
[GPU](results/tp3_20261112T142957.977461Z_gpu.csv) contiennent 135 et 165 mesures.
Les 30 contrôles CPU et les 66 contrôles GPU passent. Les petites tailles et la
dispersion limitent les conclusions sur les grands calculs et le temps total.

Depuis la racine du dépôt, avec le binaire GPU existant :

```bash
python3 tp3/benchmarks/benchmark_tp3_visu.py \
  --exe ./build-cuda/bin/semproxy --expect-backend gpu \
  --threads 4 --runs 5 --warmups 1 --io-sync fsync \
  --modes baseline,ppm_gpu_P3,ppm_gpu_P6,maps_all \
  --compare-input tp3/benchmarks/results/tp3_20261112T142957.977461Z_gpu.csv \
  --keep-data
```
