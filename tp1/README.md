# TP1 — Workflow ad-hoc

Le TP1 réutilise les scripts du groupe, auparavant dans `scripts/` :

- `benchmarks/run_benchmarks.py` : simulation, post-traitement et mesures ; la variation de maillage de l'ancien `run_two.py` est intégrée ici.
- `benchmarks/plot_benchmarks.py` : figures Gnuplot et résumé des répétitions.
- `visualisation/visu_snapshots.py` et `visu_sismos.py` : visualisation après la simulation, en CSV ou en binaire.
- `benchmarks/results/`, `benchmarks/plots/`, `visualisation/data/` et `visualisation/images/` : résultats et images, y compris ceux déjà obtenus avant cette réorganisation.

Aucune nouvelle bibliothèque. Les scripts utilisent Python standard et Gnuplot, déjà utilisé par le projet. Le calcul C++ utilise le sous-module Kokkos existant. Les traitements in-situ restent désactivés pour les mesures TP1.

## Compiler sur CPU

Depuis la racine du projet :

```bash
cmake -S . -B build-openmp -DCMAKE_BUILD_TYPE=Release \
  -DUSE_KOKKOS=ON -DUSE_VECTOR=OFF -DENABLE_CUDA=OFF \
  -DCOMPILE_FD=OFF -DSEMPROXY_ENABLE_CUFFT=OFF
cmake --build build-openmp --target semproxy -j 2
```

La compilation GPU doit être faite sur le serveur, avec sa configuration CUDA/Kokkos adaptée à la carte. Aucun outil ne compile automatiquement une version GPU.

## Lancer les benchmarks


| Configuration | Éléments par axe | Ordre | Nœuds par axe | Nombre de nœuds |
|---|---:|---:|---:|---:|
| S | 10 | 2 | 21 | 9 261 |
| M | 15 | 2 | 31 | 29 791 |
| L | 20 | 2 | 41 | 68 921 |

Domaine : 2 000 m par direction ; `dt=0.001 s` ; durée demandée `1.5 s` ; un snapshot tous les 50 pas. Les récepteurs sont lus dans `data/receivers.csv`. Le nombre de pas effectivement calculé et le `dt` utilisé sont enregistrés dans les mesures.

```bash
# CPU : quatre modes de sortie, CSV et binaire, cinq répétitions et une chauffe.
python3 tp1/benchmarks/run_benchmarks.py --threads 4 --runs 5

# Sur le serveur : CPU et GPU de la même machine, avec exactement les mêmes paramètres.
python3 tp1/benchmarks/run_benchmarks.py \
  --cpu-exe ./build-openmp/bin/semproxy \
  --gpu-exe ./build-cuda/bin/semproxy --threads 4 --runs 5

# Contrôle rapide du workflow ; ces petites tailles ne constituent pas la référence S/M/L.
python3 tp1/benchmarks/run_benchmarks.py --quick --threads 2 --runs 3
```

Les quatre modes sont `none`, `snapshots`, `sismos`, `combined`. Le mode sans sortie est exécuté une seule fois par répétition, sans échantillonnage des récepteurs. Pour les trois autres, les formats `csv` et `bin` sont testés séparément. L'ordre des cas est mélangé avec une graine fixe ; les exécutions de chauffe sont exclues du CSV.

Options utiles : `--sizes 10,15,20`, `--order 2`, `--dt 0.001`, `--timemax 1.5`, `--interval 50`, `--formats csv,bin`, `--modes none,snapshots,sismos,combined`, `--threads 1,2,4`, `--io-sync fsync`, `--no-render`, `--dry-run` et `--output chemin.csv`.

Chaque campagne produit un CSV de mesures, un JSON décrivant la machine, les paramètres et les empreintes des exécutables et du code C++, puis les figures dans `benchmarks/plots/`. Les anciennes figures OpenMP sont conservées dans `benchmarks/plots_cpu_threads/`. Le diagnostic de la dernière simulation est conservé dans `benchmarks/results/last_simulation.log`.

Les données brutes et images du dernier cas sont dans `visualisation/data/benchmark/` et `visualisation/images/benchmark/`. Elles sont remplacées au cas suivant ; les tableaux de mesures sont conservés. Pour conserver une simulation particulière, lancer directement `semproxy` avec un dossier de sortie dédié.

## Sorties et visualisation

```bash
OMP_NUM_THREADS=4 ./build-openmp/bin/semproxy \
  --ex 15 --ey 15 --ez 15 --order 2 --dt 0.001 --timemax 1.5 \
  --is-snapshots=true --snap-time-interval 50 --snap-format bin \
  --snap-folder tp1/visualisation/data/exemple \
  --rcv-file data/receivers.csv \
  --sismo-file tp1/visualisation/data/exemple/sismograms.bin --sismo-format bin \
  --metrics-file tp1/benchmarks/results/exemple.json --quiet=true

python3 tp1/visualisation/visu_snapshots.py \
  --data tp1/visualisation/data/exemple --format bin --position 1000 \
  --output tp1/visualisation/images/exemple \
  --metrics tp1/benchmarks/results/exemple_snapshots.json

python3 tp1/visualisation/visu_sismos.py \
  --input tp1/visualisation/data/exemple/sismograms.bin \
  --output tp1/visualisation/images/exemple \
  --metrics tp1/benchmarks/results/exemple_sismos.json
```

Pour le CSV, remplacer `bin` par `csv`, y compris l'extension des sismogrammes. `--snap-format both` et `--sismo-format both` permettent de vérifier les deux formats sur une même exécution ; les benchmarks les mesurent séparément.

Les snapshots CSV ont les colonnes `x,y,z,pressure`, avec assez de chiffres pour retrouver exactement les float32. En binaire, `coords.bin` contient les coordonnées `(x,y,z)` entrelacées, écrites une seule fois ; chaque `snapshot_it_XXXXXX.bin` contient une pression par nœud dans le même ordre. Les fichiers `.json` associés donnent le type `<f4` (float32 little-endian), le nombre de nœuds, les coordonnées, le pas et le temps physique.

Les sismogrammes CSV ont les colonnes `time,receiver_0,...`. En binaire, les valeurs sont rangées par pas de temps, puis par récepteur. Le JSON associé donne les positions, le nombre d'échantillons, `dt` et le premier instant. Les snapshots et les récepteurs utilisent la nouvelle pression calculée (`i1`), à l'instant `(step+1)*dt`.

La visualisation des snapshots extrait **après lecture du volume complet** les trois plans nodaux les plus proches de la position demandée. Gnuplot utilise les coordonnées physiques : les espacements GLL ne sont pas assimilés à des pixels uniformes. La palette bleu/blanc/rouge est symétrique autour de zéro, avec une amplitude par snapshot ; les couleurs de deux images ne représentent donc pas nécessairement la même amplitude absolue.

## Lire les mesures

Les mesures séparent le temps total du processus, l'initialisation, le calcul synchronisé, l'échantillonnage des récepteurs, l'écriture des snapshots et des sismogrammes, leur lecture/décodage, l'analyse et le rendu. Le volume brut inclut coordonnées et métadonnées ; le volume des images est compté séparément.

Les chronos C++ utilisent une horloge monotone. Un `Kokkos::fence()` termine les kernels avant de fermer le chrono de calcul ou de lire les données sur l'hôte. L'écriture inclut la préparation des données, les migrations UVM éventuelles, la sérialisation et la fermeture des fichiers.

Par défaut, `--io-sync buffered` mesure les écritures vers le cache du système. `--io-sync fsync` attend aussi un `fsync` de chaque fichier ; ce temps est inclus dans l'écriture et renseigné séparément comme sous-ensemble, sans le compter deux fois. La lecture suit l'écriture : le cache OS n'est pas vidé, ce n'est pas une mesure de lecture à froid. Le débit affiché est un débit logique du workflow, pas un débit matériel du disque.

Les figures présentent les moyennes et écarts-types des répétitions du benchmark. Pour ces mesures de temps et de volume, `summary.csv` contient aussi les médianes, minimums, maximums et coefficients de variation. Ces minimums et maximums décrivent les mesures du benchmark, pas les valeurs du champ de pression. Les comparaisons graphiques affichent les temps en secondes ou les volumes en Mio. `--no-render` retire le rendu des images : ne pas comparer cette campagne à une campagne qui le mesure.

## Refaire les figures sans relancer les simulations

Le lanceur peut relire un CSV existant. Aucun exécutable CPU ou GPU n'est nécessaire pour cette opération :

```bash
python3 tp1/benchmarks/run_benchmarks.py \
  --plot-only tp1/benchmarks/results/benchmarks_20261111T130533.603751Z.csv \
  --columns workflow_wall_s,compute_s,save_s,read_s,raw_output_bytes \
  --threads 4 --clean-plots
```

`--columns` sélectionne les mesures à tracer. `save_s` correspond à l'écriture des snapshots et des sismogrammes ; `read_s` est la somme de `snapshot_read_s` et `sismo_read_s`. Pour tracer uniquement le workflow complet, utiliser `--columns workflow_wall_s`.

Les comparaisons utilisent les sorties combinées : snapshots **et** sismogrammes. Chaque figure comporte deux courbes. Pour comparer CPU et GPU, le format reste fixé ; pour comparer CSV et binaire, le backend reste fixé. Les mesures des autres modes restent dans le CSV.

Avec les cinq colonnes de la commande ci-dessus, les figures produites sont :

| Fichier | Deux courbes comparées |
|---|---|
| `cpu_vs_gpu.png` | CPU / GPU, workflow complet en binaire, visualisation incluse |
| `cpu_vs_gpu_csv.png` | CPU / GPU, workflow complet en CSV, visualisation incluse |
| `cpu_vs_gpu_compute.png` | CPU / GPU, calcul seul pour les sorties binaires |
| `binary_vs_csv_cpu.png` | CSV / binaire, workflow complet sur CPU, visualisation incluse |
| `binary_vs_csv_gpu.png` | CSV / binaire, workflow complet sur GPU, visualisation incluse |
| `binary_vs_csv_write_cpu.png` | CSV / binaire, temps total d'écriture sur CPU |
| `binary_vs_csv_write_gpu.png` | CSV / binaire, temps total d'écriture sur GPU |
| `binary_vs_csv_read_cpu.png` | CSV / binaire, temps total de lecture après la simulation CPU |
| `binary_vs_csv_read_gpu.png` | CSV / binaire, temps total de lecture après la simulation GPU |
| `binary_vs_csv_storage.png` | CSV / binaire, volume brut écrit pour la simulation CPU |

Les images sont enregistrées directement dans `tp1/benchmarks/plots/`. `summary.csv` contient les statistiques des cas combinés et `plotted_columns.csv` indique la colonne et les données utilisées pour chaque courbe. `--clean-plots` remplace les anciens graphes générés et retire l'ancien sous-dossier horodaté de la campagne ; les CSV de mesures restent dans `benchmarks/results/`.

Les colonnes de temps d'écriture, de lecture et d'analyse propres aux snapshots ou aux sismogrammes peuvent aussi être sélectionnées, par exemple `--columns snapshot_write_s,sismo_write_s`. Elles sont toujours extraites des simulations avec les deux sorties activées.

Le script vérifie que les données utilisent la même machine, les mêmes paramètres, le même code C++, la même politique de synchronisation et le même choix de rendu. Aucun graphe CPU/GPU n'est produit en l'absence de mesures appariées.

## Résultats sur le serveur

La campagne `benchmarks_20261111T130533.603751Z` a été exécutée sur `jessica`, avec un Intel Core Ultra 9 285 et une NVIDIA RTX 4000 Ada Generation. La version CPU utilise Kokkos OpenMP avec 4 threads ; la version GPU utilise CUDA. Les deux versions sont compilées en Release.

Les 42 configurations ont chacune été mesurées cinq fois après une exécution de chauffe, soit 210 mesures conservées. Chaque simulation comporte 1 499 pas, avec le même maillage, le même pas de temps et les mêmes récepteurs pour CPU et GPU. Les écritures utilisent le mode `buffered` ; les lectures suivent les écritures, sans vidage du cache système.

Pour les sorties combinées (snapshots et sismogrammes), les temps médians du workflow complet sont :

| Taille | CPU CSV (s) | GPU CSV (s) | CPU/GPU CSV | CPU binaire (s) | GPU binaire (s) | CPU/GPU binaire |
|---|---:|---:|---:|---:|---:|---:|
| S | 3,75 | 3,38 | 1,11 | 3,34 | 2,97 | 1,12 |
| M | 7,54 | 5,02 | 1,50 | 6,01 | 3,60 | 1,67 |
| L | 12,88 | 6,72 | 1,92 | 10,53 | 4,21 | 2,50 |

En L, avec les sorties combinées en binaire, le calcul seul passe de 6,63 s sur CPU à 0,073 s sur GPU, soit un rapport d'environ 91. Le rapport sur le workflow complet est de 2,50 : l'initialisation, les sorties et la visualisation occupent une part importante du temps restant. La lecture, l'analyse et le rendu Gnuplot sont réalisés sur CPU dans les deux versions.

Les 30 snapshots de L occupent 53,80 Mio en CSV contre 8,68 Mio en binaire, coordonnées et métadonnées comprises, soit environ 6,2 fois moins. Les sismogrammes seuls restent beaucoup plus petits : environ 158 Kio en CSV contre 23,7 Kio en binaire. Leur réduction de volume ne se traduit pas systématiquement par un gain sur le temps total ; certaines mesures courtes présentent aussi une dispersion importante, visible dans `summary.csv`.

Les [figures à deux courbes](benchmarks/plots/), le [résumé des répétitions](benchmarks/plots/summary.csv) et la [liste des colonnes tracées](benchmarks/plots/plotted_columns.csv) proviennent du [CSV de mesures](benchmarks/results/benchmarks_20261111T130533.603751Z.csv). La [description de la machine et des paramètres](benchmarks/results/benchmarks_20261111T130533.603751Z.json) accompagne ce CSV.

Le dossier `benchmarks/plots_cpu_threads/` contient les mêmes images que les anciennes figures placées directement dans `benchmarks/plots/`. La campagne serveur ci-dessus utilise uniquement 4 threads CPU ; elle ne constitue pas une nouvelle mesure de passage à l'échelle en nombre de threads.

