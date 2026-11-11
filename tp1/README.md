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

Les références reprennent les maillages du précédent benchmark TP1 (`run_two.py`). Il faut garder ces paramètres et la même machine pour les comparaisons ultérieures, ou modifier explicitement les trois tailles avant d'établir la référence.

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

Chaque campagne produit un CSV de mesures, un JSON décrivant la machine, les paramètres et les empreintes des exécutables et du code C++, puis ses figures dans `benchmarks/plots/<nom_du_csv>/`. Les anciennes figures restent directement dans `benchmarks/plots/`. Le diagnostic de la dernière simulation est conservé dans `benchmarks/results/last_simulation.log`.

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

Les figures présentent les moyennes et écarts-types échantillonnaux ; `summary.csv` contient aussi médianes, minimums, maximums et coefficients de variation. Les ratios CSV/binaire et CPU/GPU utilisent les médianes. `--no-render` retire le rendu des images : ne pas comparer cette campagne à une campagne qui le mesure.

Pour régénérer les figures ou rassembler deux campagnes compatibles CPU et GPU :

```bash
python3 tp1/benchmarks/plot_benchmarks.py --input chemin_cpu.csv chemin_gpu.csv
```

Le script refuse de mélanger machines, paramètres, code C++, politique de synchronisation ou activation du rendu différents. Aucun résultat GPU n'est produit en l'absence de mesures GPU. Les anciennes mesures du groupe sont conservées, mais leur instrumentation ne doit pas être confondue avec celle des nouvelles campagnes.

À 19 milliards de nœuds, la pression float32 seule représente **76 Go par snapshot** (70,78 Gio). Les coordonnées représentent **228 Go**, une seule fois : K snapshots binaires occupent environ **228 + 76K Go**, hors petits JSON. Le CSV répète les coordonnées à chaque sortie ; son volume dépend de la représentation décimale et se mesure empiriquement. Une extrapolation de temps demande les débits réels du serveur.

## État local

La compilation et les sorties CPU ont été vérifiées : égalité des float32 CSV/binaire, plusieurs récepteurs dont un sur le bord du domaine, concordance des pressions et des temps entre récepteurs et snapshots, volumes et chronos, écriture avec `fsync`. `benchmarks_validation_cpu.csv` teste la chaîne complète avec trois petites tailles et trois répétitions ; ses résultats ne remplacent pas les références S/M/L sur le serveur. Les mesures GPU et la comparaison CPU/GPU restent à exécuter sur ce serveur.

Sur la plus grande taille de cette validation (17³ nœuds, six snapshots), le binaire écrit 0,170 Mio contre 0,743 Mio en CSV, soit 4,38 fois moins. Les médianes d'écriture sont 0,0021 s et 0,1049 s ; celles de lecture sont 0,0026 s et 0,1090 s. Le workflow complet passe seulement de 2,52 s à 2,24 s : le rendu des images domine à cette taille. Pour les sismogrammes seuls, les volumes baissent aussi, mais le workflow reste proche de 0,57 s dans les deux formats. Ces chiffres illustrent la chaîne locale ; ils ne prédisent pas les performances du serveur GPU.
