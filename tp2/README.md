# TP2 — Ad-hoc et in-situ

On compare les deux méthodes avec les mêmes analyses et les mêmes instants de sortie. Les données brutes sont en binaire float32, comme au TP1. Les mesures sont enregistrées en CSV et JSON.

## Analyses

Pour les snapshots : min, max, moyenne et variance.
Pour les sismogrammes : les mêmes quatre statistiques, avec la FFT en plus après retrait de la moyenne. Les snapshots n’ont pas de FFT.

Le maximum suit l’amplitude de l’onde, la variance sa dispersion et le spectre des sismogrammes leur contenu fréquentiel.

Les statistiques sont mises à jour en flux. La variance utilise Welford en double précision, avec fusion des résultats pour la réduction parallèle. Pour les récepteurs, la mémoire des statistiques ne dépend pas du nombre de pas. La FFT complète conserve toutes les traces en mémoire : son coût mémoire augmente avec le nombre de récepteurs et de pas.

Un histogramme en flux demanderait des classes fixées à l’avance. Les quantiles exacts nécessiteraient de conserver les valeurs ; une mémoire limitée imposerait une approximation. Ces analyses exploratoires restent dans les archives et ne sont pas incluses dans la comparaison.

## Configurations

On garde les configurations du TP1 sur le même serveur.

| Taille | Éléments par axe | Ordre | Nœuds |
|---|---:|---:|---:|
| S | 10 | 2 | 9 261 |
| M | 15 | 2 | 29 791 |
| L | 20 | 2 | 68 921 |

Domaine : 2 000 m par direction. Pas de temps : 0,001 s. Durée demandée : 1,5 s. Un snapshot et une analyse spatiale tous les 50 pas. Les récepteurs de `data/receivers.csv` sont échantillonnés à chaque pas.

Le CPU utilise OpenMP avec 4 threads. Le GPU utilise Kokkos CUDA pour la simulation et cuFFT pour les FFT. Le cas `scaling_cfl` ajuste le pas de temps pour étudier la scalabilité ; il est séparé des comparaisons à pas fixe.

## Lancement sur le serveur

Depuis la racine du dépôt, avec l’environnement CUDA et le dossier `build-cuda` utilisés au TP1. Il faut Python avec NumPy et Gnuplot pour le post-traitement.

```bash
export PATH="/usr/local/cuda-13.4/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.4/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v nvcc && \
cmake -S . -B build-openmp -DCMAKE_BUILD_TYPE=Release -DUSE_KOKKOS=ON -DUSE_VECTOR=OFF -DENABLE_CUDA=OFF -DKokkos_ENABLE_CUDA=OFF -DCOMPILE_FD=OFF -DSEMPROXY_ENABLE_CUFFT=OFF && \
cmake --build build-openmp --target semproxy -j 4 && \
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DUSE_KOKKOS=ON -DUSE_VECTOR=OFF -DENABLE_CUDA=ON -DCOMPILE_FD=OFF -DSEMPROXY_ENABLE_CUFFT=ON && \
cmake --build build-cuda --target semproxy -j 4 && \
python3 tp2/benchmarks/run_tp2_benchmarks.py --cpu-exe ./build-openmp/bin/semproxy --gpu-exe ./build-cuda/bin/semproxy --fft --threads 4 --runs 5 --warmups 1
```

La commande compile les deux versions puis lance les campagnes CPU et GPU. Chaque cas a une chauffe et cinq mesures. L’ordre est mélangé et la chauffe est exclue des résultats. Le script contrôle le backend réellement utilisé et compare aussi les résultats numériques hors des mesures de temps.

La configuration CUDA conserve le compilateur et l’architecture du build TP1. Les mesures CUDA/cuFFT restent à effectuer sur le serveur.

## Comparaison CPU / GPU

Les statistiques des snapshots ont trois modes :

| Mode | Traitement |
|---|---|
| `gpu` | Réduction sur le GPU, puis copie du petit résultat |
| `cpu` | Parcours du champ depuis le CPU via la mémoire unifiée |
| `hostcopy` | Copie du champ vers le CPU, puis calcul ; les deux temps sont séparés |

Le GPU renvoie une structure de statistiques. Les deux traitements CPU lisent tout le champ, soit 4 octets par nœud. Ce sont des volumes logiques, pas des mesures du trafic PCIe. Dans le mode `cpu`, les migrations UVM sont incluses dans le parcours ; un temps de copie nul ne veut pas dire qu’il n’y a aucun transfert. Le retour des données vers le GPU peut aussi ralentir le pas suivant.

Pour les FFT, on compare la lecture du binaire suivie de NumPy avec cuFFT pendant la simulation. Avec CUDA, le centrage et la FFT restent sur GPU ; seul le spectre est rapatrié pour l’écriture. Les temps du plan, de la préparation, de la FFT, des transferts et de l’écriture sont séparés.

Les deux versions calculent `dt × abs(rfft(signal - moyenne))`, sans doubler les fréquences positives. NumPy travaille en double précision pour la FFT, cuFFT en simple précision. Les différences de calcul ne viennent donc pas uniquement du workflow. Écrire tout le spectre peut aussi coûter plus de place que garder les traces binaires.

## Slices

Les options sont `--slice-interval`, `--slice-plane=XY|XZ|YZ`, `--slice-position` en mètres et `--slice-folder`. Les slices peuvent être écrites avec ou sans snapshots complets.

On choisit le plan GLL le plus proche de la position demandée, sans interpolation. La position réelle est conservée dans le JSON. Le rendu utilise les coordonnées physiques des nœuds, une palette divergente et une normalisation locale symétrique autour de zéro. La coupe du volume complet et la slice utilisent le même rendu.

La sélection initiale, l’extraction, la copie, l’écriture, la lecture et le rendu sont chronométrés. On compare leur coût total avec celui d’une sortie complète. Une slice économise des données mais empêche de choisir un autre plan après la simulation.

## Mesures et précision

`workflow_s` comprend la simulation et son post-traitement. Pour les analyses numériques : lecture, calcul et écriture des résultats. Pour `visual_full` et `visual_slice` : la visualisation est également incluse.

Les temps de calcul, d’écriture, de lecture, d’analyse et de rendu sont séparés. Les graphiques de stockage comptent les données et les résultats, sans les journaux du benchmark. Les temps détaillés sont déjà inclus dans le total : il ne faut pas les additionner une seconde fois au workflow.

Les fichiers sont fermés sans forcer leur persistance sur disque (`buffered`, comme au TP1). Les lectures peuvent profiter du cache système. Le paramètre `io_sync` du fichier de configuration permet de mesurer les écritures de simulation avec `fsync`. Les sorties Python restent buffered. Les figures donnent la moyenne et l’écart-type ; le résumé contient aussi la médiane.

La comparaison de temps garde l’intervalle 50 des deux côtés. Pour la précision, une exécution analyse tous les pas, puis on extrait les intervalles 1, 2, 5, 10, 20, 50 et 100. On compare le maximum de pression et la moyenne temporelle des moyennes spatiales. L’erreur est divisée par le maximum global de |p|, déduit des min/max. Un pic entre deux sorties peut être manqué par les snapshots. Les sismogrammes voient tous les pas dans les deux méthodes : leur échantillonnage est identique.

Une vérification par taille compare les statistiques et les spectres sur les mêmes données. Les valeurs et coordonnées des slices sont aussi comparées au snapshot complet. Les écarts sont enregistrés dans `_validation.csv` ; un dépassement de tolérance arrête la campagne.

## Résultats et figures

Les CSV et JSON sont dans `benchmarks/results/`. Les figures sont dans `benchmarks/plots/cpu/`, `gpu/` et `cpu_vs_gpu/`. Les anciens résultats restent dans `benchmarks/archives/` avec leurs paramètres d’origine.

La campagne complète avec `--fft`, sans filtre de figures, prévoit 63 images après validation : 26 dans `cpu/`, 31 dans `gpu/` et 6 dans `cpu_vs_gpu/`. Chaque figure contient deux courbes. Les points des courbes de temps donnent la moyenne des cinq répétitions, avec l’écart-type.

L’axe horizontal représente S, M et L, sauf pour `precision_*`, où il représente l’intervalle entre les sorties. Les statistiques calculées restent min, max, moyenne et variance. Leurs valeurs sont dans les CSV ; les figures de performance mesurent le coût de leur calcul ensemble.

### Comparaisons directes CPU / GPU

Dans `benchmarks/plots/cpu_vs_gpu/`, chaque figure a une courbe CPU OpenMP à 4 threads et une courbe GPU CUDA.

| Fichier | Mesure comparée entre CPU et GPU |
|---|---|
| `baseline.png` | Calcul de la simulation sans traitement supplémentaire |
| `snapshot_adhoc.png` | Simulation + écriture binaire + lecture + statistiques des snapshots |
| `snapshot_insitu_gpu.png` | Simulation + statistiques in-situ des snapshots + écriture des résultats |
| `sismo_insitu.png` | Simulation + statistiques des récepteurs en flux + écriture des résultats |
| `visual_full.png` | Workflow depuis les snapshots complets, visualisation incluse |
| `visual_slice.png` | Workflow depuis les slices, visualisation incluse |

Le rendu des images reste sur CPU dans les deux courbes de `visual_full.png` et `visual_slice.png`.

### Ad-hoc / in-situ

Ces figures existent dans `cpu/` et `gpu/`. Chaque ligne du tableau correspond à deux images : une pour les snapshots et une pour les sismogrammes.

| Fichiers | Mesure | Courbe 1 | Courbe 2 |
|---|---|---|---|
| `snapshots_workflow.png`, `sismos_workflow.png` | Temps total du workflow numérique | Ad-hoc binaire | In-situ |
| `snapshots_storage.png`, `sismos_storage.png` | Volume des données et résultats | Ad-hoc binaire | In-situ |
| `snapshots_analysis.png`, `sismos_analysis.png` | Temps de calcul des statistiques | Analyse ad-hoc | Analyse in-situ |
| `snapshots_io.png`, `sismos_io.png` | Temps des E/S et copies indiquées | Écriture brute + lecture | Copie + écriture des résultats in-situ |
| `snapshots_overhead.png`, `sismos_overhead.png` | Temps ajouté par rapport au cas sans traitement | Ad-hoc | In-situ |

Ces workflows numériques comprennent la simulation et le post-traitement associé. La visualisation est incluse dans les cas `visual_full` et `visual_slice` uniquement.

### Statistiques sur GPU ou sur CPU

Dans `gpu/`, la simulation reste sur GPU dans les deux courbes ; seul le traitement des statistiques change.

| Fichier | Courbe 1 | Courbe 2 |
|---|---|---|
| `stats_gpu_vs_hostcopy.png` | Temps de réduction GPU + copie du résultat | Temps de copie du champ + calcul CPU |
| `stats_gpu_vs_uvm_cpu.png` | Temps de réduction GPU + copie du résultat | Temps de parcours direct du champ UVM sur CPU |
| `stats_hostcopy_workflow.png` | Workflow avec statistiques GPU | Workflow avec copie et statistiques CPU |
| `stats_transfer_bytes.png` | Volume logique du petit résultat de réduction | Volume logique du champ complet |

Dans `cpu/`, ces mêmes fichiers comparent la réduction OpenMP au parcours CPU. La réduction appelée « device » dans la légende y utilise OpenMP, pas CUDA.

### Slices

Ces figures existent dans `cpu/` et `gpu/`. Le plan de coupe et le rendu sont les mêmes des deux côtés.

| Fichier | Courbe 1 | Courbe 2 |
|---|---|---|
| `slice_workflow.png` | Temps du workflow depuis le volume complet | Temps du workflow depuis la slice |
| `slice_storage.png` | Volume écrit avec le champ complet et ses résultats | Volume écrit avec la slice et ses résultats |
| `slice_read.png` | Temps de lecture du volume | Temps de lecture de la slice |
| `slice_render.png` | Temps de rendu depuis le volume | Temps de rendu depuis la slice |
| `slice_extraction_vs_write_gain.png` | Coût de sélection, extraction et copie | Temps d’écriture économisé |

`slice_workflow.png` comprend la production des images et permet de comparer le coût total des deux méthodes.

### Précision et scalabilité

Ces figures existent dans `cpu/` et `gpu/`.

| Fichiers | Courbe 1 | Courbe 2 |
|---|---|---|
| `precision_S.png`, `precision_M.png`, `precision_L.png` | Erreur normalisée sur le maximum de pression | Erreur normalisée sur la moyenne temporelle |
| `snapshots_validation.png`, `sismos_validation.png` | Écart numérique normalisé sur le maximum | Écart numérique normalisé sur la moyenne |
| `scalability_steps.png` | Nombre de pas mesuré sous CFL | Modèle proportionnel à N |
| `scalability_compute.png` | Temps de calcul mesuré sous CFL | Modèle proportionnel à N⁴ |

Pour `precision_*`, les intervalles sont 1, 2, 5, 10, 20, 50 et 100 pas ; la référence utilise tous les pas. Les figures `*_validation` comparent les méthodes sur les mêmes données et aux mêmes instants. Les quatre statistiques sont contrôlées dans le CSV de validation ; la figure affiche le maximum et la moyenne.

### FFT des sismogrammes

Ces cinq figures sont produites dans `gpu/` avec `--fft`. Pour les comparaisons NumPy/cuFFT, la simulation utilise CUDA des deux côtés. La FFT concerne uniquement les sismogrammes.

| Fichier | Courbe 1 | Courbe 2 |
|---|---|---|
| `sismo_fft_workflow.png` | Workflow binaire puis statistiques et FFT NumPy sur CPU | Workflow statistiques et cuFFT in-situ |
| `sismo_fft_compute.png` | Temps de calcul FFT NumPy seul | Temps de calcul cuFFT seul |
| `sismo_fft_setup.png` | Allocation et création du plan cuFFT | Traitement cuFFT complet |
| `sismo_fft_storage.png` | Volume des données brutes et résultats ad-hoc | Volume des résultats in-situ |
| `fft_validation.png` | Erreur normalisée sur les amplitudes spectrales | Erreur normalisée sur les fréquences |

Le traitement cuFFT complet inclut la préparation, le plan, la FFT, les transferts, l’écriture et la libération des ressources. NumPy et cuFFT utilisent ici des précisions différentes ; le temps de calcul seul dépend aussi de la bibliothèque utilisée.

### Refaire les figures

Pour refaire les figures sans relancer les simulations, remplacer `NOM` par le nom de la campagne :

```bash
python3 tp2/benchmarks/run_tp2_benchmarks.py --plot-only tp2/benchmarks/results/NOM_gpu.csv
python3 tp2/benchmarks/plot_tp2_benchmarks.py --compare tp2/benchmarks/results/NOM_cpu.csv tp2/benchmarks/results/NOM_gpu.csv
```

Pour choisir les figures :

```bash
python3 tp2/benchmarks/run_tp2_benchmarks.py --plot-only tp2/benchmarks/results/NOM_gpu.csv --figures snapshots_workflow,sismos_workflow,stats_gpu_vs_hostcopy,sismo_fft_workflow,slice_workflow
```

`--modes` permet de limiter les cas. `--keep-data` conserve les sorties brutes ; sinon elles sont supprimées après traitement. Elles sont écrites dans `tp2/data/`, sur le système de fichiers du dépôt. `--quick` utilise de petites tailles de contrôle, différentes de S/M/L.

## Scalabilité

À ordre fixe, un champ contient O(N³) valeurs. Sous CFL, le nombre de pas pour une durée physique fixée augmente comme N, donc le calcul total comme N⁴. Une sortie tous les k pas donne un volume total en N⁴ ; une sortie à intervalle physique fixe donne N³. Une slice contient O(N²) valeurs.

Le fichier `_scalability.json` confronte ces lois aux mesures S/M/L et donne l’extrapolation à 19 milliards de nœuds. En float32 : 76 Go par champ, 228 Go de coordonnées une fois, au moins 152 Go pour deux champs de pression. Une slice carrée représente environ 28,5 Mo de pression. Ces tailles dépassent la mémoire du GPU du serveur ; le modèle ne prend pas en compte une distribution sur plusieurs machines.

L’in-situ est intéressant pour les snapshots si les statistiques suffisent : on évite de gros fichiers et leur relecture. L’ad-hoc garde l’avantage de pouvoir changer d’analyse après coup. Pour les sismogrammes, déjà petits, le coût du traitement ou du plan FFT peut dépasser les E/S économisées. Les conclusions et les gains chiffrés dépendent des mesures du serveur, séparément pour les deux types de données.
