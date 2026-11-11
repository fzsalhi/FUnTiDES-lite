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

Chaque figure contient au plus deux courbes.

| Figures | Comparaison |
|---|---|
| `snapshots_*`, `sismos_*` | Ad-hoc / in-situ : workflow, volume, analyse, E/S et surcoût |
| `stats_gpu_vs_hostcopy`, `stats_gpu_vs_uvm_cpu` | Calcul GPU / CPU sur la même simulation CUDA |
| `stats_hostcopy_workflow`, `stats_transfer_bytes` | Temps global et volumes logiques transférés |
| `slice_*` | Volume complet / slice : temps, volume, lecture, rendu et coût d’extraction |
| `sismo_fft_*` | NumPy / cuFFT : workflow, calcul, préparation et volume |
| `precision_S`, `precision_M`, `precision_L` | Erreur selon l’intervalle entre sorties |
| `snapshots_validation`, `sismos_validation`, `fft_validation` | Écart numérique entre les analyses |
| `scalability_steps`, `scalability_compute` | Mesures sous CFL / modèles N et N⁴ |

Dans `cpu_vs_gpu/`, les deux courbes sont CPU et GPU : calcul seul, workflow snapshots ad-hoc, snapshots in-situ, sismogrammes in-situ, puis workflow avec visualisation depuis le volume ou les slices.

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
