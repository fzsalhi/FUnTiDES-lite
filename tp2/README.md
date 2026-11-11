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

La configuration CUDA conserve le compilateur et l’architecture du build TP1. Les mesures publiées correspondent aux backends OpenMP et CUDA sur le même serveur.

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

Les données de la campagne du 11 novembre 2026 sont dans `benchmarks/results/` : 180 mesures CPU et 195 mesures GPU, sur le serveur `jessica` (Core Ultra 9 285, RTX 4000 Ada). Chaque cas comporte cinq mesures, après une chauffe exclue des résultats. Les 60 contrôles numériques enregistrés dans les CSV de validation passent, avec des slices identiques au sous-ensemble correspondant du champ complet.

Les figures sont regroupées directement dans `benchmarks/plots/`. Une image rassemble les mesures qui répondent à la même question, avec deux ou quatre panneaux. Les analyses ad-hoc/in-situ utilisent la campagne CUDA ; les deux campagnes interviennent dans les comparaisons CPU/GPU et la scalabilité. Les résultats OpenMP restent disponibles dans les CSV et dans `plots/summary.csv`.

| Figure | Panneaux | Contenu et comparaisons |
|---|---:|---|
| `snapshots.png` | 4 | Temps complet, volume, calcul des quatre statistiques et décomposition des sorties/analyses : ad-hoc binaire / in-situ |
| `sismogrammes.png` | 4 | Même lecture pour les récepteurs : temps complet, volume, calcul des statistiques et décomposition des coûts |
| `fft.png` | 4 | Workflow, volume, calcul spectral et décomposition du traitement : NumPy ad-hoc / cuFFT in-situ |
| `slices.png` | 4 | Workflow avec visualisation, volume brut, préparation/E/S et rendu : volume complet / slice |
| `statistiques_cpu_gpu.png` | 4 | Réduction GPU / copie puis calcul CPU ; GPU / accès CPU via UVM ; volumes logiques transférés ; impact sur le calcul de la simulation |
| `precision.png` | 2 | Erreur sur le maximum et erreur sur la moyenne temporelle, avec une courbe par taille S/M/L |
| `scalabilite.png` | 4 | Nombre de pas, volume d’un champ, calcul CPU et calcul GPU, chacun confronté à son modèle |
| `cpu_vs_gpu.png` | 2 | Calcul seul puis workflow avec visualisation du volume, avec une courbe CPU et une courbe GPU |

Les courbes de temps présentent la moyenne et l’écart-type. Dans les panneaux statistiques, le temps de calcul couvre conjointement min, max, moyenne et variance ; les valeurs de ces quatre statistiques restent dans les fichiers de résultats. Dans `cpu_vs_gpu.png` et les panneaux de propagation sous CFL, le calcul désigne la propagation de l’onde, hors sorties et analyses. Les variantes d’analyse sur l’hôte utilisent une boucle CPU séquentielle ; les quatre threads OpenMP concernent la simulation CPU.

L’axe horizontal représente S/M/L, sauf pour la précision (intervalle entre snapshots) et la scalabilité (nombre d’éléments par direction). Les volumes des snapshots et des transferts sont en échelle logarithmique pour garder visibles le champ complet et les petits résultats in-situ. Le calcul seul de `cpu_vs_gpu.png` utilise aussi une échelle logarithmique.

Les barres empilées décomposent les étapes instrumentées pour L, hors calcul et initialisation de la simulation. Le temps complet est présenté dans le même fichier pour S, M et L. Les détails de chaque répétition restent dans les CSV.

Dans `precision.png`, les deux panneaux ont des échelles verticales différentes : erreur du maximum en pourcentage de max |p|, erreur de la moyenne en unités de 10⁻⁶ de max |p|. Cela évite de masquer la moyenne derrière une erreur de maximum beaucoup plus grande. Les deux erreurs sont toujours normalisées par la même amplitude de référence.

Dans `fft.png`, le temps NumPy inclut le calcul des amplitudes ; le temps du noyau cuFFT couvre la transformation seule, les amplitudes étant calculées lors de la sortie C++. Les précisions diffèrent aussi. Le coût complet et sa décomposition servent donc à interpréter le gain du noyau.

Pour régénérer les huit figures à partir des mesures publiées, sans compilation ni simulation :

```bash
python3 tp2/benchmarks/run_tp2_benchmarks.py --plot-only tp2/benchmarks/results/tp2_20261111T145417.364886Z_gpu.csv
```

Le script retrouve le CSV CPU de la même campagne. Avec d’autres noms de fichiers, les deux sources peuvent être données explicitement :

```bash
python3 tp2/benchmarks/plot_tp2_benchmarks.py --compare CHEMIN_cpu.csv CHEMIN_gpu.csv
```

Pour choisir les figures :

```bash
python3 tp2/benchmarks/run_tp2_benchmarks.py --plot-only tp2/benchmarks/results/tp2_20261111T145417.364886Z_gpu.csv --figures snapshots,fft,slices
```

La régénération complète supprime les anciens fichiers de tracé répertoriés dans les dossiers `plots/cpu/`, `gpu/` et `cpu_vs_gpu/`. Les mesures CSV/JSON et les archives sont conservées.

`--modes` limite les cas de benchmark ; `--no-plots` conserve seulement les mesures. `--keep-data` garde les sorties brutes ; sinon elles sont supprimées après traitement. Elles sont écrites dans `tp2/data/`, sur le système de fichiers du dépôt. `--quick` utilise de petites tailles de contrôle, différentes de S/M/L.

## Résultats de la campagne

Les valeurs ci-dessous sont les moyennes du workflow CUDA sur cinq répétitions, en secondes. La FFT concerne les sismogrammes ; les quatre statistiques restent min, max, moyenne et variance.

| Traitement | S ad-hoc | S in-situ | M ad-hoc | M in-situ | L ad-hoc | L in-situ |
|---|---:|---:|---:|---:|---:|---:|
| Statistiques des snapshots | 0,517 | 0,323 | 0,611 | 0,413 | 0,775 | 0,325 |
| Statistiques des sismogrammes | 0,359 | 0,375 | 0,363 | 0,374 | 0,403 | 0,360 |
| Statistiques et FFT des sismogrammes | 0,397 | 0,341 | 0,365 | 0,354 | 0,369 | 0,395 |

Pour les snapshots L, l’in-situ réduit le workflow d’un facteur 2,38 et le volume d’environ 9,11 Mo à 3,45 Ko. Les E/S évitées comptent davantage que les quelques millisecondes de différence entre les calculs statistiques.

Pour L, le GPU accélère la propagation seule d’environ 106 fois par rapport aux quatre threads CPU. Sur le workflow avec visualisation, le rapport est d’environ 3,32 : le rendu CPU et les E/S limitent le gain global.

Pour les sismogrammes, l’ad-hoc écrit environ 25 Ko, contre 414 octets de statistiques in-situ. Le gain de stockage est net, mais l’écart de temps reste proche de la dispersion pour plusieurs cas. On ne peut pas conclure à un avantage systématique en temps.

Pour les FFT L, le noyau cuFFT prend environ 17 µs contre 282 µs pour le calcul spectral NumPy. Le traitement cuFFT complet atteint toutefois environ 10,9 ms, avec près de 5 ms de plan et d’allocation. Le workflow complet est ici plus lent en in-situ (0,395 s contre 0,369 s), avec une dispersion de 0,033 s côté in-situ. Le gain du noyau ne suffit pas à établir un gain du workflow.

Pour la visualisation L, la slice réduit les données brutes de 9,10 Mo à 0,229 Mo, soit un facteur proche de 40. Le workflow passe de 2,764 s à 2,517 s : le gain est d’environ 9 %, car le rendu du même plan 2D prend environ 2 s dans les deux cas.

À un intervalle de 50 pas, l’erreur du maximum, normalisée par max |p|, vaut environ 1,23 % pour S, 5,78 % pour M et 11,05 % pour L. L’accord numérique entre méthodes à données identiques reste bien meilleur : écart maximal d’environ 2,2 × 10⁻⁹ sur les statistiques des snapshots et 2,0 × 10⁻⁷ sur les spectres. L’erreur d’échantillonnage domine donc ici l’erreur de calcul des analyses.

## Scalabilité

À ordre fixe, un champ contient O(N³) valeurs. Sous CFL, le nombre de pas pour une durée physique fixée augmente comme N, donc le calcul total comme N⁴. Une sortie tous les k pas donne un volume total en N⁴ ; une sortie à intervalle physique fixe donne N³. Une slice contient O(N²) valeurs.

Le fichier `_scalability.json` confronte ces lois aux mesures S/M/L et donne l’extrapolation à 19 milliards de nœuds. En float32 : 76 Go par champ, 228 Go de coordonnées une fois, au moins 152 Go pour deux champs de pression. Une slice carrée représente environ 28,5 Mo de pression. Ces tailles dépassent la mémoire du GPU du serveur ; le modèle ne prend pas en compte une distribution sur plusieurs machines.

Les exposants ajustés aux trois tailles sont environ 2,89 pour le volume d’un champ et 1,01 pour le nombre de pas. Le calcul CPU sous CFL donne 3,88, proche de N⁴ ; le calcul GPU donne 1,20. Les petites tailles testées ne montrent donc pas le régime asymptotique N⁴ sur GPU. Le coût des lancements et la faible occupation sont des explications possibles ; l’exposant 1,20 ne doit pas servir à prédire le temps d’un problème de 19 milliards de nœuds.

Le modèle de volume prévoit environ 15,96 To pour une sortie tous les 50 pas sous CFL, contre 2,508 To pour une sortie toutes les 0,05 s physiques, coordonnées comprises une fois. Ces volumes sont des extrapolations du modèle, pas des écritures effectuées sur le serveur.

Dans cette campagne, l’in-situ apporte un gain net pour les snapshots si les statistiques suffisent. L’ad-hoc garde l’avantage de pouvoir changer d’analyse après coup. Pour les sismogrammes, déjà petits, les économies de stockage sont mesurées mais le gain en temps n’est pas systématique. Le coût du traitement en flux, du plan FFT et des sorties doit être pris en compte avant de choisir le workflow.
