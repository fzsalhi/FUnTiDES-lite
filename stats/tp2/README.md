# Analyse AD-HOC des simulations sismiques 

## 1. Présentation

L'analyse est organisée en trois étapes :

1. **Simulation numérique** : génération des snapshots de pression et des sismogrammes.
2. **Analyse des résultats** : extraction de statistiques temporelles et spatiales, estimation du front d'onde et étude des signaux enregistrés par les receivers.
3. **Visualisation** : génération de figures permettant d'étudier la propagation de l'onde, son contenu fréquentiel et les temps d'exécution des analyses.

Les scripts utilisés sont :

- `analyse_snapshots.py` : analyse des snapshots.
- `analyse_sismos.py` : analyse des sismogrammes.
- `plot_results.py` : génération des figures à partir des résultats précédents.

## 2. Exécution de `semproxy`

Depuis le répertoire de compilation `build-openmp` :

```bash
mkdir -p results/snapshots

./bin/semproxy \
    --order 4 \
    --ex 20 --ey 10 --ez 5 \
    --lx 1000 --ly 500 --lz 200 \
    --mesh cartesian \
    --method sem \
    --dt 0.001 \
    --timemax 1.5 \
    --rcv-file receivers.csv \
    --sismo-file results/sismos.csv \
    --is-snapshots \
    --snap-time-interval 10 \
    --snap-folder results/snapshots
```

Cette commande lance la simulation et demande :

- l'enregistrement des sismogrammes dans `results/sismos.csv` ;
- la génération des snapshots dans `results/snapshots/` ;
- l'enregistrement d'un snapshot toutes les dix itérations.

Avec un pas de temps de 0,001 s, cet intervalle correspond à 0,01 s entre snapshots, sous réserve du comportement exact de l'implémentation.

La commande suppose que les options sont compatibles avec la configuration de compilation utilisée.

## 4. Analyse des snapshots

Lancer :

python analyse_snapshots.py \
    --snap-dir results/snapshots \
    --dt 0.001 \
    --tag S \
    --out results

Le paramètre `--dt` doit correspondre au pas de temps réellement utilisé par `semproxy`. Il sert à associer un temps physique à chaque instant de simulation.

Le tag `S` permet d'identifier cette configuration dans les figures et les résultats. Il peut être remplacé par `M` ou `L` pour d'autres configurations.

L'analyse des snapshots produit notamment les statistiques temporelles, l'histogramme global de la pression et les informations utilisées pour estimer la vitesse du front d'onde.

Si aucune source n'est fournie, la source par défaut peut être calculée à partir du centre de la boîte englobante des coordonnées du maillage :

\[
S=\frac{\mathbf{x}_{\min}+\mathbf{x}_{\max}}{2}.
\]

## 5. Analyse des sismogrammes

Les sismogrammes enregistrent l'évolution temporelle du signal en chaque receiver.

Lancer :

python analyse_sismos.py \
    --sismos results/sismos.csv \
    --receivers receivers.csv \
    --dt 0.001 \
    --tag S \
    --out results

Cette analyse utilise les positions des receivers pour calculer leur distance à la source. Elle permet notamment d'étudier :

- les traces temporelles des signaux ;
- les spectres d'amplitude ;
- les temps d'arrivée de l'onde ;
- la vitesse de propagation estimée ;
- l'évolution des caractéristiques fréquentielles avec la distance.

Si `--source` est une option facultative dans la version utilisée, elle peut être omise pour appliquer la stratégie de source par défaut du script.

## 6. Génération des figures

Une fois les analyses terminées, exécuter :

python plot_results.py \
    --results results \
    --out figures

Le script lit les fichiers produits dans le dossier `results` et enregistre les figures dans `figures/`.

Pour comparer plusieurs configurations, par exemple S, M et L :

python plot_results.py \
    --results results_S results_M results_L \
    --out figures

Chaque dossier doit contenir les fichiers d'analyse correspondants. Les courbes des configurations disponibles sont superposées lorsqu'une comparaison est possible.

## 7. Figures générées

### 7.1. `snapshots_evolution.png` — Évolution temporelle des snapshots

Cette figure contient trois graphiques :

- **Énergie \(\sum p^2\)** : suit l'évolution de l'énergie calculée à partir du champ de pression. L'axe vertical est logarithmique.
- **Maximum et minimum de pression** : montre les valeurs extrêmes du champ de pression au cours du temps.
- **Rayon du front d'onde** : montre l'évolution du rayon estimé du front d'onde.

Cette figure sert à observer la dynamique générale de la simulation et à repérer les variations temporelles du champ.

### 7.2. `histogram.png` — Histogramme global de la pression

Cette figure représente la distribution des valeurs de pression sur l'ensemble des snapshots analysés.

- Axe horizontal : valeur de pression \(p\).
- Axe vertical : effectif correspondant, en échelle logarithmique.

Elle permet d'étudier la distribution globale des valeurs de pression et de repérer les valeurs fréquentes ou rares.

### 7.3. `wavefront.png` — Estimation de la vitesse du front d'onde

Cette figure représente le rayon estimé du front d'onde en fonction du temps.

Les points correspondent aux estimations extraites des snapshots. Une droite de régression est ajoutée lorsque l'estimation de vitesse est disponible.

Le modèle ajusté est de la forme :

\[
r_{\mathrm{front}}(t)\approx vt+b,
\]

où \(v\) est la vitesse estimée et \(b\) l'ordonnée à l'origine.

Le coefficient \(R^2\) permet d'évaluer la qualité de l'ajustement linéaire. Une ligne pointillée indique également la limite \(0{,}8\,r_{\mathrm{domaine}}\), utilisée comme repère pour le rayon du front d'onde.

### 7.4. `sismos_traces.png` — Traces sismiques

Cette figure affiche les sismogrammes de plusieurs receivers, triés par distance à la source.

Les signaux sont décalés verticalement en fonction de la distance afin de faciliter leur comparaison.

Elle permet de comparer la forme des signaux et de visualiser la propagation de l'onde vers les receivers plus éloignés.

### 7.5. `sismos_spectra.png` — Spectres d'amplitude

Cette figure présente les spectres d'amplitude \(|FFT|\) de plusieurs receivers.

- Axe horizontal : fréquence en Hz.
- Axe vertical : amplitude spectrale, calculée à partir de la transformée de Fourier discrète et du facteur `dt`.

Les spectres permettent d'observer les fréquences présentes dans les signaux et de comparer leur contenu fréquentiel selon la distance à la source.

### 7.6. `sismos_arrivals.png` — Temps d'arrivée en fonction de la distance

Cette figure représente les temps d'arrivée estimés pour les receivers en fonction de leur distance à la source.

Une régression est ajoutée lorsque l'estimation de vitesse est disponible. Le modèle utilisé est :

\[
t_{\mathrm{arrivée}}(d)\approx t_0+\frac{d}{v},
\]

où \(d\) est la distance à la source, \(t_0\) un décalage temporel et \(v\) la vitesse de propagation estimée.

La pente de la droite permet d'estimer la vitesse de propagation. La valeur de \(R^2\) renseigne sur la qualité de l'ajustement.

### 7.7. `sismos_freq_vs_dist.png` — Contenu fréquentiel selon la distance

Cette figure représente deux indicateurs fréquentiels en fonction de la distance à la source :

- **Fréquence dominante** : fréquence associée au maximum du spectre.
- **Centroïde spectral** : fréquence moyenne pondérée par l'amplitude spectrale.

Cette représentation permet de rechercher une éventuelle variation du contenu fréquentiel du signal avec la distance.

### 7.8. `timings_adhoc.png` — Temps d'exécution des analyses

Cette figure compare les temps associés au traitement des snapshots et des sismogrammes.

Pour chaque analyse, le temps est décomposé en :

- **Lecture des fichiers** : temps nécessaire au chargement des données ;
- **Calcul** : temps consacré aux traitements et aux calculs statistiques.

Cette figure permet d'identifier la part du coût liée aux entrées-sorties et celle liée au calcul. Elle concerne les scripts d'analyse AD-HOC, et non directement le temps total de simulation de `semproxy`.

## 8. Résumé des fichiers de sortie

| Fichier | Contenu |
|---|---|
| `snapshots_stats.csv` | Statistiques temporelles des snapshots |
| `histograms.npz` | Données de l'histogramme global |
| `wavefront.json` | Rayon du front d'onde et estimation de vitesse |
| `timings_snapshots.json` | Temps de lecture et de calcul des snapshots |
| `sismos_stats.csv` | Statistiques fréquentielles des sismogrammes |
| `sismos_spectra.npz` | Traces, distances et spectres |
| `sismos_arrivals.csv` | Distances et temps d'arrivée estimés |
| `velocity_sismos.json` | Estimation de vitesse issue des temps d'arrivée |
| `timings_sismos.json` | Temps d'exécution de l'analyse des sismogrammes |

La présence de certains fichiers dépend des données disponibles et des traitements effectivement exécutés.

## 9. Points de vigilance

- Le pas de temps transmis aux scripts d'analyse doit correspondre à celui de la simulation.
- Les coordonnées des receivers doivent utiliser le même repère que le maillage.
- La source utilisée pour calculer les distances doit être cohérente avec la source de la simulation.
- Quatre receivers suffisent pour illustrer le pipeline, mais sont peu nombreux pour obtenir une estimation robuste de la vitesse par régression.
- Les temps mesurés dans `timings_adhoc.png` dépendent aussi du système de fichiers, de la taille des données et de la machine utilisée.
- Si un fichier d'analyse manque, certaines figures ne seront pas générées.

Pour vérifier les options disponibles dans chaque script :

python analyse_snapshots.py --help
python analyse_sismos.py --help
python plot_results.py --help
