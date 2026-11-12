# TP3 : compression sans perte (RLE / delta) et pilotage (steering)

Cette partie ajoute au pipeline in-situ de FUnTiDES-lite :

1. une **compression sans perte** (RLE, avec delta optionnel) appliquée après la quantification, sur les **snapshots** et sur les **sismogrammes** ;
2. un **pilotage (steering)** de la simulation en cours d'exécution, via un petit fichier de contrôle relu tous les `k` pas ;
3. des **mesures** (script de benchmark et résultats S / M) qui chiffrent le gain de stockage, le surcoût et l'erreur introduite.

Tout passe par le point d'entrée commun `SEMproxy::runInSituProcessing(step)` (appelé à chaque pas, juste avant le `swap(i1, i2)`).

---

## 1. Options de la ligne de commande

| Option | Rôle |
|---|---|
| `--insitu-compress mode[+delta][+rle]` | Pipeline de compression. `mode` ∈ `raw`, `f16`, `int8`, `int16`. Exemples : `raw+rle`, `int8+rle`, `int8+delta+rle`. Nécessite `--is-snapshots 1`. |
| `--compress-file f.csv` | CSV de mesures, une ligne par snapshot compressé. |
| `--compress-order natural\|scan\|scanz` | Ordre de parcours des nœuds avant compression (voir §5.4). |
| `--compress-no-csv 1` | Le snapshot compressé (`.cmp`) **remplace** le CSV. |
| `--insitu-extra steering` | Active le steering. |
| `--steer-file f.txt` | Fichier de contrôle relu pendant le run. |
| `--steer-interval k` | Lecture du fichier tous les `k` pas (défaut 10). |

Colonnes du CSV de mesures (`--compress-file`) :
`step, raw_bytes, stored_bytes, ratio, time_ms, write_ms, csv_bytes, rmse_norm, max_abs_norm, d_min, d_max, d_mean, d_var_rel, rcv_orig, rcv_rec`.

- `rmse_norm`, `max_abs_norm` : erreur de reconstruction normalisée par max|p|.
- `d_min, d_max, d_mean` : écart (reconstruit − original) du min / max / de la moyenne, normalisé par max|p|. `d_var_rel` : écart relatif de la variance.
- `rcv_orig`, `rcv_rec` : valeur interpolée au récepteur 0 depuis le champ original et depuis le champ reconstruit.

En fin de run, si la compression est active, une ligne `[compress-sismo]` par récepteur donne le ratio et l'erreur du **sismogramme** compressé.

## 2. Format des données

- **Fichier `.cmp`** : en-tête de 40 octets (`magic`, `version`, `mode`, `nbits`, `flags` {DELTA, RLE}, `n_elements`, `payload_size`, `scale`, `step`) suivi du payload.
- **RLE** (`include/rle.hpp`) : paires (compte 1–255, octet). Pire cas : taille ×2.
- **DELTA** (`include/rle.hpp`) : chaque élément est remplacé par sa différence avec le précédent, en arithmétique entière non signée modulo 2^bits (donc exactement réversible). Appliqué **avant** le RLE ; à la lecture on défait dans l'ordre inverse (RLE, puis DELTA, puis déquantification).
- Chaque **trace de sismogramme** est quantifiée avec **sa propre échelle** (son maximum), contrairement aux snapshots où l'échelle suit le maximum du champ entier.

## 3. Fichier de contrôle (steering)

Un fichier texte, `clé=valeur` par ligne, `#` commence un commentaire. Il peut être modifié pendant que la simulation tourne :

```
snap_interval=20     # change la fréquence des snapshots
slice=z:0.5          # snapshots CSV limités au plan de nœuds le plus proche de z=0.5 (axe x, y ou z) ; slice=off pour revenir au volume complet
snapshot_now=1       # écrit un snapshot immédiatement, à chaque changement de valeur
stop=1               # arrêt propre de la boucle en temps
```

Chaque lecture est chronométrée ; le résumé de fin de run affiche le nombre de lectures, le temps total et le temps moyen par lecture.

## 4. Exécution

```bash
cd ~/FUnTiDES-lite/build && make -j8

# Compression int8 + RLE des snapshots, comparaison avec le CSV
./bin/semproxy --ex 20 --ey 20 --ez 20 --timemax 0.5 --is-snapshots 1 \
  --snap-time-interval 50 --snap-folder /tmp/snaps \
  --insitu-compress int8+rle --compress-file /tmp/compress.csv

# Idem en remplaçant le CSV par les fichiers compressés
./bin/semproxy ... --insitu-compress int8+rle --compress-file /tmp/compress.csv --compress-no-csv 1

# Steering : lecture tous les 10 pas, fichier modifiable en cours de run
printf 'snap_interval=100\n' > /tmp/steer.txt
./bin/semproxy --ex 40 --ey 40 --ez 40 --timemax 1.0 --is-snapshots 1 \
  --snap-time-interval 100 --snap-folder /tmp/snaps \
  --insitu-extra steering --steer-file /tmp/steer.txt --steer-interval 10 &
printf 'snap_interval=20\nslice=z:0.0\nsnapshot_now=1\n' > /tmp/steer.txt   # pendant le run
printf 'stop=1\n' > /tmp/steer.txt                                           # arrêt propre

# Benchmark complet (1 warm-up + 5 répétitions, médiane, 4 threads OpenMP)
python3 scripts/benchmarks/bench_tp3_compress.py --sizes S --output tp3/compression/bench_compress/S
python3 scripts/benchmarks/bench_tp3_compress.py --sizes M --runs 3 \
  --configs baseline,raw+rle,int16+rle,int8+rle,steer_k10 --output tp3/compression/bench_compress/M
```

```bash
python3 scripts/benchmarks/bench_tp3_compress.py --sizes L --runs 3 \
  --configs baseline,raw+rle,int16+rle,int8+rle,int8+rle_nocsv,steer_k10 --output tp3/compression/bench_compress/L
```

Maillages : S = 20³, M = 40³, L = 50³ éléments (ordre 2), `--timemax 0.5` (500 pas), un snapshot tous les 50 pas (10 snapshots), 4 threads OpenMP, CPU. S : 5 répétitions ; M et L : 3 répétitions (1 warm-up ignoré, médiane). Les résultats bruts sont dans `tp3/compression/bench_compress/{S,M,L}/`.

## 5. Résultats

### 5.1 Compression des snapshots (médianes, maillage S puis M)

Taille de référence : CSV complet de 10 snapshots = 18,4 Mo (S), 124,7 Mo (M) et 227,9 Mo (L).

| Config | Ratio (brut/stocké) S / M / L | Stocké S | rmse_norm max (S) | var. relative max (S) |
|---|---|---|---|---|
| raw+rle | 0,59 / 0,82 / 0,98 | 4,69 Mo | 0 | 0 |
| f16+rle | 2,05 / n/m / n/m | 1,34 Mo | 3,2e-5 | 4,2e-3 |
| int16+rle | 15,0 / 18,4 / 19,0 | 184 Ko | 4,5e-6 | 1,9e-5 |
| int8+rle | 56,4 / 60,3 / 62,4 | 49 Ko | 8,4e-4 | 8,2e-3 |
| int8+delta+rle | 55,0 / n/m / n/m | 50 Ko | 8,4e-4 | 8,2e-3 |

(n/m : non mesuré.) Sur M, int8+rle stocke 353 Ko (contre 124,7 Mo de CSV, environ 350 fois moins) et int16+rle 1,15 Mo (environ 108 fois moins). Sur L : int8+rle 661 Ko (environ 345 fois moins que 227,9 Mo), int16+rle 2,17 Mo (environ 105 fois moins), raw+rle 42,1 Mo. Les erreurs restent les mêmes d'une taille à l'autre : int16 ≈ 4e-6 (variance ≤ 1,7e-5), int8 ≈ 9e-4 (variance ≤ 3,2e-3 en M et en L).

### 5.2 Ratio par snapshot et propagation de l'onde

Avec `raw+rle` (sans perte), le ratio d'un snapshot de S vaut **114 au step 0** (champ nul presque partout), 1,26 au step 20, puis tombe à **0,51 au step 180**. Le RLE ne profite que des répétitions exactes : les zéros exacts de la zone que l'onde n'a pas encore atteinte. Quand le front d'onde remplit le domaine, les octets de mantisse des flottants sont quasi aléatoires et le RLE atteint son pire cas (×2). Le ratio sans perte mesure donc la **fraction du domaine non encore atteinte par l'onde**.

### 5.3 Conclusions physiques selon la précision (question du TP)

- **Statistiques globales** (min, max, moyenne, variance du champ reconstruit) : écarts ≤ 2,5e-4 de max|p| en f16, ≤ 1,3e-5 en int16, ≤ 3e-3 en int8 (variance à moins de 1 %). Les conclusions du TP2 sur les statistiques restent valables même en int8.
- **Signal au récepteur** : la quantification **globale** des snapshots (pas = max|p|/(2^(b-1)−1), dominé par la source) écrase le signal faible du récepteur (≈ 1e-5, du même ordre que le pas int16). Mesuré sur le sismogramme reconstruit depuis le snapshot : f16 → 0,2 % d'erreur ; **int16 → 95 % et int8 → 100 %**. Les forts ratios int16/int8 viennent donc en grande partie de cet écrasement (champ presque nul, donc très compressible), pas d'une vraie compression.
- Quantifier **chaque trace avec sa propre échelle** supprime le problème : sur le sismogramme, int8 garde 0,16 % d'erreur (≈ demi-pas, 1/254 du pic) pour un ratio de 5,2 à 5,6 ; int16 garde 7e-6.

### 5.4 Delta et ordre de parcours

- **Delta sur les snapshots** : légèrement défavorable (−5 à −10 % de ratio en int8/int16) dans l'ordre naturel. Il n'a de gain que sur des signaux lisses.
- **Delta sur les sismogrammes** : favorable, int8 passe de 5,2 à **8,2** à erreur identique (une trace temporelle est lisse).
- **Ordre de parcours** : sur ce maillage cartésien, la numérotation naturelle est déjà un balayage x → y → z (`scan` : 0 nœud sur 68921 change de place). L'ordre `scanz` (z le plus rapide) donne des ratios identiques à 4 chiffres en raw, f16, int8 et int16 (écart de 0,5135 à 0,5134 en raw+delta+rle). Raison : le RLE octet par octet ne profite que de répétitions exactes ; la zone nulle est un bloc contigu quel que soit l'ordre, et les champs quantifiés int8/int16 sont si creux que le nombre de plages ne dépend pas de l'ordre.

### 5.5 Sismogrammes (récepteur 0, 500 échantillons, benchmark `--timemax 0.5`)

| Config | Ratio (S) | rmse_norm (S, /pic de la trace) | Ratio (M) | rmse_norm (M) |
|---|---|---|---|---|
| raw+rle | 0,50 | 0 | 0,51 | 0 |
| f16+rle | 1,16 | 8,3e-5 | n/m | n/m |
| int16+rle | 1,40 | 7,8e-6 | 1,76 | 6,8e-6 |
| int8+rle | 5,17 | 1,6e-3 | 5,64 | 1,5e-3 |
| int8+delta+rle | 8,18 | 1,6e-3 | n/m | n/m |

Sur L : raw+rle 0,51 (erreur 0), int16+rle 1,82 (6,7e-6), int8+rle 5,67 (1,5e-3). (n/m : non mesuré.) Le RLE sans perte est inutile sur ce signal : le récepteur voit l'onde presque tout de suite (dans un run court de 200 échantillons, seuls 4 échantillons sont nuls au début), donc il n'y a quasiment pas de plage d'octets identiques. Les 40 octets d'en-tête pèsent environ 2 % d'une trace de 500 échantillons (2 000 octets).

### 5.6 Surcoût et gain d'E/S

| | S (20³) | M (40³) | L (50³) |
|---|---|---|---|
| Kernel Time (référence) | 7,44 s | 67,5 s | 127,1 s |
| Kernel Time avec compression ou steering | 7,44 s | 67,4 – 67,5 s | 127,1 – 127,3 s |
| Output Time, référence (CSV seul) | 1,55 s | 12,18 s | 22,67 s |
| Output Time, CSV + `.cmp` (int8+rle) | 1,58 s | 12,47 s | 23,19 s |
| Output Time, `.cmp` seul (`--compress-no-csv 1`) | **0,035 s** | **0,28 s** | **0,51 s** |

- La compression ne modifie pas le temps de calcul. Avec le CSV conservé en plus, l'Output Time augmente de 2 à 3 % (contrôle d'erreur compris : +0,26 à +0,39 s en M, +0,48 à +0,68 s en L).
- En **remplaçant** le CSV par le snapshot compressé, l'Output Time est divisé par environ **45** (S), **43** (M) et **44** (L), et le volume sur disque passe de 18 Mo à 49 Ko (S) et de 228 Mo à 661 Ko (L). Les valeurs S et M ont été relevées à la main (3 exécutions chacune) ; celle de L vient du benchmark (`int8+rle_nocsv`).
- Temps de compression seul : environ 1 ms par snapshot en S, 9–12 ms en M, 17–20 ms en L ; écriture du `.cmp` : environ 0,05 ms (S), 0,1–0,2 ms (M) et 0,15–0,3 ms (L) par snapshot pour int8/int16. Pour raw+rle, l'écriture de 4 Mo (S) à 42 Mo (L) coûte de 0,35 ms à 2,7 ms par snapshot.
- Le Kernel Time est identique d'une configuration à l'autre. Il croît avec le maillage : × 9,1 de S à M (7,7 fois plus de nœuds) et × 1,9 de M à L (1,9 fois plus de nœuds).

### 5.7 Coût du steering

| Intervalle `k` | Lectures (S) | Temps total | Coût par lecture |
|---|---|---|---|
| 1 | 499 | 5,1 ms | 10 µs |
| 10 | 50 | 0,64 ms (1,5 ms en M et en L) | 13 – 30 µs |
| 100 | 5 | 0,12 ms | 25 µs |

Le coût décroît en 1/k et reste inférieur à 0,1 % du Kernel Time même avec `k = 1`. Le plan de coupe (`slice=z:...`) réduit les snapshots de 531 442 lignes à 6 562 (81 × 81 nœuds) et le temps de sortie d'un facteur d'environ 8 (mesuré sur M, 40³, `--timemax 0.2`, un snapshot tous les 20 pas : 11,9 s → 1,46 s).

## 6. Limites

- Une seule source et un seul récepteur (récepteur 0) pour les mesures d'erreur au récepteur.
- Écritures mesurées dans le cache de pages (`/tmp`), sans `fsync` : le gain d'E/S réel sur un système de fichiers partagé (NFS, Lustre) peut être plus grand, les temps absolus changeront.
- Les mesures sont faites sur CPU (4 threads), pas sur GPU.
- Le contrôle d'erreur (déquantification et comparaison à chaque snapshot) est inclus dans l'Output Time ; il ne serait pas nécessaire en production.
- L'erreur des sismogrammes est normalisée par le pic de la trace : les parties faibles (coda tardive) sont affectées par le pas de quantification.
- Un changement de `snap_interval` par le steering s'applique à partir du pas suivant (le snapshot régulier de `run()` est écrit avant la lecture du fichier).
- Maillage cartésien uniquement pour l'étude de l'ordre de parcours.

## 7. Fichiers

- `tp3/include/rle.hpp` : RLE et DELTA.
- `src/main/fe/include/sem_proxy_options.h`, `sem_proxy.h`, `src/main/fe/src/sem_proxy.cc` : options, `compressSnapshot`, `compressSeismograms`, `steerFromFile`, plan de coupe dans `saveSnapshot`.
- `scripts/benchmarks/bench_tp3_compress.py` : benchmark.
- `tp3/compression/bench_compress/` : résultats.
