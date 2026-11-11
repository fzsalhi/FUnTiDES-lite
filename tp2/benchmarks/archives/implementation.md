# TP2 – Premiers pas en in-situ

Branche : `sismos`

Ce document décrit ce qui a été ajouté à `semproxy` pour calculer des
**statistiques en cours de simulation (in-situ)**, sans relire de fichier,
ainsi que la comparaison entre un calcul **sur GPU** et un calcul **sur CPU
après copie des données**. Il donne toutes les commandes pour compiler,
exécuter et analyser.


---

## 1. Ce qui a été fait

### 1.1 Statistiques in-situ du champ de pression

À intervalle régulier, `semproxy` calcule sur le champ de pression :
**min, max, moyenne, variance** (variance de population, division par `n`).

- Le calcul se fait pendant la boucle en temps, juste après le calcul du pas
  (`computeOneStep`), sur la colonne `i1` de `pnGlobal`, qui contient le champ
  le plus récent (avant l'échange `swap(i1, i2)`).
- Les snapshots écrits sur disque lisent la colonne `i2` : ils ont donc
  **un pas de retard** par rapport aux statistiques in-situ (pas `n-1` contre
  pas `n`). À garder en tête quand on compare avec des statistiques ad-hoc.

### 1.2 Statistiques « streaming » sur les sismogrammes

Pour chaque récepteur, min / max / moyenne / variance sont mises à jour
**à chaque pas de temps**, sans stocker la série (algorithme de Welford).
Elles sont affichées à la fin de la simulation :

```
[In-situ sismo] rcv=0 n=200 min=... max=... moyenne=... variance=...
```

Ces valeurs ont été vérifiées en recalculant les mêmes statistiques avec `awk`
sur le fichier `sismos.csv`.

### 1.3 Trois modes de calcul des statistiques du champ

Option `--stats-mode` :

| Mode | Fonction | Principe |
|---|---|---|
| `cpu` (défaut) | `computePressureStatisticsCPU` | Boucle séquentielle (Welford, une passe) qui lit directement `pnGlobal`. En build CUDA, la mémoire est partagée (UVM) : les pages sont migrées **implicitement** vers le CPU. |
| `gpu` | `computePressureStatistics` | `Kokkos::parallel_reduce` sur le device (4 réductions : min, max, somme, variance en deux passes). Seuls 4 scalaires reviennent à l'hôte (24 octets). |
| `hostcopy` | `computePressureStatisticsHostCopy` | `Kokkos::deep_copy` **explicite** de la colonne vers un `View` en `HostSpace`, puis boucle CPU (Welford) sur la copie. Temps de copie et de calcul mesurés séparément. |

Pour que les mesures soient comparables, un `Kokkos::fence()` est fait **avant**
de démarrer chaque chronomètre (les kernels Kokkos sont asynchrones : sans
barrière, on compterait la fin du kernel du solveur).

### 1.4 Sortie CSV commune

Les trois modes écrivent dans le même fichier (`--stats-file`) via la fonction
commune `recordStats` :

```
step,min,max,mean,variance,time_ms,mode,copy_ms,compute_ms,bytes
```

- `time_ms` : temps total du calcul de statistiques pour ce pas.
- `copy_ms` : temps du transfert GPU → hôte (0 pour `cpu` et `gpu`).
- `compute_ms` : temps de calcul.
- `bytes` : octets qui traversent le bus GPU → hôte.
  - `gpu` : 24 (2 `float` + 2 `double`).
  - `hostcopy` : `nbNodes × 4`.
  - `cpu` : 0 = non mesuré (la migration est implicite, via l'UVM).

Le temps cumulé et le nombre d'appels sont affichés en fin de simulation :

```
---- Elapsed Stats Time : ... seconds (N calls).
```

**La première ligne du CSV (pas 0) est un échauffement** (premier lancement de
kernel GPU) : l'ignorer dans les moyennes.

### 1.5 Fichiers modifiés

- `src/main/fe/include/sem_proxy_options.h` : options `--stats-interval`,
  `--stats-file`, `--stats-mode`.
- `src/main/fe/include/sem_proxy.h` : membres et déclarations des fonctions de
  statistiques et de `recordStats`. `computePressureStatistics` est **publique**
  (nvcc interdit les `KOKKOS_LAMBDA` dans une méthode privée/protégée).
- `src/main/fe/src/sem_proxy.cc` : implémentation, appel dans `run()`.

---

## 2. Options de `semproxy` liées au TP2

| Option | Rôle |
|---|---|
| `--stats-interval N` | Calcule les statistiques tous les `N` pas (0 = désactivé). |
| `--stats-mode cpu\|gpu\|hostcopy` | Mode de calcul (défaut `cpu`). Toute autre valeur provoque une erreur. |
| `--stats-file f.csv` | Fichier CSV de sortie des statistiques. |
| `--rcv-file`, `--sismo-file` | (TP1) récepteurs en entrée, sismogrammes en sortie. |

Les autres options (`--ex --ey --ez --timemax --dt ...`) sont inchangées.

---

## 3. Compiler

### 3.1 Version CPU (PlaFRIM)

Charger les modules habituels (compilateur C++, CMake), puis :

```bash
cd FUnTiDES-lite
git submodule update --init --recursive
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_KOKKOS=ON -DENABLE_CUDA=OFF
make -j8
# exécutable : build/bin/semproxy
```

Piège rencontré : un fichier périmé `src/utils/include/common_config.h` peut
masquer celui généré par le build (champ de pression tout à zéro). Il ne doit
pas exister : seul `common_config.h.in` doit être présent dans ce dossier.

```bash
ls src/utils/include
```

### 3.2 Version GPU (machine CREMI `hayt`, RTX 4000 Ada, CUDA 13.4)

Environnement CUDA (à refaire à chaque nouveau terminal) :

```bash
export PATH=/usr/local/cuda-13.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.4/lib64:$LD_LIBRARY_PATH
which nvcc   # doit afficher /usr/local/cuda-13.4/bin/nvcc
```

**a) `nvcc_wrapper` avec la bonne architecture par défaut.**
Le wrapper de Kokkos 4.1 impose `sm_70`, que CUDA 13 n'accepte plus. On en fait
une copie personnelle (hors dépôt) avec `sm_89` (Ada) :

```bash
cp external/kokkos/bin/nvcc_wrapper ~/nvcc_wrapper_ada
sed -i 's/^default_arch="sm_70"/default_arch="sm_89"/' ~/nvcc_wrapper_ada
chmod +x ~/nvcc_wrapper_ada
```

**b) Patch local de Kokkos 4.1 pour CUDA 13** (à ne PAS commiter).
Dans CUDA 13, `cudaMemPrefetchAsync` prend un `cudaMemLocation`. À appliquer
une fois dans `external/kokkos/core/src/Cuda/Kokkos_CudaSpace.cpp` :

```bash
python3 - <<'EOF'
import os
p = os.path.expanduser("~/FUnTiDES-lite/external/kokkos/core/src/Cuda/Kokkos_CudaSpace.cpp")
s = open(p).read()
old = """    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPrefetchAsync(
        ptr, bytes, space.cuda_device(), space.cuda_stream()));
"""
new = """#if CUDART_VERSION >= 13000
    // CUDA 13 changed the signature: the target is now a cudaMemLocation
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = space.cuda_device();
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPrefetchAsync(
        ptr, bytes, loc, 0 /* flags */, space.cuda_stream()));
#else
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMemPrefetchAsync(
        ptr, bytes, space.cuda_device(), space.cuda_stream()));
#endif
"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("patched")
EOF
```

Avec un CUDA 12.x, les étapes a) et b) ne devraient pas être nécessaires
(non testé ici).

**c) Configurer et compiler** dans un dossier séparé :

```bash
cd ~/FUnTiDES-lite
mkdir -p build-cuda && cd build-cuda
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_KOKKOS=ON \
  -DENABLE_CUDA=ON \
  -DKokkos_ARCH_ADA89=ON \
  -DCMAKE_CXX_COMPILER=$HOME/nvcc_wrapper_ada
make -j8 2>&1 | tee build.log | tail -5
# exécutable : build-cuda/bin/semproxy
```

`-DENABLE_CUDA=ON` active automatiquement `Kokkos_ENABLE_CUDA`,
`CUDA_LAMBDA` et `CUDA_UVM`. Sur une autre carte, remplacer `ADA89` (et
`sm_89`) par l'architecture correspondante.

Vérification de l'espace d'exécution :

```bash
./bin/semproxy --kokkos-print-configuration --ex 4 --ey 4 --ez 4 --timemax 0.01 2>&1 | grep -i -E "Default Device|Selected"
```

---

## 4. Exécuter

### 4.1 Un run simple

```bash
cd build-cuda
./bin/semproxy --ex 20 --ey 20 --ez 20 --timemax 0.2 \
  --stats-interval 10 --stats-mode gpu --stats-file stats_gpu.csv
head -5 stats_gpu.csv
```

### 4.2 Comparer les trois modes (maillage 40³, 531 441 nœuds)

```bash
for m in cpu gpu hostcopy; do
  ./bin/semproxy --ex 40 --ey 40 --ez 40 --timemax 0.5 \
    --stats-interval 10 --stats-mode $m --stats-file stats_$m.csv > run_$m.log 2>&1
  grep "Compute Time" run_$m.log
done
```

Sans statistiques (référence) :

```bash
./bin/semproxy --ex 40 --ey 40 --ez 40 --timemax 0.5 | grep "Compute Time"
```

### 4.3 Résumé par mode (premier appel exclu)

```bash
for m in cpu gpu hostcopy; do
  awk -F, -v M=$m 'NR>2 {t+=$6; c+=$8; p+=$9; b=$10; n++}
    END{printf "%-8s calls=%d total_ms=%.3f copy_ms=%.3f compute_ms=%.3f bytes=%d\n", M, n, t/n, c/n, p/n, b}' stats_$m.csv
done
```

### 4.4 Vérifier que les modes donnent les mêmes valeurs

```bash
paste -d' ' <(cut -d, -f1-5 stats_cpu.csv) <(cut -d, -f2-5 stats_gpu.csv) | head -5
```

Les `min / max / mean / variance` doivent coïncider (aux arrondis près).

---

## 5. Résultats (maillage 40³, RTX 4000 Ada, un run)

Par appel de statistiques, premier appel exclu :

| Mode | Total | Copie GPU→hôte | Calcul | Octets sur le bus |
|---|---|---|---|---|
| `gpu` (`parallel_reduce`) | 0,065 ms | – | 0,065 ms | 24 |
| `hostcopy` | 1,78 ms | 0,187 ms | 1,59 ms | 2 125 764 |
| `cpu` (lecture UVM directe) | 7,51 ms | implicite | 7,51 ms | non mesuré |

Temps de calcul total (`Elapsed Compute Time`) de la simulation :

| Configuration | Compute Time |
|---|---|
| sans statistiques | ≈ 0,090 s |
| `gpu` | ≈ 0,088 s (surcoût non mesurable) |
| `hostcopy` | ≈ 0,177 s |
| `cpu` | ≈ 0,57 s |

À retenir :
- Les trois modes calculent les mêmes valeurs.
- Calcul sur le GPU : quasi gratuit, et seuls 24 octets traversent le bus.
- Copie explicite : la copie (≈ 11 Go/s) est faible devant la boucle
  séquentielle sur l'hôte (≈ 90 % du temps).
- Lecture CPU directe en UVM : la plus coûteuse (migration de pages à la
  demande).

### Limites de ces mesures

- Un seul run, sur une seule machine (RTX 4000 Ada), pas sur les GPU de PlaFRIM.
- Mémoire UVM imposée par le CMake du projet (`CUDA_UVM=ON`) : le mode `cpu` en
  dépend fortement.
- Le premier appel GPU est plus lent (échauffement) : il est exclu des moyennes.
- Un premier test à 20³ avec `--stats-interval 50 --timemax 0.2` avait donné
  ≈ 0,53 ms par appel en mode `gpu` (contre ≈ 0,045 ms avec
  `--stats-interval 10 --timemax 0.5`). Cet écart n'a pas été expliqué.
- Le mode `hostcopy` suppose un `Layout` en `LayoutLeft` (build CUDA), pour que
  la colonne soit contiguë ; il n'a été testé qu'en build CUDA.
- La comparaison avec les statistiques ad-hoc (fichiers) n'est pas faite ici ;
  attention au décalage d'un pas entre snapshots (`i2`) et statistiques (`i1`).

---

## 6. Note de reproductibilité

Les éléments suivants sont **locaux** et ne sont pas dans le dépôt :
`~/nvcc_wrapper_ada` et le patch de `external/kokkos`. Ne pas ajouter
`external/kokkos` ni `build*/` avec `git add`.
