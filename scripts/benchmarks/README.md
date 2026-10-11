# FUnTiDES-lite Benchmarks

Run the SEM benchmark:

```bash
python3 scripts/benchmarks/run_benchmarks.py --threads 1,2,4,8 --runs 5 --compare-snapshots
```

`--threads` selects the thread counts and `--runs` selects the number of repetitions per configuration.  
Without options, the benchmark uses **4 threads, 3 runs, with snapshots enabled**.

The SEM mesh is fixed at **15 × 15 × 15 elements**, with approximation order **2**, corresponding to **31 × 31 × 31 nodes**.

## Snapshot comparison

With `--compare-snapshots`, each thread count is benchmarked in two modes:

- **Without snapshots**
- **With snapshots**

The snapshot interval is fixed at **50 time steps**.

## Terminal results

- **Simulation (Python clock):** elapsed time of `semproxy`, measured with `time.perf_counter()`.
- **TotalExe:** total execution time reported by SEM.
- **Initial:** SEM initialization time.
- **Compute:** time spent inside the SEM run.
- **Kernel:** computation kernel time.
- **Output:** SEM output time. When snapshots are enabled, snapshot CSV writing is included in this time.
- **Snapshot saving:** included in the SEM Output time.
- **Snapshot visualization:** elapsed time of `visu_snapshots.py`, measured with `time.perf_counter()`.
- **Workflow:** total time from starting SEM to finishing snapshot visualization.
- **File sizes:** total size of generated snapshot CSV and PNG files in MiB.

The terminal displays the **average over all repetitions** for each thread count and configuration.

## HPC metrics

The benchmark also computes the following performance metrics.

### Speedup

S(p) = T(1) / T(p)

where `T(1)` is the average execution time with one thread and `T(p)` is the average execution time with `p` threads.

### Parallel efficiency

E(p) = S(p) / p × 100

This shows how efficiently the available threads are used.

### Relative snapshot overhead

overhead (%) = (T_with - T_without) / T_without × 100

where `T_with` is the average SEM time with snapshots and `T_without` is the average SEM time without snapshots.

### Effective I/O bandwidth

bandwidth = snapshot CSV size / SEM Output time

The bandwidth is reported in **MiB/s**.

The SEM Output time also contains other output operations, so this value is an **effective I/O bandwidth**, not an isolated measurement of `saveSnapshot()`.

## Mesh and ad hoc workflow benchmark

The second benchmark is implemented in:

scripts/benchmarks/run_two.py

It studies the influence of the SEM mesh size while keeping the execution fixed at **4 OpenMP threads**.

Run it with:

```bash
python3 scripts/benchmarks/run_two.py --runs 3
```

The tested SEM meshes are:

- **10 × 10 × 10 elements → 21 × 21 × 21 nodes**
- **15 × 15 × 15 elements → 31 × 31 × 31 nodes**
- **20 × 20 × 20 elements → 41 × 41 × 41 nodes**
- **25 × 25 × 25 elements → 51 × 51 × 51 nodes**

For each mesh size, two configurations are measured:

- **Without snapshots:** SEM execution only.
- **With snapshots:** SEM execution, snapshot CSV writing and snapshot visualization.

The reported workflow time therefore measures the complete **ad hoc workflow** when snapshots are enabled.

The generated snapshot images are archived separately for each mesh size and each repetition under:

scripts/visualisation/images/mesh_benchmark/

For example:

nodes_31/run_1/
nodes_31/run_2/
nodes_31/run_3/

At the end, `run_two.py` also runs a **201 × 201 × 201 node reference case** used for visual comparison. This reference run is kept separate from the mesh-scaling averages because it uses a dedicated configuration.

To skip this final reference run:

```bash
python3 scripts/benchmarks/run_two.py --runs 3 --skip-reference
```

The mesh benchmark generates:

```text
scripts/benchmarks/results/mesh_workflow_benchmark.csv
scripts/benchmarks/plots/mesh_workflow_scaling.png
```

`mesh_workflow_scaling.png` compares the average total workflow time for each mesh size, with one bar **without snapshots** and one bar **with snapshots + visualization**.

## Output files

- `scripts/benchmarks/results/benchmarks.csv` — one row per run; previous results are kept.
- `scripts/benchmarks/plots/simulation_time.png` — SEM time with and without snapshots.
- `scripts/benchmarks/plots/workflow_time.png` — complete workflow time.
- `scripts/benchmarks/plots/snapshot_output_time.png` — SEM Output time with and without snapshots.
- `scripts/benchmarks/plots/snapshot_overhead.png` — absolute snapshot overhead in seconds.
- `scripts/benchmarks/plots/snapshot_overhead_percent.png` — relative snapshot overhead in percent.
- `scripts/benchmarks/plots/speedup.png` — measured speedup and ideal speedup.
- `scripts/benchmarks/plots/parallel_efficiency.png` — measured parallel efficiency and ideal efficiency.
- `scripts/benchmarks/plots/io_bandwidth.png` — effective snapshot I/O bandwidth.
- `scripts/benchmarks/plots/sem_phases_without_snapshots.png` — SEM phases without snapshots.
- `scripts/benchmarks/plots/sem_phases_with_snapshots.png` — SEM phases with snapshots.
- `scripts/benchmarks/plots/mesh_workflow_scaling.png` — complete ad hoc workflow time for different mesh sizes.

To regenerate the main benchmark graphs without rerunning the simulations:

```bash
python3 scripts/benchmarks/plot_benchmarks.py
```

**Note:** Snapshot CSV files are generated by the SEM simulation itself and are then used by the visualization script. Seismogram benchmarking is not integrated yet.
