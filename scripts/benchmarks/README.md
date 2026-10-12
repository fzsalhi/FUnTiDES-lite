# FUnTiDES-lite Benchmarks

## Unified SEM benchmark

The main benchmark is implemented in:

scripts/benchmarks/run_benchmarks.py
scripts/benchmarks/plot_benchmarks.py

Run the complete benchmark with:

```bash
python3 scripts/benchmarks/run_benchmarks.py --threads 1,2,4,8 --runs 5
```

`--threads` selects the thread counts and `--runs` selects the number of repetitions per configuration.

Without options, the benchmark uses **4 threads**, **3 runs**, and tests all four configurations.

The SEM configuration is fixed at:

- **15 × 15 × 15 elements**
- approximation order **2**
- **31 × 31 × 31 nodes**
- domain **2000 × 2000 × 2000**
- `dt = 0.001`
- `timemax = 1.5`
- snapshot interval **50 time steps**

The benchmark results are stored in:

scripts/benchmarks/results/benchmarks.csv

The generated plots are stored in:

scripts/benchmarks/plots/

## Configurations

Each thread count is benchmarked in four modes:

- **Without outputs**
- **Snapshots only**
- **Sismos only**
- **Snapshots + sismos**

The corresponding mode names are:

none
snapshots
sismos
combined

A subset of configurations can be selected with `--modes`.

For example:

```bash
python3 scripts/benchmarks/run_benchmarks.py   --threads 1,2,4,8   --runs 5   --modes none,snapshots
```

The complete benchmark with 4 thread counts, 5 repetitions and 4 configurations performs:

4 thread counts × 5 repetitions × 4 configurations = 80 executions

## Snapshot output

Snapshot CSV files are generated directly by `semproxy` in:

scripts/visualisation/data/

The generated files follow the form:

snapshot_it_000000.csv
snapshot_it_000050.csv
snapshot_it_000100.csv
...

Snapshot visualization is performed by:

scripts/visualisation/visu_snapshots.py

The generated snapshot images are stored under:

scripts/visualisation/images/snapshots/

## Sismogram output

Receiver positions are read from:

data/receivers.csv

If the file does not exist, the benchmark creates it automatically with three default receivers.

The sismogram CSV is generated directly by `semproxy` in:

scripts/visualisation/data/sismos_benchmark.csv

The generated sismogram image is stored in:

scripts/visualisation/images/sismos/sismos_benchmark.png

## Terminal results

For every thread count and configuration, the benchmark reports the average over all repetitions.

The reported timings include:

- **Simulation:** elapsed time of `semproxy`, measured with `time.perf_counter()`.
- **TotalExe:** total execution time reported by SEM.
- **Initial:** SEM initialization time.
- **Compute:** time spent in the SEM computation loop.
- **Kernel:** SEM kernel time.
- **Output:** SEM output time.
- **Saving:** time used to generate and write CSV output.
- **Snapshot visualization:** elapsed snapshot visualization time.
- **Sismo visualization:** elapsed sismogram visualization time.
- **Total visualization:** snapshot and sismogram visualization time combined.
- **Workflow:** total time from starting SEM to finishing the required visualizations.
- **File sizes:** generated CSV and PNG sizes in MiB.

For the combined configuration, the saving time contains the snapshot and sismogram CSV saving time accumulated by SEM.

## HPC metrics

### Speedup

S(p) = T(1) / T(p)

where `T(1)` is the average simulation time with one thread and `T(p)` is the average simulation time with `p` threads.

Speedup is calculated separately for each configuration.

### Parallel efficiency

E(p) = S(p) / p × 100

Parallel efficiency is calculated separately for each configuration.

### Output overhead

The **Without outputs** configuration is used as the reference.

overhead = T_mode - T_without_outputs

The relative overhead is:

overhead (%) = (T_mode - T_without_outputs) / T_without_outputs × 100

The benchmark computes this overhead for:

- snapshots only
- sismos only
- snapshots + sismos

### I/O bandwidth

bandwidth = total CSV size / saving time

The bandwidth is reported in **MiB/s**.

For snapshots, it uses the total snapshot CSV size.

For sismos, it uses the sismogram CSV size.

For the combined configuration, it uses the total size of snapshot and sismogram CSV files divided by the accumulated saving time.

## Generated plots

`plot_benchmarks.py` reads the most recent benchmark batch from:

scripts/benchmarks/results/benchmarks.csv

It generates:

scripts/benchmarks/plots/simulation_time.png
scripts/benchmarks/plots/workflow_time.png
scripts/benchmarks/plots/initial_time.png
scripts/benchmarks/plots/compute_time.png
scripts/benchmarks/plots/kernel_time.png
scripts/benchmarks/plots/output_time.png
scripts/benchmarks/plots/save_time.png
scripts/benchmarks/plots/visualization_time.png
scripts/benchmarks/plots/overhead.png
scripts/benchmarks/plots/overhead_percent.png
scripts/benchmarks/plots/speedup.png
scripts/benchmarks/plots/parallel_efficiency.png
scripts/benchmarks/plots/io_bandwidth.png
scripts/benchmarks/plots/csv_size.png

Each plot contains the available benchmark configurations on the same figure.

To regenerate the plots without rerunning the simulations:

```bash
python3 scripts/benchmarks/plot_benchmarks.py
```

## Mesh and ad hoc workflow benchmark

The mesh benchmark is implemented in:

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

The reported workflow time measures the complete ad hoc workflow when snapshots are enabled.

The generated snapshot images are archived separately for each mesh size and repetition under:

scripts/visualisation/images/mesh_benchmark/

For example:

nodes_31/run_1/
nodes_31/run_2/
nodes_31/run_3/

At the end, `run_two.py` also runs a **201 × 201 × 201 node reference case** used for visual comparison.

To skip this final reference run:

```bash
python3 scripts/benchmarks/run_two.py --runs 3 --skip-reference
```

The mesh benchmark generates:

scripts/benchmarks/results/mesh_workflow_benchmark.csv
scripts/benchmarks/plots/mesh_workflow_scaling.png

`mesh_workflow_scaling.png` compares the average total workflow time for each mesh size, with one bar without snapshots and one bar with snapshots and visualization.
