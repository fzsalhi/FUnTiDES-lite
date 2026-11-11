# TP2 Benchmark

This folder contains the scripts and commands used to compare ad-hoc and in-situ data processing for snapshots and seismograms.

The benchmarks use three problem sizes:

- S: 20 x 20 x 20 elements
- M: 40 x 40 x 40 elements
- L: 50 x 50 x 50 elements

The main benchmarks use 4 OpenMP threads.

## 1. PlaFRIM allocation

Request a Sirocco node:

salloc -C sirocco --time=04:00:00

Check the allocated node:

hostname

The experiments were run on a Sirocco node with an NVIDIA K40 GPU.

## 2. CUDA 11.4 environment with Guix

Load CUDA 11.4.1 with Guix:

guix shell cuda-toolkit@11.4.1 -- bash

Check CUDA:

nvcc --version
nvidia-smi

Find the CUDA installation directory:

```bash
CUDA_ROOT=$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")
```

Check the value:

```bash
echo "$CUDA_ROOT"
```

This CUDA path is used by CMake.

## 3. Build semproxy with cuFFT support

Go to the project directory:

```bash
cd ~/traitement_in_situ/FUnTiDES-lite
```

Create a clean build directory:

```bash
rm -rf build-cufft-k40
mkdir build-cufft-k40
cd build-cufft-k40
```

Configure the project:

```bash
cmake ..   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_C_COMPILER=/usr/bin/gcc   -DCMAKE_CXX_COMPILER=/usr/bin/g++   -DCUDAToolkit_ROOT="$CUDA_ROOT"   -DUSE_VECTOR=OFF   -DUSE_KOKKOS=ON   -DKokkos_ENABLE_OPENMP=ON   -DKokkos_ENABLE_CUDA=OFF   -DSEMPROXY_ENABLE_CUFFT=ON
```

Build semproxy:

```bash
cmake --build . --target semproxy -j4
```

Return to the project root:

```bash
cd ..
```

The executable is:

```text
build-cufft-k40/bin/semproxy
```

This build uses OpenMP for the SEM simulation and cuFFT for the in-situ seismogram FFT.

Kokkos CUDA is disabled in this build:

```text
Kokkos_ENABLE_CUDA=OFF
```

The cuFFT support is enabled with:

```text
SEMPROXY_ENABLE_CUFFT=ON
```

The working configuration links the static cuFFT library without callbacks and the static CUDA runtime.

A useful check is:

```bash
ldd build-cufft-k40/bin/semproxy
```

The cuFFT and CUDA runtime libraries do not necessarily appear in `ldd` because they are linked statically.

## 2. Main S/M/L benchmark

Run the complete benchmark:
    
python3 stats/tp2/benchmark/run_tp2_benchmarks.py   --config stats/tp2/benchmark/tp2_benchmark_config.json   --mode all

The benchmark runs these modes for S, M and L:

baseline
snapshot_adhoc
snapshot_insitu_cpu
sismo_adhoc
sismo_insitu_cufft
 
The raw results are written to:
   
stats/tp2/benchmark/results_tp2_bench/tp2_bench_raw.csv
 
Create the summary file:
    
python3 stats/tp2/benchmark/summarize_tp2_benchmarks.py   --raw stats/tp2/benchmark/results_tp2_bench/tp2_bench_raw.csv   --out stats/tp2/benchmark/results_tp2_bench/tp2_bench_summary.csv

## 3. Thread benchmark

The thread benchmark compares 1, 2 and 4 OpenMP threads.

Example:
 
python3 stats/tp2/benchmark/run_tp2_threads.py   --config stats/tp2/benchmark/tp2_benchmark_config.json   --size S   --threads 1,2,4   --repetitions 1
 
The results are written to:
   
stats/tp2/benchmark/results_tp2_bench/threads/tp2_bench_threads_raw.csv
stats/tp2/benchmark/results_tp2_bench/threads/tp2_bench_threads.csv
 
## 4. Precision benchmark

The precision experiment studies the effect of the snapshot sampling interval.

A reference result is generated with statistics computed at every time step.

Example for size S:

    
mkdir -p stats/tp2/benchmark/results_tp2_bench/precision/S

OMP_NUM_THREADS=4 ./build-cufft-k40/bin/semproxy   --order 3   --ex 20 --ey 20 --ez 20   --lx 2000 --ly 2000 --lz 2000   --implem makutu   --mesh cartesian   --method sem   --auto-dt   --timemax 1.0   --stats-interval 1   --stats-mode cpu   --stats-file stats/tp2/benchmark/results_tp2_bench/precision/S/stats_every_step.csv
 
The same experiment can be run for M and L with 40 x 40 x 40 and 50 x 50 x 50 elements.

The precision analysis compares different sampling intervals such as:
  
1, 2, 5, 10, 20, 50, 100
 
The goal is to measure how much information is lost when fewer snapshots are kept.

## 5. Scalability analysis

The scalability analysis uses the S, M and L measurements to study how execution time and data volume increase with the problem size.

It is also used to discuss larger simulations and storage requirements.

The scalability results are stored in:
   
stats/tp2/benchmark/results_tp2_bench/tp2_bench_scalability.csv
 

## 6. Generate the figures

Matplotlib is launched through Guix on PlaFRIM.

Generate the main figures with:

guix shell --pure python python-matplotlib python-numpy -- python3 stats/tp2/benchmark/plot_tp2_benchmarks.py   --summary stats/tp2/benchmark/results_tp2_bench/tp2_bench_summary.csv   --precision stats/tp2/benchmark/results_tp2_bench/tp2_bench_precision_all.csv   --scalability stats/tp2/benchmark/results_tp2_bench/tp2_bench_scalability.csv   --out-dir stats/tp2/benchmark/plots_tp2_bench
 
Generate only the thread figure with:
    
guix shell --pure python python-matplotlib python-numpy -- python3 stats/tp2/benchmark/plot_tp2_benchmarks.py   --threads stats/tp2/benchmark/results_tp2_bench/threads/tp2_bench_threads.csv   --out-dir stats/tp2/benchmark/plots_tp2_bench

The figures are stored in:

stats/tp2/benchmark/plots_tp2_bench/
 
Existing figures are not overwritten. A new suffix such as   _001   is used when needed.

## 7. Figures

### tp2_bench_threads_overview.png

This figure studies OpenMP scaling.

The left graph shows the total workflow time for 1, 2 and 4 threads.

The right graph shows the speedup compared with one thread.

It compares the baseline, snapshot ad-hoc, snapshot in-situ, seismogram ad-hoc and seismogram in-situ cuFFT modes.

Some curves can overlap because their execution times are very close.

### tp2_bench_workflow_overview.png

This figure compares the complete workflow time for S, M and L.

Snapshots and seismograms are shown separately.

For ad-hoc processing, the workflow includes simulation, file writing, file reading and analysis.

For in-situ processing, the analysis is performed directly during the simulation or from data already kept in memory.

This is the main time comparison between ad-hoc and in-situ processing.

### tp2_bench_storage_overview.png

This figure compares the amount of data written to disk.

Snapshots are much larger because they contain the complete 3D pressure field.

Seismograms are much smaller because they contain values only at receiver positions.

In-situ processing can reduce the amount of raw data that must be stored.

### tp2_bench_precision_overview.png

This figure studies the effect of snapshot sampling.

It compares different snapshot intervals with a reference based on every time step.

It shows how sampling can affect quantities such as the temporal maximum and temporal mean.

A larger interval reduces data volume but can miss information between two saved snapshots.

### tp2_bench_cudamemcpy_throughput.png

This figure shows the CUDA memory transfer throughput used by the in-situ cuFFT analysis.

It compares:
   
Host to Device throughput
Device to Host throughput

The values are computed from the number of transferred bytes and the measured CUDA copy time.

The FFT itself is very fast for the current small seismograms.

One measured run with one receiver and 134 samples gave:
   
H2D: 0.020656 ms
FFT: 0.053121 ms
D2H: 0.018790 ms
Measured cuFFT operations: 0.092932 ms
 
The complete program can still be slower because CUDA initialization and library initialization are not included in these three measurements.

### tp2_bench_scalability_overview.png

This figure studies how the benchmark changes from S to M to L.

It is used to compare measured growth with the expected theoretical growth of the 3D problem.

It also helps discuss the amount of data produced by much larger simulations.

## 8. Ad-hoc and in-situ workflow

Ad-hoc processing writes raw data to disk and analyzes it after the simulation.
   
simulation + file writing + file reading + analysis
 
In-situ processing analyzes data directly during the simulation or before the data is discarded.

simulation + in-situ analysis + small result output

The objective is to reduce unnecessary disk I/O when possible.

## 9. Snapshots and seismograms

Snapshots and seismograms must be studied separately.

Snapshots contain a complete 3D field and generate much more data. In-situ processing can therefore reduce storage and post-processing costs.

Seismograms contain much less data. Ad-hoc processing can already be very cheap.

For the current small seismogram case, the CPU ad-hoc FFT can be faster than the complete cuFFT workflow because the signal is small and CUDA has a fixed initialization cost.

This is a valid benchmark result: in-situ processing is not always faster for every type of data.
