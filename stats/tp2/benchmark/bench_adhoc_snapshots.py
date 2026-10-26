import argparse
import csv
import glob
import os
import time

import numpy as np


def read_pressure(path):
    t0 = time.perf_counter()
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(3,))
    t1 = time.perf_counter()
    return data, t1 - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snap-dir", required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bins", type=int, default=100)
    a = p.parse_args()

    paths = sorted(glob.glob(os.path.join(a.snap_dir, "snapshot_it_*.csv")))
    if not paths:
        raise RuntimeError("No snapshot CSV found.")

    read1 = 0.0
    compute1 = 0.0
    global_min = np.inf
    global_max = -np.inf
    rows = []

    for path in paths:
        pressure, dt_read = read_pressure(path)
        read1 += dt_read
        t0 = time.perf_counter()
        vmin = float(np.min(pressure))
        vmax = float(np.max(pressure))
        mean = float(np.mean(pressure))
        var = float(np.var(pressure))
        energy = float(np.sum(pressure * pressure))
        t1 = time.perf_counter()
        compute1 += t1 - t0
        global_min = min(global_min, vmin)
        global_max = max(global_max, vmax)
        rows.append((os.path.basename(path), len(pressure), vmin, vmax, mean, var, energy))

    hist = np.zeros(a.bins, dtype=np.int64)
    read2 = 0.0
    compute2 = 0.0

    if global_max == global_min:
        global_max = global_min + 1.0

    for path in paths:
        pressure, dt_read = read_pressure(path)
        read2 += dt_read
        t0 = time.perf_counter()
        h, _ = np.histogram(pressure, bins=a.bins, range=(global_min, global_max))
        hist += h
        t1 = time.perf_counter()
        compute2 += t1 - t0

    total_read = read1 + read2
    total_compute = compute1 + compute2
    total = total_read + total_compute
    total_bytes = sum(os.path.getsize(x) for x in paths)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "size", "n_snapshots", "bytes", "read_s", "compute_s", "total_s",
            "stats_read_s", "stats_compute_s", "hist_read_s", "hist_compute_s",
            "global_min", "global_max", "hist_bins"
        ])
        w.writerow([
            a.size, len(paths), total_bytes, total_read, total_compute, total,
            read1, compute1, read2, compute2, global_min, global_max, a.bins
        ])

    detail = os.path.splitext(a.out)[0] + "_detail.csv"
    with open(detail, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "n_values", "min", "max", "mean", "variance", "energy"])
        w.writerows(rows)

    hist_out = os.path.splitext(a.out)[0] + "_hist.csv"
    edges = np.linspace(global_min, global_max, a.bins + 1)
    with open(hist_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left", "bin_right", "count"])
        for i in range(a.bins):
            w.writerow([edges[i], edges[i + 1], int(hist[i])])

    print(a.out)


if __name__ == "__main__":
    main()
