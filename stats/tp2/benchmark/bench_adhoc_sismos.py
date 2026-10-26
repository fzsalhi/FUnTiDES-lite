import argparse
import csv
import os
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sismos", required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    t0 = time.perf_counter()
    data = np.genfromtxt(a.sismos, delimiter=",", names=True)
    t1 = time.perf_counter()

    names = data.dtype.names
    if names is None or len(names) < 2:
        raise RuntimeError("Invalid sismos CSV.")

    time_values = np.atleast_1d(data[names[0]]).astype(np.float64)
    signals = [np.atleast_1d(data[name]).astype(np.float64) for name in names[1:]]
    if len(time_values) < 2:
        raise RuntimeError("Not enough samples.")

    dt = float((time_values[-1] - time_values[0]) / (len(time_values) - 1))
    read_s = t1 - t0

    t2 = time.perf_counter()
    stat_rows = []
    for i, signal in enumerate(signals):
        stat_rows.append([
            i,
            float(np.min(signal)),
            float(np.max(signal)),
            float(np.mean(signal)),
            float(np.var(signal)),
            float(np.sqrt(np.mean(signal * signal))),
            float(np.sum(signal * signal) * dt)
        ])
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    fft_rows = []
    for i, signal in enumerate(signals):
        centered = signal - np.mean(signal)
        z = np.fft.rfft(centered)
        amp = np.abs(z) * dt
        freq = np.fft.rfftfreq(len(signal), d=dt)
        k = 1 + int(np.argmax(amp[1:])) if len(amp) > 1 else 0
        denom = float(np.sum(amp))
        centroid = float(np.sum(freq * amp) / denom) if denom > 0 else 0.0
        fft_rows.append([i, float(freq[k]), centroid])
    t5 = time.perf_counter()

    stats_s = t3 - t2
    fft_s = t5 - t4
    total_s = read_s + stats_s + fft_s

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "size", "n_receivers", "n_samples", "bytes", "dt",
            "read_s", "stats_s", "fft_s", "compute_s", "total_s"
        ])
        w.writerow([
            a.size, len(signals), len(time_values), os.path.getsize(a.sismos), dt,
            read_s, stats_s, fft_s, stats_s + fft_s, total_s
        ])

    detail = os.path.splitext(a.out)[0] + "_detail.csv"
    with open(detail, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["receiver", "min", "max", "mean", "variance", "rms", "energy", "dominant_hz", "centroid_hz"])
        for s, q in zip(stat_rows, fft_rows):
            w.writerow(s + q[1:])

    print(a.out)


if __name__ == "__main__":
    main()
