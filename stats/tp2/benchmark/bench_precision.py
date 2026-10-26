import argparse
import csv

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--intervals", default="1,2,5,10,20,50,100")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    data = np.genfromtxt(a.stats, delimiter=",", names=True)
    step = np.atleast_1d(data["step"]).astype(int)
    vmax = np.atleast_1d(data["max"]).astype(float)
    mean = np.atleast_1d(data["mean"]).astype(float)

    ref_max = float(np.max(vmax))
    ref_mean = float(np.mean(mean))

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "size", "interval", "n_samples",
            "reference_global_max", "sampled_global_max",
            "max_abs_error", "max_rel_error",
            "reference_temporal_mean", "sampled_temporal_mean",
            "mean_abs_error", "mean_rel_error"
        ])
        for interval in [int(x) for x in a.intervals.split(",") if x.strip()]:
            mask = step % interval == 0
            sm = float(np.max(vmax[mask]))
            mm = float(np.mean(mean[mask]))
            max_abs = abs(sm - ref_max)
            mean_abs = abs(mm - ref_mean)
            max_rel = max_abs / max(abs(ref_max), 1e-30)
            mean_rel = mean_abs / max(abs(ref_mean), 1e-30)
            w.writerow([
                a.size, interval, int(np.sum(mask)),
                ref_max, sm, max_abs, max_rel,
                ref_mean, mm, mean_abs, mean_rel
            ])

    print(a.out)


if __name__ == "__main__":
    main()
