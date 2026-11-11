import argparse
import csv
from pathlib import Path
import numpy as np


def precision_rows(stats_path, size, intervals):
    data = np.atleast_1d(np.genfromtxt(stats_path, delimiter=",", names=True))
    step = data["step"].astype(int)
    amplitude = float(np.max(np.maximum(abs(data["min"]), abs(data["max"]))))
    reference_max = float(np.max(data["max"]))
    reference_mean = float(np.mean(data["mean"]))
    scale = amplitude or 1
    rows = []
    for interval in intervals:
        if interval < 1:
            raise ValueError("Les intervalles doivent etre positifs")
        sampled = step % interval == 0
        observed_max = float(np.max(data["max"][sampled]))
        observed_mean = float(np.mean(data["mean"][sampled]))
        rows.append(dict(size=size, interval=interval, n_samples=int(sampled.sum()),
                         reference_global_max=reference_max, sampled_global_max=observed_max,
                         normalization_amplitude=amplitude,
                         peak_error_norm=abs(observed_max-reference_max)/scale,
                         reference_temporal_mean=reference_mean, sampled_temporal_mean=observed_mean,
                         mean_error_norm=abs(observed_mean-reference_mean)/scale))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--intervals", default="1,2,5,10,20,50,100")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = precision_rows(args.stats, args.size, [int(x) for x in args.intervals.split(",")])
    with Path(args.out).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
