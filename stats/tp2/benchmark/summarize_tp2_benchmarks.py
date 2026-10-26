import argparse
import csv
from collections import defaultdict

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    with open(a.raw, newline="") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)

    for r in rows:
        groups[(r["size"], r["mode"])].append(r)

    numeric = [
        "nodes_est",
        "wall_s",
        "totalexe_s",
        "kernel_s",
        "output_s",
        "save_s",
        "stats_s",
        "bytes_written",
        "n_snapshots",
        "n_stats_calls",
        "adhoc_read_s",
        "adhoc_compute_s",
        "adhoc_total_s",
        "stats_copy_ms",
        "stats_compute_ms",
        "stats_total_ms",
        "stats_bytes",
        "cufft_bytes_h2d",
        "cufft_bytes_d2h",
        "cufft_h2d_ms",
        "cufft_fft_ms",
        "cufft_d2h_ms",
        "cufft_total_ms",
        "cufft_h2d_gbs",
        "cufft_d2h_gbs"
    ]

    out = []
    baseline = {}

    for key, items in groups.items():
        size, mode = key

        row = {
            "size": size,
            "mode": mode,
            "repetitions": len(items)
        }

        for col in numeric:
            x = np.array(
                [float(v.get(col, 0) or 0) for v in items],
                dtype=float
            )

            row[col + "_mean"] = float(np.mean(x))
            row[col + "_std"] = float(np.std(x))

        row["workflow_s_mean"] = (
            row["wall_s_mean"]
            + row["adhoc_total_s_mean"]
        )

        out.append(row)

        if mode == "baseline":
            baseline[size] = row["wall_s_mean"]

    for row in out:
        b = baseline.get(row["size"], 0.0)

        row["simulation_overhead_s"] = (
            row["wall_s_mean"] - b
        )

        row["simulation_overhead_pct"] = (
            100.0 * (row["wall_s_mean"] - b) / b
            if b > 0
            else 0.0
        )

        n_snap = row["n_snapshots_mean"]
        n_stats = row["n_stats_calls_mean"]

        row["adhoc_cost_per_snapshot_ms"] = (
            1000.0
            * (
                row["save_s_mean"]
                + row["adhoc_total_s_mean"]
            )
            / n_snap
            if n_snap > 0
            else 0.0
        )

        row["insitu_cost_per_call_ms"] = (
            row["stats_total_ms_mean"] / n_stats
            if n_stats > 0
            else 0.0
        )

    fields = list(out[0].keys())

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    print(a.out)


if __name__ == "__main__":
    main()