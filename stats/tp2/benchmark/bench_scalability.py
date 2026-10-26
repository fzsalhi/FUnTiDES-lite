import argparse
import csv
import math

import numpy as np


def fit_power(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 2:
        return 0.0, 0.0
    p = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(p[0]), float(np.exp(p[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target-nodes", type=float, default=19e9)
    p.add_argument("--fs-bandwidth-gbs", type=float, default=500.0)
    a = p.parse_args()

    with open(a.summary, newline="") as f:
        rows = list(csv.DictReader(f))

    snap = [r for r in rows if r["mode"] == "snapshot_adhoc"]
    if not snap:
        raise RuntimeError("snapshot_adhoc rows missing.")

    nodes = np.array([float(r["nodes_est_mean"]) for r in snap])
    n = np.cbrt(nodes)
    volume = np.array([float(r["bytes_written_mean"]) for r in snap])
    compute = np.array([float(r["kernel_s_mean"]) for r in snap])

    volume_exp, volume_coeff = fit_power(n, volume)
    compute_exp, compute_coeff = fit_power(n, compute)

    i = int(np.argmax(nodes))
    ref_nodes = nodes[i]
    ref_volume = volume[i]
    ref_n = n[i]
    target_n = a.target_nodes ** (1.0 / 3.0)
    node_ratio = a.target_nodes / ref_nodes
    n_ratio = target_n / ref_n

    target_snapshot_bytes = ref_volume * node_ratio
    target_total_fixed_steps_bytes = ref_volume * node_ratio * n_ratio
    target_total_fixed_physical_bytes = ref_volume * node_ratio
    bw = a.fs_bandwidth_gbs * 1e9
    target_snapshot_write_s = target_snapshot_bytes / bw

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["measured_snapshot_volume_exponent_vs_N", volume_exp])
        w.writerow(["measured_compute_exponent_vs_N", compute_exp])
        w.writerow(["theory_snapshot_volume_exponent_vs_N", 3.0])
        w.writerow(["theory_steps_exponent_vs_N", 1.0])
        w.writerow(["theory_compute_total_exponent_vs_N", 4.0])
        w.writerow(["theory_total_output_fixed_step_interval_exponent_vs_N", 4.0])
        w.writerow(["theory_total_output_fixed_physical_interval_exponent_vs_N", 3.0])
        w.writerow(["target_nodes", a.target_nodes])
        w.writerow(["target_equivalent_N", target_n])
        w.writerow(["projected_one_snapshot_bytes", target_snapshot_bytes])
        w.writerow(["projected_one_snapshot_gb", target_snapshot_bytes / 1e9])
        w.writerow(["projected_one_snapshot_write_s_at_bandwidth", target_snapshot_write_s])
        w.writerow(["filesystem_bandwidth_gbs", a.fs_bandwidth_gbs])
        w.writerow(["projected_total_bytes_fixed_step_relative_to_one_ref_run", target_total_fixed_steps_bytes])
        w.writerow(["projected_total_bytes_fixed_physical_relative_to_one_ref_run", target_total_fixed_physical_bytes])

    print(a.out)


if __name__ == "__main__":
    main()
