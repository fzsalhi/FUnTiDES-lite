import argparse
import csv
import glob
import os
import time

import numpy as np


def measure(paths):
    read_s = 0.0
    compute_s = 0.0
    n_values = 0
    for path in paths:
        t0 = time.perf_counter()
        data = np.genfromtxt(path, delimiter=",", names=True)
        t1 = time.perf_counter()
        names = data.dtype.names
        if names is None:
            continue
        name = "pressure" if "pressure" in names else names[-1]
        x = np.atleast_1d(data[name]).astype(float)
        t2 = time.perf_counter()
        np.min(x)
        np.max(x)
        np.mean(x)
        np.std(x)
        t3 = time.perf_counter()
        read_s += t1 - t0
        compute_s += t3 - t2
        n_values += len(x)
    return {
        "n_files": len(paths),
        "bytes": sum(os.path.getsize(x) for x in paths),
        "n_values": n_values,
        "read_s": read_s,
        "compute_s": compute_s,
        "total_s": read_s + compute_s
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", required=True)
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--slice-dir", required=True)
    p.add_argument("--snapshot-write-s", type=float, default=0.0)
    p.add_argument("--slice-write-s", type=float, default=0.0)
    p.add_argument("--slice-extract-s", type=float, default=0.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    snapshots = sorted(glob.glob(os.path.join(a.snapshot_dir, "*.csv")))
    slices = sorted(glob.glob(os.path.join(a.slice_dir, "*.csv")))
    if not snapshots or not slices:
        raise RuntimeError("Snapshot or slice CSV files are missing.")

    s = measure(snapshots)
    z = measure(slices)

    snapshot_workflow = a.snapshot_write_s + s["total_s"]
    slice_workflow = a.slice_extract_s + a.slice_write_s + z["total_s"]

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "size",
            "snapshot_files", "slice_files",
            "snapshot_bytes", "slice_bytes", "volume_ratio",
            "snapshot_read_s", "slice_read_s",
            "snapshot_post_s", "slice_post_s",
            "snapshot_write_s", "slice_write_s", "slice_extract_s",
            "snapshot_workflow_s", "slice_workflow_s", "workflow_gain_s"
        ])
        w.writerow([
            a.size,
            s["n_files"], z["n_files"],
            s["bytes"], z["bytes"], s["bytes"] / max(z["bytes"], 1),
            s["read_s"], z["read_s"],
            s["compute_s"], z["compute_s"],
            a.snapshot_write_s, a.slice_write_s, a.slice_extract_s,
            snapshot_workflow, slice_workflow, snapshot_workflow - slice_workflow
        ])

    print(a.out)


if __name__ == "__main__":
    main()
