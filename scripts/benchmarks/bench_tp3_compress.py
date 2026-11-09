#!/usr/bin/env python3
"""TP3 (person 3) benchmark: in-situ compression (RLE/delta) and steering.

For each mesh size and each configuration, semproxy is run `warmups` times
(ignored) then `runs` times; the table reports the median of each metric.
Method aligned with benchmark_tp3_visu.py: 1 warm-up, >= 5 runs, medians.
"""
import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

SIZES = {"S": 20, "M": 40, "L": 50}
COMPRESS = ["raw+rle", "f16+rle", "int16+rle", "int8+rle", "int8+delta+rle"]
STEER_K = [1, 10, 100]

RE_KERNEL = re.compile(r"Elapsed Kernel Time : ([0-9.eE+-]+)")
RE_OUTPUT = re.compile(r"Elapsed Output Time : ([0-9.eE+-]+)")
RE_STEER = re.compile(r"Steering : (\d+) reads .*total ([0-9.eE+-]+) ms")
RE_SISMO = re.compile(
    r"\[compress-sismo\] rcv=0 .* ratio=([0-9.eE+-]+) rmse_norm=([0-9.eE+-]+)")

COLUMNS = ["size", "config", "kernel_s", "output_s", "snap_ratio",
           "stored_bytes", "csv_bytes", "compress_ms", "write_ms",
           "rmse_norm_max", "dvar_rel_max", "sismo_ratio", "sismo_rmse_norm",
           "steer_reads", "steer_ms"]


def read_compress(path):
    """Aggregate the per-snapshot compression CSV of one run."""
    s = dict(raw=0.0, stored=0.0, comp_ms=0.0, write_ms=0.0, csv_bytes=0.0,
             rmse=0.0, dvar=0.0)
    with path.open() as f:
        for row in csv.DictReader(f):
            s["raw"] += float(row["raw_bytes"])
            s["stored"] += float(row["stored_bytes"])
            s["comp_ms"] += float(row["time_ms"])
            s["write_ms"] += float(row["write_ms"])
            s["csv_bytes"] += float(row["csv_bytes"])
            s["rmse"] = max(s["rmse"], float(row["rmse_norm"]))
            s["dvar"] = max(s["dvar"], abs(float(row["d_var_rel"])))
    return s


def run_one(exe, n, config, work, timemax, interval, threads):
    snap = work / "snaps"
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True)
    cmp_csv = work / "compress.csv"
    ctl = work / "steer.txt"
    cmd = [str(exe), "--ex", str(n), "--ey", str(n), "--ez", str(n),
           "--timemax", str(timemax), "--is-snapshots", "1",
           "--snap-time-interval", str(interval), "--snap-folder", str(snap)]
    if config in COMPRESS:
        cmd += ["--insitu-compress", config, "--compress-file", str(cmp_csv)]
    if config.startswith("steer_k"):
        ctl.write_text(f"snap_interval={interval}\n")
        cmd += ["--insitu-extra", "steering", "--steer-file", str(ctl),
                "--steer-interval", config[len("steer_k"):]]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    text = res.stdout + res.stderr
    if res.returncode != 0:
        raise RuntimeError(f"run failed ({config}, {n}):\n{text[-800:]}")

    m = {"kernel_s": float(RE_KERNEL.search(text).group(1)),
         "output_s": float(RE_OUTPUT.search(text).group(1))}
    if config in COMPRESS:
        c = read_compress(cmp_csv)
        m.update(snap_ratio=c["raw"] / c["stored"], stored_bytes=c["stored"],
                 csv_bytes=c["csv_bytes"], compress_ms=c["comp_ms"],
                 write_ms=c["write_ms"], rmse_norm_max=c["rmse"],
                 dvar_rel_max=c["dvar"])
        s = RE_SISMO.search(text)
        if s:
            m.update(sismo_ratio=float(s.group(1)),
                     sismo_rmse_norm=float(s.group(2)))
    st = RE_STEER.search(text)
    if st:
        m.update(steer_reads=int(st.group(1)), steer_ms=float(st.group(2)))
    shutil.rmtree(snap, ignore_errors=True)
    return m


def median(runs, key):
    vals = [r[key] for r in runs if key in r]
    return statistics.median(vals) if vals else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="build/bin/semproxy")
    ap.add_argument("--output", default="stats/tp3/bench_compress")
    ap.add_argument("--sizes", default="S,M")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timemax", type=float, default=0.5)
    ap.add_argument("--interval", type=int, default=50)
    ap.add_argument("--configs", default="")
    a = ap.parse_args()

    exe = Path(a.exe).resolve()
    if not exe.is_file():
        sys.exit(f"Executable not found: {exe}")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp") / "bench_tp3_compress_work"
    work.mkdir(parents=True, exist_ok=True)

    configs = ["baseline"] + COMPRESS + [f"steer_k{k}" for k in STEER_K]
    if a.configs:
        configs = a.configs.split(",")
    rows = []
    for size in a.sizes.split(","):
        n = SIZES[size]
        for config in configs:
            runs = []
            for i in range(a.warmups + a.runs):
                m = run_one(exe, n, config, work, a.timemax, a.interval,
                            a.threads)
                tag = "warmup" if i < a.warmups else f"run {i - a.warmups + 1}"
                print(f"[{size} {config}] {tag}: kernel {m['kernel_s']:.2f}s "
                      f"output {m['output_s']:.2f}s", flush=True)
                if i >= a.warmups:
                    runs.append(m)
            row = {"size": size, "config": config}
            for key in COLUMNS[2:]:
                row[key] = median(runs, key)
            rows.append(row)
            with (out / f"summary_{a.sizes.replace(',', '')}.csv").open(
                    "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                w.writeheader()
                w.writerows(rows)
    print("\nDone. Summary in", out)


if __name__ == "__main__":
    main()
