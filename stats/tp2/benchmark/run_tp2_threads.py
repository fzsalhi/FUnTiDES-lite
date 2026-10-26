import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_number(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else 0.0


def common_command(exe, c):
    return [
        exe,
        "--order", str(c["order"]),
        "--ex", str(c["ex"]),
        "--ey", str(c["ey"]),
        "--ez", str(c["ez"]),
        "--lx", str(c["lx"]),
        "--ly", str(c["ly"]),
        "--lz", str(c["lz"]),
        "--implem", c["implem"],
        "--mesh", c["mesh"],
        "--method", c["method"],
        "--auto-dt",
        "--timemax", str(c["timemax"])
    ]


def run_command(cmd, cwd, env, log_path):
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    t1 = time.perf_counter()
    Path(log_path).write_text(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed: " + " ".join(cmd) + "\n" + p.stdout[-4000:]
        )
    return p.stdout, t1 - t0


def read_one_row(path):
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return next(csv.DictReader(f), {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--size", default="M")
    p.add_argument("--threads", default="1,2,4,8")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--out")
    a = p.parse_args()

    script_dir = Path(__file__).resolve().parent

    with open(a.config) as f:
        cfg = json.load(f)

    size_cfg = next(
        (x for x in cfg["sizes"] if x["name"] == a.size),
        None
    )

    if size_cfg is None:
        raise RuntimeError("Unknown size.")

    c = dict(cfg["common"])
    c.update(size_cfg)

    exe = str(Path(cfg["semproxy"]).resolve())
    out_root = Path(cfg["output_dir"]).resolve() / "threads"
    out_root.mkdir(parents=True, exist_ok=True)

    out_csv = (
        Path(a.out).resolve()
        if a.out
        else out_root / "tp2_bench_threads.csv"
    )

    thread_values = [
        int(x)
        for x in a.threads.split(",")
        if x.strip()
    ]

    modes = [
        "baseline",
        "snapshot_adhoc",
        "snapshot_insitu_cpu",
        "sismo_adhoc",
        "sismo_insitu_cufft"
    ]

    raw = []

    for threads in thread_values:
        for mode in modes:
            for rep in range(1, a.repetitions + 1):
                run_dir = (
                    out_root
                    / a.size
                    / mode
                    / f"t{threads}"
                    / f"rep_{rep:02d}"
                )

                if run_dir.exists():
                    shutil.rmtree(run_dir)

                run_dir.mkdir(parents=True)

                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(threads)

                cmd = common_command(exe, c)

                if cfg.get("receivers_file"):
                    cmd += [
                        "--rcv-file",
                        str(Path(cfg["receivers_file"]).resolve())
                    ]

                snap_dir = run_dir / "snapshots"
                stats_file = run_dir / "stats.csv"
                sismo_file = run_dir / "sismos.csv"
                fft_file = run_dir / "fft.csv"

                if mode == "snapshot_adhoc":
                    cmd += [
                        "--is-snapshots",
                        "--snap-time-interval",
                        str(cfg["snapshot_interval"]),
                        "--snap-folder",
                        str(snap_dir)
                    ]

                elif mode == "snapshot_insitu_cpu":
                    cmd += [
                        "--stats-interval",
                        str(cfg["snapshot_interval"]),
                        "--stats-mode",
                        "cpu",
                        "--stats-file",
                        str(stats_file)
                    ]

                elif mode == "sismo_adhoc":
                    cmd += [
                        "--sismo-file",
                        str(sismo_file)
                    ]

                elif mode == "sismo_insitu_cufft":
                    cmd += [
                        "--fft-file",
                        str(fft_file)
                    ]

                stdout, wall_s = run_command(
                    cmd,
                    run_dir,
                    env,
                    run_dir / "semproxy.log"
                )

                adhoc_s = 0.0

                if mode == "snapshot_adhoc":
                    adhoc_out = run_dir / "adhoc_snapshots.csv"

                    run_command(
                        [
                            sys.executable,
                            str(script_dir / "bench_adhoc_snapshots.py"),
                            "--snap-dir",
                            str(snap_dir),
                            "--size",
                            a.size,
                            "--out",
                            str(adhoc_out),
                            "--bins",
                            str(cfg["hist_bins"])
                        ],
                        run_dir,
                        env,
                        run_dir / "adhoc_snapshots.log"
                    )

                    d = read_one_row(adhoc_out)
                    adhoc_s = float(d.get("total_s", 0) or 0)

                elif mode == "sismo_adhoc":
                    adhoc_out = run_dir / "adhoc_sismos.csv"

                    run_command(
                        [
                            sys.executable,
                            str(script_dir / "bench_adhoc_sismos.py"),
                            "--sismos",
                            str(sismo_file),
                            "--size",
                            a.size,
                            "--out",
                            str(adhoc_out)
                        ],
                        run_dir,
                        env,
                        run_dir / "adhoc_sismos.log"
                    )

                    d = read_one_row(adhoc_out)
                    adhoc_s = float(d.get("total_s", 0) or 0)

                raw.append({
                    "size": a.size,
                    "mode": mode,
                    "threads": threads,
                    "rep": rep,
                    "wall_s": wall_s,
                    "totalexe_s": parse_number(
                        r"Elapsed TotalExe Time\s*:\s*([0-9.eE+-]+)",
                        stdout
                    ),
                    "kernel_s": parse_number(
                        r"Elapsed Kernel Time\s*:\s*([0-9.eE+-]+)",
                        stdout
                    ),
                    "save_s": parse_number(
                        r"Elapsed Save Time\s*:\s*([0-9.eE+-]+)",
                        stdout
                    ),
                    "stats_s": parse_number(
                        r"Elapsed Stats Time\s*:\s*([0-9.eE+-]+)",
                        stdout
                    ),
                    "adhoc_s": adhoc_s,
                    "workflow_s": wall_s + adhoc_s
                })

                print(a.size, mode, threads, rep)

    raw_path = out_csv.with_name(
        out_csv.stem + "_raw.csv"
    )

    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(raw[0].keys())
        )
        w.writeheader()
        w.writerows(raw)

    summary = []

    for mode in modes:
        mode_rows = []

        for threads in thread_values:
            items = [
                r for r in raw
                if r["mode"] == mode
                and r["threads"] == threads
            ]

            workflows = np.array(
                [r["workflow_s"] for r in items],
                dtype=float
            )

            walls = np.array(
                [r["wall_s"] for r in items],
                dtype=float
            )

            adhoc = np.array(
                [r["adhoc_s"] for r in items],
                dtype=float
            )

            mode_rows.append({
                "size": a.size,
                "mode": mode,
                "threads": threads,
                "repetitions": len(items),
                "simulation_s_mean": float(np.mean(walls)),
                "simulation_s_std": float(np.std(walls)),
                "adhoc_s_mean": float(np.mean(adhoc)),
                "adhoc_s_std": float(np.std(adhoc)),
                "workflow_s_mean": float(np.mean(workflows)),
                "workflow_s_std": float(np.std(workflows))
            })

        t1 = next(
            x["workflow_s_mean"]
            for x in mode_rows
            if x["threads"] == 1
        )

        for row in mode_rows:
            speedup = t1 / row["workflow_s_mean"]
            row["speedup"] = speedup
            row["efficiency_pct"] = (
                100.0 * speedup / row["threads"]
            )

        summary.extend(mode_rows)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(summary[0].keys())
        )
        w.writeheader()
        w.writerows(summary)

    print(raw_path)
    print(out_csv)


if __name__ == "__main__":
    main()