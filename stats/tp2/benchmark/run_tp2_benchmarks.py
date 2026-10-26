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


def parse_number(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else 0.0


def file_bytes(path):
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
    return 0


def read_one_row(path):
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(newline="") as f:
        return next(csv.DictReader(f), {})


def sum_stats(path):
    p = Path(path)
    if not p.exists():
        return 0, 0.0, 0.0, 0.0, 0

    n = 0
    copy_ms = 0.0
    compute_ms = 0.0
    total_ms = 0.0
    bytes_count = 0

    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            n += 1
            copy_ms += float(row.get("copy_ms", 0) or 0)
            compute_ms += float(row.get("compute_ms", 0) or 0)
            total_ms += float(row.get("time_ms", 0) or 0)
            bytes_count += int(float(row.get("bytes", 0) or 0))

    return n, copy_ms, compute_ms, total_ms, bytes_count


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="tp2_benchmark_config.json")
    p.add_argument("--mode", choices=["all", "quick"], default="all")
    a = p.parse_args()

    script_dir = Path(__file__).resolve().parent

    with open(a.config) as f:
        cfg = json.load(f)

    out_root = Path(cfg["output_dir"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    master = out_root / "tp2_bench_raw.csv"
    exe = str(Path(cfg["semproxy"]).resolve())

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(cfg["omp_threads"])

    fields = [
        "size",
        "mode",
        "rep",
        "ex",
        "ey",
        "ez",
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

    all_rows = []

    modes = [
        "baseline",
        "snapshot_adhoc",
        "snapshot_insitu_cpu",
        "sismo_adhoc",
        "sismo_insitu_cufft"
    ]

    for size_cfg in cfg["sizes"]:
        for key in ("ex", "ey", "ez"):
            if size_cfg.get(key) is None:
                raise RuntimeError(
                    "Fill S/M/L dimensions in tp2_benchmark_config.json."
                )

        size = size_cfg["name"]

        c = dict(cfg["common"])
        c.update(size_cfg)

        nodes_est = (
            (c["ex"] * c["order"] + 1)
            * (c["ey"] * c["order"] + 1)
            * (c["ez"] * c["order"] + 1)
        )

        reps = 1 if a.mode == "quick" else cfg["repetitions"]

        for mode in modes:
            for rep in range(1, reps + 1):
                run_dir = out_root / size / mode / f"rep_{rep:02d}"

                if run_dir.exists():
                    shutil.rmtree(run_dir)

                run_dir.mkdir(parents=True)

                cmd = common_command(exe, c)

                if cfg.get("receivers_file"):
                    cmd += [
                        "--rcv-file",
                        str(Path(cfg["receivers_file"]).resolve())
                    ]

                stats_file = run_dir / "stats.csv"
                sismo_file = run_dir / "sismos.csv"
                fft_file = run_dir / "fft.csv"
                snap_dir = run_dir / "snapshots"

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

                totalexe_s = parse_number(
                    r"Elapsed TotalExe Time\s*:\s*([0-9.eE+-]+)",
                    stdout
                )

                kernel_s = parse_number(
                    r"Elapsed Kernel Time\s*:\s*([0-9.eE+-]+)",
                    stdout
                )

                output_s = parse_number(
                    r"Elapsed Output Time\s*:\s*([0-9.eE+-]+)",
                    stdout
                )

                save_s = parse_number(
                    r"Elapsed Save Time\s*:\s*([0-9.eE+-]+)",
                    stdout
                )

                stats_s = parse_number(
                    r"Elapsed Stats Time\s*:\s*([0-9.eE+-]+)",
                    stdout
                )

                adhoc_read_s = 0.0
                adhoc_compute_s = 0.0
                adhoc_total_s = 0.0
                n_snapshots = 0

                if mode == "snapshot_adhoc":
                    adhoc_out = run_dir / "adhoc_snapshots.csv"

                    run_command(
                        [
                            sys.executable,
                            str(script_dir / "bench_adhoc_snapshots.py"),
                            "--snap-dir",
                            str(snap_dir),
                            "--size",
                            size,
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

                    adhoc_read_s = float(d.get("read_s", 0) or 0)
                    adhoc_compute_s = float(d.get("compute_s", 0) or 0)
                    adhoc_total_s = float(d.get("total_s", 0) or 0)
                    n_snapshots = int(float(d.get("n_snapshots", 0) or 0))

                if mode == "sismo_adhoc":
                    adhoc_out = run_dir / "adhoc_sismos.csv"

                    run_command(
                        [
                            sys.executable,
                            str(script_dir / "bench_adhoc_sismos.py"),
                            "--sismos",
                            str(sismo_file),
                            "--size",
                            size,
                            "--out",
                            str(adhoc_out)
                        ],
                        run_dir,
                        env,
                        run_dir / "adhoc_sismos.log"
                    )

                    d = read_one_row(adhoc_out)

                    adhoc_read_s = float(d.get("read_s", 0) or 0)
                    adhoc_compute_s = float(d.get("compute_s", 0) or 0)
                    adhoc_total_s = float(d.get("total_s", 0) or 0)

                (
                    n_stats_calls,
                    stats_copy_ms,
                    stats_compute_ms,
                    stats_total_ms,
                    stats_bytes
                ) = sum_stats(stats_file)

                timing_file = Path(str(fft_file) + ".timings.csv")
                cufft = read_one_row(timing_file)

                cufft_bytes_h2d = int(
                    float(cufft.get("bytes_h2d", 0) or 0)
                )

                cufft_bytes_d2h = int(
                    float(cufft.get("bytes_d2h", 0) or 0)
                )

                cufft_h2d_ms = float(
                    cufft.get("h2d_ms", 0) or 0
                )

                cufft_fft_ms = float(
                    cufft.get("fft_ms", 0) or 0
                )

                cufft_d2h_ms = float(
                    cufft.get("d2h_ms", 0) or 0
                )

                cufft_total_ms = float(
                    cufft.get("total_ms", 0) or 0
                )

                cufft_h2d_gbs = (
                    cufft_bytes_h2d / (cufft_h2d_ms * 1e6)
                    if cufft_h2d_ms > 0
                    else 0.0
                )

                cufft_d2h_gbs = (
                    cufft_bytes_d2h / (cufft_d2h_ms * 1e6)
                    if cufft_d2h_ms > 0
                    else 0.0
                )

                bytes_written = 0

                if mode == "snapshot_adhoc":
                    bytes_written = file_bytes(snap_dir)

                elif mode == "snapshot_insitu_cpu":
                    bytes_written = file_bytes(stats_file)

                elif mode == "sismo_adhoc":
                    bytes_written = file_bytes(sismo_file)

                elif mode == "sismo_insitu_cufft":
                    bytes_written = (
                        file_bytes(fft_file)
                        + file_bytes(timing_file)
                    )

                row = {
                    "size": size,
                    "mode": mode,
                    "rep": rep,
                    "ex": c["ex"],
                    "ey": c["ey"],
                    "ez": c["ez"],
                    "nodes_est": nodes_est,
                    "wall_s": wall_s,
                    "totalexe_s": totalexe_s,
                    "kernel_s": kernel_s,
                    "output_s": output_s,
                    "save_s": save_s,
                    "stats_s": stats_s,
                    "bytes_written": bytes_written,
                    "n_snapshots": n_snapshots,
                    "n_stats_calls": n_stats_calls,
                    "adhoc_read_s": adhoc_read_s,
                    "adhoc_compute_s": adhoc_compute_s,
                    "adhoc_total_s": adhoc_total_s,
                    "stats_copy_ms": stats_copy_ms,
                    "stats_compute_ms": stats_compute_ms,
                    "stats_total_ms": stats_total_ms,
                    "stats_bytes": stats_bytes,
                    "cufft_bytes_h2d": cufft_bytes_h2d,
                    "cufft_bytes_d2h": cufft_bytes_d2h,
                    "cufft_h2d_ms": cufft_h2d_ms,
                    "cufft_fft_ms": cufft_fft_ms,
                    "cufft_d2h_ms": cufft_d2h_ms,
                    "cufft_total_ms": cufft_total_ms,
                    "cufft_h2d_gbs": cufft_h2d_gbs,
                    "cufft_d2h_gbs": cufft_d2h_gbs
                }

                all_rows.append(row)

                with master.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(all_rows)

                print(size, mode, rep)

    print(master)


if __name__ == "__main__":
    main()