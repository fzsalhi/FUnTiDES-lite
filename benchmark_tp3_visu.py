#!/usr/bin/env python3
import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def command_output(args, root):
    try:
        return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def parse_case(value):
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Use NAME:EX:EY:EZ")
    name, *sizes = parts
    if not name or any(not c.isalnum() and c not in "_-" for c in name):
        raise argparse.ArgumentTypeError("Invalid case name")
    try:
        ex, ey, ez = map(int, sizes)
    except ValueError:
        raise argparse.ArgumentTypeError("EX, EY and EZ must be integers")
    if min(ex, ey, ez) <= 0:
        raise argparse.ArgumentTypeError("Mesh dimensions must be positive")
    return (name, ex, ey, ez)


def median_or_none(values):
    return statistics.median(values) if values else None


def write_csv(path, rows, columns):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default="build-tp3-cpu/bin/semproxy")
    parser.add_argument("--output", default="tp3_bench_ppm")
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--timemax", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--plane", choices=["XY", "XZ", "YZ"], default="XY")
    parser.add_argument("--position", type=float, default=0.5)
    parser.add_argument("--skip-paired", action="store_true")
    args = parser.parse_args()
    if args.runs < 5 or args.warmups < 0 or args.threads <= 0:
        parser.error("--runs must be >=5, --warmups >=0, --threads >0")
    if args.dt <= 0 or args.timemax <= 0 or args.interval <= 0:
        parser.error("--dt, --timemax and --interval must be positive")
    if args.width < 2 or args.height < 2:
        parser.error("Image dimensions must be >=2")
    root = Path.cwd()
    executable = (root / args.exe).resolve()
    if not executable.is_file():
        parser.error(f"Executable not found: {executable}")
    out = (root / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = args.case or [("41nodes", 20, 20, 20)]
    modes = [
        ("baseline", None, False),
        ("P3_cache", "P3", False),
        ("P6_cache", "P6", False),
        ("P3_fsync", "P3", True),
        ("P6_fsync", "P6", True),
    ]
    if not args.skip_paired:
        modes.append(("P3_P6_paired", "both", False))
    cache = root / "build-tp3-cpu" / "CMakeCache.txt"
    cmake_options = {}
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith(("USE_KOKKOS:", "USE_VECTOR:", "ENABLE_CUDA:",
                                "Kokkos_ENABLE_OPENMP:", "CMAKE_BUILD_TYPE:",
                                "SEMPROXY_ENABLE_CUFFT:", "COMPILE_SEM:")):
                key, value = line.split("=", 1)
                cmake_options[key] = value
    metadata = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_threads": os.cpu_count(),
        "OMP_NUM_THREADS": args.threads,
        "git_commit": command_output(["git", "rev-parse", "HEAD"], root),
        "filesystem": command_output(["df", "-T", str(out)], root),
        "executable": str(executable),
        "options": vars(args),
        "cases": cases,
        "cmake_options": cmake_options,
        "timing": "Python time.perf_counter around process, including startup and teardown; C++ steady_clock per phase",
        "fsync_definition": "write_ms = open and write, fsync_ms = additional fsync syscall time; total_ms includes encoding and close",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    raw = []
    images = []
    for name, ex, ey, ez in cases:
        for mode_name, image_format, sync in modes:
            for iteration in range(-args.warmups, args.runs):
                warmup = iteration < 0
                label = f"warmup_{-iteration}" if warmup else f"rep_{iteration + 1:02d}"
                folder = out / name / mode_name / label
                folder.mkdir(parents=True, exist_ok=True)
                cmd = [
                    str(executable), "--mesh", "cartesian", "--order", str(args.order),
                    "--ex", str(ex), "--ey", str(ey), "--ez", str(ez),
                    "--timemax", str(args.timemax), "--dt", str(args.dt),
                    "--sismo-file", str(folder / "sismos.csv"),
                    "--stats-interval", "0",
                    "--insitu-viz", "ppm" if image_format else "none",
                ]
                if image_format:
                    cmd += [
                        "--visu-plane", args.plane, "--visu-position", str(args.position),
                        "--visu-width", str(args.width), "--visu-height", str(args.height),
                        "--visu-interval", str(args.interval),
                        "--visu-format", image_format,
                        "--visu-folder", str(folder / "images"),
                    ]
                    if sync:
                        cmd.append("--visu-fsync=true")
                start = time.perf_counter()
                with (folder / "run.log").open("w") as log:
                    completed = subprocess.run(cmd, cwd=root, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, check=False)
                elapsed_ms = 1000 * (time.perf_counter() - start)
                row = {
                    "case": name, "ex": ex, "ey": ey, "ez": ez,
                    "mode": mode_name, "format": image_format or "none", "fsync": int(sync),
                    "repetition": iteration + 1 if not warmup else 0,
                    "warmup": int(warmup), "wall_ms": elapsed_ms,
                    "return_code": completed.returncode,
                    "log_path": str(folder / "run.log"),
                }
                raw.append(row)
                print(f"{name:<10s} {mode_name:<14s} {label:<11s} {elapsed_ms:10.3f} ms", flush=True)
                if completed.returncode != 0:
                    print((folder / "run.log").read_text()[-5000:], file=sys.stderr)
                    write_csv(out / "runs_partial.csv", raw, list(raw[0]))
                    raise RuntimeError(f"Simulation failed for {name} / {mode_name} / {label}")
                if image_format:
                    metrics_path = folder / "images" / "visu_metrics.csv"
                    if not metrics_path.exists():
                        raise RuntimeError(f"No visualization metrics: {metrics_path}")
                    with metrics_path.open(newline="") as file:
                        for item in csv.DictReader(file):
                            item.update({
                                "case": name, "mode": mode_name,
                                "repetition": row["repetition"], "warmup": int(warmup)
                            })
                            images.append(item)
    run_cols = list(raw[0])
    image_cols = list(images[0]) if images else []
    write_csv(out / "runs.csv", raw, run_cols)
    if image_cols:
        write_csv(out / "images.csv", images, image_cols)
    summaries = []
    for name, *_ in cases:
        baseline = [float(row["wall_ms"]) for row in raw
                    if row["case"] == name and row["mode"] == "baseline" and not row["warmup"]]
        baseline_median = statistics.median(baseline)
        for mode, image_format, sync in modes:
            durations = [float(row["wall_ms"]) for row in raw
                         if row["case"] == name and row["mode"] == mode and not row["warmup"]]
            mode_images = [r for r in images if r["case"] == name and r["mode"] == mode
                           and int(r["warmup"]) == 0]
            for fmt in (["none"] if image_format is None else ["P3", "P6"] if image_format == "both" else [image_format]):
                part = [r for r in mode_images if r["format"] == fmt]
                one = {
                    "case": name,
                    "mode": mode,
                    "format": fmt,
                    "fsync": int(sync),
                    "runs": len(durations),
                    "wall_median_ms": statistics.median(durations),
                    "wall_min_ms": min(durations),
                    "wall_max_ms": max(durations),
                    "baseline_median_ms": baseline_median,
                    "overhead_pct": 100 * (statistics.median(durations) / baseline_median - 1),
                    "image_count": len(part),
                }
                for key in ["size_bytes", "extract_ms", "color_ms", "encode_ms", "write_ms", "fsync_ms", "output_total_ms", "frame_total_ms"]:
                    one[key + "_median"] = median_or_none([float(r[key]) for r in part])
                summaries.append(one)
    summary_cols = list(summaries[0])
    write_csv(out / "synthese.csv", summaries, summary_cols)
    paired = []
    grouped = defaultdict(dict)
    for row in images:
        if row["mode"] != "P3_P6_paired" or int(row["warmup"]):
            continue
        key = (row["case"], row["repetition"], row["step"], row["plane"])
        grouped[key][row["format"]] = row
    for (name, repetition, step, plane), formats in sorted(grouped.items()):
        if "P3" not in formats or "P6" not in formats:
            raise RuntimeError(f"Paired image missing for {name} rep={repetition} step={step}")
        p3 = int(formats["P3"]["size_bytes"])
        p6 = int(formats["P6"]["size_bytes"])
        paired.append({"case": name, "repetition": repetition, "step": step, "plane": plane,
                       "P3_bytes": p3, "P6_bytes": p6, "P3_to_P6_ratio": p3 / p6})
    if paired:
        write_csv(out / "tailles_P3_P6_meme_image.csv", paired, list(paired[0]))
    print("\nResults written to:", out)
    print("runs.csv, images.csv, synthese.csv, metadata.json",
          ", tailles_P3_P6_meme_image.csv" if paired else "")


if __name__ == "__main__":
    main()
