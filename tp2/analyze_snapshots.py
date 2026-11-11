#!/usr/bin/env python3
"""Statistiques ad-hoc des snapshots binaires et visualisation des plans nodaux."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
from io_utils import list_snapshots, load_snapshot, load_coords, snapshot_metadata, pressure_statistics, dump_json


def render_slice(coords, pressure, path, plane="XY", position=1000):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tp1" / "visualisation"))
    from visu_snapshots import build_data_block
    a, b, fixed = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[plane]
    levels = np.unique(coords[:, fixed])
    chosen = levels[np.argmin(abs(levels - position))]
    mask = coords[:, fixed] == chosen
    xyz, values = coords[mask], pressure[mask]
    u, v = np.unique(xyz[:, a]), np.unique(xyz[:, b])
    if len(u) * len(v) != len(values):
        raise ValueError("La slice doit etre une grille cartesienne complete")
    scale = float(np.max(np.abs(values))) or 1
    points = [(float(point[a]), float(point[b]), float(value)) for point, value in zip(xyz, values)]
    quoted = str(path).replace("\\", "\\\\").replace('"', '\\"')
    script = build_data_block("slice", points) + f"""
set terminal pngcairo size 800,650 font "Sans,12" noenhanced
set output "{quoted}"
set view map
set pm3d map
set size ratio -1
set xlabel "{'XYZ'[a]} (m)"
set ylabel "{'XYZ'[b]} (m)"
set title "{plane}, {'XYZ'[fixed]}={chosen:g} m"
set palette defined (-1 "blue", 0 "white", 1 "red")
set cbrange [{-scale}:{scale}]
set cblabel "Pression"
splot $slice using 1:2:3 with pm3d notitle
unset output
"""
    subprocess.run(["gnuplot"], input=script, text=True, check=True, capture_output=True)


def analyze(folder, output, pattern="snapshot_it_*.bin", dt=None, render=False, visual_only=False, plane="XY", position=1000):
    start = perf_counter()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    paths = list_snapshots(folder, pattern)
    timing = dict(read_s=0.0, compute_s=0.0, render_s=0.0, write_s=0.0, bytes_read=0, n_snapshots=len(paths))
    rows, coords = [], None
    for step, path in paths:
        t = perf_counter()
        meta = snapshot_metadata(path)
        pressure = load_snapshot(path)
        timing["bytes_read"] += path.stat().st_size + path.with_suffix('.json').stat().st_size
        if render and coords is None:
            coord_path = path.parent / meta.get("coordinates", "coords.bin")
            coords = load_coords(coord_path, "<f4")
            timing["bytes_read"] += coord_path.stat().st_size
        timing["read_s"] += perf_counter() - t
        t = perf_counter()
        stats = {} if visual_only else pressure_statistics(pressure)
        rows.append(dict(step=step, time_s=meta.get("time_s", (step + 1) * dt if dt else None), **stats))
        timing["compute_s"] += perf_counter() - t
        if render:
            t = perf_counter()
            render_slice(coords, pressure, out / f"frame_{step:06d}.png", plane, position)
            timing["render_s"] += perf_counter() - t
    t = perf_counter()
    with (out / "snapshots_stats.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    timing["write_s"] = perf_counter() - t
    timing["total_s"] = perf_counter() - start
    timing["n_nodes"] = len(pressure)
    dump_json(out / "timings_snapshots.json", timing)
    return rows, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snap-dir", required=True)
    parser.add_argument("--pattern", default="snapshot_it_*.bin")
    parser.add_argument("--out", default="tp2/results")
    parser.add_argument("--dt", type=float)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--visual-only", action="store_true")
    parser.add_argument("--plane", choices=["XY", "XZ", "YZ"], default="XY")
    parser.add_argument("--position", type=float, default=1000)
    args = parser.parse_args()
    _, timing = analyze(args.snap_dir, args.out, args.pattern, args.dt, args.render, args.visual_only, args.plane, args.position)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
