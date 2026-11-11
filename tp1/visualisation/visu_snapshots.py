import csv
import math
import argparse
import subprocess
import array
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images" / "snapshots"

def build_data_block(name, points):
    points.sort(key=lambda point: (point[0], point[1]))

    lines = []
    previous_coordinate = None

    for first_coordinate, second_coordinate, pressure in points:
        if (
            previous_coordinate is not None
            and first_coordinate != previous_coordinate
        ):
            lines.append("")

        lines.append(
            f"{first_coordinate} {second_coordinate} {pressure}"
        )

        previous_coordinate = first_coordinate

    return f"${name} << EOD\n" + "\n".join(lines) + "\nEOD\n"


def read_snapshot(path, coordinate_cache):
    if path.suffix == ".csv":
        columns = [array.array("f") for _ in range(4)]
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, []) != ["x", "y", "z", "pressure"]:
                raise ValueError(f"En-tête snapshot invalide : {path}")
            for row in reader:
                if len(row) != 4:
                    raise ValueError(f"Ligne snapshot invalide : {path}")
                for column, value in zip(columns, row):
                    column.append(float(value))
        return columns
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("schema") != "funtides.snapshot.v1" or meta.get("dtype") != "<f4":
        raise ValueError(f"Métadonnées snapshot invalides : {path}")

    def floats(file):
        values = array.array("f")
        if file.stat().st_size % 4:
            raise ValueError(f"Fichier tronqué : {file}")
        with file.open("rb") as stream:
            values.fromfile(stream, file.stat().st_size // 4)
        if sys.byteorder != "little":
            values.byteswap()
        return values

    coordinates = path.parent / meta["coordinates"]
    if coordinates not in coordinate_cache:
        xyz = floats(coordinates)
        if len(xyz) != 3 * meta["count"]:
            raise ValueError("Nombre de coordonnées incohérent")
        coordinate_cache[coordinates] = (xyz[0::3], xyz[1::3], xyz[2::3])
    pressure = floats(path)
    if len(pressure) != meta["count"]:
        raise ValueError("Nombre de pressions incohérent")
    return (*coordinate_cache[coordinates], pressure)


def visualize_snapshot(csv_file, args, coordinate_cache):
    start = time.perf_counter()
    x, y, z, pressure = read_snapshot(csv_file, coordinate_cache)
    read_s = time.perf_counter() - start
    start = time.perf_counter()
    point_count = len(pressure)
    if not point_count or not all(math.isfinite(v) for v in pressure):
        raise ValueError("Champ vide ou non fini : vérifier dt/CFL")
    min_pressure, max_pressure = min(pressure), max(pressure)
    mean_pressure = math.fsum(pressure) / point_count
    planes = ["xy", "xz", "yz"] if args.plan == "tous" else [args.plan]
    slices = {}
    positions = {}
    for plane in planes:
        first, second, fixed = {"xy": (x, y, z), "xz": (x, z, y), "yz": (y, z, x)}[plane]
        # Nearest existing nodal plane: GLL nodes are not uniformly spaced.
        chosen = min(fixed, key=lambda v: abs(v - args.position))
        positions[plane] = chosen
        slices[plane] = [(a, b, p) for a, b, c, p in zip(first, second, fixed, pressure) if c == chosen]
    analysis_s = time.perf_counter() - start
    if args.no_render:
        return dict(read_s=read_s, analysis_s=analysis_s, render_s=0.0, points=point_count,
                    min=min_pressure, max=max_pressure, mean=mean_pressure)
    start = time.perf_counter()
    image_name = (
        csv_file.stem
        if args.plan == "tous"
        else f"{csv_file.stem}_{args.plan}"
    )

    image_file = args.output / f"{image_name}.png"

    amplitude = max(abs(min_pressure), abs(max_pressure))

    if amplitude == 0:
        amplitude = 1.0

    width = 1800 if len(planes) == 3 else 800
    height = 650

    gnuplot_script = ""

    for plane, points in slices.items():
        gnuplot_script += build_data_block(
            f"slice_{plane}",
            points
        )

    gnuplot_script += f"""
set terminal pngcairo size {width},{height} font "Sans,12"
set output "{image_file}"

set view map
set pm3d map
set size ratio -1

set palette defined (-1 "blue", 0 "white", 1 "red")
set cbrange [{-amplitude}:{amplitude}]
set cblabel "Pressure"
"""

    if len(planes) == 3:
        gnuplot_script += (
            f'\nset multiplot layout 1,3 '
            f'title "Snapshot: {csv_file.stem}"\n'
        )

    axis_labels = {
        "xy": ("X", "Y", "z"),
        "xz": ("X", "Z", "y"),
        "yz": ("Y", "Z", "x")
    }

    for plane in planes:
        x_label, y_label, fixed_axis = axis_labels[plane]

        gnuplot_script += f"""
set title "Slice {plane.upper()} ({fixed_axis} = {positions[plane]})"
set xlabel "{x_label}"
set ylabel "{y_label}"

splot $slice_{plane} using 1:2:3 with pm3d notitle
"""

    if len(planes) == 3:
        gnuplot_script += "\nunset multiplot\n"

    gnuplot_script += "\nunset output\n"

    subprocess.run(
        ["gnuplot"],
        input=gnuplot_script,
        text=True,
        check=True
    )


    return dict(read_s=read_s, analysis_s=analysis_s, render_s=time.perf_counter() - start,
                points=point_count, min=min_pressure, max=max_pressure, mean=mean_pressure)


def main():
    parser = argparse.ArgumentParser(description="Visualisation ad-hoc des snapshots CSV ou binaires")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=IMAGES_DIR)
    parser.add_argument("--format", choices=["csv", "bin"], default="csv")
    parser.add_argument("--plan", choices=["xy", "xz", "yz", "tous"], default="tous")
    parser.add_argument("--position", type=float, default=1000)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    files = sorted(args.data.glob(f"snapshot_it_*.{args.format}"))
    if not files:
        parser.error(f"Aucun snapshot {args.format} dans {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cache, rows = {}, []
    for path in files:
        rows.append({"file": path.name, **visualize_snapshot(path, args, cache)})
    metrics = {key: sum(r[key] for r in rows) for key in ["read_s", "analysis_s", "render_s"]}
    metrics.update(total_s=time.perf_counter() - start, snapshot_count=len(rows), snapshots=rows)
    if args.metrics:
        args.metrics.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "snapshots"}, indent=2))


if __name__ == "__main__":
    main()
