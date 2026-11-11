import csv
import argparse
import re
import subprocess
import array
import json
import math
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "seismograms.csv"
IMAGES_DIR = BASE_DIR / "images" / "seismograms"

def read_seismograms(csv_file):
    if csv_file.suffix == ".bin":
        meta = json.loads(csv_file.with_suffix(".json").read_text())
        if (meta.get("schema") != "funtides.seismograms.v1" or meta.get("dtype") != "<f4"
                or meta.get("layout") != "sample-major"):
            raise ValueError("Métadonnées sismogrammes invalides")
        receivers = len(meta["receivers"])
        count = meta["samples"] * receivers
        if receivers < 1 or meta["samples"] < 1 or csv_file.stat().st_size != count * 4:
            raise ValueError("Taille des sismogrammes incohérente")
        values = array.array("f")
        with csv_file.open("rb") as stream:
            values.fromfile(stream, count)
        if sys.byteorder != "little":
            values.byteswap()
        return [f"receiver_{r}" for r in range(receivers)], [
            [meta["time_start_s"] + t * meta["dt_s"], *values[t*receivers:(t+1)*receivers]]
            for t in range(meta["samples"])]
    with open(csv_file, "r", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "time" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'time' column")

        receivers = [
            name for name in reader.fieldnames
            if name != "time"
        ]

        if not receivers:
            raise ValueError("No receivers found in CSV")

        samples = []

        for row in reader:
            values = [float(row["time"])]

            for receiver in receivers:
                values.append(float(row[receiver]))

            samples.append(values)

    if not samples:
        raise ValueError("CSV file is empty")

    samples.sort(key=lambda sample: sample[0])

    return receivers, samples


def build_data_block(samples):
    lines = []

    for sample in samples:
        line = " ".join(str(value) for value in sample)
        lines.append(line)

    return "$signals << EOD\n" + "\n".join(lines) + "\nEOD\n"


def escape_label(label):
    return label.replace("\\", "\\\\").replace('"', '\\"')


def visualize_seismograms(receivers, samples, mode, output):
    gnuplot_script = build_data_block(samples)

    gnuplot_script += """
set grid
set xlabel "Time (s)"
set ylabel "Pressure"
"""

    generated_images = []

    if mode in ["all", "both"]:
        image_file = output / "seismograms_all.png"

        gnuplot_script += f"""
set terminal pngcairo size 1200,700 font "Sans,12"
set output "{image_file}"

set title "FUnTiDES - All Seismograms"
set key outside right
"""

        plots = []

        for index, receiver in enumerate(receivers, start=2):
            label = escape_label(receiver)

            plots.append(
                f'$signals using 1:{index} '
                f'with lines linewidth 2 title "{label}"'
            )

        gnuplot_script += "\nplot " + ", ".join(plots) + "\n"
        gnuplot_script += "\nunset output\n"

        generated_images.append(image_file)

    if mode in ["separate", "both"]:
        gnuplot_script += """
set terminal pngcairo size 1000,600 font "Sans,12"
unset key
"""

        for index, receiver in enumerate(receivers, start=2):
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", receiver)
            image_file = (
                output / f"seismogram_{index - 1}_{safe_name}.png"
            )

            label = escape_label(receiver)

            gnuplot_script += f"""
set output "{image_file}"

set title "FUnTiDES - Receiver {label}"

plot $signals using 1:{index} with lines linewidth 2 notitle

unset output
"""

            generated_images.append(image_file)

    subprocess.run(
        ["gnuplot"],
        input=gnuplot_script,
        text=True,
        check=True
    )

    print(f"Number of receivers: {len(receivers)}")
    print(f"Number of time samples: {len(samples)}")

    for image_file in generated_images:
        print(f"Image generated: {image_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualisation ad-hoc des sismogrammes CSV ou binaires")
    parser.add_argument("--input", type=Path, default=DATA_FILE)
    parser.add_argument("--output", type=Path, default=IMAGES_DIR)
    parser.add_argument("--mode", choices=["all", "separate", "both"], default="all")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    receivers, samples = read_seismograms(args.input)
    read_s = time.perf_counter() - start
    stage = time.perf_counter()
    if not all(math.isfinite(value) for row in samples for value in row):
        raise ValueError("Signal non fini : vérifier dt/CFL")
    stats = []
    for r, name in enumerate(receivers, 1):
        values = [row[r] for row in samples]
        stats.append(dict(receiver=name, min=min(values), max=max(values), mean=math.fsum(values)/len(values)))
    analysis_s = time.perf_counter() - stage
    stage = time.perf_counter()
    if not args.no_render:
        visualize_seismograms(receivers, samples, args.mode, args.output)
    metrics = dict(read_s=read_s, analysis_s=analysis_s, render_s=time.perf_counter() - stage,
                   total_s=time.perf_counter() - start, receiver_count=len(receivers), samples=len(samples), statistics=stats)
    if args.metrics:
        args.metrics.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
