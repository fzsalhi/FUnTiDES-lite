import csv
import argparse
import re
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "seismograms.csv"
IMAGES_DIR = BASE_DIR / "images" / "seismograms"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(
    description="FUnTiDES seismogram visualization"
)

parser.add_argument(
    "--mode",
    choices=["all", "separate", "both"],
    default="all"
)

args = parser.parse_args()


def read_seismograms(csv_file):
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


def visualize_seismograms(receivers, samples, mode):
    gnuplot_script = build_data_block(samples)

    gnuplot_script += """
set grid
set xlabel "Time (s)"
set ylabel "Pressure"
"""

    generated_images = []

    if mode in ["all", "both"]:
        image_file = IMAGES_DIR / "seismograms_all.png"

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
                IMAGES_DIR / f"seismogram_{index - 1}_{safe_name}.png"
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
    if not DATA_FILE.exists():
        print(f"CSV file not found: {DATA_FILE}")
        return

    receivers, samples = read_seismograms(DATA_FILE)

    visualize_seismograms(receivers, samples, args.mode)

    print("\nVisualization completed.")


if __name__ == "__main__":
    main()