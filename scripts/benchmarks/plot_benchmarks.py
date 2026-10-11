import csv
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results" / "benchmarks.csv"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_recent_batch():
    if not RESULTS_FILE.is_file():
        raise FileNotFoundError(f"Resultats introuvables : {RESULTS_FILE}")
    with RESULTS_FILE.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Le fichier des benchmarks est vide")
    batch_id = rows[-1].get("batch_id", "")
    selected = [row for row in rows if row.get("batch_id") == batch_id] if batch_id else [rows[-1]]
    return [row for row in selected if row.get("solver") in ("", "semproxy")]


def average(rows, field):
    values = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
            if math.isfinite(value) and value >= 0:
                values.append(value)
        except (TypeError, ValueError):
            continue
    return statistics.mean(values) if values else 0.0


def gnuplot_string(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def fallback_lines_svg(path, title, y_label, x_values, series):
    width = 1200
    height = 760
    left, right, top, bottom = 105, 55, 110, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [value for _, values in series for value in values]
    maximum = max(all_values, default=0.0)
    maximum = maximum * 1.15 if maximum > 0 else 1.0
    colors = ["#3f78a9", "#dd9140", "#3c9a70", "#8858ac"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="44" text-anchor="middle" font-family="sans-serif" font-size="22">{escape(title)}</text>',
        f'<text x="{left}" y="80" font-family="sans-serif" font-size="14">{escape(y_label)}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_height * (1 - tick / 5)
        parts.append(f'<line x1="{left}" y1="{y:.3f}" x2="{width-right}" y2="{y:.3f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+5:.3f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.3g}</text>')
    if len(x_values) == 1:
        x_positions = [left + plot_width / 2]
    else:
        x_positions = [left + plot_width * index / (len(x_values) - 1) for index in range(len(x_values))]
    for x, label in zip(x_positions, x_values):
        parts.append(f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{top+plot_height}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x:.3f}" y="{top+plot_height+30}" text-anchor="middle" font-family="sans-serif" font-size="13">{escape(str(label))}</text>')
    for index, (label, values) in enumerate(series):
        points = []
        for x, value in zip(x_positions, values):
            y = top + plot_height * (1 - value / maximum)
            points.append((x, y))
        if len(points) > 1:
            parts.append(
                f'<polyline fill="none" stroke="{colors[index % len(colors)]}" stroke-width="3" points="' +
                " ".join(f"{x:.3f},{y:.3f}" for x, y in points) + '"/>'
            )
        for x, y in points:
            parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="{colors[index % len(colors)]}"/>')
        legend_x = left + index * 240
        parts.append(f'<line x1="{legend_x}" y1="88" x2="{legend_x+28}" y2="88" stroke="{colors[index % len(colors)]}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x+14}" cy="88" r="4" fill="{colors[index % len(colors)]}"/>')
        parts.append(f'<text x="{legend_x+38}" y="93" font-family="sans-serif" font-size="13">{escape(label)}</text>')
    parts.append(f'<text x="{width/2}" y="{height-25}" text-anchor="middle" font-family="sans-serif" font-size="14">Nombre de threads</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Graphique SVG : {path}")


def plot_lines(filename, title, y_label, x_values, series):
    blocks = []
    plots = []
    point_types = [7, 5, 9, 11]
    for index, (label, values) in enumerate(series):
        name = f"series_{index}"
        pairs = "\n".join(f"{x} {value:.12g}" for x, value in zip(x_values, values))
        blocks.append(f"${name} << EOD\n{pairs}\nEOD\n")
        plots.append(
            f"${name} using 1:2 with linespoints lw 3 pt {point_types[index % len(point_types)]} ps 1.2 title {gnuplot_string(label)}"
        )
    png = PLOTS_DIR / filename
    png.unlink(missing_ok=True)
    xtics = ", ".join(f'"{value}" {value}' for value in x_values)
    script = f"""
set terminal pngcairo size 1500,850 font "Sans,12"
set output {gnuplot_string(str(png))}
set title {gnuplot_string(title)}
set ylabel {gnuplot_string(y_label)}
set xlabel "Nombre de threads"
set yrange [0:*]
set grid ytics xtics
set key outside top horizontal
set xtics ({xtics})
plot {', '.join(plots)}
unset output
"""
    try:
        result = subprocess.run(
            ["gnuplot"],
            input="\n".join(blocks) + script,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0 and png.is_file() and png.stat().st_size > 0:
            print(f"Graphique PNG : {png}")
            return
        print(f"Gnuplot indisponible (code {result.returncode}), utilisation de SVG.")
    except FileNotFoundError:
        print("Gnuplot introuvable, utilisation de SVG.")
    png.unlink(missing_ok=True)
    fallback_lines_svg(png.with_suffix(".svg"), title, y_label, x_values, series)


def positive_ratio(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def speedup_values(groups, thread_counts, enabled):
    if (1, enabled) not in groups:
        return None

    baseline = average(groups[(1, enabled)], "simulation_wall_s")
    if baseline <= 0:
        return None

    values = []
    for threads in thread_counts:
        elapsed = average(groups[(threads, enabled)], "simulation_wall_s")
        values.append(positive_ratio(baseline, elapsed))
    return values


def efficiency_values(speedups, thread_counts):
    return [
        positive_ratio(speedup, threads) * 100.0
        for speedup, threads in zip(speedups, thread_counts)
    ]


def io_bandwidth_values(groups, thread_counts):
    values = []
    for threads in thread_counts:
        rows = groups.get((threads, 1), [])
        size_mib = average(rows, "snapshot_csv_mib")
        output_s = average(rows, "output_s")
        values.append(positive_ratio(size_mib, output_s))
    return values


def main():
    rows = load_recent_batch()
    if not rows:
        raise ValueError("Aucun resultat SEM dans la derniere serie")

    groups = defaultdict(list)
    for row in rows:
        threads = int(row["threads"])
        enabled = int(row.get("snapshots_enabled", "1") or 1)
        groups[(threads, enabled)].append(row)

    thread_counts = sorted({int(row["threads"]) for row in rows})
    modes = sorted({int(row.get("snapshots_enabled", "1") or 1) for row in rows})

    if modes == [0, 1]:
        simulation_series = [
            (
                "Sans snapshots",
                [average(groups[(threads, 0)], "simulation_wall_s") for threads in thread_counts],
            ),
            (
                "Avec snapshots",
                [average(groups[(threads, 1)], "simulation_wall_s") for threads in thread_counts],
            ),
        ]

        workflow_series = [
            (
                "Sans snapshots",
                [average(groups[(threads, 0)], "workflow_wall_s") for threads in thread_counts],
            ),
            (
                "Avec snapshots + visualisation",
                [average(groups[(threads, 1)], "workflow_wall_s") for threads in thread_counts],
            ),
        ]

        output_series = [
            (
                "Sans snapshots",
                [average(groups[(threads, 0)], "output_s") for threads in thread_counts],
            ),
            (
                "Avec snapshots",
                [average(groups[(threads, 1)], "output_s") for threads in thread_counts],
            ),
        ]

        plot_lines(
            "simulation_time.png",
            "Temps moyen SEM : avec et sans snapshots",
            "Temps moyen (secondes)",
            thread_counts,
            simulation_series,
        )

        plot_lines(
            "workflow_time.png",
            "Temps moyen du workflow : avec et sans snapshots",
            "Temps moyen (secondes)",
            thread_counts,
            workflow_series,
        )

        plot_lines(
            "snapshot_output_time.png",
            "Temps Output SEM : avec et sans snapshots",
            "Temps moyen (secondes)",
            thread_counts,
            output_series,
        )

        phase_without = [
            (
                "Initialisation",
                [average(groups[(threads, 0)], "initial_s") for threads in thread_counts],
            ),
            (
                "Kernel",
                [average(groups[(threads, 0)], "kernel_s") for threads in thread_counts],
            ),
            (
                "Output",
                [average(groups[(threads, 0)], "output_s") for threads in thread_counts],
            ),
        ]

        phase_with = [
            (
                "Initialisation",
                [average(groups[(threads, 1)], "initial_s") for threads in thread_counts],
            ),
            (
                "Kernel",
                [average(groups[(threads, 1)], "kernel_s") for threads in thread_counts],
            ),
            (
                "Output",
                [average(groups[(threads, 1)], "output_s") for threads in thread_counts],
            ),
        ]

        plot_lines(
            "sem_phases_without_snapshots.png",
            "Etapes internes SEM - sans snapshots",
            "Temps moyen (secondes)",
            thread_counts,
            phase_without,
        )

        plot_lines(
            "sem_phases_with_snapshots.png",
            "Etapes internes SEM - avec snapshots",
            "Temps moyen (secondes)",
            thread_counts,
            phase_with,
        )

        overhead_seconds = []
        overhead_percent = []

        for threads in thread_counts:
            without = average(groups[(threads, 0)], "simulation_wall_s")
            with_snapshots = average(groups[(threads, 1)], "simulation_wall_s")
            overhead_seconds.append(with_snapshots - without)
            overhead_percent.append(positive_ratio(with_snapshots - without, without) * 100.0)

        plot_lines(
            "snapshot_overhead.png",
            "Surcout moyen des snapshots",
            "Surcout moyen (secondes)",
            thread_counts,
            [("Avec - sans snapshots", overhead_seconds)],
        )

        plot_lines(
            "snapshot_overhead_percent.png",
            "Surcout relatif des snapshots",
            "Surcout (%)",
            thread_counts,
            [("Overhead snapshots", overhead_percent)],
        )

        speedup_without = speedup_values(groups, thread_counts, 0)
        speedup_with = speedup_values(groups, thread_counts, 1)

        if speedup_without is not None and speedup_with is not None:
            plot_lines(
                "speedup.png",
                "Strong scaling SEM - speedup",
                "Speedup T(1) / T(p)",
                thread_counts,
                [
                    ("Sans snapshots", speedup_without),
                    ("Avec snapshots", speedup_with),
                    ("Speedup ideal", [float(threads) for threads in thread_counts]),
                ],
            )

            plot_lines(
                "parallel_efficiency.png",
                "Efficacite parallele SEM",
                "Efficacite (%)",
                thread_counts,
                [
                    ("Sans snapshots", efficiency_values(speedup_without, thread_counts)),
                    ("Avec snapshots", efficiency_values(speedup_with, thread_counts)),
                    ("Efficacite ideale", [100.0 for _ in thread_counts]),
                ],
            )
        else:
            print("Speedup non genere : le benchmark doit contenir une mesure a 1 thread.")

        plot_lines(
            "io_bandwidth.png",
            "Debit I/O effectif des snapshots",
            "Debit (MiB/s)",
            thread_counts,
            [("CSV / Output SEM", io_bandwidth_values(groups, thread_counts))],
        )

    else:
        enabled = modes[-1]
        label = "Avec snapshots" if enabled else "Sans snapshots"

        plot_lines(
            "simulation_time.png",
            f"Temps moyen de la simulation SEM - {label.lower()}",
            "Temps moyen (secondes)",
            thread_counts,
            [
                (
                    label,
                    [average(groups[(threads, enabled)], "simulation_wall_s") for threads in thread_counts],
                )
            ],
        )

        plot_lines(
            "workflow_time.png",
            f"Temps moyen du workflow - {label.lower()}",
            "Temps moyen (secondes)",
            thread_counts,
            [
                (
                    label,
                    [average(groups[(threads, enabled)], "workflow_wall_s") for threads in thread_counts],
                )
            ],
        )

        speedups = speedup_values(groups, thread_counts, enabled)

        if speedups is not None:
            plot_lines(
                "speedup.png",
                "Strong scaling SEM - speedup",
                "Speedup T(1) / T(p)",
                thread_counts,
                [
                    (label, speedups),
                    ("Speedup ideal", [float(threads) for threads in thread_counts]),
                ],
            )

            plot_lines(
                "parallel_efficiency.png",
                "Efficacite parallele SEM",
                "Efficacite (%)",
                thread_counts,
                [
                    (label, efficiency_values(speedups, thread_counts)),
                    ("Efficacite ideale", [100.0 for _ in thread_counts]),
                ],
            )
        else:
            print("Speedup non genere : le benchmark doit contenir une mesure a 1 thread.")

        if enabled == 1:
            plot_lines(
                "io_bandwidth.png",
                "Debit I/O effectif des snapshots",
                "Debit (MiB/s)",
                thread_counts,
                [("CSV / Output SEM", io_bandwidth_values(groups, thread_counts))],
            )


if __name__ == "__main__":
    main()