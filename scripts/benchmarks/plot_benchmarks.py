import csv
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results" / "benchmarks.csv"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODE_ORDER = ["none", "snapshots", "sismos", "combined"]
MODE_LABELS = {
    "none": "Sans sorties",
    "snapshots": "Snapshots seuls",
    "sismos": "Sismos seuls",
    "combined": "Snapshots + sismos",
}
POINT_TYPES = [7, 5, 9, 11]


def load_recent_batch():
    if not RESULTS_FILE.is_file():
        raise FileNotFoundError(f"Resultats introuvables : {RESULTS_FILE}")

    with RESULTS_FILE.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Le fichier des benchmarks est vide")

    batch_id = rows[-1].get("batch_id", "")

    if batch_id:
        rows = [row for row in rows if row.get("batch_id") == batch_id]
    else:
        rows = [rows[-1]]

    return [
        row
        for row in rows
        if row.get("solver") in ("", "semproxy") and row.get("mode") in MODE_ORDER
    ]


def average(rows, field):
    values = []

    for row in rows:
        try:
            value = float(row.get(field, ""))

            if math.isfinite(value):
                values.append(value)

        except (TypeError, ValueError):
            continue

    return statistics.mean(values) if values else 0.0


def gnuplot_string(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def plot_lines(filename, title, y_label, thread_counts, series, zero_based=True):
    blocks = []
    plots = []

    for index, (label, values) in enumerate(series):
        name = f"series_{index}"
        pairs = "\n".join(
            f"{threads} {value:.12g}"
            for threads, value in zip(thread_counts, values)
        )
        blocks.append(f"${name} << EOD\n{pairs}\nEOD\n")
        plots.append(
            f"${name} using 1:2 with linespoints lw 3 "
            f"pt {POINT_TYPES[index % len(POINT_TYPES)]} ps 1.2 "
            f"title {gnuplot_string(label)}"
        )

    output = PLOTS_DIR / filename
    output.unlink(missing_ok=True)
    xtics = ", ".join(f'"{value}" {value}' for value in thread_counts)
    yrange = "[0:*]" if zero_based else "[*:*]"

    script = f'''set terminal pngcairo size 1500,850 font "Sans,12"
set output {gnuplot_string(str(output))}
set title {gnuplot_string(title)}
set ylabel {gnuplot_string(y_label)}
set xlabel "Nombre de threads"
set yrange {yrange}
set grid ytics xtics
set key outside top horizontal
set xtics ({xtics})
plot {", ".join(plots)}
unset output
'''

    result = subprocess.run(
        ["gnuplot"],
        input="\n".join(blocks) + script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Echec de gnuplot pour {filename} :\n{result.stderr[-5000:]}"
        )

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Graphique non genere : {output}")

    print(f"Graphique PNG : {output}")


def mode_series(groups, thread_counts, modes, field):
    return [
        (
            MODE_LABELS[mode],
            [average(groups[(threads, mode)], field) for threads in thread_counts],
        )
        for mode in modes
    ]


def speedup_values(groups, thread_counts, mode):
    baseline = average(groups[(1, mode)], "simulation_wall_s")

    if baseline <= 0:
        return [0.0 for _ in thread_counts]

    values = []

    for threads in thread_counts:
        elapsed = average(groups[(threads, mode)], "simulation_wall_s")
        values.append(baseline / elapsed if elapsed > 0 else 0.0)

    return values


def efficiency_values(speedups, thread_counts):
    return [
        speedup / threads * 100.0 if threads > 0 else 0.0
        for speedup, threads in zip(speedups, thread_counts)
    ]


def overhead_values(groups, thread_counts, mode):
    values = []

    for threads in thread_counts:
        baseline = average(groups[(threads, "none")], "simulation_wall_s")
        elapsed = average(groups[(threads, mode)], "simulation_wall_s")
        values.append(max(0.0, elapsed - baseline))

    return values


def overhead_percent_values(groups, thread_counts, mode):
    values = []

    for threads in thread_counts:
        baseline = average(groups[(threads, "none")], "simulation_wall_s")
        elapsed = average(groups[(threads, mode)], "simulation_wall_s")

        if baseline > 0:
            values.append(max(0.0, (elapsed - baseline) / baseline * 100.0))
        else:
            values.append(0.0)

    return values


def bandwidth_values(groups, thread_counts, mode):
    values = []

    for threads in thread_counts:
        rows = groups[(threads, mode)]
        size_mib = average(rows, "total_csv_mib")
        save_s = average(rows, "save_s")
        values.append(size_mib / save_s if save_s > 0 else 0.0)

    return values


def main():
    rows = load_recent_batch()

    if not rows:
        raise ValueError("Aucun resultat unifie dans la derniere serie")

    groups = defaultdict(list)

    for row in rows:
        groups[(int(row["threads"]), row["mode"])].append(row)

    thread_counts = sorted({int(row["threads"]) for row in rows})
    modes = [mode for mode in MODE_ORDER if any(row["mode"] == mode for row in rows)]

    plot_lines(
        "simulation_time.png",
        "Temps moyen de la simulation SEM",
        "Temps moyen (secondes)",
        thread_counts,
        mode_series(groups, thread_counts, modes, "simulation_wall_s"),
    )

    plot_lines(
        "workflow_time.png",
        "Temps moyen du workflow complet",
        "Temps moyen (secondes)",
        thread_counts,
        mode_series(groups, thread_counts, modes, "workflow_wall_s"),
    )

    plot_lines(
        "initial_time.png",
        "Temps d'initialisation SEM",
        "Temps moyen (secondes)",
        thread_counts,
        mode_series(groups, thread_counts, modes, "initial_s"),
    )

    plot_lines(
        "compute_time.png",
        "Temps de calcul SEM",
        "Temps moyen (secondes)",
        thread_counts,
        mode_series(groups, thread_counts, modes, "compute_loop_s"),
    )

    plot_lines(
        "kernel_time.png",
        "Temps du kernel SEM",
        "Temps moyen (secondes)",
        thread_counts,
        mode_series(groups, thread_counts, modes, "kernel_s"),
    )

    plot_lines(
        "output_time.png",
        "Temps Output SEM",
        "Temps moyen (secondes)",
        thread_counts,
        mode_series(groups, thread_counts, modes, "output_s"),
    )

    plot_lines(
        "save_time.png",
        "Temps de sauvegarde CSV",
        "Temps moyen (secondes)",
        thread_counts,
        mode_series(groups, thread_counts, modes, "save_s"),
    )

    plot_lines(
        "visualization_time.png",
        "Temps de visualisation",
        "Temps moyen (secondes)",
        thread_counts,
        mode_series(groups, thread_counts, modes, "visualization_total_s"),
    )

    if "none" in modes:
        plot_lines(
            "overhead.png",
            "Surcout par rapport a la configuration sans sorties",
            "Surcout moyen (secondes)",
            thread_counts,
            [
                (
                    MODE_LABELS[mode],
                    [0.0 for _ in thread_counts]
                    if mode == "none"
                    else overhead_values(groups, thread_counts, mode),
                )
                for mode in modes
            ],
            zero_based=True,
        )

        plot_lines(
            "overhead_percent.png",
            "Surcout relatif par rapport a la configuration sans sorties",
            "Surcout (%)",
            thread_counts,
            [
                (
                    MODE_LABELS[mode],
                    [0.0 for _ in thread_counts]
                    if mode == "none"
                    else overhead_percent_values(groups, thread_counts, mode),
                )
                for mode in modes
            ],
            zero_based=True,
        )

    if 1 in thread_counts:
        speedups = {
            mode: speedup_values(groups, thread_counts, mode)
            for mode in modes
        }

        plot_lines(
            "speedup.png",
            "Strong scaling SEM - speedup",
            "Speedup T(1) / T(p)",
            thread_counts,
            [(MODE_LABELS[mode], speedups[mode]) for mode in modes],
        )

        plot_lines(
            "parallel_efficiency.png",
            "Efficacite parallele SEM",
            "Efficacite (%)",
            thread_counts,
            [
                (
                    MODE_LABELS[mode],
                    efficiency_values(speedups[mode], thread_counts),
                )
                for mode in modes
            ],
        )
    else:
        print("Speedup non genere : une mesure a 1 thread est necessaire.")

    plot_lines(
        "io_bandwidth.png",
        "Debit I/O CSV",
        "Debit (MiB/s)",
        thread_counts,
        [
            (
                MODE_LABELS[mode],
                bandwidth_values(groups, thread_counts, mode),
            )
            for mode in modes
        ],
    )

    plot_lines(
        "csv_size.png",
        "Taille totale des CSV generes",
        "Taille moyenne (MiB)",
        thread_counts,
        mode_series(groups, thread_counts, modes, "total_csv_mib"),
    )


if __name__ == "__main__":
    main()