import argparse
import csv
import itertools
import json
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


def plot_lines(filename, title, y_label, thread_counts, series, zero_based=True, x_label="Nombre de threads", errors=None):
    blocks = []
    plots = []

    for index, (label, values) in enumerate(series):
        name = f"series_{index}"
        pairs = "\n".join(
            f"{threads} {value:.12g}" + (f" {errors[index][j]:.12g}" if errors is not None else "")
            for j, (threads, value) in enumerate(zip(thread_counts, values)) if value is not None
        )
        blocks.append(f"${name} << EOD\n{pairs}\nEOD\n")
        plots.append(
            f"${name} using {'1:2:3 with yerrorlines' if errors is not None else '1:2 with linespoints'} lw 3 "
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
set xlabel {gnuplot_string(x_label)}
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


def legacy_main():
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


def main():
    global RESULTS_FILE, PLOTS_DIR
    parser = argparse.ArgumentParser(description="Figures TP1 : dispersion, CPU/GPU, CSV/binaire")
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.input:
        candidates = list((BASE_DIR / "results").glob("benchmarks*.csv"))
        if not candidates:
            parser.error("Aucune mesure disponible")
        args.input = [max(candidates, key=lambda p: p.stat().st_mtime_ns)]
    records = []
    for path in args.input:
        with path.open(newline="") as stream:
            records.extend(csv.DictReader(stream))
    if not records:
        parser.error("Aucune mesure disponible")
    if "backend" not in records[0]:
        if len(args.input) != 1:
            parser.error("Les anciennes mesures doivent être tracées séparément")
        RESULTS_FILE = args.input[0]
        PLOTS_DIR = args.output or BASE_DIR / "plots"
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        legacy_main()
        return
    identity = {(r["machine"], r["config_hash"], r["source_hash"], r["io_sync"], r["visualize"]) for r in records}
    if len(identity) != 1:
        parser.error("Comparer des mesures de même machine, code, configuration, synchronisation et visualisation")
    if len({p.resolve() for p in args.input}) != len(args.input):
        parser.error("Un même fichier ne peut pas être fourni deux fois")
    PLOTS_DIR = args.output or BASE_DIR / "plots" / args.input[0].stem
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    keys = ("backend", "threads", "size", "mode", "format")
    for row in records:
        grouped[tuple(row[k] for k in keys)].append(row)
    numeric = ["simulation_wall_s", "workflow_wall_s", "compute_s", "receiver_sample_s", "snapshot_write_s", "sismo_write_s",
               "snapshot_read_s", "sismo_read_s", "snapshot_analysis_s", "sismo_analysis_s", "snapshot_render_s",
               "sismo_render_s", "visualization_total_s", "raw_output_bytes", "snapshot_bytes", "sismo_bytes", "save_s"]
    summary = []
    for key, rows in sorted(grouped.items()):
        item = dict(zip(keys, key), repetitions=len(rows))
        for metric in numeric:
            values = [float(r[metric]) for r in rows]
            avg = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            item.update({metric + "_mean": avg, metric + "_median": statistics.median(values), metric + "_std": std,
                         metric + "_min": min(values), metric + "_max": max(values), metric + "_cv": std / avg if avg else 0})
        summary.append(item)
    with (PLOTS_DIR / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    sizes = sorted({r["size"] for r in records}, key=lambda s: next(int(r["nodes"]) for r in records if r["size"] == s))
    combinations = sorted({(r["backend"], r["threads"], r["mode"], r["format"]) for r in records})

    def curves(metric, transform=None):
        series, errors = [], []
        for backend, threads, mode, fmt in combinations:
            y, e = [], []
            for size in sizes:
                rows = grouped.get((backend, threads, size, mode, fmt), [])
                values = [transform(r) if transform else float(r[metric]) for r in rows]
                y.append(statistics.mean(values) if values else None)
                e.append(statistics.stdev(values) if len(values) > 1 else 0)
            series.append((f"{backend.upper()} {threads}t {MODE_LABELS[mode]} {fmt}", y))
            errors.append(e)
        return series, errors

    suffix = " — " + records[0]["profile"] + " (moyenne +/- ecart-type)"
    for metric, filename, title, label in [
        ("simulation_wall_s", "simulation_time.png", "Execution SEM", "Secondes"),
        ("workflow_wall_s", "workflow_time.png", "Workflow ad-hoc", "Secondes"),
        ("compute_s", "compute_time.png", "Calcul hors sorties", "Secondes"),
        ("snapshot_write_s", "snapshot_write_time.png", "Ecriture snapshots", "Secondes"),
        ("sismo_write_s", "sismo_write_time.png", "Ecriture sismogrammes", "Secondes"),
        ("snapshot_read_s", "snapshot_read_time.png", "Lecture snapshots et decodage", "Secondes"),
        ("sismo_read_s", "sismo_read_time.png", "Lecture sismogrammes et decodage", "Secondes"),
        ("snapshot_analysis_s", "snapshot_analysis_time.png", "Analyse snapshots", "Secondes"),
        ("sismo_analysis_s", "sismo_analysis_time.png", "Analyse sismogrammes", "Secondes"),
        ("visualization_total_s", "visualization_time.png", "Post-traitement complet", "Secondes")]:
        series, errors = curves(metric)
        plot_lines(filename, title + suffix, label, range(1, len(sizes)+1), series,
                   x_label="Configurations " + ", ".join(sizes), errors=errors)
    for filename, title, label, transform in [
        ("storage_size.png", "Volume brut ecrit", "MiB", lambda r: float(r["raw_output_bytes"]) / 2**20),
        ("io_share.png", "Part des E/S dans le workflow", "Pourcentage", lambda r: 100 * (float(r["save_s"]) + float(r["snapshot_read_s"]) + float(r["sismo_read_s"])) / float(r["workflow_wall_s"])),
        ("io_bandwidth.png", "Debit logique (preparation et serialisation incluses)", "MiB/s", lambda r: float(r["raw_output_bytes"]) / 2**20 / float(r["save_s"]) if float(r["save_s"]) else 0)]:
        series, errors = curves(None, transform)
        plot_lines(filename, title + suffix, label, range(1, len(sizes)+1), series,
                   x_label="Configurations " + ", ".join(sizes), errors=errors)

    comparisons = []
    def med(backend, threads, size, mode, fmt, metric):
        rows = grouped.get((backend, threads, size, mode, fmt))
        return statistics.median(float(r[metric]) for r in rows) if rows else None

    for comparison, metric, ylabel in [("cpu_vs_gpu", "workflow_wall_s", "Temps CPU / GPU"),
                                        ("cpu_vs_gpu_compute", "compute_s", "Calcul CPU / GPU"),
                                        ("binary_vs_csv", "workflow_wall_s", "Temps CSV / binaire"),
                                        ("binary_vs_csv_storage", "raw_output_bytes", "Volume CSV / binaire")]:
        series = []
        for backend, threads, mode, fmt in combinations:
            is_gpu_comparison = comparison.startswith("cpu_vs_gpu")
            if is_gpu_comparison and backend != "cpu":
                continue
            if not is_gpu_comparison and fmt != "csv":
                continue
            y = []
            for size in sizes:
                a = med(backend, threads, size, mode, fmt, metric)
                b = med("gpu" if is_gpu_comparison else backend, threads, size, mode,
                        fmt if is_gpu_comparison else "bin", metric)
                y.append(a / b if a is not None and b else None)
                if y[-1] is not None:
                    comparisons.append(dict(comparison=comparison, backend=backend, threads=threads, size=size,
                                            mode=mode, format=fmt, ratio=y[-1]))
            if any(v is not None for v in y):
                series.append((f"{backend} {threads}t {mode} {fmt}", y))
        if series:
            plot_lines(comparison + ".png", comparison + " (ratio des medianes)", ylabel,
                       range(1, len(sizes)+1), series, x_label="Configurations " + ", ".join(sizes))
        elif comparison == "cpu_vs_gpu":
            print("CPU/GPU : mesures appariees manquantes, aucun resultat invente.")
    if comparisons:
        with (PLOTS_DIR / "comparisons.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    print(f"{len(records)} mesures ; résumé et figures : {PLOTS_DIR}")


if __name__ == "__main__":
    main()
