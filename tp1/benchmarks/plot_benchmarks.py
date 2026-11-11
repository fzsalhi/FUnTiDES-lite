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


def plot_lines(filename, title, y_label, thread_counts, series, zero_based=True, x_label="Nombre de threads", errors=None, x_tick_labels=None):
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
    labels = x_tick_labels if x_tick_labels is not None else thread_counts
    xtics = ", ".join(f'{gnuplot_string(label)} {value}' for label, value in zip(labels, thread_counts))
    yrange = "[0:*]" if zero_based else "[*:*]"

    script = f'''set terminal pngcairo size 1500,950 font "Sans,12" noenhanced
set output {gnuplot_string(str(output))}
set title {gnuplot_string(title)}
set ylabel {gnuplot_string(y_label)}
set xlabel {gnuplot_string(x_label)}
set yrange {yrange}
set grid ytics xtics
set key outside top center horizontal maxcols 3
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


PLOT_COLUMNS = {
    "workflow_wall_s": ("workflow", "Workflow complet", "Temps (s)"),
    "compute_s": ("compute", "Calcul seul", "Temps (s)"),
    "save_s": ("write", "Ecriture des snapshots et sismogrammes", "Temps (s)"),
    "read_s": ("read", "Lecture des snapshots et sismogrammes", "Temps (s)"),
    "raw_output_bytes": ("storage", "Volume brut ecrit", "Volume (Mio)"),
    "snapshot_write_s": ("snapshot_write", "Ecriture des snapshots", "Temps (s)"),
    "sismo_write_s": ("sismo_write", "Ecriture des sismogrammes", "Temps (s)"),
    "snapshot_read_s": ("snapshot_read", "Lecture des snapshots", "Temps (s)"),
    "sismo_read_s": ("sismo_read", "Lecture des sismogrammes", "Temps (s)"),
    "snapshot_analysis_s": ("snapshot_analysis", "Analyse des snapshots", "Temps (s)"),
    "sismo_analysis_s": ("sismo_analysis", "Analyse des sismogrammes", "Temps (s)"),
    "visualization_total_s": ("postprocess", "Post-traitement complet", "Temps (s)"),
}
DEFAULT_COLUMNS = "workflow_wall_s,compute_s,save_s,read_s,raw_output_bytes"
OLD_PLOTS = {
    "simulation_time.png", "workflow_time.png", "compute_time.png", "snapshot_write_time.png",
    "sismo_write_time.png", "snapshot_read_time.png", "sismo_read_time.png", "snapshot_analysis_time.png",
    "sismo_analysis_time.png", "visualization_time.png", "storage_size.png", "io_share.png", "io_bandwidth.png",
    "cpu_vs_gpu.png", "cpu_vs_gpu_compute.png", "binary_vs_csv.png", "binary_vs_csv_storage.png",
}


def main():
    global RESULTS_FILE, PLOTS_DIR
    parser = argparse.ArgumentParser(description="TP1 : comparaisons a deux courbes, sorties combinees")
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "plots")
    parser.add_argument("--columns", default=DEFAULT_COLUMNS, help=",".join(PLOT_COLUMNS))
    parser.add_argument("--threads", type=int, help="Nombre de threads CPU ; maximum disponible par defaut")
    parser.add_argument("--clean-plots", action="store_true")
    args = parser.parse_args()
    if not args.input:
        candidates = list((BASE_DIR / "results").glob("benchmarks*.csv"))
        if not candidates:
            parser.error("Aucune mesure disponible")
        args.input = [max(candidates, key=lambda p: p.stat().st_mtime_ns)]
    if len({p.resolve() for p in args.input}) != len(args.input):
        parser.error("Un meme fichier ne peut pas etre fourni deux fois")
    records = []
    for path in args.input:
        with path.open(newline="") as stream:
            records.extend(csv.DictReader(stream))
    if not records:
        parser.error("Aucune mesure disponible")
    if "backend" not in records[0]:
        if len(args.input) != 1:
            parser.error("Les anciennes mesures doivent etre tracees separement")
        RESULTS_FILE, PLOTS_DIR = args.input[0], args.output
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        legacy_main()
        return
    identity = {(r["machine"], r["config_hash"], r["source_hash"], r["io_sync"], r["visualize"]) for r in records}
    if len(identity) != 1:
        parser.error("Les campagnes doivent avoir la meme machine, configuration, code, synchronisation et visualisation")
    columns = list(dict.fromkeys(c.strip() for c in args.columns.split(",")))
    if not columns or any(c not in PLOT_COLUMNS for c in columns):
        parser.error("Colonnes disponibles : " + ",".join(PLOT_COLUMNS))
    combined = [r for r in records if r["mode"] == "combined"]
    if not combined:
        print("Aucune sortie combinee : pas de comparaison CPU/GPU ou CSV/binaire a tracer.")
        return
    threads = args.threads if args.threads is not None else max(int(r["threads"]) for r in combined)
    selected = [r for r in combined if int(r["threads"]) == threads]
    if not selected:
        parser.error(f"Aucune mesure avec {threads} threads")
    if len({(r["steps"], r["dt_s"]) for r in selected}) != 1:
        parser.error("Les nombres de pas ou les pas de temps sont differents")
    PLOTS_DIR = args.output
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.clean_plots:
        import shutil
        names = set(OLD_PLOTS) | {"cpu_vs_gpu_csv.png", "cpu_vs_gpu_bin.png"}
        for backend in ("cpu", "gpu"):
            names.add(f"binary_vs_csv_{backend}.png")
            for alias, _, _ in PLOT_COLUMNS.values():
                names.add(f"binary_vs_csv_{alias}_{backend}.png")
        for path in PLOTS_DIR.glob("*.png"):
            archive = BASE_DIR / "plots_cpu_threads" / path.name
            if path.name in names or (archive.is_file() and path.read_bytes() == archive.read_bytes()):
                path.unlink()
        for source in args.input:
            previous = BASE_DIR / "plots" / source.stem
            if previous.is_dir() and previous.resolve() != PLOTS_DIR.resolve():
                expected = OLD_PLOTS | {"summary.csv", "comparisons.csv"}
                if all(p.is_file() and p.name in expected for p in previous.iterdir()):
                    shutil.rmtree(previous)
    sizes = sorted({r["size"] for r in selected},
                   key=lambda size: next(int(r["nodes"]) for r in selected if r["size"] == size))
    groups = defaultdict(list)
    for row in selected:
        groups[(row["backend"], row["format"], row["size"])].append(row)

    def value(row, column):
        if column == "read_s":
            result = float(row["snapshot_read_s"]) + float(row["sismo_read_s"])
        else:
            result = float(row[column])
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"Mesure invalide : {column}")
        return result

    summary = []
    for (backend, fmt, size), rows in sorted(groups.items()):
        entry = dict(backend=backend, threads=threads, size=size, mode="combined", format=fmt, repetitions=len(rows))
        for column in columns:
            numbers = [value(r, column) for r in rows]
            avg = statistics.mean(numbers)
            std = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
            entry.update({column + "_mean": avg, column + "_median": statistics.median(numbers),
                          column + "_std": std, column + "_min": min(numbers), column + "_max": max(numbers),
                          column + "_cv": std / avg if avg else 0})
        summary.append(entry)
    with (PLOTS_DIR / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    rendered = []
    plotted = []
    repetitions = sorted({len(rows) for rows in groups.values()})

    def draw(column, filename, title, curves):
        series, errors = [], []
        scale = 2**20 if column == "raw_output_bytes" else 1
        for label, backend, fmt in curves:
            means, deviations = [], []
            for size in sizes:
                rows = groups.get((backend, fmt, size))
                if not rows:
                    return  
                numbers = [value(r, column) / scale for r in rows]
                avg = statistics.mean(numbers)
                std = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
                means.append(avg)
                deviations.append(std)
            series.append((label, means))
            errors.append(deviations)
        if len(series) != 2:
            raise ValueError("Une comparaison doit contenir exactement deux courbes")
        plot_lines(filename, title + " — snapshots + sismogrammes", PLOT_COLUMNS[column][2],
                   range(1, len(sizes)+1), series, x_label="Configuration", errors=errors, x_tick_labels=sizes)
        rendered.append(filename)
        for label, backend, fmt in curves:
            plotted.append(dict(figure=filename, curve=label, backend=backend, format=fmt, column=column))

    preferred = "bin" if any(r["format"] == "bin" for r in selected) else "csv"
    format_names = {"csv": "CSV", "bin": "Binaire"}
    visual = "visualisation incluse" if selected[0]["visualize"] == "1" else "sans rendu PNG"
    for column in columns:
        alias, label, _ = PLOT_COLUMNS[column]
        if column == "workflow_wall_s":
            for fmt in (preferred, "csv" if preferred == "bin" else "bin"):
                filename = "cpu_vs_gpu.png" if fmt == preferred else f"cpu_vs_gpu_{fmt}.png"
                draw(column, filename, f"Workflow complet CPU / GPU — {format_names[fmt]}, {visual}",
                     [(f"CPU ({threads} threads)", "cpu", fmt), ("GPU", "gpu", fmt)])
        if column == "compute_s":
            draw(column, "cpu_vs_gpu_compute.png", f"Calcul CPU / GPU — {format_names[preferred]}",
                 [(f"CPU ({threads} threads)", "cpu", preferred), ("GPU", "gpu", preferred)])
            continue
        if column == "raw_output_bytes":
            backend = "cpu" if any(r["backend"] == "cpu" for r in selected) else "gpu"
            draw(column, "binary_vs_csv_storage.png", f"Volume CSV / binaire — {backend.upper()}",
                 [("CSV", backend, "csv"), ("Binaire", backend, "bin")])
            continue
        for backend in ("cpu", "gpu"):
            suffix = backend if column == "workflow_wall_s" else f"{alias}_{backend}"
            detail = f", {visual}" if column == "workflow_wall_s" else ""
            draw(column, f"binary_vs_csv_{suffix}.png", f"{label} CSV / binaire — {backend.upper()}{detail}",
                 [("CSV", backend, "csv"), ("Binaire", backend, "bin")])
    with (PLOTS_DIR / "plotted_columns.csv").open("w", newline="") as stream:
        if plotted:
            writer = csv.DictWriter(stream, fieldnames=list(plotted[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(plotted)
    print(f"{len(selected)} mesures combinees, {threads} threads CPU, repetitions par cas : {repetitions}")
    print(f"{len(rendered)} figures a deux courbes : {PLOTS_DIR}")


if __name__ == "__main__":
    main()
