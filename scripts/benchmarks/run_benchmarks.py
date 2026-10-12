import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent
VIS_DIR = ROOT_DIR / "scripts" / "visualisation"
DATA_DIR = VIS_DIR / "data"
SNAPSHOT_IMAGES_DIR = VIS_DIR / "images" / "snapshots"
SISMO_IMAGES_DIR = VIS_DIR / "images" / "sismos"
RESULTS_FILE = BASE_DIR / "results" / "benchmarks.csv"
EXE = ROOT_DIR / "build-openmp" / "bin" / "semproxy"
SNAPSHOT_VISUALIZER = VIS_DIR / "visu_snapshots.py"
RCV_FILE = ROOT_DIR / "data" / "receivers.csv"
SISMO_FILE = DATA_DIR / "sismos_benchmark.csv"
SISMO_IMAGE = SISMO_IMAGES_DIR / "sismos_benchmark.png"

ORDER = 2
EX = 15
EY = 15
EZ = 15
NX = EX * ORDER + 1
NY = EY * ORDER + 1
NZ = EZ * ORDER + 1
LX = 2000
LY = 2000
LZ = 2000
DT = 0.001
TIMEMAX = 1.5
SNAP_INTERVAL = 50
SLICE_POSITION = 1000

DEFAULT_RECEIVERS = [
    (1200, 1000, 1000),
    (1400, 1000, 1000),
    (1600, 1000, 1000),
]

MODE_ORDER = ["none", "snapshots", "sismos", "combined"]
MODE_LABELS = {
    "none": "Sans sorties",
    "snapshots": "Snapshots seuls",
    "sismos": "Sismos seuls",
    "combined": "Snapshots + sismos",
}

COLUMNS = [
    "date",
    "solver",
    "threads",
    "nx",
    "ny",
    "nz",
    "mode",
    "snapshots_enabled",
    "sismos_enabled",
    "receiver_count",
    "simulation_wall_s",
    "simulation_internal_s",
    "initial_s",
    "compute_loop_s",
    "compute_s",
    "kernel_s",
    "output_s",
    "save_s",
    "snapshots_viz_s",
    "sismos_viz_s",
    "visualization_total_s",
    "workflow_wall_s",
    "snapshot_csv_count",
    "snapshot_csv_mib",
    "snapshot_png_count",
    "snapshot_png_mib",
    "sismo_csv_count",
    "sismo_csv_mib",
    "sismo_png_count",
    "sismo_png_mib",
    "total_csv_mib",
    "total_png_mib",
    "batch_id",
    "run",
]


def parse_threads(value):
    try:
        numbers = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Utilise par exemple --threads 1,2,4,8"
        ) from error

    if (
        not numbers
        or any(number < 1 for number in numbers)
        or len(numbers) != len(set(numbers))
    ):
        raise argparse.ArgumentTypeError(
            "Les threads doivent etre des entiers positifs distincts"
        )

    return numbers


def parse_modes(value):
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]

    if not modes:
        raise argparse.ArgumentTypeError("La liste des modes est vide")

    invalid = [mode for mode in modes if mode not in MODE_ORDER]

    if invalid:
        raise argparse.ArgumentTypeError(
            "Modes disponibles : none,snapshots,sismos,combined"
        )

    return list(dict.fromkeys(modes))


def run_command(command, title, env=None):
    start = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        env=env,
    )

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"Echec de {title} (code {result.returncode}) :\n"
            f"{(result.stdout + result.stderr)[-5000:]}"
        )

    return elapsed, result.stdout + "\n" + result.stderr


def extract_time(output, pattern):
    matches = re.findall(pattern, output, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else ""


def get_solver_times(output):
    number = r"([0-9.eE+-]+)"

    times = {
        "simulation_internal_s": extract_time(
            output,
            rf"Elapsed\s+TotalExe\s+Time\s*:\s*{number}",
        ),
        "initial_s": extract_time(
            output,
            rf"Elapsed\s+Initial\s+Time\s*:\s*{number}",
        ),
        "compute_loop_s": extract_time(
            output,
            rf"Elapsed\s+Compute\s+Time\s*:\s*{number}",
        ),
        "kernel_s": extract_time(
            output,
            rf"Elapsed\s+Kernel\s+Time\s*:\s*{number}",
        ),
        "output_s": extract_time(
            output,
            rf"Elapsed\s+Output\s+Time\s*:\s*{number}",
        ),
        "save_s": extract_time(
            output,
            rf"Elapsed\s+Save\s+Time\s*:\s*{number}",
        ),
    }

    times["compute_s"] = times["kernel_s"] if times["kernel_s"] != "" else ""
    return times


def ensure_receivers_file():
    RCV_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not RCV_FILE.is_file():
        with RCV_FILE.open("w", encoding="utf-8") as file:
            for x, y, z in DEFAULT_RECEIVERS:
                file.write(f"{x},{y},{z}\n")


def receiver_count(path):
    count = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.replace(",", " ").replace(";", " ").split()

            if len(parts) < 3:
                continue

            try:
                float(parts[0])
                float(parts[1])
                float(parts[2])
            except ValueError:
                continue

            count += 1

    return count


def file_info(paths):
    files = [path for path in paths if path.is_file()]
    size = sum(path.stat().st_size for path in files) / (1024 * 1024)
    return len(files), size


def file_state(paths):
    return {
        str(path.resolve()): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
        if path.is_file()
    }


def changed_files(paths, previous):
    changed = []

    for path in paths:
        if not path.is_file():
            continue

        current = (path.stat().st_mtime_ns, path.stat().st_size)

        if previous.get(str(path.resolve())) != current:
            changed.append(path)

    return changed


def clear_generated_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for path in DATA_DIR.glob("snapshot_it_*.csv"):
        path.unlink()

    SISMO_FILE.unlink(missing_ok=True)


def gnuplot_string(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def visualize_sismos(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)

    if not header or len(header) < 2:
        raise ValueError(f"CSV sismos invalide : {csv_path}")

    SISMO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plots = []

    for column in range(2, len(header) + 1):
        label = header[column - 1]
        plots.append(
            f"{gnuplot_string(str(csv_path))} using 1:{column} "
            f"with lines lw 2 title {gnuplot_string(label)}"
        )

    script = f'''set datafile separator ","
set terminal pngcairo size 1500,850 font "Sans,12"
set output {gnuplot_string(str(SISMO_IMAGE))}
set title "Sismogrammes aux recepteurs"
set xlabel "Temps (s)"
set ylabel "Pression"
set grid
set key outside top horizontal
plot {", ".join(plots)}
unset output
'''

    start = time.perf_counter()

    result = subprocess.run(
        ["gnuplot"],
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT_DIR,
    )

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            "Echec de la visualisation sismos :\n"
            f"{result.stderr[-5000:]}"
        )

    if not SISMO_IMAGE.is_file() or SISMO_IMAGE.stat().st_size == 0:
        raise RuntimeError("La visualisation sismos n'a pas genere de PNG")

    return elapsed, SISMO_IMAGE


def visualize_snapshots():
    if not SNAPSHOT_VISUALIZER.is_file():
        raise FileNotFoundError(
            f"Visualiseur snapshots introuvable : {SNAPSHOT_VISUALIZER}"
        )

    SNAPSHOT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    before = file_state(SNAPSHOT_IMAGES_DIR.rglob("*.png"))

    elapsed, _ = run_command(
        [
            sys.executable,
            str(SNAPSHOT_VISUALIZER),
            "--position",
            str(SLICE_POSITION),
        ],
        "visualisation snapshots",
    )

    generated = changed_files(SNAPSHOT_IMAGES_DIR.rglob("*.png"), before)

    if not generated:
        raise RuntimeError(
            f"Aucune image PNG nouvelle ou modifiee dans {SNAPSHOT_IMAGES_DIR}"
        )

    return elapsed, generated


def save_result(row):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    existing_columns = []

    if RESULTS_FILE.is_file() and RESULTS_FILE.stat().st_size:
        with RESULTS_FILE.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            existing_columns = reader.fieldnames or []
            existing_rows = list(reader)

    fieldnames = list(dict.fromkeys(existing_columns + COLUMNS))

    if existing_rows and existing_columns != fieldnames:
        temporary = None

        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=RESULTS_FILE.parent,
                prefix="benchmarks_",
                suffix=".csv",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)

            os.replace(temporary, RESULTS_FILE)

        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    header = not RESULTS_FILE.is_file() or RESULTS_FILE.stat().st_size == 0

    with RESULTS_FILE.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if header:
            writer.writeheader()

        writer.writerow(row)


def mean_value(rows, field):
    values = []

    for row in rows:
        try:
            value = float(row.get(field, ""))

            if math.isfinite(value):
                values.append(value)

        except (TypeError, ValueError):
            continue

    return statistics.mean(values) if values else None


def show_mean(label, rows, field):
    value = mean_value(rows, field)

    if value is None:
        print(f"  {label:<38} non disponible")
    else:
        print(f"  {label:<38} {value:.6f} s")


def mode_flags(mode):
    snapshots_enabled = mode in ("snapshots", "combined")
    sismos_enabled = mode in ("sismos", "combined")
    return snapshots_enabled, sismos_enabled


def build_command(mode):
    snapshots_enabled, sismos_enabled = mode_flags(mode)

    command = [
        str(EXE),
        "--order",
        str(ORDER),
        "--ex",
        str(EX),
        "--ey",
        str(EY),
        "--ez",
        str(EZ),
        "--lx",
        str(LX),
        "--ly",
        str(LY),
        "--lz",
        str(LZ),
        "--method",
        "sem",
        "--mesh",
        "cartesian",
        "--implem",
        "makutu",
        "--dt",
        str(DT),
        "--timemax",
        str(TIMEMAX),
    ]

    command.extend(["--rcv-file", str(RCV_FILE)])

    if sismos_enabled:
        command.extend(["--sismo-file", str(SISMO_FILE)])
    else:
        command.extend(["--sismo-file", ""])

    if snapshots_enabled:
        command.extend(
            [
                "--is-snapshots=true",
                "--snap-time-interval",
                str(SNAP_INTERVAL),
                "--snap-folder",
                str(DATA_DIR),
            ]
        )
    else:
        command.extend(["--is-snapshots=false"])

    return command


def run_once(threads, repetition, batch_id, mode, n_receivers):
    snapshots_enabled, sismos_enabled = mode_flags(mode)
    clear_generated_data()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "threads"

    workflow_start = time.perf_counter()
    simulation_wall, output = run_command(build_command(mode), "SEM", env=env)
    solver_times = get_solver_times(output)

    snapshots_viz = 0.0
    sismos_viz = 0.0
    snapshot_csv_files = []
    snapshot_png_files = []
    sismo_csv_files = []
    sismo_png_files = []

    if snapshots_enabled:
        snapshot_csv_files = sorted(DATA_DIR.glob("snapshot_it_*.csv"))

        if not snapshot_csv_files:
            raise FileNotFoundError(
                f"Aucun snapshot genere dans {DATA_DIR}"
            )

        snapshots_viz, snapshot_png_files = visualize_snapshots()

    if sismos_enabled:
        if not SISMO_FILE.is_file():
            raise FileNotFoundError(f"Aucun fichier sismo genere : {SISMO_FILE}")

        sismo_csv_files = [SISMO_FILE]
        sismos_viz, sismo_image = visualize_sismos(SISMO_FILE)
        sismo_png_files = [sismo_image]

    workflow_wall = time.perf_counter() - workflow_start

    snapshot_csv_count, snapshot_csv_mib = file_info(snapshot_csv_files)
    snapshot_png_count, snapshot_png_mib = file_info(snapshot_png_files)
    sismo_csv_count, sismo_csv_mib = file_info(sismo_csv_files)
    sismo_png_count, sismo_png_mib = file_info(sismo_png_files)

    row = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "solver": "semproxy",
        "threads": threads,
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "mode": mode,
        "snapshots_enabled": int(snapshots_enabled),
        "sismos_enabled": int(sismos_enabled),
        "receiver_count": n_receivers,
        "simulation_wall_s": simulation_wall,
        **solver_times,
        "snapshots_viz_s": snapshots_viz,
        "sismos_viz_s": sismos_viz,
        "visualization_total_s": snapshots_viz + sismos_viz,
        "workflow_wall_s": workflow_wall,
        "snapshot_csv_count": snapshot_csv_count,
        "snapshot_csv_mib": snapshot_csv_mib,
        "snapshot_png_count": snapshot_png_count,
        "snapshot_png_mib": snapshot_png_mib,
        "sismo_csv_count": sismo_csv_count,
        "sismo_csv_mib": sismo_csv_mib,
        "sismo_png_count": sismo_png_count,
        "sismo_png_mib": sismo_png_mib,
        "total_csv_mib": snapshot_csv_mib + sismo_csv_mib,
        "total_png_mib": snapshot_png_mib + sismo_png_mib,
        "batch_id": batch_id,
        "run": repetition,
    }

    save_result(row)
    return row


def print_means(threads, rows, runs, mode):
    print(
        f"\n====== MOYENNES {MODE_LABELS[mode].upper()} - "
        f"{threads} THREAD(S) ({runs} repetitions) ======"
    )

    show_mean("Total SEM (horloge Python)", rows, "simulation_wall_s")
    show_mean("Total SEM (TotalExe)", rows, "simulation_internal_s")
    show_mean("Initialisation", rows, "initial_s")
    show_mean("Calcul SEM", rows, "compute_loop_s")
    show_mean("Kernel", rows, "kernel_s")
    show_mean("Output SEM", rows, "output_s")
    show_mean("Sauvegarde CSV", rows, "save_s")
    show_mean("Visualisation snapshots", rows, "snapshots_viz_s")
    show_mean("Visualisation sismos", rows, "sismos_viz_s")
    show_mean("Visualisation totale", rows, "visualization_total_s")
    show_mean("TOTAL workflow execute", rows, "workflow_wall_s")

    last = rows[-1]

    print("\n====== TAILLES DES FICHIERS ======")
    print(
        f"  Snapshots CSV    : {last['snapshot_csv_count']} fichier(s), "
        f"{last['snapshot_csv_mib']:.6f} MiB"
    )
    print(
        f"  Sismos CSV       : {last['sismo_csv_count']} fichier(s), "
        f"{last['sismo_csv_mib']:.6f} MiB"
    )
    print(
        f"  Images snapshots : {last['snapshot_png_count']} PNG, "
        f"{last['snapshot_png_mib']:.6f} MiB"
    )
    print(
        f"  Image sismos     : {last['sismo_png_count']} PNG, "
        f"{last['sismo_png_mib']:.6f} MiB"
    )


def safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None

    return numerator / denominator


def print_hpc_metrics(all_rows, thread_counts, modes):
    print("\n====== METRIQUES HPC ======")

    for mode in modes:
        baseline = mean_value(all_rows.get((1, mode), []), "simulation_wall_s")

        if baseline is None:
            print(
                f"\nSpeedup / efficacite ({MODE_LABELS[mode]}) : "
                "thread 1 absent, calcul impossible"
            )
            continue

        print(f"\n{MODE_LABELS[mode].upper()}")
        print("  threads    speedup    efficacite")

        for threads in thread_counts:
            elapsed = mean_value(
                all_rows.get((threads, mode), []),
                "simulation_wall_s",
            )
            speedup = safe_ratio(baseline, elapsed)

            if speedup is None:
                continue

            efficiency = speedup / threads * 100.0
            print(f"  {threads:<10}{speedup:<11.3f}{efficiency:.2f} %")

    if "none" in modes:
        print("\n====== SURCOUT PAR RAPPORT A SANS SORTIES ======")
        print("  threads    mode                     secondes       pourcentage")

        for threads in thread_counts:
            baseline = mean_value(
                all_rows.get((threads, "none"), []),
                "simulation_wall_s",
            )

            if baseline is None or baseline <= 0:
                continue

            for mode in modes:
                if mode == "none":
                    continue

                elapsed = mean_value(
                    all_rows.get((threads, mode), []),
                    "simulation_wall_s",
                )

                if elapsed is None:
                    continue

                overhead = max(0.0, elapsed - baseline)
                percent = max(0.0, overhead / baseline * 100.0)
                print(
                    f"  {threads:<10}{MODE_LABELS[mode]:<25}"
                    f"{overhead:<15.6f}{percent:.2f} %"
                )

    print("\n====== DEBIT I/O CSV ======")
    print("  threads    mode                     MiB/s")

    for threads in thread_counts:
        for mode in modes:
            rows = all_rows.get((threads, mode), [])
            size_mib = mean_value(rows, "total_csv_mib")
            save_s = mean_value(rows, "save_s")
            bandwidth = safe_ratio(size_mib, save_s)

            if bandwidth is not None:
                print(
                    f"  {threads:<10}{MODE_LABELS[mode]:<25}"
                    f"{bandwidth:.3f}"
                )

    print("\nSpeedup = T(1) / T(p)")
    print("Efficacite = Speedup / p * 100")
    print("Overhead (%) = (Tmode - Tsans_sorties) / Tsans_sorties * 100")
    print("Debit I/O = taille totale des CSV / temps de sauvegarde")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SEM des sorties snapshots et sismos"
    )
    parser.add_argument("--threads", type=parse_threads, default=[4])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=MODE_ORDER,
        help="none,snapshots,sismos,combined",
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs doit etre superieur ou egal a 1")

    if not EXE.is_file():
        parser.error(f"Executable semproxy introuvable : {EXE}")

    modes = [mode for mode in MODE_ORDER if mode in args.modes]
    needs_snapshots = any(mode in ("snapshots", "combined") for mode in modes)

    ensure_receivers_file()
    n_receivers = receiver_count(RCV_FILE)

    if n_receivers < 1:
        parser.error(f"Aucun recepteur valide dans {RCV_FILE}")

    if needs_snapshots and not SNAPSHOT_VISUALIZER.is_file():
        parser.error(
            f"Visualiseur snapshots introuvable : {SNAPSHOT_VISUALIZER}"
        )

    batch_id = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    total = len(args.threads) * args.runs * len(modes)

    print(
        f"\nBenchmark SEM : {args.threads} threads, "
        f"{args.runs} repetitions par configuration"
    )
    print(f"Maillage SEM : {EX} x {EY} x {EZ} elements, ordre {ORDER}")
    print(f"Noeuds : {NX} x {NY} x {NZ}")
    print(f"Domaine : {LX} x {LY} x {LZ}")
    print("Configurations : " + " / ".join(MODE_LABELS[mode] for mode in modes))
    print(f"Nombre total d'executions : {total}", flush=True)

    all_rows = {}

    for threads in args.threads:
        rows_by_mode = {mode: [] for mode in modes}

        for repetition in range(1, args.runs + 1):
            offset = (repetition - 1) % len(modes)
            ordered_modes = modes[offset:] + modes[:offset]

            for mode in ordered_modes:
                print(
                    f"\nThreads {threads} - {MODE_LABELS[mode]} - "
                    f"repetition {repetition}/{args.runs}",
                    flush=True,
                )

                row = run_once(
                    threads,
                    repetition,
                    batch_id,
                    mode,
                    n_receivers,
                )
                rows_by_mode[mode].append(row)

                print(
                    f"  Simulation : {row['simulation_wall_s']:.6f} s | "
                    f"Output : {float(row['output_s']):.6f} s | "
                    f"Sauvegarde : {float(row['save_s']):.6f} s | "
                    f"Visualisation : {row['visualization_total_s']:.6f} s | "
                    f"Workflow : {row['workflow_wall_s']:.6f} s"
                )

        for mode in modes:
            rows = rows_by_mode[mode]
            print_means(threads, rows, args.runs, mode)
            all_rows[(threads, mode)] = rows

    print_hpc_metrics(all_rows, args.threads, modes)
    print(f"\nResultats enregistres dans : {RESULTS_FILE}")
    print("Creation des graphiques...", flush=True)

    subprocess.run(
        [sys.executable, str(BASE_DIR / "plot_benchmarks.py")],
        check=True,
    )


if __name__ == "__main__":
    main()