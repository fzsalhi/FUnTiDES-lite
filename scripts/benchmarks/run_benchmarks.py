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
IMAGES_DIR = VIS_DIR / "images"
RESULTS_FILE = BASE_DIR / "results" / "benchmarks.csv"
EXE = ROOT_DIR / "build-openmp" / "bin" / "semproxy"
MESH = 10
DT = 0.001
TIME_MAX = 0.05

COLUMNS = [
    "date", "mesh", "threads", "dt", "timemax", "data_status",
    "simulation_wall_s", "simulation_internal_s", "initial_s", "compute_s",
    "kernel_s", "output_s", "snapshot_save_s", "sismo_save_s", "save_time_status",
    "snapshots_viz_s", "sismos_viz_s", "workflow_wall_s",
    "snapshot_csv_count", "snapshot_csv_mib", "sismo_csv_count", "sismo_csv_mib",
    "snapshot_png_count", "snapshot_png_mib", "sismo_png_count", "sismo_png_mib",
    "batch_id", "run",
]


def parse_threads(value):
    try:
        numbers = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Utilise par exemple --threads 1,2,4,8") from error
    if not numbers or any(number < 1 for number in numbers) or len(numbers) != len(set(numbers)):
        raise argparse.ArgumentTypeError("Les threads doivent etre des entiers positifs distincts")
    return numbers


def get_visualizer(names):
    for name in names:
        path = VIS_DIR / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Script de visualisation introuvable : {', '.join(names)}")


def run_command(command, title):
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"Echec de {title} (code {result.returncode}) :\n"
            f"{(result.stdout + result.stderr)[-4000:]}"
        )
    return elapsed, result.stdout + "\n" + result.stderr


def get_solver_times(output):
    names = {
        "TotalExe": "simulation_internal_s",
        "Initial": "initial_s",
        "Compute": "compute_s",
        "Kernel": "kernel_s",
        "Output": "output_s",
    }
    times = {}
    for name, field in names.items():
        pattern = rf"Elapsed\s+{name}\s+Time\s*:\s*([0-9.eE+-]+)"
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        times[field] = float(matches[-1]) if matches else ""
    return times


def file_info(paths):
    files = [path for path in paths if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files) / (1024 * 1024)


def state(paths):
    return {
        str(path.resolve()): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths if path.is_file()
    }


def changed(paths, previous):
    return [
        path for path in paths if path.is_file()
        and previous.get(str(path.resolve())) != (path.stat().st_mtime_ns, path.stat().st_size)
    ]


def save_result(row):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_FILE.is_file() and RESULTS_FILE.stat().st_size:
        with RESULTS_FILE.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            old_columns = reader.fieldnames or []
            if old_columns != COLUMNS:
                old_rows = list(reader)
            else:
                old_rows = None
        if old_rows is not None:
            columns = list(dict.fromkeys(COLUMNS + old_columns))
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", newline="", dir=RESULTS_FILE.parent,
                    prefix="benchmarks_", suffix=".csv", delete=False
                ) as file:
                    temporary = Path(file.name)
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(old_rows)
                os.replace(temporary, RESULTS_FILE)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            fieldnames = columns
        else:
            fieldnames = COLUMNS
    else:
        fieldnames = COLUMNS
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


def run_once(threads, repetition, batch_id, snapshot_script, sismo_script):
    snapshot_files = sorted(DATA_DIR.glob("snapshot_*.csv"))
    sismo_files = sorted(DATA_DIR.glob("seismograms.csv"))
    before_snapshots = state(snapshot_files)
    before_sismos = state(sismo_files)
    start = time.perf_counter()

    command = [
        str(EXE), f"--kokkos-num-threads={threads}",
        "--ex", str(MESH), "--ey", str(MESH), "--ez", str(MESH),
        "--dt", str(DT), "--timemax", str(TIME_MAX),
        "--implem", "makutu", "--method", "sem", "--mesh", "cartesian",
    ]
    simulation_wall, output = run_command(command, "FUnTiDES")
    solver_times = get_solver_times(output)

    snapshot_files = sorted(DATA_DIR.glob("snapshot_*.csv"))
    sismo_files = sorted(DATA_DIR.glob("seismograms.csv"))
    if not snapshot_files or not sismo_files:
        raise FileNotFoundError(
            f"CSV manquants dans {DATA_DIR} : snapshot_*.csv et seismograms.csv requis"
        )
    fresh_snapshots = len(changed(snapshot_files, before_snapshots)) == len(snapshot_files)
    fresh_sismos = len(changed(sismo_files, before_sismos)) == len(sismo_files)
    data_status = "generated_this_run" if fresh_snapshots and fresh_sismos else "existing_or_mixed"

    snapshots_dir = IMAGES_DIR / "snapshots"
    sismos_dir = IMAGES_DIR / "seismograms"
    before_snapshot_images = state(snapshots_dir.glob("*.png"))
    before_sismo_images = state(sismos_dir.glob("*.png"))

    snapshots_viz, _ = run_command(
        [sys.executable, str(snapshot_script)], "visualisation snapshots"
    )
    snapshot_images = changed(snapshots_dir.glob("*.png"), before_snapshot_images)

    sismos_viz, _ = run_command(
        [sys.executable, str(sismo_script)], "visualisation sismogrammes"
    )
    sismo_images = changed(sismos_dir.glob("*.png"), before_sismo_images)
    workflow_wall = time.perf_counter() - start

    snapshot_count, snapshot_mib = file_info(snapshot_files)
    sismo_count, sismo_mib = file_info(sismo_files)
    snapshot_png_count, snapshot_png_mib = file_info(snapshot_images)
    sismo_png_count, sismo_png_mib = file_info(sismo_images)

    row = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mesh": MESH, "threads": threads, "dt": DT, "timemax": TIME_MAX,
        "data_status": data_status,
        "simulation_wall_s": simulation_wall,
        **solver_times,
        "snapshot_save_s": 0.0,
        "sismo_save_s": 0.0,
        "save_time_status": "not_instrumented",
        "snapshots_viz_s": snapshots_viz,
        "sismos_viz_s": sismos_viz,
        "workflow_wall_s": workflow_wall,
        "snapshot_csv_count": snapshot_count,
        "snapshot_csv_mib": snapshot_mib,
        "sismo_csv_count": sismo_count,
        "sismo_csv_mib": sismo_mib,
        "snapshot_png_count": snapshot_png_count,
        "snapshot_png_mib": snapshot_png_mib,
        "sismo_png_count": sismo_png_count,
        "sismo_png_mib": sismo_png_mib,
        "batch_id": batch_id,
        "run": repetition,
    }
    save_result(row)
    return row


def main():
    parser = argparse.ArgumentParser(description="Benchmarks du workflow FUnTiDES")
    parser.add_argument("--threads", type=parse_threads, default=[4])
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs doit etre superieur ou egal a 1")
    if not EXE.is_file():
        parser.error(f"Executable FUnTiDES introuvable : {EXE}")

    snapshot_script = get_visualizer((
        "visu_snapshots.py", "visu_snapshot.py", "visualiser_snapshot.py"
    ))
    sismo_script = get_visualizer(("visu_sismos.py",))
    batch_id = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    total = len(args.threads) * args.runs
    all_rows = []

    print(f"\nBenchmark : {args.threads} threads, {args.runs} repetitions par configuration")
    print(f"Nombre total d'executions : {total}", flush=True)

    for threads in args.threads:
        rows = []
        for repetition in range(1, args.runs + 1):
            print(f"\nThreads {threads} - repetition {repetition}/{args.runs}", flush=True)
            row = run_once(threads, repetition, batch_id, snapshot_script, sismo_script)
            rows.append(row)
            all_rows.append(row)
            print(
                f"  Simulation : {row['simulation_wall_s']:.6f} s | "
                f"Snapshots : {row['snapshots_viz_s']:.6f} s | "
                f"Sismos : {row['sismos_viz_s']:.6f} s | "
                f"Workflow : {row['workflow_wall_s']:.6f} s"
            )

        print(f"\n====== MOYENNES POUR {threads} THREAD(S) ({args.runs} repetitions) ======")
        show_mean("Total FUnTiDES (horloge Python)", rows, "simulation_wall_s")
        show_mean("Total FUnTiDES (TotalExe)", rows, "simulation_internal_s")
        show_mean("Initialisation", rows, "initial_s")
        show_mean("Propagation / calcul", rows, "compute_s")
        show_mean("Noyaux (inclus dans calcul)", rows, "kernel_s")
        show_mean("Sorties FUnTiDES", rows, "output_s")
        print(f"  {'Sauvegarde snapshots':<38} non mesuree (0 provisoire)")
        print(f"  {'Sauvegarde sismogrammes':<38} non mesuree (0 provisoire)")
        show_mean("Visualisation snapshots", rows, "snapshots_viz_s")
        show_mean("Visualisation sismogrammes", rows, "sismos_viz_s")
        show_mean("TOTAL workflow execute", rows, "workflow_wall_s")
        last = rows[-1]
        print("\n====== TAILLES DES FICHIERS ======")
        print(f"  Snapshots CSV       : {last['snapshot_csv_count']} fichier(s), {last['snapshot_csv_mib']:.3f} MiB")
        print(f"  Sismogrammes CSV    : {last['sismo_csv_count']} fichier(s), {last['sismo_csv_mib']:.3f} MiB")
        print(f"  Images snapshots    : {last['snapshot_png_count']} PNG, {last['snapshot_png_mib']:.3f} MiB")
        print(f"  Images sismogrammes : {last['sismo_png_count']} PNG, {last['sismo_png_mib']:.3f} MiB")
        print(f"  Total CSV           : {last['snapshot_csv_mib'] + last['sismo_csv_mib']:.3f} MiB")
        print(f"  Total images        : {last['snapshot_png_mib'] + last['sismo_png_mib']:.3f} MiB")

    print(f"\nResultats individuels enregistres dans : {RESULTS_FILE}")
    if any(row["data_status"] != "generated_this_run" for row in all_rows):
        print("ATTENTION : CSV preexistants ou partiellement produits : le workflow n'est pas encore couple a la simulation.")
    print("Les temps de sauvegarde a 0 ne sont pas des mesures reelles.")
    print("\nCreation des graphiques...", flush=True)
    subprocess.run([sys.executable, str(BASE_DIR / "plot_benchmarks.py")], check=True)


if __name__ == "__main__":
    main()