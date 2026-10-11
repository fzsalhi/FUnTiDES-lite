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
EXE = ROOT_DIR / "build-openmp" / "src" / "main" / "fd" / "fdproxy"

NX = 30
NY = 30
NZ = 30

DX = 10
DY = 10
DZ = 10

XS = NX // 2
YS = NY // 2
ZS = NZ // 2

SLICE_POSITION = XS * DX

COLUMNS = [
    "date", "solver", "threads", "nx", "ny", "nz", "snapshots_enabled", "data_status",
    "simulation_wall_s", "simulation_internal_s", "initial_s", "compute_loop_s",
    "compute_s", "kernel_s", "add_rhs_s", "output_s", "snapshot_save_s",
    "save_time_status", "snapshots_viz_s", "workflow_wall_s", "snapshot_csv_count",
    "snapshot_csv_mib", "snapshot_png_count", "snapshot_png_mib", "batch_id", "run",
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
        "simulation_internal_s": extract_time(output, rf"Elapsed\s+TotalExecution\s+Time\s*:\s*{number}"),
        "initial_s": extract_time(output, rf"Elapsed\s+Initialization\s+Time\s*:\s*{number}"),
        "compute_loop_s": extract_time(output, rf"Elapsed\s+ComputeLoop\s+RunTime\s*:\s*{number}"),
        "kernel_s": extract_time(output, rf"Elapsed\s+Kernel\s+Time\s*:\s*{number}"),
        "add_rhs_s": extract_time(output, rf"Elapsed\s+AddRHS\s+Time\s*:\s*{number}"),
        "output_s": extract_time(output, rf"Elapsed\s+Output\s+Time\s*:\s*{number}"),
    }

    if times["kernel_s"] != "" and times["add_rhs_s"] != "":
        times["compute_s"] = times["kernel_s"] + times["add_rhs_s"]
    else:
        times["compute_s"] = ""

    return times


def file_info(paths):
    files = [path for path in paths if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files) / (1024 * 1024)


def state(paths):
    return {
        str(path.resolve()): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
        if path.is_file()
    }


def changed(paths, previous):
    return [
        path
        for path in paths
        if path.is_file()
        and previous.get(str(path.resolve())) != (path.stat().st_mtime_ns, path.stat().st_size)
    ]


def clear_generated_snapshots():
    for path in DATA_DIR.glob("snapshot_it_*.csv"):
        path.unlink()


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

    if not fieldnames:
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
        print(f"  {label:<42} non disponible")
    else:
        print(f"  {label:<42} {value:.6f} s")


def run_once(threads, repetition, batch_id, snapshot_script, snapshots_enabled):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clear_generated_snapshots()

    snapshots_dir = IMAGES_DIR / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    before_snapshot_images = state(snapshots_dir.glob("*.png"))

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "threads"

    command = [
        str(EXE),
        "-nx", str(NX),
        "-ny", str(NY),
        "-nz", str(NZ),
        "-dx", str(DX),
        "-dy", str(DY),
        "-dz", str(DZ),
        "-xs", str(XS),
        "-ys", str(YS),
        "-zs", str(ZS),
    ]

    if snapshots_enabled:
        command.extend(["-savesnapshots", "-repo", str(DATA_DIR)])

    start = time.perf_counter()

    simulation_wall, output = run_command(command, "FDTD", env=env)
    solver_times = get_solver_times(output)

    snapshots_viz = 0.0
    snapshot_files = []
    snapshot_images = []

    if snapshots_enabled:
        snapshot_files = sorted(DATA_DIR.glob("snapshot_it_*.csv"))

        if not snapshot_files:
            raise FileNotFoundError(
                f"Aucun snapshot genere dans {DATA_DIR}. Attendu : snapshot_it_*.csv"
            )

        snapshots_viz, _ = run_command(
            [sys.executable, str(snapshot_script), "--position", str(SLICE_POSITION)],
            "visualisation snapshots",
        )

        snapshot_images = changed(snapshots_dir.glob("*.png"), before_snapshot_images)

    workflow_wall = time.perf_counter() - start

    snapshot_count, snapshot_mib = file_info(snapshot_files)
    snapshot_png_count, snapshot_png_mib = file_info(snapshot_images)

    row = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "solver": "fdproxy",
        "threads": threads,
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "snapshots_enabled": int(snapshots_enabled),
        "data_status": "generated_this_run" if snapshots_enabled else "snapshots_disabled",
        "simulation_wall_s": simulation_wall,
        **solver_times,
        "snapshot_save_s": "",
        "save_time_status": "included_in_output_not_isolated" if snapshots_enabled else "disabled",
        "snapshots_viz_s": snapshots_viz,
        "workflow_wall_s": workflow_wall,
        "snapshot_csv_count": snapshot_count,
        "snapshot_csv_mib": snapshot_mib,
        "snapshot_png_count": snapshot_png_count,
        "snapshot_png_mib": snapshot_png_mib,
        "batch_id": batch_id,
        "run": repetition,
    }

    save_result(row)
    return row


def print_means(threads, rows, runs, snapshots_enabled):
    mode = "AVEC SNAPSHOTS" if snapshots_enabled else "SANS SNAPSHOTS"

    print(f"\n====== MOYENNES {mode} - {threads} THREAD(S) ({runs} repetitions) ======")

    show_mean("Total FDTD (horloge Python)", rows, "simulation_wall_s")
    show_mean("Total FDTD (TotalExecution)", rows, "simulation_internal_s")
    show_mean("Initialisation", rows, "initial_s")
    show_mean("Boucle de calcul (ComputeLoop)", rows, "compute_loop_s")
    show_mean("Kernel", rows, "kernel_s")
    show_mean("AddRHS", rows, "add_rhs_s")
    show_mean("Output FDTD", rows, "output_s")

    if snapshots_enabled:
        print(f"  {'Sauvegarde snapshots seule':<42} non mesuree separement")
        show_mean("Visualisation snapshots", rows, "snapshots_viz_s")

    show_mean("TOTAL workflow execute", rows, "workflow_wall_s")

    last = rows[-1]

    if snapshots_enabled:
        print("\n====== TAILLES DES FICHIERS ======")
        print(
            f"  Snapshots CSV    : {last['snapshot_csv_count']} fichier(s), "
            f"{last['snapshot_csv_mib']:.3f} MiB"
        )
        print(
            f"  Images snapshots : {last['snapshot_png_count']} PNG, "
            f"{last['snapshot_png_mib']:.3f} MiB"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmarks FDTD avec comparaison snapshots")
    parser.add_argument("--threads", type=parse_threads, default=[4])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--compare-snapshots", action="store_true")

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs doit etre superieur ou egal a 1")

    if not EXE.is_file():
        parser.error(f"Executable fdproxy introuvable : {EXE}")

    snapshot_script = get_visualizer(
        ("visu_snapshots.py", "visu_snapshot.py", "visualiser_snapshot.py")
    )

    batch_id = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    modes = [False, True] if args.compare_snapshots else [True]
    total = len(args.threads) * args.runs * len(modes)

    print(
        f"\nBenchmark FDTD : {args.threads} threads, {args.runs} repetitions par configuration"
    )
    print(f"Grille fixe : {NX} x {NY} x {NZ}")
    print(f"Position de coupe : {SLICE_POSITION}")

    if args.compare_snapshots:
        print("Comparaison : sans snapshots / avec snapshots")
    else:
        print("Snapshots : actives, sauvegarde tous les 50 pas")

    print(f"Nombre total d'executions : {total}", flush=True)

    for threads in args.threads:
        for snapshots_enabled in modes:
            rows = []
            mode = "avec snapshots" if snapshots_enabled else "sans snapshots"

            for repetition in range(1, args.runs + 1):
                print(
                    f"\nThreads {threads} - {mode} - repetition {repetition}/{args.runs}",
                    flush=True,
                )

                row = run_once(
                    threads,
                    repetition,
                    batch_id,
                    snapshot_script,
                    snapshots_enabled,
                )

                rows.append(row)

                line = (
                    f"  Simulation : {row['simulation_wall_s']:.6f} s | "
                    f"Output FDTD : {float(row['output_s']):.6f} s"
                )

                if snapshots_enabled:
                    line += f" | Visualisation snapshots : {row['snapshots_viz_s']:.6f} s"

                line += f" | Workflow : {row['workflow_wall_s']:.6f} s"
                print(line)

            print_means(threads, rows, args.runs, snapshots_enabled)

    print(f"\nResultats individuels enregistres dans : {RESULTS_FILE}")

    if args.compare_snapshots:
        print(
            "La difference de temps FDTD entre les deux courbes mesure le surcout global des snapshots."
        )
        print("Le temps exact de writeSnapshotCSV n'est pas encore isole separement.")
    else:
        print(
            "Output FDTD contient le cout des sorties, y compris les snapshots, "
            "mais le temps de sauvegarde snapshot n'est pas encore isole."
        )

    print("\nCreation des graphiques...", flush=True)

    subprocess.run(
        [sys.executable, str(BASE_DIR / "plot_benchmarks.py")],
        check=True,
    )


if __name__ == "__main__":
    main()