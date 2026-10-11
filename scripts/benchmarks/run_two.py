import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build-openmp" / "bin" / "semproxy"
VIS_SCRIPT = ROOT / "scripts" / "visualisation" / "visu_snapshots.py"

DATA_DIR = ROOT / "scripts" / "visualisation" / "data"
IMAGES_DIR = ROOT / "scripts" / "visualisation" / "images" / "snapshots"
ARCHIVE_DIR = ROOT / "scripts" / "visualisation" / "images" / "mesh_benchmark"

RESULTS_DIR = ROOT / "scripts" / "benchmarks" / "results"
PLOTS_DIR = ROOT / "scripts" / "benchmarks" / "plots"
RESULTS_FILE = RESULTS_DIR / "mesh_workflow_benchmark.csv"
PLOT_FILE = PLOTS_DIR / "mesh_workflow_scaling.png"

THREADS = 4
ORDER = 2
ELEMENT_COUNTS = [10, 15, 20, 25]

LX = 2000
LY = 2000
LZ = 2000
DT = 0.001
TIMEMAX = 1.5
SNAP_INTERVAL = 50
SLICE_POSITION = 1000

PROF_ELEMENTS = 100
PROF_TIMEMAX = 0.661
PROF_SNAP_INTERVAL = 660
PROF_SLICE_POSITION = 1000


def clear_outputs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for path in DATA_DIR.glob("snapshot_it_*.csv"):
        path.unlink()

    for path in IMAGES_DIR.glob("*.png"):
        path.unlink()


def archive_images(nodes, run_number, label=""):
    suffix = f"_{label}" if label else ""
    target = ARCHIVE_DIR / f"nodes_{nodes}{suffix}" / f"run_{run_number}"

    if target.exists():
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)

    images = sorted(IMAGES_DIR.rglob("*.png"))

    if not images:
        raise RuntimeError(
            f"Aucune image PNG trouvee dans {IMAGES_DIR} apres la visualisation"
        )

    for image in images:
        relative = image.relative_to(IMAGES_DIR)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination)

    return target, len(images)


def run_command(command, env=None):
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(result.stdout + "\n" + result.stderr)

    return elapsed


def sem_command(elements, timemax, snapshots=False, snap_interval=SNAP_INTERVAL):
    command = [
        str(EXE),
        "--order", str(ORDER),
        "--ex", str(elements),
        "--ey", str(elements),
        "--ez", str(elements),
        "--lx", str(LX),
        "--ly", str(LY),
        "--lz", str(LZ),
        "--method", "sem",
        "--mesh", "cartesian",
        "--implem", "makutu",
        "--dt", str(DT),
        "--timemax", str(timemax),
    ]

    if snapshots:
        command.extend([
            "--is-snapshots=true",
            "--snap-time-interval", str(snap_interval),
            "--snap-folder", str(DATA_DIR),
        ])

    return command


def run_workflow(elements, snapshots, run_number):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS)
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "threads"

    if snapshots:
        clear_outputs()

    command = sem_command(elements, TIMEMAX, snapshots)

    workflow_start = time.perf_counter()
    simulation_s = run_command(command, env=env)
    visualization_s = 0.0
    archive_path = ""
    image_count = 0

    if snapshots:
        visualization_s = run_command([
            sys.executable,
            str(VIS_SCRIPT),
            "--position",
            str(SLICE_POSITION),
        ])

        nodes = elements * ORDER + 1
        target, image_count = archive_images(nodes, run_number)
        archive_path = str(target.relative_to(ROOT))

    workflow_s = time.perf_counter() - workflow_start

    return simulation_s, visualization_s, workflow_s, archive_path, image_count


def run_prof_reference():
    elements = PROF_ELEMENTS
    nodes = elements * ORDER + 1

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS)
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "threads"

    clear_outputs()

    command = sem_command(
        elements,
        PROF_TIMEMAX,
        snapshots=True,
        snap_interval=PROF_SNAP_INTERVAL,
    )

    print("\n====== REFERENCE PROF ======")
    print(f"  {nodes}^3 noeuds, {THREADS} threads")
    print(f"  timemax = {PROF_TIMEMAX} s")
    print(f"  snapshot interval = {PROF_SNAP_INTERVAL}")
    print("  snapshots attendus : timestep 0 et timestep 660")

    workflow_start = time.perf_counter()
    simulation_s = run_command(command, env=env)

    visualization_s = run_command([
        sys.executable,
        str(VIS_SCRIPT),
        "--position",
        str(PROF_SLICE_POSITION),
    ])

    workflow_s = time.perf_counter() - workflow_start

    target, image_count = archive_images(nodes, 1, "prof_reference")

    print(f"  SEM : {simulation_s:.6f} s")
    print(f"  Visualisation : {visualization_s:.6f} s")
    print(f"  Workflow : {workflow_s:.6f} s")
    print(f"  Images conservees : {image_count}")
    print(f"  Dossier : {target.relative_to(ROOT)}")


def save_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threads",
                "order",
                "elements_per_axis",
                "nodes_per_axis",
                "total_nodes",
                "snapshots_enabled",
                "run",
                "simulation_s",
                "visualization_s",
                "workflow_s",
                "image_count",
                "images_directory",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_results(averages):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if shutil.which("gnuplot") is None:
        raise RuntimeError("gnuplot est introuvable")

    data_file = RESULTS_DIR / "mesh_workflow_scaling.dat"

    with data_file.open("w", encoding="utf-8") as file:
        for (nodes_without, without_time), (nodes_with, with_time) in zip(
            averages[False], averages[True]
        ):
            if nodes_without != nodes_with:
                raise RuntimeError("Tailles de maillage incoherentes")

            file.write(f"{nodes_without} {without_time} {with_time}\n")

    gnuplot_script = f"""
set terminal pngcairo size 1500,850
set output "{PLOT_FILE}"
set title "Ad hoc workflow mesh scaling - SEM - 4 threads"
set xlabel "Taille du maillage (noeuds par axe)"
set ylabel "Temps total moyen du workflow (secondes)"
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.85 border -1
set boxwidth 0.85
set grid ytics
set key top left
plot "{data_file}" using 2:xtic(1) title "Sans snapshots", \
     "" using 3 title "Avec snapshots + visualisation"
"""

    subprocess.run(["gnuplot"], input=gnuplot_script, text=True, check=True)
    data_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark du workflow ad hoc complet selon la taille du maillage"
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Ne pas lancer le run final 201^3 de reference prof",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs doit etre >= 1")

    if not EXE.is_file():
        parser.error(f"Executable introuvable : {EXE}")

    if not VIS_SCRIPT.is_file():
        parser.error(f"Visualiseur introuvable : {VIS_SCRIPT}")

    rows = []
    averages = {False: [], True: []}

    for elements in ELEMENT_COUNTS:
        nodes = elements * ORDER + 1
        total_nodes = nodes ** 3

        print(f"\nMaillage {elements}^3 elements -> {nodes}^3 noeuds")

        for snapshots in [False, True]:
            label = "avec snapshots + visualisation" if snapshots else "sans snapshots"
            workflow_times = []

            for run in range(1, args.runs + 1):
                simulation_s, visualization_s, workflow_s, archive_path, image_count = (
                    run_workflow(elements, snapshots, run)
                )

                workflow_times.append(workflow_s)

                rows.append({
                    "threads": THREADS,
                    "order": ORDER,
                    "elements_per_axis": elements,
                    "nodes_per_axis": nodes,
                    "total_nodes": total_nodes,
                    "snapshots_enabled": int(snapshots),
                    "run": run,
                    "simulation_s": simulation_s,
                    "visualization_s": visualization_s,
                    "workflow_s": workflow_s,
                    "image_count": image_count,
                    "images_directory": archive_path,
                })

                print(
                    f"  {label} - run {run}/{args.runs} | "
                    f"SEM: {simulation_s:.6f} s | "
                    f"visu: {visualization_s:.6f} s | "
                    f"workflow: {workflow_s:.6f} s"
                )

                if snapshots:
                    print(f"    images : {archive_path}")

            average = statistics.mean(workflow_times)
            averages[snapshots].append((nodes, average))

            print(
                f"  TEMPS TOTAL MOYEN DU WORKFLOW {label}: "
                f"{average:.6f} s"
            )

    save_results(rows)
    plot_results(averages)

    print("\n====== MOYENNES DU WORKFLOW PAR MAILLAGE ======")

    for index, (nodes, without_time) in enumerate(averages[False]):
        with_time = averages[True][index][1]

        print(
            f"  {nodes}^3 noeuds | "
            f"sans snapshots: {without_time:.6f} s | "
            f"avec snapshots + visualisation: {with_time:.6f} s"
        )

    print(f"\nResultats : {RESULTS_FILE}")
    print(f"Graphe    : {PLOT_FILE}")
    print(f"Images    : {ARCHIVE_DIR.relative_to(ROOT)}")

    if not args.skip_reference:
        run_prof_reference()


if __name__ == "__main__":
    main()