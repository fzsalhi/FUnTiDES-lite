import argparse
import hashlib
import itertools
import json
import platform
import random
import shutil
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
VIS_DIR = ROOT_DIR / "tp1" / "visualisation"
DATA_DIR = VIS_DIR / "data" / "benchmark"
SNAPSHOT_IMAGES_DIR = VIS_DIR / "images" / "benchmark" / "snapshots"
SISMO_IMAGES_DIR = VIS_DIR / "images" / "benchmark" / "sismos"
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

COLUMNS = []

def parse_threads(value):
    try:
        numbers = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Exemple : --threads 1,2,4,8"
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

def clear_generated_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("snapshot_it_*", "coords*.bin", "sismos_benchmark.*", "*_metrics.json", "simulation.json"):
        for path in DATA_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def build_command(executable, elements, args, mode, fmt):
    command = [str(executable), "--order", str(args.order),
               "--ex", str(elements), "--ey", str(elements), "--ez", str(elements),
               "--lx", str(LX), "--ly", str(LY), "--lz", str(LZ),
               "--dt", str(args.dt), "--timemax", str(args.timemax),
               "--rcv-file", str(RCV_FILE), "--quiet=true",
               f"--record-receivers={str(mode in ('sismos', 'combined')).lower()}",
               f"--is-snapshots={str(mode in ('snapshots', 'combined')).lower()}",
               "--snap-time-interval", str(args.interval), "--snap-folder", str(DATA_DIR),
               "--snap-format", "bin" if fmt == "none" else fmt,
               "--sismo-format", "bin" if fmt == "none" else fmt,
               "--io-sync", args.io_sync, "--metrics-file", str(DATA_DIR / "simulation.json")]
    if mode in ("sismos", "combined"):
        command += ["--sismo-file", str(SISMO_FILE.with_suffix('.' + fmt))]
    return command


def run_once(backend, executable, threads, size, elements, mode, fmt, repetition, args, metadata):
    clear_generated_data()
    for directory in (SNAPSHOT_IMAGES_DIR, SISMO_IMAGES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.png"):
            path.unlink()
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="spread", OMP_PLACES="cores")
    workflow_start = time.perf_counter()
    simulation_wall, log = run_command(build_command(executable, elements, args, mode, fmt), "SEM", env)
    (RESULTS_FILE.parent / "last_simulation.log").write_text(log)
    metrics = json.loads((DATA_DIR / "simulation.json").read_text())
    if metrics["backend"] != backend:
        raise RuntimeError(f"{backend} demandé, mais {metrics['execution_space']} détecté : {executable}")
    post = {}
    for kind in ("snapshot", "sismo"):
        post.update({f"{kind}_{stage}_s": 0.0 for stage in ("read", "analysis", "render")})
    visualization_wall = 0.0
    for kind, enabled, script, extra in [
        ("snapshot", mode in ("snapshots", "combined"), "visu_snapshots.py",
         ["--data", str(DATA_DIR), "--format", fmt, "--position", str(args.position), "--output", str(SNAPSHOT_IMAGES_DIR)]),
        ("sismo", mode in ("sismos", "combined"), "visu_sismos.py",
         ["--input", str(SISMO_FILE.with_suffix('.' + fmt)), "--output", str(SISMO_IMAGES_DIR)])]:
        if enabled:
            path = DATA_DIR / f"{kind}_metrics.json"
            command = [sys.executable, str(VIS_DIR / script), *extra, "--metrics", str(path)]
            if args.no_render:
                command += ["--no-render"]
            elapsed, _ = run_command(command, kind + " visualisation", env)
            visualization_wall += elapsed
            timings = json.loads(path.read_text())
            post.update({f"{kind}_{stage}_s": timings[f"{stage}_s"] for stage in ("read", "analysis", "render")})
            if kind == "snapshot" and timings["snapshot_count"] != metrics["snapshot_count"]:
                raise RuntimeError("Le post-traitement n'a pas relu tous les snapshots")
    raw_files = [*DATA_DIR.glob("snapshot_it_*"), *DATA_DIR.glob("coords*.bin"), *DATA_DIR.glob("sismos_benchmark.*")]
    raw_bytes = sum(p.stat().st_size for p in raw_files)
    if raw_bytes != metrics["raw_output_bytes"]:
        raise RuntimeError("Le volume instrumenté diffère des fichiers effectivement écrits")
    row = dict(date=datetime.now(timezone.utc).isoformat(), batch_id=metadata["batch_id"], run=repetition,
               solver="semproxy", machine=platform.node(), config_hash=metadata["config_hash"],
               source_hash=metadata["source_hash"], profile="validation" if args.quick else "reference",
               backend=backend, execution_space=metrics["execution_space"], format=fmt, threads=threads,
               size=size, elements_per_axis=elements, nx=elements * args.order + 1,
               order=args.order, mode=mode, io_sync=args.io_sync, visualize=int(not args.no_render),
               simulation_wall_s=simulation_wall, workflow_wall_s=time.perf_counter() - workflow_start,
               visualization_total_s=visualization_wall, **post)
    for key in ("nodes", "steps", "dt_s", "receivers", "total_s", "initial_s", "compute_s", "receiver_sample_s",
                "snapshot_write_s", "sismo_write_s", "other_processing_s", "fsync_s", "snapshot_count",
                "snapshot_bytes", "sismo_bytes", "raw_output_bytes"):
        row[key] = metrics[key]
    row.update(simulation_internal_s=metrics["total_s"], compute_loop_s=metrics["run_s"], kernel_s=metrics["compute_s"],
               save_s=metrics["snapshot_write_s"] + metrics["sismo_write_s"],
               output_s=metrics["receiver_sample_s"] + metrics["snapshot_write_s"] + metrics["sismo_write_s"],
               total_data_mib=raw_bytes / 2**20,
               total_png_mib=file_info([*SNAPSHOT_IMAGES_DIR.glob("*.png"), *SISMO_IMAGES_DIR.glob("*.png")])[1])
    return row


def main():
    global RESULTS_FILE, COLUMNS
    parser = argparse.ArgumentParser(description="TP1 : benchmark SEM ad-hoc CPU/GPU et CSV/binaire")
    parser.add_argument("--cpu-exe", type=Path)
    parser.add_argument("--gpu-exe", type=Path)
    parser.add_argument("--threads", type=parse_threads, default=[4])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--modes", type=parse_modes, default=MODE_ORDER)
    parser.add_argument("--formats", default="csv,bin")
    parser.add_argument("--sizes", default="10,15,20", help="Nombre d'éléments par axe pour S,M,L")
    parser.add_argument("--order", type=int, default=ORDER)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--timemax", type=float, default=TIMEMAX)
    parser.add_argument("--interval", type=int, default=SNAP_INTERVAL)
    parser.add_argument("--position", type=float, default=SLICE_POSITION)
    parser.add_argument("--io-sync", choices=["buffered", "fsync"], default="buffered")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot-only", type=Path, help="Relire un CSV existant et tracer les figures, sans simulation")
    parser.add_argument("--columns", default="workflow_wall_s,compute_s,save_s,read_s,raw_output_bytes",
                        help="Colonnes à tracer ; read_s = snapshot_read_s + sismo_read_s")
    parser.add_argument("--plots-dir", type=Path, default=BASE_DIR / "plots")
    parser.add_argument("--clean-plots", action="store_true", help="Remplacer les anciens graphes générés pour cette campagne")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Validation réduite, pas les mesures de référence")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    def plot_results(csv_file):
        command = [sys.executable, str(BASE_DIR / "plot_benchmarks.py"), "--input", str(csv_file),
                   "--output", str(args.plots_dir), "--columns", args.columns,
                   "--threads", str(max(args.threads))]
        if args.clean_plots:
            command.append("--clean-plots")
        subprocess.run(command, check=True)

    if args.plot_only:
        plot_results(args.plot_only)
        return
    if args.runs < 1 or args.warmups < 0 or args.interval < 1 or args.order not in (1, 2, 3):
        parser.error("runs/interval >= 1, warmups >= 0, ordre 1/2/3 requis")
    if not math.isfinite(args.dt) or not math.isfinite(args.timemax) or args.dt <= 0 or args.timemax <= 0:
        parser.error("dt/timemax doivent être finis et positifs")
    try:
        elements = [int(item) for item in args.sizes.split(",")]
        if len(elements) != 3 or len(set(elements)) != 3 or any(n < 1 for n in elements):
            raise ValueError()
    except ValueError:
        parser.error("--sizes attend trois entiers positifs distincts, par exemple 10,15,20")
    labels = ["S", "M", "L"]
    if args.quick:
        elements, labels, args.timemax, args.interval = [4, 6, 8], ["small", "medium", "large"], 0.12, 20
    formats = args.formats.split(",")
    if not formats or len(set(formats)) != len(formats) or any(x not in ("csv", "bin") for x in formats):
        parser.error("formats possibles : csv,bin")
    executables = {k: v.resolve() for k, v in [("cpu", args.cpu_exe), ("gpu", args.gpu_exe)] if v}
    if not executables:
        executables = {"cpu": EXE}
    for executable in executables.values():
        if not executable.is_file() or not os.access(executable, os.X_OK):
            parser.error(f"Exécutable absent : {executable}")
    if not args.no_render and shutil.which("gnuplot") is None:
        parser.error("Gnuplot est requis pour générer les images")
    cases = [(backend, threads, size, n, mode, fmt)
             for backend, threads, (size, n), mode in itertools.product(executables, args.threads, zip(labels, elements), args.modes)
             for fmt in (["none"] if mode == "none" else formats)]
    print(f"{len(cases)} cas × ({args.warmups} chauffe + {args.runs} mesures)", flush=True)
    if args.dry_run:
        for backend, threads, size, n, mode, fmt in cases:
            print(backend, threads, size, mode, fmt, " ".join(build_command(executables[backend], n, args, mode, fmt)))
        return
    ensure_receivers_file()
    if receiver_count(RCV_FILE) < 1:
        parser.error("Aucun récepteur valide")
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    RESULTS_FILE = (args.output or BASE_DIR / "results" / f"benchmarks_{batch}.csv").resolve()
    if RESULTS_FILE.exists():
        parser.error("Le fichier de résultats existe déjà ; choisir un nouveau nom")
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    common = dict(elements=elements, order=args.order, dt=args.dt, timemax=args.timemax,
                  interval=args.interval, receivers=RCV_FILE.read_text(), position=args.position)
    files = [ROOT_DIR / "src/main/fe/src/sem_proxy.cc", ROOT_DIR / "src/main/fe/src/main.cc",
             ROOT_DIR / "src/main/fe/include/sem_proxy.h", ROOT_DIR / "src/main/fe/include/sem_proxy_options.h"]
    metadata = dict(batch_id=batch, config=common, config_hash=hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest(),
                    source_hash=hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest(),
                    machine=platform.node(), platform=platform.platform(), logical_cpus=os.cpu_count(),
                    command=sys.argv, io_sync=args.io_sync,
                    cache="Lectures après écriture, cache OS non contrôlé. buffered=cache OS; fsync=fsync par fichier.",
                    executables={k: dict(path=str(v), sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k, v in executables.items()})
    for name, cmd in [("cpu_info", ["lscpu"]), ("gpu_info", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])]:
        try:
            metadata[name] = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            metadata[name] = "indisponible"
    RESULTS_FILE.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    rng = random.Random(328)
    for repetition in range(-args.warmups + 1, args.runs + 1):
        order = list(cases)
        rng.shuffle(order)
        for backend, threads, size, n, mode, fmt in order:
            row = run_once(backend, executables[backend], threads, size, n, mode, fmt, repetition, args, metadata)
            if repetition > 0:
                COLUMNS = list(row)
                save_result(row)
            print(f"{'chauffe' if repetition <= 0 else f'mesure {repetition}/{args.runs}'} "
                  f"{backend} {threads}t {size} {mode} {fmt}: simulation={row['simulation_wall_s']:.3f}s "
                  f"workflow={row['workflow_wall_s']:.3f}s volume={row['total_data_mib']:.3f} MiB", flush=True)
    print(f"Résultats : {RESULTS_FILE}", flush=True)
    if not args.no_plots:
        plot_results(RESULTS_FILE)


if __name__ == "__main__":
    main()
