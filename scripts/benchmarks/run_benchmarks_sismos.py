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
SISMO_IMAGES_DIR = VIS_DIR / "images" / "sismos"
RESULTS_FILE = BASE_DIR / "results" / "sismos_benchmarks.csv"
EXE = ROOT_DIR / "build-openmp" / "bin" / "semproxy"
RCV_FILE = ROOT_DIR / "data" / "receivers.csv"
SISMO_FILE = DATA_DIR / "sismos_benchmark.csv"
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

DEFAULT_RECEIVERS = [
    (1200, 1000, 1000),
    (1400, 1000, 1000),
    (1600, 1000, 1000),
]

COLUMNS = ["date","solver","threads","nx","ny","nz","sismos_enabled","receiver_count","data_status","simulation_wall_s","simulation_internal_s","initial_s","compute_loop_s","compute_s","kernel_s","output_s","sismo_save_s","save_time_status","sismos_viz_s","workflow_wall_s","sismo_csv_count","sismo_csv_mib","sismo_png_count","sismo_png_mib","batch_id","run"]


def parse_threads(value):
    try:
        numbers = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Utilise par exemple --threads 1,2,4,8") from error
    if not numbers or any(number < 1 for number in numbers) or len(numbers) != len(set(numbers)):
        raise argparse.ArgumentTypeError("Les threads doivent etre des entiers positifs distincts")
    return numbers


def run_command(command, title, env=None):
    start = time.perf_counter()
    result = subprocess.run(command,cwd=ROOT_DIR,capture_output=True,text=True,env=env)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"Echec de {title} (code {result.returncode}) :\n{(result.stdout + result.stderr)[-5000:]}")
    return elapsed, result.stdout + "\n" + result.stderr


def run_gnuplot(script):
    start = time.perf_counter()
    result = subprocess.run(["gnuplot"],input=script,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=ROOT_DIR)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"Echec de la visualisation sismos :\n{result.stderr[-5000:]}")
    return elapsed


def extract_time(output, pattern):
    matches = re.findall(pattern, output, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else ""


def get_solver_times(output):
    number = r"([0-9.eE+-]+)"
    times = {
        "simulation_internal_s": extract_time(output, rf"Elapsed\s+TotalExe\s+Time\s*:\s*{number}"),
        "initial_s": extract_time(output, rf"Elapsed\s+Initial\s+Time\s*:\s*{number}"),
        "compute_loop_s": extract_time(output, rf"Elapsed\s+Compute\s+Time\s*:\s*{number}"),
        "kernel_s": extract_time(output, rf"Elapsed\s+Kernel\s+Time\s*:\s*{number}"),
        "output_s": extract_time(output, rf"Elapsed\s+Output\s+Time\s*:\s*{number}"),
        "sismo_save_s": extract_time(output, rf"Elapsed\s+Save\s+Time\s*:\s*{number}"),
    }
    times["compute_s"] = times["kernel_s"] if times["kernel_s"] != "" else ""
    return times


def file_info(paths):
    files = [path for path in paths if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files) / (1024 * 1024)


def ensure_receivers_file():
    RCV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not RCV_FILE.is_file():
        with RCV_FILE.open("w", encoding="utf-8") as file:
            for x, y, z in DEFAULT_RECEIVERS:
                file.write(f"{x},{y},{z}\n")


def receiver_count(path):
    count = 0
    with path.open("r",encoding="utf-8") as file:
        for line in file:
            line = line.split("#",1)[0].replace(","," ").replace(";"," ").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                float(parts[0]); float(parts[1]); float(parts[2])
            except ValueError:
                continue
            count += 1
    return count


def clear_generated_sismos():
    SISMO_FILE.unlink(missing_ok=True)
    SISMO_IMAGES_DIR.mkdir(parents=True,exist_ok=True)
    for path in SISMO_IMAGES_DIR.glob("sismos_benchmark.*"):
        path.unlink()


def gnuplot_string(value):
    return '"' + str(value).replace("\\","\\\\").replace('"','\\"') + '"'


def visualize_sismos(csv_path):
    with csv_path.open("r",encoding="utf-8",newline="") as file:
        reader = csv.reader(file)
        header = next(reader,None)
    if not header or len(header) < 2:
        raise ValueError(f"CSV sismos invalide : {csv_path}")
    SISMO_IMAGES_DIR.mkdir(parents=True,exist_ok=True)
    image = SISMO_IMAGES_DIR / "sismos_benchmark.png"
    plots = []
    for column in range(2,len(header)+1):
        label = header[column-1]
        plots.append(f"{gnuplot_string(str(csv_path))} using 1:{column} with lines lw 2 title {gnuplot_string(label)}")
    script = f'''set datafile separator ","
set terminal pngcairo size 1500,850 font "Sans,12"
set output {gnuplot_string(str(image))}
set title "Sismogrammes aux recepteurs"
set xlabel "Temps (s)"
set ylabel "Pression"
set grid
set key outside top horizontal
plot {", ".join(plots)}
unset output
'''
    elapsed = run_gnuplot(script)
    if not image.is_file() or image.stat().st_size == 0:
        raise RuntimeError("La visualisation sismos n'a pas genere de PNG")
    return elapsed, image


def save_result(row):
    RESULTS_FILE.parent.mkdir(parents=True,exist_ok=True)
    existing_rows = []
    existing_columns = []
    if RESULTS_FILE.is_file() and RESULTS_FILE.stat().st_size:
        with RESULTS_FILE.open("r",encoding="utf-8",newline="") as file:
            reader = csv.DictReader(file)
            existing_columns = reader.fieldnames or []
            existing_rows = list(reader)
    fieldnames = list(dict.fromkeys(existing_columns + COLUMNS))
    if existing_rows and existing_columns != fieldnames:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w",encoding="utf-8",newline="",dir=RESULTS_FILE.parent,prefix="sismos_benchmarks_",suffix=".csv",delete=False) as file:
                temporary = Path(file.name)
                writer = csv.DictWriter(file,fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
            os.replace(temporary,RESULTS_FILE)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    header = not RESULTS_FILE.is_file() or RESULTS_FILE.stat().st_size == 0
    with RESULTS_FILE.open("a",encoding="utf-8",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        if header:
            writer.writeheader()
        writer.writerow(row)


def mean_value(rows, field):
    values = []
    for row in rows:
        try:
            value = float(row.get(field,""))
            if math.isfinite(value):
                values.append(value)
        except (TypeError,ValueError):
            continue
    return statistics.mean(values) if values else None


def show_mean(label, rows, field):
    value = mean_value(rows,field)
    if value is None:
        print(f"  {label:<42} non disponible")
    else:
        print(f"  {label:<42} {value:.6f} s")


def run_once(threads,repetition,batch_id,sismos_enabled,n_receivers):
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    clear_generated_sismos()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "spread"
    env["OMP_PLACES"] = "threads"
    command = [str(EXE),"--order",str(ORDER),"--ex",str(EX),"--ey",str(EY),"--ez",str(EZ),"--lx",str(LX),"--ly",str(LY),"--lz",str(LZ),"--method","sem","--mesh","cartesian","--implem","makutu","--dt",str(DT),"--timemax",str(TIMEMAX),"--is-snapshots=false"]
    if sismos_enabled:
        command.extend(["--rcv-file",str(RCV_FILE),"--sismo-file",str(SISMO_FILE)])
    else:
        command.extend(["--sismo-file",""])
    start = time.perf_counter()
    simulation_wall, output = run_command(command,"SEM",env=env)
    solver_times = get_solver_times(output)
    sismos_viz = 0.0
    sismo_files = []
    sismo_images = []
    if sismos_enabled:
        if not SISMO_FILE.is_file():
            raise FileNotFoundError(f"Aucun fichier sismo genere : {SISMO_FILE}")
        sismo_files = [SISMO_FILE]
        sismos_viz, image = visualize_sismos(SISMO_FILE)
        sismo_images = [image]
    workflow_wall = time.perf_counter() - start
    sismo_count, sismo_mib = file_info(sismo_files)
    sismo_png_count, sismo_png_mib = file_info(sismo_images)
    row = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "solver": "semproxy",
        "threads": threads,
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "sismos_enabled": int(sismos_enabled),
        "receiver_count": n_receivers if sismos_enabled else 1,
        "data_status": "generated_this_run" if sismos_enabled else "sismos_disabled",
        "simulation_wall_s": simulation_wall,
        **solver_times,
        "save_time_status": "isolated" if sismos_enabled else "disabled",
        "sismos_viz_s": sismos_viz,
        "workflow_wall_s": workflow_wall,
        "sismo_csv_count": sismo_count,
        "sismo_csv_mib": sismo_mib,
        "sismo_png_count": sismo_png_count,
        "sismo_png_mib": sismo_png_mib,
        "batch_id": batch_id,
        "run": repetition,
    }
    save_result(row)
    return row


def print_means(threads,rows,runs,sismos_enabled):
    mode = "AVEC SISMOS" if sismos_enabled else "SANS SISMOS"
    print(f"\n====== MOYENNES {mode} - {threads} THREAD(S) ({runs} repetitions) ======")
    show_mean("Total SEM (horloge Python)",rows,"simulation_wall_s")
    show_mean("Total SEM (TotalExe)",rows,"simulation_internal_s")
    show_mean("Initialisation",rows,"initial_s")
    show_mean("Calcul SEM",rows,"compute_loop_s")
    show_mean("Kernel",rows,"kernel_s")
    show_mean("Output SEM",rows,"output_s")
    if sismos_enabled:
        show_mean("Sauvegarde sismos",rows,"sismo_save_s")
        show_mean("Visualisation sismos",rows,"sismos_viz_s")
    show_mean("TOTAL workflow execute",rows,"workflow_wall_s")
    if sismos_enabled:
        last = rows[-1]
        print("\n====== TAILLES DES FICHIERS ======")
        print(f"  Sismos CSV       : {last['sismo_csv_count']} fichier(s), {last['sismo_csv_mib']:.6f} MiB")
        print(f"  Image sismos     : {last['sismo_png_count']} PNG, {last['sismo_png_mib']:.6f} MiB")


def safe_ratio(numerator,denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def print_hpc_metrics(all_rows,thread_counts,compare_sismos):
    print("\n====== METRIQUES HPC ======")
    modes = [0,1] if compare_sismos else [1]
    for enabled in modes:
        label = "avec sismos" if enabled else "sans sismos"
        baseline = mean_value(all_rows.get((1,enabled),[]),"simulation_wall_s")
        if baseline is None:
            print(f"\nSpeedup / efficacite ({label}) : thread 1 absent, calcul impossible")
            continue
        print(f"\n{label.upper()}")
        print("  threads    speedup    efficacite")
        for threads in thread_counts:
            elapsed = mean_value(all_rows.get((threads,enabled),[]),"simulation_wall_s")
            speedup = safe_ratio(baseline,elapsed)
            if speedup is None:
                continue
            print(f"  {threads:<10}{speedup:<11.3f}{speedup / threads * 100.0:.2f} %")
    if compare_sismos:
        print("\nSURCOUT RELATIF DES SISMOS")
        print("  threads    overhead")
        for threads in thread_counts:
            without = mean_value(all_rows.get((threads,0),[]),"simulation_wall_s")
            with_sismos = mean_value(all_rows.get((threads,1),[]),"simulation_wall_s")
            if without is None or with_sismos is None or without <= 0:
                continue
            print(f"  {threads:<10}{(with_sismos - without) / without * 100.0:.2f} %")
        print("\nDEBIT I/O EFFECTIF DES SISMOS")
        print("  threads    MiB/s")
        for threads in thread_counts:
            rows = all_rows.get((threads,1),[])
            size_mib = mean_value(rows,"sismo_csv_mib")
            save_s = mean_value(rows,"sismo_save_s")
            bandwidth = safe_ratio(size_mib,save_s)
            if bandwidth is not None:
                print(f"  {threads:<10}{bandwidth:.6f}")
    print("\nSpeedup = T(1) / T(p)")
    print("Efficacite = Speedup / p * 100")
    print("Overhead sismos (%) = (Tavec - Tsans) / Tsans * 100")
    print("Debit I/O = taille CSV / temps de sauvegarde")


def main():
    parser = argparse.ArgumentParser(description="Benchmarks SEM avec comparaison sismos")
    parser.add_argument("--threads",type=parse_threads,default=[4])
    parser.add_argument("--runs",type=int,default=3)
    parser.add_argument("--compare-sismos",action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs doit etre superieur ou egal a 1")
    if not EXE.is_file():
        parser.error(f"Executable semproxy introuvable : {EXE}")
    ensure_receivers_file()
    n_receivers = receiver_count(RCV_FILE)
    if n_receivers < 1:
        parser.error(f"Aucun recepteur valide dans : {RCV_FILE}")
    batch_id = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    modes = [False,True] if args.compare_sismos else [True]
    total = len(args.threads) * args.runs * len(modes)
    print(f"\nBenchmark SEM sismos : {args.threads} threads, {args.runs} repetitions par configuration")
    print(f"Maillage SEM : {EX} x {EY} x {EZ} elements, ordre {ORDER}")
    print(f"Noeuds : {NX} x {NY} x {NZ}")
    print(f"Domaine : {LX} x {LY} x {LZ}")
    print(f"Recepteurs : {n_receivers}")
    if args.compare_sismos:
        print("Comparaison : sans sismos / avec sismos")
    else:
        print("Sismos : actives")
    print(f"Nombre total d'executions : {total}",flush=True)
    all_rows = {}
    for threads in args.threads:
        for sismos_enabled in modes:
            rows = []
            mode = "avec sismos" if sismos_enabled else "sans sismos"
            for repetition in range(1,args.runs+1):
                print(f"\nThreads {threads} - {mode} - repetition {repetition}/{args.runs}",flush=True)
                row = run_once(threads,repetition,batch_id,sismos_enabled,n_receivers)
                rows.append(row)
                line = f"  Simulation : {row['simulation_wall_s']:.6f} s | Output SEM : {float(row['output_s']):.6f} s"
                if sismos_enabled:
                    line += f" | Sauvegarde sismos : {float(row['sismo_save_s']):.6f} s"
                    line += f" | Visualisation sismos : {row['sismos_viz_s']:.6f} s"
                line += f" | Workflow : {row['workflow_wall_s']:.6f} s"
                print(line)
            print_means(threads,rows,args.runs,sismos_enabled)
            all_rows[(threads,int(sismos_enabled))] = rows
    print_hpc_metrics(all_rows,args.threads,args.compare_sismos)
    print(f"\nResultats individuels enregistres dans : {RESULTS_FILE}")
    if args.compare_sismos:
        print("La difference de temps SEM entre les deux modes mesure le surcout global de la fonctionnalite sismos.")
    print("Le temps de sauvegarde des sismos est mesure separement.")
    print("\nCreation des graphiques...",flush=True)
    subprocess.run([sys.executable,str(BASE_DIR / "plot_benchmarks_sismos.py")],check=True)
if __name__ == "__main__":
    main()