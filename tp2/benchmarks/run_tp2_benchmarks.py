import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from analyze_snapshots import analyze as analyze_snapshots
from analyze_sismos import analyze as analyze_sismos
from bench_precision import precision_rows
from io_utils import list_snapshots, load_snapshot, load_coords

MODES = ["baseline", "snapshot_adhoc", "snapshot_insitu_gpu", "snapshot_insitu_cpu",
         "snapshot_insitu_hostcopy", "snapshot_insitu_all", "sismo_adhoc", "sismo_insitu",
         "visual_full", "visual_slice", "scaling_cfl", "sismo_fft_adhoc"]
FFT_METRICS = ["fft_prepare_s", "fft_plan_s", "fft_h2d_s", "fft_compute_s", "fft_d2h_s",
               "fft_write_s", "fft_total_s", "fft_h2d_bytes", "fft_d2h_bytes", "fft_output_bytes"]


def write_rows(path, rows):
    if rows:
        with Path(path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)


def execute(command, environment, log):
    start = perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Simulation en echec : {log}\n{result.stderr[-2000:]}")
    return perf_counter()-start


def compare_stats(reference, candidate, size, kind, tolerance=5e-5):
    if len(reference) != len(candidate):
        raise ValueError(f"Nombre de lignes different pour {kind}")
    rows = []
    scale = max(max(abs(float(r["min"])), abs(float(r["max"]))) for r in reference) or 1
    fields = ["min", "max", "mean", "variance"]
    for column in fields:
        errors = []
        for a, b in zip(reference, candidate):
            if int(a["count"]) != int(b["count"]): raise ValueError("Nombre d'echantillons different")
            if kind == "sismos" and int(a["receiver"]) != int(b["receiver"]):
                raise ValueError("Recepteurs differents dans les resultats")
            if kind == "snapshots" and (int(a["step"]) != int(b["step"]) or abs(float(a["time_s"])-float(b["time_s"])) > 1e-7):
                raise ValueError("Les instants analyses ne correspondent pas")
            denom = scale**2 if column == "variance" else scale
            error = abs(float(a[column])-float(b[column])) / denom
            errors.append(error if np.isfinite(error) else float('inf'))
        rows.append(dict(size=size, data=kind, statistic=column, max_error_norm=max(errors), tolerance=tolerance,
                         passed=int(max(errors) <= tolerance)))
    return rows


def main():
    parser = argparse.ArgumentParser(description="TP2 : analyses binaires ad-hoc, in-situ et slices")
    parser.add_argument("--config", type=Path, default=HERE/"tp2_benchmark_config.json")
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--cpu-exe", type=Path)
    parser.add_argument("--gpu-exe", type=Path)
    parser.add_argument("--fft", action="store_true", help="Ajouter la FFT in-situ cuFFT")
    parser.add_argument("--figures", help="Figures a regenerer, separees par des virgules")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--expect-backend", choices=["cpu", "gpu"])
    parser.add_argument("--plot-only", type=Path)
    parser.add_argument("--no-plots", action="store_true", help="Conserver uniquement les mesures")
    args = parser.parse_args()
    if args.cpu_exe or args.gpu_exe:
        if not (args.cpu_exe and args.gpu_exe) or args.exe:
            parser.error("Fournir --cpu-exe et --gpu-exe ensemble")
        batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        prefix = args.output or HERE / "results" / f"tp2_{batch}.csv"
        outputs = []
        for backend, executable in [("cpu", args.cpu_exe), ("gpu", args.gpu_exe)]:
            output = prefix.with_name(prefix.stem + "_" + backend + ".csv")
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
                       "--exe", str(executable), "--expect-backend", backend, "--output", str(output),
                       "--modes", args.modes, "--no-plots"]
            for key in ("runs", "warmups", "threads"):
                if getattr(args, key) is not None: command += ["--" + key, str(getattr(args, key))]
            if args.quick: command.append("--quick")
            if args.keep_data: command.append("--keep-data")
            if args.fft and backend == "gpu": command.append("--fft")
            if args.figures: command += ["--figures", args.figures]
            subprocess.run(command, cwd=ROOT, check=True)
            outputs.append(output)
        if not args.no_plots:
            command = [sys.executable, str(HERE / "plot_tp2_benchmarks.py"), "--compare", *map(str, outputs)]
            if args.figures: command += ['--figures', args.figures]
            subprocess.run(command, check=True)
        return
    if args.plot_only:
        command = [sys.executable, str(HERE/"plot_tp2_benchmarks.py"), "--input", str(args.plot_only)]
        if args.figures: command += ["--figures", args.figures]
        subprocess.run(command, check=True)
        return
    cfg = json.loads(args.config.read_text())
    exe = (args.exe or Path(cfg["semproxy"])).resolve()
    if not exe.is_file(): parser.error(f"Executable absent : {exe}")
    runs = args.runs if args.runs is not None else cfg["repetitions"]
    warmups = args.warmups if args.warmups is not None else cfg["warmups"]
    threads = args.threads if args.threads is not None else cfg["omp_threads"]
    if runs < 1 or warmups < 0 or threads < 1: parser.error("runs/threads >= 1 et warmups >= 0 requis")
    modes = args.modes.split(",")
    if args.fft: modes.append("sismo_fft_insitu")
    if any(m not in MODES + ["sismo_fft_insitu"] for m in modes): parser.error("Modes : " + ",".join(MODES))
    modes = list(dict.fromkeys(modes))
    if args.quick:
        cfg["sizes"] = [dict(name=name, ex=n, ey=n, ez=n) for name,n in zip(["small","medium","large"],[4,6,8])]
        cfg["common"]["timemax"] = .12
        cfg["snapshot_interval"] = 20
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or Path(cfg["output_dir"])/f"tp2_{batch}.csv").resolve()
    if output.exists(): parser.error("Le CSV existe deja : choisir un nouveau nom")
    output.parent.mkdir(parents=True, exist_ok=True)
    details = output.parent/"details"/output.stem
    details.mkdir(parents=True, exist_ok=False)
    data_root = ROOT / "tp2" / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="spread", OMP_PLACES="cores")
    receivers = Path(cfg["receivers_file"]).resolve()
    source_files = [ROOT/"src/main/fe/src/sem_proxy.cc", ROOT/"src/main/fe/src/sismo_fft_cuda.cc", ROOT/"src/main/fe/include/sem_proxy.h", ROOT/"src/main/fe/include/sem_proxy_options.h", ROOT/"tp2/io_utils.py", ROOT/"tp2/analyze_snapshots.py", ROOT/"tp2/analyze_sismos.py"]
    meta = dict(schema="funtides.tp2.v1", batch=batch, machine=platform.node(), platform=platform.platform(),
                executable=str(exe), executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(b"".join(p.read_bytes() for p in source_files)).hexdigest(),
                config=cfg, threads=threads, repetitions=runs, warmups=warmups, profile="validation" if args.quick else "reference",
                cache="Ecritures buffered ou fsync selon configuration ; lectures apres ecriture, cache OS non controle",
                data_directory=str(data_root), modes=modes, command=sys.argv)
    cpu_info = Path('/proc/cpuinfo')
    if cpu_info.exists():
        meta['cpu'] = next((line.split(':',1)[1].strip() for line in cpu_info.read_text().splitlines() if line.startswith('model name')), platform.processor())
    cache = exe.parent.parent/'CMakeCache.txt'
    if cache.exists():
        meta['build_options'] = [line for line in cache.read_text().splitlines() if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'ENABLE_CUDA:', 'SEMPROXY_ENABLE_CUFFT:', 'Kokkos_ARCH_', 'Kokkos_ENABLE_')) and not line.endswith('=OFF')]
    meta['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', 'non defini')
    try: meta["gpu"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): meta["gpu"]="indisponible"
    output.with_suffix(".json").write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n")
    rows, precisions, checks = [], [], []
    rng = random.Random(328)
    cases = [(size,mode) for size in cfg["sizes"] for mode in modes]
    print(f"{len(cases)} cas x ({warmups} chauffe + {runs} mesures), puis verification des resultats", flush=True)
    schedule = [(warmup, rep, case) for warmup, count in [(True,warmups),(False,runs)] for rep in range(1,count+1) for case in rng.sample(cases,len(cases))]
    
    for warmup, rep, (size,mode) in schedule:
        name = size["name"]
        report = details/name/mode/(f"warmup_{rep}" if warmup else f"run_{rep}")
        report.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix="run-", dir=data_root) as temporary:
            data = Path(temporary)
            command = [str(exe)]
            parameters = {**cfg["common"], **{k:size[k] for k in ("ex","ey","ez")}}
            for k,v in parameters.items(): command += [f"--{k}={str(v).lower() if isinstance(v,bool) else v}"]
            command += ["--quiet=true", "--snap-format=bin", "--sismo-format=bin", f"--rcv-file={receivers}",
                        f"--record-receivers={str(mode.startswith('sismo')).lower()}", f"--io-sync={cfg['io_sync']}",
                        f"--metrics-file={report/'simulation.json'}"]
            interval = cfg["snapshot_interval"]
            if mode in ("snapshot_adhoc","visual_full"):
                command += ["--is-snapshots=true",f"--snap-time-interval={interval}",f"--snap-folder={data/'snapshots'}"]
            if mode.startswith("snapshot_insitu"):
                method = "gpu" if mode == "snapshot_insitu_all" else mode.rsplit("_",1)[1]
                command += [f"--stats-interval={1 if mode=='snapshot_insitu_all' else interval}",f"--stats-mode={method}",f"--stats-file={report/'stats.csv'}"]
            if mode in ("sismo_adhoc", "sismo_fft_adhoc"): command += [f"--sismo-file={data/'sismos.bin'}"]
            if mode in ("sismo_insitu", "sismo_fft_insitu"): command += [f"--sismo-stats-file={report/'sismos_stats.csv'}"]
            if mode == "sismo_fft_insitu": command += [f"--fft-file={report/'fft.csv'}"]
            if mode == "visual_slice":
                command += [f"--slice-interval={interval}",f"--slice-plane={cfg['slice_plane']}",f"--slice-position={cfg['slice_position']}",f"--slice-folder={data/'slices'}"]
            if mode == "scaling_cfl": command += ["--auto-dt=true"]
            (report/"command.json").write_text(json.dumps(command,indent=2)+"\n")
            sim_wall = execute(command,env,report/"simulation.log")
            sim = json.loads((report/"simulation.json").read_text())
            if args.expect_backend and sim["backend"] != args.expect_backend: raise ValueError(f"Backend detecte : {sim['backend']}")
            if "backend" in meta and meta["backend"] != sim["backend"]: raise ValueError("Le backend a change")
            meta["backend"], meta["execution_space"] = sim["backend"], sim["execution_space"]
            post = dict(read_s=0.,compute_s=0.,render_s=0.,write_s=0.)
            start = perf_counter()
            if mode in ("snapshot_adhoc","visual_full","visual_slice"):
                source = data/("slices" if mode=="visual_slice" else "snapshots")
                _, post = analyze_snapshots(source,report/"analysis",pattern="slice_it_*.bin" if mode=="visual_slice" else "snapshot_it_*.bin",
                    render=mode.startswith("visual"),visual_only=mode.startswith("visual"),plane=cfg["slice_plane"],position=cfg["slice_position"])
            if mode in ("sismo_adhoc", "sismo_fft_adhoc"):
                _, post = analyze_sismos(data/"sismos.bin",report/"analysis", with_fft=mode=="sismo_fft_adhoc")
            post_wall = perf_counter()-start
            actual_bytes = sum(p.stat().st_size for p in data.rglob('*') if p.is_file())
            actual_bytes += sum((report/p).stat().st_size for p in ["stats.csv","sismos_stats.csv", "fft.csv"] if (report/p).exists())
            if actual_bytes != sim["total_output_bytes"]: raise ValueError(f"Volume mesure incoherent pour {name} {mode}: {actual_bytes} != {sim['total_output_bytes']}")
            row = dict(batch=batch, profile=meta["profile"], machine=platform.node(), source_hash=meta["source_sha256"], config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                backend=sim["backend"], threads=threads, size=name, elements_per_axis=size["ex"], mode=mode, run=rep,
                io_sync=cfg["io_sync"], order=cfg["common"]["order"], simulation_wall_s=sim_wall, postprocess_wall_s=post_wall, workflow_s=sim_wall+post_wall,
                adhoc_read_s=post["read_s"], adhoc_analysis_s=post["compute_s"], render_s=post["render_s"], post_write_s=post["write_s"],
                adhoc_fft_prepare_s=post.get("fft_prepare_s", 0), adhoc_fft_compute_s=post.get("fft_compute_s", 0))
            for key in ["nodes","steps","dt_s","compute_s","initial_s","snapshot_write_s","sismo_write_s","stats_compute_s","stats_copy_s","stats_write_s","stats_bytes","stats_calls","stats_transfer_bytes",
                        "sismo_stats_s","sismo_stats_copy_s","sismo_stats_write_s","sismo_stats_bytes","receiver_trace_bytes","receiver_stats_memory_bytes",
                        "slice_setup_s","slice_extract_s","slice_copy_s","slice_write_s","slice_count","slice_nodes","slice_bytes","total_output_bytes","snapshot_count","snapshot_bytes","sismo_bytes"]:
                row[key]=sim[key]
            for key in FFT_METRICS: row[key] = sim[key]
            row["post_output_bytes"] = sum(p.stat().st_size for p in (report/"analysis").rglob("*") if p.is_file())
            row["workflow_output_bytes"] = sim["total_output_bytes"] + row["post_output_bytes"]
            if not warmup:
                rows.append(row); write_rows(output,rows)
                if mode == "snapshot_insitu_all" and name not in {p['size'] for p in precisions}:
                    precisions += precision_rows(report/"stats.csv",name,cfg["precision_intervals"])
                    write_rows(output.with_name(output.stem+"_precision.csv"),precisions)
            if args.keep_data: shutil.copytree(data,report/"data")
            print(f"{'chauffe' if warmup else f'mesure {rep}/{runs}'} {name} {mode}: simulation={sim_wall:.3f}s workflow={row['workflow_s']:.3f}s",flush=True)
    # Both analyses see the same field in this independent validation run.
    for size in cfg["sizes"]:
        name=size["name"]; report=details/name/"validation"; report.mkdir()
        with tempfile.TemporaryDirectory(prefix="check-", dir=data_root) as temporary:
            data=Path(temporary)
            command=[str(exe)]
            for k,v in {**cfg["common"], **{k:size[k] for k in ['ex','ey','ez']}}.items(): command += [f"--{k}={v}"]
            command += ["--quiet=true",f"--rcv-file={receivers}","--snap-format=bin","--sismo-format=bin","--is-snapshots=true",f"--snap-folder={data/'snapshots'}",
                        f"--snap-time-interval={cfg['snapshot_interval']}",f"--stats-interval={cfg['snapshot_interval']}","--stats-mode=gpu",f"--stats-file={report/'stats.csv'}",
                        f"--sismo-file={data/'sismos.bin'}",f"--sismo-stats-file={report/'sismos_stats.csv'}",
                        f"--slice-interval={cfg['snapshot_interval']}",f"--slice-plane={cfg['slice_plane']}",
                        f"--slice-position={cfg['slice_position']}",f"--slice-folder={data/'slices'}"]
            if args.fft: command += [f"--fft-file={report/'fft.csv'}"]
            execute(command,env,report/"simulation.log")
            ref,_=analyze_snapshots(data/'snapshots',report/'analysis')
            with (report/'stats.csv').open() as stream: candidate=list(csv.DictReader(stream))
            checks += compare_stats(ref,candidate,name,"snapshots")
            ref,_=analyze_sismos(data/'sismos.bin',report/'analysis', with_fft=args.fft)
            with (report/'sismos_stats.csv').open() as stream: candidate=list(csv.DictReader(stream))
            checks += compare_stats(ref,candidate,name,"sismos")
            full_coords = load_coords(data/'snapshots'/'coords.bin', '<f4')
            sliced_coords = load_coords(data/'slices'/'coords.bin', '<f4')
            axis = {'XY':2, 'XZ':1, 'YZ':0}[cfg['slice_plane']]
            levels = np.unique(full_coords[:,axis])
            position = levels[np.argmin(abs(levels-cfg['slice_position']))]
            mask = full_coords[:,axis] == position
            same_coords = np.array_equal(full_coords[mask], sliced_coords)
            full_paths = list_snapshots(data/'snapshots')
            sliced_paths = list_snapshots(data/'slices', 'slice_it_*.bin')
            if [step for step,_ in full_paths] != [step for step,_ in sliced_paths]:
                raise ValueError('Instants des slices et snapshots differents')
            slice_error = max(float(np.max(np.abs(load_snapshot(a)[mask]-load_snapshot(b)))) for (_,a),(_,b) in zip(full_paths,sliced_paths))
            checks.append(dict(size=name, data='slices', statistic='max_abs_difference', max_error_norm=slice_error,
                               tolerance=0, passed=int(same_coords and slice_error == 0)))
            if args.fft:
                ref = np.genfromtxt(report/'analysis'/'fft.csv', delimiter=',', names=True)
                found = np.genfromtxt(report/'fft.csv', delimiter=',', names=True)
                if ref.shape != found.shape or not np.array_equal(ref['receiver'], found['receiver']):
                    raise ValueError("Dimensions des spectres incompatibles")
                for column, tolerance in [('frequency_hz', 1e-10), ('amplitude', 5e-5)]:
                    scale = float(np.max(np.abs(ref[column]))) or 1.
                    error = float(np.max(np.abs(ref[column] - found[column]))) / scale
                    checks.append(dict(size=name, data='fft', statistic=column, max_error_norm=error,
                                       tolerance=tolerance, passed=int(error <= tolerance)))
            if args.keep_data: shutil.copytree(data,report/'data')
    write_rows(output.with_name(output.stem+"_validation.csv"),checks)
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n")
    if any(not r['passed'] for r in checks): raise RuntimeError("Ecart numerique excessif : voir le CSV de validation")
    subprocess.run([sys.executable,str(HERE/'bench_scalability.py'),'--raw',str(output),'--out',str(output.with_name(output.stem+'_scalability.json'))],check=True)
    if not args.no_plots:
        command = [sys.executable,str(HERE/'plot_tp2_benchmarks.py'),'--input',str(output)]
        if args.figures: command += ['--figures', args.figures]
        subprocess.run(command,check=True)
    print(f"Mesures : {output}")


if __name__ == "__main__":
    main()
