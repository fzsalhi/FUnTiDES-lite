#!/usr/bin/env python3
"""Mesures PPM et cartes cumulees sur les configurations S/M/L."""
import argparse
from collections import defaultdict
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


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plot_results(source, comparison=None):
    command = [
        sys.executable,
        str(HERE / 'plot_tp3_mesures.py'),
        '--input',
        str(source),
    ]
    if comparison:
        command += ['--compare-input', str(comparison)]
    subprocess.run(command, check=True)


def execute(command, env, report):
    start = perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    duration = perf_counter() - start
    (report / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    (report / 'simulation.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(
            f'Simulation en echec : {report}/simulation.log\n'
            f'{result.stderr[-2000:]}'
        )
    return duration


def read_ppm(path):
    with Path(path).open('rb') as stream:
        magic = stream.readline().strip()
        (width, height) = map(int, stream.readline().split())
        maximum = int(stream.readline())
        data = stream.read()
    if maximum != 255 or magic not in (b'P3', b'P6'):
        raise ValueError(f'En-tete PPM invalide : {path}')
    if magic == b'P3':
        values = np.fromstring(data.decode('ascii'), sep=' ', dtype=np.int64)
        if np.any(values < 0) or np.any(values > 255):
            raise ValueError('Canal P3 hors limites')
        rgb = values.astype(np.uint8)
    else:
        rgb = np.frombuffer(data, dtype=np.uint8)
    if rgb.size != 3 * width * height:
        raise ValueError(f'Taille PPM incoherente : {path}')
    return rgb.reshape(height, width, 3)


def command_base(exe, size, args, data, report):
    return [str(exe), '--mesh=cartesian', '--implem=makutu', '--method=sem',
        f'--order={args.order}', f'--ex={size[1]}', f'--ey={size[1]}',
        f'--ez={size[1]}', '--lx=2000', '--ly=2000', '--lz=2000', f'--dt={args.dt}',
        f'--timemax={args.timemax}', '--quiet=true', '--record-receivers=false',
        '--is-snapshots=false', '--stats-interval=0', '--insitu-viz=none',
        '--insitu-compress=none', '--insitu-extra=none', f'--io-sync={args.io_sync}',
        f"--metrics-file={report / 'simulation.json'}",
        f"--visu-folder={data / 'images'}", f"--maps-folder={data / 'maps'}",
        f'--visu-width={args.width or args.order * size[1] + 1}',
        f'--visu-height={args.height or args.order * size[1] + 1}',
        f'--visu-plane={args.plane}', f'--visu-position={args.position}',
        f'--visu-interval={args.interval}', f'--maps-threshold={args.threshold}']


def mode_options(mode, args):
    render = 'gpu' if args.expect_backend == 'gpu' else 'cpu'
    if mode == 'baseline':
        return []
    if mode.startswith('maps_'):
        return [f'--insitu-maps={mode[5:]}', f'--visu-backend={render}']
    if mode in ('normalization_fixed', 'normalization_running'):
        kind = mode.split('_', 1)[1]
        return ['--insitu-viz=ppm', f'--visu-backend={render}', '--visu-format=P6',
            f'--visu-normalization={kind}',
            f"--visu-scale={(args.scale if kind == 'fixed' else 0)}",
            f'--visu-window={args.window}']
    (_, backend, fmt) = mode.split('_')
    return ['--insitu-viz=ppm', f'--visu-backend={backend}', f'--visu-format={fmt}',
        '--visu-normalization=local']


def validate(exe, args, env, details, data_root):
    checks = []
    for plane in ['XY', 'XZ', 'YZ']:
        images = {}
        for backend in ['cpu', 'gpu'] if args.expect_backend == 'gpu' else ['cpu']:
            report = details / 'validation' / plane / backend
            report.mkdir(parents=True)
            with tempfile.TemporaryDirectory(dir=data_root,
                prefix='check-') as temporary:
                data = Path(temporary)
                command = command_base(exe, ('GLL', 3), args, data, report)
                command += ['--order=3', '--timemax=0.3', f'--visu-plane={plane}',
                    '--visu-width=64', '--visu-height=48', '--visu-position=0.43',
                    '--insitu-viz=ppm', '--visu-format=both',
                    f'--visu-backend={backend}']
                if plane == 'XY' and backend == 'cpu':
                    command += ['--insitu-maps=all', '--maps-reference-interval=1',
                        '--maps-images=false']
                execute(command, env, report)
                if plane == 'XY' and backend == 'cpu':
                    for (quantity, dtype) in ([('maximum_amplitude', '<f4'),
                        ('first_arrival', '<i4'), ('cumulative_energy', '<f8')]):
                        original = np.fromfile(data / 'maps' / f'{quantity}.bin',
                            dtype=dtype)
                        sampled = np.fromfile(data / 'maps' / f'{quantity}_sampled.bin',
                            dtype=dtype)
                        error = float(np.max(np.abs(original.astype(float)
                            - sampled.astype(float))))
                        checks.append(dict(test='maps_k1_' + quantity, plane=plane,
                            backend=backend, step='all', max_error=error, tolerance=0,
                            passed=int(error == 0)))
                local = {}
                for path in sorted((data / 'images').glob('*_P3.ppm')):
                    rgb = read_ppm(path)
                    other = read_ppm(path.with_name(path.name.replace('_P3.ppm',
                        '_P6.ppm')))
                    error = int(np.max(np.abs(rgb.astype(int) - other.astype(int))))
                    checks.append(dict(test='P3_P6', plane=plane, backend=backend,
                        step=path.stem.split('_')[3], max_error=error, tolerance=0,
                        passed=int(error == 0)))
                    local[path.name] = rgb
                if not local:
                    raise ValueError('Aucune image de validation')
                images[backend] = local
        if 'gpu' in images:
            if images['cpu'].keys() != images['gpu'].keys():
                raise ValueError('Images CPU/GPU manquantes')
            for (name, rgb) in images['cpu'].items():
                error = int(np.max(np.abs(rgb.astype(int)
                    - images['gpu'][name].astype(int))))
                checks.append(dict(test='CPU_GPU', plane=plane, backend='gpu',
                    step=name.split('_')[3], max_error=error, tolerance=2,
                    passed=int(error <= 2)))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=ROOT / 'build-cuda/bin/semproxy')
    parser.add_argument('--expect-backend', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--timemax', type=float, default=1.5)
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--width', type=int, default=0,
        help='Largeur en pixels ; 0 : nombre de noeuds par direction')
    parser.add_argument('--height', type=int, default=0,
        help='Hauteur en pixels ; 0 : nombre de noeuds par direction')
    parser.add_argument('--plane', choices=['XY', 'XZ', 'YZ'], default='XY')
    parser.add_argument('--position', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=2.5)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--io-sync', choices=['buffered', 'fsync'], default='buffered')
    parser.add_argument('--keep-data', action='store_true')
    parser.add_argument('--skip-validation', action='store_true')
    parser.add_argument('--modes', help='Modes separes par des virgules')
    parser.add_argument('--plot-only', type=Path)
    parser.add_argument(
        '--compare-input', type=Path,
        help='Mesures avec l autre mode io-sync pour comparer fsync',
    )
    args = parser.parse_args()
    if args.plot_only:
        plot_results(args.plot_only, args.compare_input)
        return
    if args.compare_input:
        args.compare_input = args.compare_input.resolve()
        if not args.compare_input.is_file():
            parser.error('Fichier de comparaison absent')
        with args.compare_input.open(newline='') as stream:
            previous = list(csv.DictReader(stream))
        expected_policy = 'buffered' if args.io_sync == 'fsync' else 'fsync'
        if {r['io_sync'] for r in previous} != {expected_policy}:
            parser.error('La comparaison exige buffered et fsync')
        if {r['backend'] for r in previous} != {args.expect_backend}:
            parser.error('Backends des campagnes differents')
    if (args.runs < 3 or args.warmups < 0 or args.threads < 1 or (args.interval < 1)
        or (args.width < 0) or (args.height < 0) or (args.width == 1)
        or (args.height == 1)):
        parser.error(
            'runs >= 3, warmups >= 0, threads/interval >= 1 '
            'et dimensions = 0 ou >= 2 requis'
        )
    if (args.dt <= 0 or args.timemax <= 0 or args.scale <= 0 or (args.window < 1)
        or (args.threshold <= 0)):
        parser.error('Parametres physiques invalides')
    exe = args.exe.resolve()
    if not exe.exists():
        parser.error(f'Executable absent : {exe}')
    sizes = [('S', 10), ('M', 15), ('L', 20)]
    modes = ['baseline', 'ppm_cpu_P3', 'ppm_cpu_P6']
    if args.expect_backend == 'gpu':
        modes += ['ppm_gpu_P3', 'ppm_gpu_P6']
    modes += ['normalization_fixed', 'normalization_running', 'maps_amplitude',
        'maps_arrival', 'maps_energy', 'maps_all']
    if args.modes:
        selected = args.modes.split(',')
        if set(selected) - set(modes):
            parser.error('Mode inconnu ou incompatible avec le backend')
        modes = selected
        if 'baseline' not in modes:
            modes.insert(0, 'baseline')
    batch = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = (args.output or HERE / 'results'
        / f'tp3_{batch}_{args.expect_backend}.csv').resolve()
    if output.exists():
        parser.error('Le fichier de mesures existe deja')
    output.parent.mkdir(parents=True, exist_ok=True)
    details = output.parent / 'details' / output.stem
    details.mkdir(parents=True)
    data_root = ROOT / 'tp3/data'
    data_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_PROC_BIND='spread',
        OMP_PLACES='cores')
    sources = sorted((ROOT / 'tp3/src').glob('*')) + sorted((ROOT
        / 'tp3/include').glob('*'))
    sources += [ROOT / 'src/main/fe/src/sem_proxy.cc', ROOT
        / 'src/main/fe/include/sem_proxy_options.h', ROOT
        / 'src/utils/include/slice_utils.hpp']
    meta = dict(schema='funtides.tp3.v1', batch=batch, machine=platform.node(),
        platform=platform.platform(), parameters={k: str(v) if isinstance(v,
        Path) else v for (k, v) in vars(args).items()}, executable=str(exe),
        executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(b''.join((p.read_bytes() for p in sources
        if p.is_file()))).hexdigest(), sizes=sizes, modes=modes,
        data_directory=str(data_root),
        cache='Lectures non purgees ; buffered ou fsync selon io-sync',
        command=sys.argv)
    try:
        meta['gpu'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        meta['gpu'] = 'indisponible'
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        meta['cpu'] = next((line.split(':', 1)[1].strip()
            for line in cpuinfo.read_text().splitlines()
            if line.startswith('model name')), platform.processor())
    cache = exe.parent.parent / 'CMakeCache.txt'
    if cache.exists():
        meta['build_options'] = [line for line in cache.read_text().splitlines()
            if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:',
            'ENABLE_CUDA:', 'Kokkos_ARCH_', 'Kokkos_ENABLE_'))
            and (not line.endswith('=OFF'))]
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2,
        ensure_ascii=False) + '\n')
    rows = []
    frames = []
    rng = random.Random(328)
    cases = [(size, mode) for size in sizes for mode in modes]
    print(f'{len(cases)} cas x ({args.warmups} chauffe + {args.runs} mesures)',
        flush=True)
    for (warmup, count) in [(True, args.warmups), (False, args.runs)]:
        for rep in range(1, count + 1):
            for (size, mode) in rng.sample(cases, len(cases)):
                report = details / size[0] / mode / (f'warmup_{rep}'
                    if warmup else f'run_{rep}')
                report.mkdir(parents=True)
                with tempfile.TemporaryDirectory(dir=data_root,
                    prefix='run-') as temporary:
                    data = Path(temporary)
                    wall = execute(command_base(exe, size, args, data, report)
                        + mode_options(mode, args), env, report)
                    metrics = json.loads((report / 'simulation.json').read_text())
                    if metrics['backend'] != args.expect_backend:
                        raise ValueError(f"Backend detecte : {metrics['backend']}")
                    actual = sum((p.stat().st_size for p in data.rglob('*')
                        if p.is_file()))
                    if actual != metrics['total_output_bytes']:
                        raise ValueError(
                            f'Comptage des octets incoherent : {actual} != '
                            f'{metrics["total_output_bytes"]}'
                        )
                    row = dict(metrics)
                    row.update(batch=batch, machine=platform.node(),
                        source_hash=meta['source_sha256'], threads=args.threads,
                        size=size[0], mode=mode, run=rep, width=args.width
                        or args.order * size[1] + 1, height=args.height or args.order
                        * size[1] + 1, interval=args.interval, wall_s=wall)
                    if not warmup:
                        rows.append(row)
                        write_csv(output, rows)
                        write_csv(output.with_name(output.stem + '_baseline.csv'), [r
                            for r in rows if r['mode'] == 'baseline'])
                        image_csv = data / 'images/visu_metrics.csv'
                        if image_csv.exists():
                            with image_csv.open(newline='') as stream:
                                for frame in csv.DictReader(stream):
                                    frames.append(dict(size=size[0], mode=mode, run=rep,
                                        **frame))
                            write_csv(output.with_name(output.stem + '_frames.csv'),
                                frames)
                    if args.keep_data:
                        shutil.copytree(data, report / 'data')
                    print(
                        f'{("chauffe" if warmup else f"mesure {rep}/{args.runs}")} '
                        f'{size[0]} {mode}: {wall:.3f}s, '
                        f'{actual / 2 ** 20:.3f} Mio',
                        flush=True,
                    )
    baseline = {size: np.mean([float(r['wall_s']) for r in rows if r['size'] == size
        and r['mode'] == 'baseline']) for (size, _) in sizes}
    for row in rows:
        row['overhead_s'] = float(row['wall_s']) - float(baseline[row['size']])
    write_csv(output, rows)
    write_csv(output.with_name(output.stem + '_baseline.csv'), [r for r in rows
        if r['mode'] == 'baseline'])
    maps = []
    reference_checks = []
    reference_sizes = (
        sizes if any(mode.startswith('maps_') for mode in modes) else []
    )
    for size in reference_sizes:
        report = details / size[0] / 'maps_reference'
        report.mkdir(parents=True)
        with tempfile.TemporaryDirectory(dir=data_root, prefix='maps-') as temporary:
            data = Path(temporary)
            render = 'gpu' if args.expect_backend == 'gpu' else 'cpu'
            command = command_base(exe, size, args, data,
                report) + ['--insitu-maps=all',
                f'--maps-reference-interval={args.interval}',
                f'--visu-backend={render}']
            if not args.skip_validation:
                command += ['--is-snapshots=true', '--snap-format=bin',
                    f"--snap-folder={data / 'snapshots'}",
                    f'--snap-time-interval={args.interval}']
            execute(command, env, report)
            if not args.skip_validation:
                sys.path.insert(0, str(ROOT / 'tp2'))
                from io_utils import list_snapshots, load_snapshot
                simulated = json.loads((report / 'simulation.json').read_text())
                n = simulated['nodes']
                steps = simulated['steps']
                maximum = np.zeros(n, dtype=np.float32)
                first = np.full(n, -1, dtype=np.int32)
                energy = np.zeros(n, dtype=np.float64)
                for (step, path) in list_snapshots(data / 'snapshots'):
                    values = load_snapshot(path)
                    absolute = np.abs(values)
                    maximum = np.maximum(maximum, absolute)
                    first[(first < 0) & (absolute > np.float32(args.threshold))] = step
                    energy += values.astype(np.float64) ** 2 * min(args.interval, steps
                        - step)
                for (quantity, expected, dtype) in ([('maximum_amplitude', maximum,
                    '<f4'), ('first_arrival', first, '<i4'), ('cumulative_energy',
                    energy, '<f8')]):
                    found = np.fromfile(data / 'maps' / f'{quantity}_sampled.bin',
                        dtype=dtype)
                    scale = float(np.max(np.abs(expected))) or 1.0
                    error = float(np.max(np.abs(found.astype(np.float64)
                        - expected))) / scale
                    tolerance = 1e-12 if quantity == 'cumulative_energy' else 0.0
                    reference_checks.append(dict(test='sampled_' + quantity,
                        plane=args.plane, backend=args.expect_backend, step=size[0],
                        max_error=error, tolerance=tolerance,
                        passed=int(error <= tolerance)))
            with (data / 'maps/maps_summary.csv').open(newline='') as stream:
                for row in csv.DictReader(stream):
                    maps.append(dict(size=size[0], **row))
            destination = (
                output.parent / 'details' / output.stem / size[0]
                / 'maps_reference' / 'data'
            )
            shutil.copytree(data / 'maps', destination)
    write_csv(output.with_name(output.stem + '_maps.csv'), maps)
    if not args.skip_validation:
        checks = reference_checks + validate(exe, args, env, details, data_root)
        write_csv(output.with_name(output.stem + '_validation.csv'), checks)
        if any((not r['passed'] for r in checks)):
            raise RuntimeError(
                'Ecart CPU/GPU ou P3/P6 excessif : voir le CSV de validation'
            )
    plot_results(output, args.compare_input)
    print(f'Mesures : {output}')


if __name__ == '__main__':
    main()
