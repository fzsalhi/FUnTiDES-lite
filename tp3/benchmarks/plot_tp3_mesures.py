#!/usr/bin/env python3
"""Figures de cout, de stockage et de precision des traitements TP3."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tp2/benchmarks'))
from plot_tp2_benchmarks import draw


def read_rows(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Aucune mesure : {path}')
    return rows


def sync_sources(first_path, second_path):
    sources = {}
    for path in (first_path, second_path):
        rows = read_rows(path)
        policies = {r['io_sync'] for r in rows}
        if len(policies) != 1:
            raise ValueError('Une campagne doit avoir un seul mode io-sync')
        policy = policies.pop()
        if policy in sources:
            raise ValueError('La comparaison exige buffered et fsync')
        sources[policy] = (path, rows)
    if set(sources) != {'buffered', 'fsync'}:
        raise ValueError('La comparaison exige buffered et fsync')

    keys = (
        'machine', 'source_hash', 'backend', 'threads', 'size', 'nodes',
        'steps', 'dt_s', 'width', 'height', 'interval',
    )
    signatures = [
        {tuple(row[key] for key in keys) for row in rows}
        for _, rows in sources.values()
    ]
    if signatures[0] != signatures[1]:
        raise ValueError('Configurations buffered/fsync incompatibles')

    metadata = [
        json.loads(path.with_suffix('.json').read_text())
        for path, _ in sources.values()
    ]
    for key in ('cpu', 'gpu', 'data_directory'):
        if metadata[0].get(key) != metadata[1].get(key):
            raise ValueError(f'Environnement buffered/fsync different : {key}')
    for key in (
        'order', 'dt', 'timemax', 'interval', 'plane', 'position',
        'scale', 'window', 'threshold',
    ):
        if (metadata[0]['parameters'].get(key)
                != metadata[1]['parameters'].get(key)):
            raise ValueError(f'Parametre buffered/fsync different : {key}')
    return sources['buffered'], sources['fsync']


def draw_sync_comparison(output, buffered, synced, sizes, suffix):
    groups = defaultdict(list)
    for rows in (buffered, synced):
        for row in rows:
            groups[row['size'], row['mode'], row['io_sync']].append(row)
    backend = None
    for candidate in ('gpu', 'cpu'):
        if all(
            (size, f'ppm_{candidate}_{fmt}', policy) in groups
            for size in sizes
            for fmt in ('P3', 'P6')
            for policy in ('buffered', 'fsync')
        ):
            backend = candidate
            break
    if backend is None:
        raise ValueError('Mesures P3/P6 manquantes pour comparer fsync')

    def curve(policy, mode, extract, label):
        samples = [
            [extract(row) for row in groups[size, mode, policy]]
            for size in sizes
        ]
        return dict(
            label=label,
            values=[statistics.mean(values) for values in samples],
            errors=[
                statistics.stdev(values) if len(values) > 1 else 0
                for values in samples
            ],
        )

    def panel(title, ylabel, curves):
        return dict(
            title=title, ylabel=ylabel, xlabel='Configuration', ticks=sizes,
            x=list(range(1, len(sizes) + 1)), series=curves,
        )

    def paired(mode, extract):
        return [
            curve(policy, mode, extract, label)
            for policy, label in (
                ('buffered', 'Sans fsync'), ('fsync', 'Avec fsync')
            )
        ]

    write_ms = lambda row: 1000 * (
        float(row['ppm_write_s']) + float(row['ppm_fsync_s'])
    )
    p3 = f'ppm_{backend}_P3'
    p6 = f'ppm_{backend}_P6'
    final_panel = panel(
        'P6 : temps de fsync seul', 'Temps cumule (ms)', [
            curve('fsync', p6,
                  lambda row: 1000 * float(row['ppm_fsync_s']),
                  'Images PPM'),
            curve('fsync', p6,
                  lambda row: 1000 * (
                      float(row['fsync_s']) - float(row['ppm_fsync_s'])
                  ), 'CSV de mesures'),
        ],
    )
    if all(
        (size, 'maps_all', policy) in groups
        for size in sizes for policy in ('buffered', 'fsync')
    ):
        final_panel = panel(
            'Trois cartes : simulation et sorties', 'Temps total (s)',
            paired('maps_all', lambda row: float(row['wall_s'])),
        )
    name = 'ppm_fsync'
    draw(
        output, name,
        'Cout de fsync — rendu ' + backend.upper() + suffix,
        [
            panel('P3 : ecriture et fsync des images', 'Temps cumule (ms)',
                  paired(p3, write_ms)),
            panel('P6 : ecriture et fsync des images', 'Temps cumule (ms)',
                  paired(p6, write_ms)),
            panel('P6 : simulation et visualisation', 'Temps total (s)',
                  paired(p6, lambda row: float(row['wall_s']))),
            final_panel,
        ],
    )
    return name + '.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'plots')
    parser.add_argument('--compare-input', type=Path)
    args = parser.parse_args()
    synced_rows = None
    synced_path = None
    if args.compare_input:
        (args.input, rows), (synced_path, synced_rows) = sync_sources(
            args.input, args.compare_input
        )
    else:
        rows = read_rows(args.input)
    if (len({(r['machine'], r['source_hash'], r['backend'], r['threads'], r['io_sync'])
        for r in rows}) != 1):
        parser.error('Mesures incompatibles')
    sizes = sorted({r['size'] for r in rows}, key=lambda s: next((int(r['nodes'])
        for r in rows if r['size'] == s)))
    groups = defaultdict(list)
    for row in rows:
        groups[row['size'], row['mode']].append(row)

    def has(*modes):
        return all(((s, m) in groups for s in sizes for m in modes))

    def series(label, mode, extract):
        samples = [[extract(r) for r in groups[s, mode]] for s in sizes]
        return dict(label=label, values=[statistics.mean(v) for v in samples],
            errors=[statistics.stdev(v) if len(v) > 1 else 0 for v in samples])

    def line(title, ylabel, curves, log=False):
        return dict(title=title, ylabel=ylabel, x=list(range(1, len(sizes) + 1)),
            ticks=sizes, xlabel='Configuration', log_y=log, series=[series(*curve)
            for curve in curves])
    val = lambda key, scale=1: lambda r: float(r[key]) * scale
    total = lambda *keys: lambda r: sum((float(r[k]) for k in keys)) * 1000
    args.output.mkdir(parents=True, exist_ok=True)
    figures = []
    suffix = ' — ' + rows[0]['backend'].upper() + ' / ' + rows[0]['machine']

    def save(name, title, panels):
        draw(args.output, name, title + suffix, panels)
        figures.append(name + '.png')
    image_backend = 'gpu' if has('ppm_gpu_P3', 'ppm_gpu_P6') else 'cpu'
    (p3, p6) = ('ppm_' + image_backend + '_P3', 'ppm_' + image_backend + '_P6')
    if has(p3, p6):
        curves = lambda extract: [('P3 texte', p3, extract), ('P6 binaire', p6,
            extract)]
        save('ppm_formats', 'Meme image : PPM texte ou binaire',
            [line('Simulation et visualisation', 'Temps total (s)',
            curves(val('wall_s'))), line('Fichiers PPM produits',
            'Taille cumulee (Mio)', curves(val('ppm_bytes', 1 / 2 ** 20))),
            line('Encodage des pixels', 'Temps cumule (ms)', curves(val('ppm_encode_s',
            1000))), line('Ecriture et synchronisation', 'Temps cumule (ms)',
            curves(total('ppm_write_s', 'ppm_fsync_s')))])
    if has('ppm_cpu_P6', 'ppm_gpu_P6'):
        curves = lambda extract: [('Image calculee sur CPU', 'ppm_cpu_P6', extract),
            ('Image calculee sur GPU', 'ppm_gpu_P6', extract)]
        save('ppm_cpu_gpu',
            'Simulation CUDA : generation des memes images sur CPU ou GPU',
            [line('Simulation et visualisation', 'Temps total (s)',
            curves(val('wall_s'))), line('Extraction, couleurs et retour RGB',
            'Temps cumule (ms)', curves(total('ppm_extract_s', 'ppm_color_s',
            'ppm_copy_s'))), line('Pressions utiles CPU / RGB et maximum GPU',
            'Donnees logiques cumulees (Mio)', curves(val('ppm_host_bytes', 1 / 2
            ** 20))), line('Propagation de l onde sur GPU', 'Temps cumule (ms)',
            curves(val('compute_s', 1000)))])
    frame_path = args.input.with_name(args.input.stem + '_frames.csv')
    if frame_path.exists() and has(p6, 'normalization_fixed', 'normalization_running'):
        with frame_path.open(newline='') as stream:
            frames = list(csv.DictReader(stream))
        curves = []
        x = None
        for (label, mode) in ([('Locale', p6), ('Fixe', 'normalization_fixed'),
            ('Maximum glissant', 'normalization_running')]):
            values = sorted((r for r in frames if r['size'] == sizes[-1]
                and r['mode'] == mode and (r['run'] == '1')),
                key=lambda r: int(r['step']))
            if not values:
                continue
            x = [float(r['time_s']) for r in values]
            curves.append(dict(label=label, values=[float(r['scale']) for r in values]))
        if len(curves) == 3:
            stride = max(1, len(x) // 5)
            save('normalisation', 'Normalisation symetrique des pressions',
                [dict(title='Borne positive utilisee — ' + sizes[-1], ylabel='Pression',
                x=x, ticks=[f'{v:.2f}' if i % stride == 0 else '' for (i,
                v) in enumerate(x)], xlabel='Temps physique (s)', series=curves),
                line('Cout des images P6', 'Temps cumule (ms)', [(label, mode,
                val('ppm_s', 1000)) for (label, mode) in [('Locale', p6), ('Fixe',
                'normalization_fixed'), ('Glissante', 'normalization_running')]])])
    maps = [('Amplitude maximale', 'maps_amplitude'), ('Premiere arrivee',
        'maps_arrival'), ('Somme de p^2', 'maps_energy'), ('Trois cartes', 'maps_all')]
    if has('baseline', *[m for (_, m) in maps]):
        baseline = {(r['size'], r['run']): float(r['wall_s']) for r in rows
            if r['mode'] == 'baseline'}
        save('cartes_cout', 'Cartes calculees a chaque pas : cout et memoire',
            [line('Simulation, ecriture et images finales', 'Temps total (s)',
            [('Sans traitement', 'baseline', val('wall_s')), ('Trois cartes',
            'maps_all', val('wall_s'))]), line('Mise a jour des cartes',
            'Temps cumule (ms)', [(label, mode, val('maps_update_s', 1000)) for (label,
            mode) in maps]), line('Mises a jour, copies, fichiers et images finales',
            'Temps cumule (ms)', [(label, mode, total('maps_update_s', 'maps_copy_s',
            'maps_write_s', 'maps_render_s')) for (label, mode) in maps]),
            line('Accumulateurs RAM / snapshots bruts sur disque',
            'Volume logique (Mio)', [('Trois cartes en memoire', 'maps_all',
            val('maps_memory_bytes', 1 / 2 ** 20)), ('Pressions de tous les snapshots',
            'maps_all', lambda r: 4 * float(r['nodes']) * ((int(r['steps'])
            + int(r['interval']) - 1) // int(r['interval'])) / 2 ** 20)])])
    map_path = args.input.with_name(args.input.stem + '_maps.csv')
    if map_path.exists():
        with map_path.open(newline='') as stream:
            maps_rows = list(csv.DictReader(stream))
        by = {(r['size'], r['quantity']): r for r in maps_rows}

        def map_line(title, ylabel, quantity, curves):
            return dict(title=title, ylabel=ylabel, x=list(range(1, len(sizes) + 1)),
                ticks=sizes, xlabel='Configuration', series=[dict(label=label,
                values=[extract(by[s, quantity]) for s in sizes]) for (label,
                extract) in curves])
        if (all(((s, q) in by for s in sizes for q in ['maximum_amplitude',
            'first_arrival', 'cumulative_energy']))):
            save('cartes_precision',
                'Tous les pas ou snapshots espaces : effet sur les cartes',
                [map_line('Amplitude maximale', 'Erreur normalisee',
                'maximum_amplitude', [('RMSE', lambda r: float(r['rmse_norm'])),
                ('Erreur maximale', lambda r: float(r['max_error_norm']))]),
                map_line('Premieres arrivees manquees', 'Noeuds atteints manques (%)',
                'first_arrival', [('Echantillonnage des snapshots', lambda r: 100
                * int(r['missed_nodes']) / max(1, int(r['reached_nodes'])))]),
                map_line('Retard des arrivees detectees dans les deux cartes',
                'Retard (ms)', 'first_arrival', [('Moyen',
                lambda r: float(r['mean_delay_s']) * 1000), ('Maximum',
                lambda r: float(r['max_delay_s']) * 1000)]),
                map_line('Somme p^2, reference par maintien constant',
                'Erreur normalisee', 'cumulative_energy', [('RMSE',
                lambda r: float(r['rmse_norm'])), ('Erreur maximale',
                lambda r: float(r['max_error_norm']))])])
    if synced_rows is not None:
        figures.append(draw_sync_comparison(
            args.output, rows, synced_rows, sizes, suffix
        ))
    summary = []
    for ((size, mode), group) in sorted(groups.items()):
        row = dict(size=size, mode=mode, repetitions=len(group))
        for key in group[0]:
            if key.endswith(('_s', '_bytes')):
                values = [float(r[key]) for r in group]
                row.update({key + '_mean': statistics.mean(values), key
                    + '_median': statistics.median(values), key
                    + '_std': statistics.stdev(values) if len(values) > 1 else 0})
        summary.append(row)
    if summary:
        with (args.output / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(summary[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(summary)
    manifest = dict(source=str(args.input), figures=figures)
    if synced_path is not None:
        manifest['fsync_source'] = str(synced_path)
    (args.output / 'figures.json').write_text(
        json.dumps(manifest, indent=2) + '\n'
    )
    print(f'{len(figures)} figures : {args.output}')


if __name__ == '__main__':
    main()
