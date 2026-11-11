"""Figures TP2 : une comparaison par image, avec deux a quatre panneaux."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tp1' / 'benchmarks'))
from plot_benchmarks import gnuplot_string

COLORS = ['#2563eb', '#ea580c', '#059669', '#7c3aed']
FIGURES = ['snapshots', 'sismogrammes', 'fft', 'slices', 'statistiques_cpu_gpu',
           'precision', 'scalabilite', 'cpu_vs_gpu']


class Campaign:
    def __init__(self, path):
        self.path = Path(path).resolve()
        with self.path.open(newline='') as stream:
            self.rows = list(csv.DictReader(stream))
        if not self.rows:
            raise ValueError(f'Aucune mesure : {path}')
        identities = {(r['machine'], r['source_hash'], r['config_hash'], r['threads'],
                       r['io_sync'], r['profile'], r['backend']) for r in self.rows}
        if len(identities) != 1:
            raise ValueError(f'Mesures incompatibles : {path}')
        identity = next(iter(identities))
        self.identity, self.backend = identity[:-1], identity[-1]
        self.groups = defaultdict(list)
        for row in self.rows:
            self.groups[(row['size'], row['mode'])].append(row)
        self.sizes = sorted({r['size'] for r in self.rows},
                            key=lambda s: next(int(r['nodes']) for r in self.rows if r['size'] == s))

    def has(self, *modes):
        return all((size, mode) in self.groups for size in self.sizes for mode in modes)

    def series(self, label, mode, extract):
        means, errors = [], []
        for size in self.sizes:
            values = [extract(r) for r in self.groups[(size, mode)]]
            if not values or not all(math.isfinite(v) for v in values):
                raise ValueError(f'Mesure absente ou non finie : {size} {mode}')
            means.append(statistics.mean(values))
            errors.append(statistics.stdev(values) if len(values) > 1 else 0)
        return dict(label=label, values=means, errors=errors)

    def mean(self, size, mode, extract):
        return statistics.mean(extract(r) for r in self.groups[(size, mode)])

    def sidecar(self, suffix):
        return self.path.with_name(self.path.stem + suffix)


def value(key, scale=1):
    return lambda row: float(row[key]) * scale


def total(*keys, scale=1):
    return lambda row: sum(float(row[k]) for k in keys) * scale


def panel(campaign, title, ylabel, curves, log_y=False):
    return dict(title=title, ylabel=ylabel, x=list(range(1, len(campaign.sizes)+1)),
                ticks=campaign.sizes, xlabel='Configuration', log_y=log_y,
                series=[campaign.series(*curve) for curve in curves])


def phases(campaign, title, modes, components):
    size = campaign.sizes[-1]
    return dict(title=f'{title} — {size}', ylabel='Temps cumule (ms)',
                bars=[label for label, mode in modes],
                components=[dict(label=label, values=[campaign.mean(size, mode, extract) * 1000
                            for _, mode in modes]) for label, extract in components])


def draw(output, name, title, panels):
    """Gnuplot multiplot ; les barres empilees representent des etapes disjointes."""
    if not panels or len(panels) > 4:
        raise ValueError('Une figure doit contenir entre un et quatre panneaux')
    rows = 1 if len(panels) <= 2 else 2
    cols = 1 if len(panels) == 1 else 2
    width, height = 1800, 760 if rows == 1 else 1260
    destination = output / (name + '.png')
    temporary = output / ('.' + name + '.png')
    blocks, commands = [], []
    for index, item in enumerate(panels):
        commands += ['unset logscale', 'unset label', 'set autoscale', 'set format x "%g"',
                     'set format y "%g"', 'set grid ytics back lc rgb "#d9dce3"',
                     'set border 3 lc rgb "#6b7280"', 'set tics nomirror',
                     'set key outside bottom center horizontal maxcols 2 font ",11"',
                     'set lmargin 11', 'set rmargin 3', 'set tmargin 3', 'set bmargin 5',
                     'set title ' + gnuplot_string(item['title']),
                     'set ylabel ' + gnuplot_string(item['ylabel'])]
        if 'bars' in item:
            block = f'data_{index}'
            values = [' '.join([gnuplot_string(label)] + [f"{c['values'][i]:.12g}" for c in item['components']])
                      for i, label in enumerate(item['bars'])]
            blocks.append(f'${block} << EOD\n' + '\n'.join(values) + '\nEOD')
            commands += ['set xlabel ""', 'set xrange [-0.7:' + str(len(item['bars'])-.3) + ']',
                         'set yrange [0:*]', 'set xtics auto', 'set style data histograms',
                         'set style histogram rowstacked', 'set style fill solid 0.85 border -1',
                         'set boxwidth 0.55 relative']
            plots = [f'${block} using {i+2}:xtic(1) title {gnuplot_string(c["label"])} lc rgb "{COLORS[i]}"'
                     for i, c in enumerate(item['components'])]
        else:
            commands += ['set xlabel ' + gnuplot_string(item['xlabel']), 'set yrange [0:*]']
            if len(item['series']) == 3:
                commands += ['set key maxcols 3']
            if item.get('log_y'):
                commands += ['set logscale y', 'set yrange [*:*]', 'set format y "10^{%L}"']
            if item.get('log_x'):
                commands += ['set logscale x']
            x = item['x']
            margin = .06 * (max(x)-min(x)) if len(x) > 1 else .5
            commands += [f'set xrange [{min(x)-margin if not item.get("log_x") else min(x)/1.12}:{max(x)+margin}]',
                         'set xtics (' + ', '.join(f'{gnuplot_string(t)} {v}' for t,v in zip(item['ticks'], x)) + ')']
            plots = []
            for j, curve in enumerate(item['series']):
                block = f'data_{index}_{j}'
                errors = curve.get('errors')
                values = [f'{a:.12g} {b:.12g}' + (f' {errors[i]:.12g}' if errors is not None else '')
                          for i,(a,b) in enumerate(zip(x, curve['values']))]
                blocks.append(f'${block} << EOD\n' + '\n'.join(values) + '\nEOD')
                style = '1:2:3 with yerrorlines' if errors is not None else '1:2 with linespoints'
                plots.append(f'${block} using {style} lw 2.5 pt {7+j*2} ps 1.0 lc rgb "{COLORS[j]}" title {gnuplot_string(curve["label"])}')
        commands.append('plot ' + ', '.join(plots))
    script = '\n'.join(blocks) + f'\nset terminal pngcairo size {width},{height} enhanced font "Sans,12"\n'
    script += 'set output ' + gnuplot_string(str(temporary)) + '\n'
    script += f'set multiplot layout {rows},{cols} rowsfirst title {gnuplot_string(title)} font ",17"\n'
    script += '\n'.join(commands) + '\nunset multiplot\nunset output\n'
    result = subprocess.run(['gnuplot'], input=script, text=True, capture_output=True)
    if result.returncode or not temporary.exists():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f'Echec du trace {name} : {result.stderr[-2000:]}')
    temporary.replace(destination)


def workflow(campaign, data, adhoc, insitu):
    if not campaign.has(adhoc, insitu):
        return []
    if data == 'snapshots':
        compute, copies, write = 'stats_compute_s', 'stats_copy_s', 'stats_write_s'
    else:
        compute, copies, write = 'sismo_stats_s', 'sismo_stats_copy_s', 'sismo_stats_write_s'
    modes = [('Ad-hoc', adhoc), ('In-situ', insitu)]
    panels = [panel(campaign, 'Temps complet : simulation + post-traitement', 'Temps (s)',
                    [(label, mode, value('workflow_s')) for label,mode in modes]),
              panel(campaign, 'Volume des donnees et resultats', 'Volume (Mio)',
                    [(label, mode, value('workflow_output_bytes', 1/2**20)) for label,mode in modes], log_y=True),
              panel(campaign, 'Calcul : min, max, moyenne et variance', 'Temps cumule (ms)',
                    [('Ad-hoc NumPy', adhoc, value('adhoc_analysis_s', 1000)),
                     ('In-situ', insitu, value(compute, 1000))]),
              phases(campaign, 'Sorties et analyse, hors calcul de la simulation', modes,
                     [('Bruts : ecriture + lecture', total('snapshot_write_s', 'sismo_write_s', 'adhoc_read_s')),
                      ('Calcul des statistiques', total('adhoc_analysis_s', compute)),
                      ('Copies des resultats', value(copies)),
                      ('Ecriture des resultats', total('post_write_s', write))])]
    return panels


def fft(campaign):
    a, b = 'sismo_fft_adhoc', 'sismo_fft_insitu'
    if not campaign.has(a,b): return []
    overhead = lambda r: (float(r['adhoc_fft_prepare_s']) + float(r['sismo_stats_copy_s']) +
                         max(0, float(r['fft_total_s'])-float(r['fft_compute_s'])-float(r['fft_write_s'])))
    return [panel(campaign, 'Workflow : simulation + statistiques + FFT', 'Temps (s)',
                  [('Ad-hoc NumPy',a,value('workflow_s')), ('In-situ cuFFT',b,value('workflow_s'))]),
            panel(campaign, 'Volume des donnees et resultats', 'Volume (Kio)',
                  [('Ad-hoc NumPy',a,value('workflow_output_bytes',1/1024)), ('In-situ cuFFT',b,value('workflow_output_bytes',1/1024))]),
            panel(campaign, 'Calcul spectral NumPy / noyau cuFFT', 'Temps (microsecondes)',
                  [('NumPy + amplitudes (double)',a,value('adhoc_fft_compute_s',1e6)), ('Noyau cuFFT (simple)',b,value('fft_compute_s',1e6))]),
            phases(campaign, 'Traitement complet, hors calcul de la simulation', [('Ad-hoc',a),('In-situ',b)],
                   [('Bruts : ecriture + lecture',total('sismo_write_s','adhoc_read_s')),
                    ('Calcul : statistiques + FFT',lambda r: float(r['adhoc_analysis_s'])-float(r['adhoc_fft_prepare_s'])+float(r['sismo_stats_s'])+float(r['fft_compute_s'])),
                    ('Preparation, plan et copies',overhead),
                    ('Ecriture des resultats',total('post_write_s','sismo_stats_write_s','fft_write_s'))])]


def slices(campaign):
    a,b = 'visual_full','visual_slice'
    if not campaign.has(a,b): return []
    return [panel(campaign,'Workflow avec visualisation','Temps (s)',
                  [('Volume complet',a,value('workflow_s')),('Slice',b,value('workflow_s'))]),
            panel(campaign,'Volume brut, hors images','Volume (Mio)',
                  [('Volume complet',a,value('snapshot_bytes',1/2**20)),('Slice',b,value('slice_bytes',1/2**20))]),
            phases(campaign,'Preparation et E/S', [('Volume',a),('Slice',b)],
                   [('Selection, extraction et copie',total('slice_setup_s','slice_extract_s','slice_copy_s')),
                    ('Ecriture des donnees',total('snapshot_write_s','slice_write_s')),
                    ('Lecture des donnees',value('adhoc_read_s'))]),
            panel(campaign,'Rendu du meme plan 2D','Temps (s)',
                  [('Depuis le volume',a,value('render_s')),('Depuis la slice',b,value('render_s'))])]


def host_device(campaign):
    if campaign.backend != 'gpu' or not campaign.has('snapshot_insitu_gpu','snapshot_insitu_cpu','snapshot_insitu_hostcopy'):
        return []
    a,b,c = 'snapshot_insitu_gpu','snapshot_insitu_hostcopy','snapshot_insitu_cpu'
    return [panel(campaign,'Reduction GPU ou copie puis calcul CPU','Traitement cumule (ms)',
                  [('Copie du champ + CPU',b,total('stats_compute_s','stats_copy_s',scale=1000)),
                   ('GPU + copie du resultat',a,total('stats_compute_s','stats_copy_s',scale=1000))]),
            panel(campaign,'Reduction GPU ou acces CPU via UVM','Traitement cumule (ms)',
                  [('CPU via UVM',c,value('stats_compute_s',1000)),('GPU + copie du resultat',a,total('stats_compute_s','stats_copy_s',scale=1000))]),
            panel(campaign,'Donnees logiques lues par l hote','Volume cumule (Kio)',
                  [('Champ complet vers CPU',b,value('stats_transfer_bytes',1/1024)),('Resultats de reduction GPU',a,value('stats_transfer_bytes',1/1024))],log_y=True),
            panel(campaign,'Impact sur le calcul de la simulation','Calcul de la simulation (ms)',
                  [('Analyse CPU via UVM',c,value('compute_s',1000)),('Analyse GPU',a,value('compute_s',1000))])]


def precision(campaign):
    path = campaign.sidecar('_precision.csv')
    if not path.exists(): return []
    with path.open(newline='') as stream: rows = list(csv.DictReader(stream))
    result=[]
    for key, title, ylabel, scale in [('peak_error_norm','Maximum de pression','Erreur / max |p| (%)',100),
            ('mean_error_norm','Moyenne temporelle','Erreur / max |p| (x 10^{-6})',1e6)]:
        series=[]
        for size in campaign.sizes:
            values=sorted((r for r in rows if r['size']==size),key=lambda r:int(r['interval']))
            series.append(dict(label=size, values=[float(r[key])*scale for r in values]))
        x=[int(r['interval']) for r in values]
        result.append(dict(title=title,ylabel=ylabel,x=x,ticks=x,xlabel='Intervalle entre snapshots (pas, echelle log)',
                           log_x=True,series=series))
    return result


def scalability(campaigns, primary):
    if not primary.has('scaling_cfl'): return []
    x=[float(primary.groups[(s,'scaling_cfl')][0]['elements_per_axis']) for s in primary.sizes]
    def curve(campaign,title,key,ylabel,exponent,scale=1):
        measured=campaign.series('Mesure','scaling_cfl',value(key,scale))
        return dict(title=title,ylabel=ylabel,x=x,ticks=x,xlabel='Elements par direction N',
                    series=[measured,
                            dict(label=f'Modele N^{exponent}',values=[measured['values'][0]*(n/x[0])**exponent for n in x])])
    result=[curve(primary,'Nombre de pas sous CFL','steps','Pas de temps',1),
            curve(primary,'Volume d un champ float32','nodes','Volume (Kio)',3,4/1024)]
    for backend in ['cpu','gpu']:
        campaign=campaigns.get(backend)
        if campaign and campaign.has('scaling_cfl'):
            result.append(curve(campaign,'Propagation sous CFL — '+backend.upper(),'compute_s','Temps de propagation (ms)',4,1000))
    return result


def compare_backends(campaigns):
    if set(campaigns) != {'cpu','gpu'}: return []
    cpu,gpu=campaigns['cpu'],campaigns['gpu']
    result=[]
    for mode,title,key in [('baseline','Propagation de l onde, hors sorties','compute_s'),
            ('visual_full','Workflow complet avec visualisation','workflow_s')]:
        if cpu.has(mode) and gpu.has(mode):
            item=panel(cpu,title,'Temps (s)', [(f"CPU OpenMP, {cpu.rows[0]['threads']} threads",mode,value(key))])
            item['series'].append(gpu.series('GPU CUDA',mode,value(key)))
            if mode == 'baseline':
                item['log_y'] = True
                item['title'] += ' (echelle log)'
            result.append(item)
    return result


def cleanup_legacy(output):
    for name in ['cpu','gpu','cpu_vs_gpu']:
        folder=output/name
        manifest=folder/'figures.json'
        if not manifest.exists(): continue
        for filename in json.loads(manifest.read_text()).get('figures',[]):
            if Path(filename).name == filename: (folder/filename).unlink(missing_ok=True)
        (folder/'summary.csv').unlink(missing_ok=True)
        manifest.unlink()
        if not any(folder.iterdir()): folder.rmdir()


def write_summary(output,campaigns):
    rows=[]
    for campaign in campaigns.values():
        for (size,mode),group in sorted(campaign.groups.items()):
            row=dict(backend=campaign.backend,size=size,mode=mode,repetitions=len(group))
            for key in group[0]:
                if key.endswith(('_s','_bytes')):
                    values=[float(r[key]) for r in group]
                    row.update({key+'_mean':statistics.mean(values),key+'_median':statistics.median(values),
                                key+'_std':statistics.stdev(values) if len(values)>1 else 0})
            rows.append(row)
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path)
    source.add_argument('--compare',type=Path,nargs=2,metavar=('CPU_CSV','GPU_CSV'))
    parser.add_argument('--output',type=Path,default=HERE/'plots')
    parser.add_argument('--figures',help='Noms separes par des virgules : '+','.join(FIGURES))
    args=parser.parse_args()
    paths=list(args.compare or [args.input])
    if args.input and args.input.stem.endswith(('_cpu','_gpu')):
        backend=args.input.stem.rsplit('_',1)[1]
        sibling=args.input.with_name(args.input.stem.rsplit('_',1)[0]+('_gpu.csv' if backend=='cpu' else '_cpu.csv'))
        if sibling.exists(): paths.append(sibling)
    campaigns={}
    for path in paths:
        campaign=Campaign(path)
        if campaign.backend in campaigns: parser.error('Deux campagnes du meme backend')
        campaigns[campaign.backend]=campaign
    if len({c.identity for c in campaigns.values()}) != 1: parser.error('Parametres, sources ou machines differents')
    if len({tuple(c.sizes) for c in campaigns.values()}) != 1: parser.error('Tailles CPU et GPU differentes')
    primary=campaigns.get('gpu',campaigns.get('cpu'))
    selected={name.removesuffix('.png') for name in args.figures.split(',')} if args.figures else set(FIGURES)
    if selected-set(FIGURES): parser.error('Figures inconnues : '+','.join(sorted(selected-set(FIGURES))))
    args.output.mkdir(parents=True,exist_ok=True)
    definitions={
        'snapshots':('Snapshots : min, max, moyenne et variance — ad-hoc / in-situ',lambda:workflow(primary,'snapshots','snapshot_adhoc','snapshot_insitu_gpu')),
        'sismogrammes':('Sismogrammes : min, max, moyenne et variance — ad-hoc / in-situ',lambda:workflow(primary,'sismos','sismo_adhoc','sismo_insitu')),
        'fft':('Sismogrammes : min, max, moyenne, variance et FFT — ad-hoc / in-situ',lambda:fft(primary)),
        'slices':('Meme visualisation : volume complet ou slice',lambda:slices(primary)),
        'statistiques_cpu_gpu':('Simulation CUDA : min, max, moyenne et variance sur GPU ou CPU',lambda:host_device(primary)),
        'precision':('Effet de l intervalle entre snapshots',lambda:precision(primary)),
        'scalabilite':('Scalabilite a ordre fixe : mesures et modeles',lambda:scalability(campaigns,primary)),
        'cpu_vs_gpu':('CPU / GPU : propagation de l onde et workflow avec visualisation',lambda:compare_backends(campaigns))}
    generated=[]
    profile=primary.identity[-1]+' '+primary.backend.upper()
    for name in FIGURES:
        if name not in selected: continue
        title,builder=definitions[name]
        panels=builder()
        if not panels: continue
        figure_profile = primary.identity[-1]+' CPU / GPU' if name in ('scalabilite','cpu_vs_gpu') and len(campaigns)==2 else profile
        draw(args.output,name,title+' — '+figure_profile,panels)
        generated.append(name+'.png')
    if not generated: parser.error('Aucune figure disponible pour les modes mesures')
    write_summary(args.output,campaigns)
    manifest_path=args.output/'figures.json'
    manifest=json.loads(manifest_path.read_text()) if args.figures and manifest_path.exists() else {}
    sources=manifest.get('sources',{})
    sources.update({name:[str(c.path) for c in campaigns.values()] for name in generated})
    manifest.update(figures=sorted(sources),sources=sources,primary_backend=primary.backend,
                    machine=primary.identity[0],profile=primary.identity[-1])
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if not args.figures: cleanup_legacy(args.output)
    print(f'{len(generated)} figures regroupees : {args.output}')


if __name__=='__main__':
    main()
