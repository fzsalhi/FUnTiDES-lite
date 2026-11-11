import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from collections import defaultdict

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'tp1'/'benchmarks'))
import plot_benchmarks as charts


def compare_backends(paths, output):
    datasets = {}
    identities = []
    for path in paths:
        with path.open(newline='') as stream: rows = list(csv.DictReader(stream))
        if not rows: raise ValueError('Aucune mesure : ' + str(path))
        identity = {(r['machine'], r['source_hash'], r['config_hash'], r['threads'], r['io_sync']) for r in rows}
        if len(identity) != 1: raise ValueError('Mesures incompatibles : ' + str(path))
        identities.append(next(iter(identity)))
        backends = {r['backend'] for r in rows}
        if len(backends) != 1: raise ValueError('Plusieurs backends dans une campagne')
        datasets[next(iter(backends))] = rows
    if identities[0] != identities[1] or set(datasets) != {'cpu', 'gpu'}:
        raise ValueError('La comparaison exige CPU et GPU sur la meme machine avec les memes parametres et sources')
    output.mkdir(parents=True, exist_ok=True)
    charts.PLOTS_DIR = output
    sizes = sorted({r['size'] for r in datasets['cpu']}, key=lambda size: next(int(r['nodes']) for r in datasets['cpu'] if r['size']==size))
    figures = []
    for mode, title, key in [('baseline', 'Calcul de la simulation', 'compute_s'),
            ('snapshot_adhoc', 'Snapshots : workflow ad-hoc', 'workflow_s'),
            ('snapshot_insitu_gpu', 'Snapshots : workflow in-situ', 'workflow_s'),
            ('sismo_insitu', 'Sismogrammes : workflow in-situ', 'workflow_s'),
            ('visual_full', 'Workflow complet avec visualisation du volume', 'workflow_s'),
            ('visual_slice', 'Workflow complet avec visualisation des slices', 'workflow_s')]:
        series, errors = [], []
        for backend, rows in datasets.items():
            values = [[float(r[key]) for r in rows if r['size']==size and r['mode']==mode] for size in sizes]
            if any(not v for v in values): break
            series.append((backend.upper(), [statistics.mean(v) for v in values]))
            errors.append([statistics.stdev(v) if len(v)>1 else 0 for v in values])
        if len(series) != 2: continue
        name = mode + '.png'
        charts.plot_lines(name, title + ' CPU / GPU', 'Temps (s)', range(1,len(sizes)+1), series,
                          errors=errors, x_tick_labels=sizes, x_label='Configuration')
        figures.append(name)
    (output/'figures.json').write_text(json.dumps(dict(sources=[str(p) for p in paths], figures=figures), indent=2)+'\n')
    print(f'{len(figures)} comparaisons CPU / GPU : {output}')


def main():
    parser=argparse.ArgumentParser(description='TP2 : comparaisons de workflows a deux courbes')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--compare',type=Path,nargs=2,metavar=('CPU_CSV','GPU_CSV'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--figures',help='Noms des figures sans extension, separes par des virgules')
    args=parser.parse_args()
    if args.compare:
        compare_backends(args.compare, args.output or HERE/'plots'/'cpu_vs_gpu')
        return
    if not args.input: parser.error('--input ou --compare requis')
    with args.input.open(newline='') as stream: rows=list(csv.DictReader(stream))
    if not rows: parser.error('Aucune mesure')
    identities={(r['machine'],r['source_hash'],r['config_hash'],r['backend'],r['threads'],r['io_sync']) for r in rows}
    if len(identities)!=1: parser.error('Mesures incompatibles')
    if args.output is None: args.output = HERE/'plots'/rows[0]['backend']
    selected = {v.removesuffix('.png') for v in args.figures.split(',')} if args.figures else None
    if not selected:
        manifest = args.output/'figures.json'
        if manifest.exists():
            for name in json.loads(manifest.read_text()).get('figures', []):
                if Path(name).name == name: (args.output/name).unlink(missing_ok=True)
    charts.PLOTS_DIR=args.output
    args.output.mkdir(parents=True,exist_ok=True)
    sizes=sorted({r['size'] for r in rows},key=lambda s:next(int(r['nodes']) for r in rows if r['size']==s))
    groups=defaultdict(list)
    for r in rows: groups[(r['size'],r['mode'])].append(r)
    profile=rows[0]['profile']+' '+rows[0]['backend'].upper()
    summary=[]
    metrics=['workflow_s','simulation_wall_s','compute_s','total_output_bytes','adhoc_read_s','adhoc_analysis_s','stats_compute_s','stats_copy_s','stats_write_s','sismo_stats_s','sismo_stats_copy_s','sismo_stats_write_s','snapshot_write_s','sismo_write_s','slice_extract_s','slice_copy_s','slice_write_s','render_s','stats_transfer_bytes','receiver_trace_bytes','receiver_stats_memory_bytes']
    for (size,mode),repetitions in sorted(groups.items()):
        item=dict(size=size,mode=mode,repetitions=len(repetitions))
        for key in metrics:
            values=[float(r[key]) for r in repetitions]
            item.update({key+'_mean':statistics.mean(values),key+'_median':statistics.median(values),
                         key+'_std':statistics.stdev(values) if len(values)>1 else 0})
        summary.append(item)
    with (args.output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(summary[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(summary)
    plotted=[]
    def line(name,title,ylabel,curves,zero_based=True):
        if selected is not None and name.removesuffix('.png') not in selected: return
        series,errors=[],[]
        for label,mode,extract in curves:
            means,stds=[],[]
            for size in sizes:
                source=groups.get((size,mode))
                if not source: return
                numbers=[extract(r) for r in source]
                means.append(statistics.mean(numbers)); stds.append(statistics.stdev(numbers) if len(numbers)>1 else 0)
            series.append((label,means)); errors.append(stds)
        charts.plot_lines(name,title+' — '+profile,ylabel,range(1,len(sizes)+1),series,
            errors=errors,x_tick_labels=sizes,x_label='Configuration',zero_based=zero_based)
        plotted.append(name)
    f=lambda key:lambda r:float(r[key])
    mb=lambda r:float(r.get('workflow_output_bytes', r['total_output_bytes']))/2**20
    for data,adhoc,insitu in [('snapshots','snapshot_adhoc','snapshot_insitu_gpu'),('sismos','sismo_adhoc','sismo_insitu')]:
        line(data+'_workflow.png','Workflow numerique '+data,'Temps (s)', [('Ad-hoc binaire',adhoc,f('workflow_s')),('In-situ',insitu,f('workflow_s'))])
        line(data+'_storage.png','Volume des sorties '+data,'Volume (Mio)', [('Ad-hoc binaire',adhoc,mb),('In-situ',insitu,mb)])
        line(data+'_analysis.png','Calcul des analyses '+data,'Temps (s)', [('Ad-hoc CPU',adhoc,f('adhoc_analysis_s')),('In-situ',insitu,f('stats_compute_s' if data=='snapshots' else 'sismo_stats_s'))])
        line(data+'_io.png','Ecritures lectures et copies '+data,'Temps (s)',
            [('Donnees brutes',adhoc,lambda r:float(r['snapshot_write_s'])+float(r['sismo_write_s'])+float(r['adhoc_read_s'])),
             ('Resultats in-situ',insitu,lambda r:float(r['stats_copy_s'])+float(r['stats_write_s'])+float(r['sismo_stats_copy_s'])+float(r['sismo_stats_write_s']))])
        baseline={(r['size'],r['run']):r for r in rows if r['mode']=='baseline'}
        if all((r['size'],r['run']) in baseline for r in rows if r['mode'] in (adhoc,insitu)):
            line(data+'_overhead.png','Surcout du traitement '+data,'Temps ajoute au workflow (s)',
                 [('Ad-hoc binaire',adhoc,lambda r:float(r['workflow_s'])-float(baseline[(r['size'],r['run'])]['workflow_s'])),
                  ('In-situ',insitu,lambda r:float(r['workflow_s'])-float(baseline[(r['size'],r['run'])]['workflow_s']))],zero_based=False)
    line('stats_gpu_vs_hostcopy.png','Analyse sur device ou copie puis analyse CPU','Temps (s)',
        [('Reduction device','snapshot_insitu_gpu',lambda r:float(r['stats_compute_s'])+float(r['stats_copy_s'])),
         ('Copie + Welford CPU','snapshot_insitu_hostcopy',lambda r:float(r['stats_compute_s'])+float(r['stats_copy_s']))])
    line('stats_gpu_vs_uvm_cpu.png','Analyse device ou parcours CPU UVM','Temps (s)',
        [('Reduction device','snapshot_insitu_gpu',lambda r:float(r['stats_compute_s'])+float(r['stats_copy_s'])),
         ('Parcours CPU UVM','snapshot_insitu_cpu',f('stats_compute_s'))])
    line('stats_hostcopy_workflow.png','Impact du mode d analyse sur la simulation','Workflow (s)',
        [('Reduction device','snapshot_insitu_gpu',f('workflow_s')),('Copie + Welford CPU','snapshot_insitu_hostcopy',f('workflow_s'))])
    line('stats_transfer_bytes.png','Donnees logiques renvoyees vers l hote','Octets',
        [('Resultat de reduction','snapshot_insitu_gpu',f('stats_transfer_bytes')),('Champ complet','snapshot_insitu_hostcopy',f('stats_transfer_bytes'))])
    line('sismo_fft_workflow.png','FFT des sismogrammes : workflow complet','Temps (s)',
         [('Binaire puis NumPy CPU','sismo_fft_adhoc',f('workflow_s')),('In-situ cuFFT','sismo_fft_insitu',f('workflow_s'))])
    line('sismo_fft_compute.png','FFT seule : bibliotheques et precisions differentes','Calcul FFT (s)',
         [('NumPy CPU','sismo_fft_adhoc',f('adhoc_fft_compute_s')),('cuFFT GPU','sismo_fft_insitu',f('fft_compute_s'))])
    line('sismo_fft_setup.png','FFT : preparation et cout complet','Temps (s)',
         [('Allocation + plan cuFFT','sismo_fft_insitu',f('fft_plan_s')),('Traitement cuFFT complet','sismo_fft_insitu',f('fft_total_s'))])
    line('sismo_fft_storage.png','FFT : volume total du workflow','Volume (Mio)',
         [('Donnees brutes + resultats','sismo_fft_adhoc',mb),('Resultats in-situ','sismo_fft_insitu',mb)])
    for metric,name,label in [('workflow_s','slice_workflow.png','Temps total (s)'),('total_output_bytes','slice_storage.png','Volume (Mio)'),('render_s','slice_render.png','Rendu (s)'),('adhoc_read_s','slice_read.png','Lecture (s)')]:
        extract=mb if metric=='total_output_bytes' else f(metric)
        line(name,'Visualisation depuis volume ou slice',label,[('Volume binaire','visual_full',extract),('Slice binaire','visual_slice',extract)])
    full={(r['size'],r['run']):r for r in rows if r['mode']=='visual_full'}
    if all((r['size'],r['run']) in full for r in rows if r['mode']=='visual_slice'):
        line('slice_extraction_vs_write_gain.png','Cout de la slice et economie d ecriture','Temps (s)',
            [('Selection + extraction + copie','visual_slice',lambda r:float(r['slice_setup_s'])+float(r['slice_extract_s'])+float(r['slice_copy_s'])),
             ('Ecriture economisee','visual_slice',lambda r:float(full[(r['size'],r['run'])]['snapshot_write_s'])-float(r['slice_write_s']))],zero_based=False)
    precision=args.input.with_name(args.input.stem+'_precision.csv')
    if precision.exists():
        with precision.open(newline='') as stream: prec=list(csv.DictReader(stream))
        for size in sizes:
            if selected is not None and 'precision_'+size not in selected: continue
            values=[r for r in prec if r['size']==size]
            if not values: continue
            x=[int(r['interval']) for r in values]
            charts.plot_lines('precision_'+size+'.png','Erreur d echantillonnage — '+size,'Erreur / max |p|',x,
                [('Maximum',[float(r['peak_error_norm']) for r in values]),('Moyenne temporelle',[float(r['mean_error_norm']) for r in values])],x_label='Intervalle entre snapshots (pas)')
            plotted.append('precision_'+size+'.png')
    line('scalability_steps.png','Nombre de pas sous condition CFL','Pas de temps',
        [('Mesure','scaling_cfl',f('steps')),
         ('Modele proportionnel a N','scaling_cfl',lambda r:statistics.mean(float(a['steps']) for a in groups[(sizes[0],'scaling_cfl')])*float(r['elements_per_axis'])/float(groups[(sizes[0],'scaling_cfl')][0]['elements_per_axis']))])
    line('scalability_compute.png','Calcul sous condition CFL : mesure et ordre N puissance 4','Temps (s)',
        [('Mesure','scaling_cfl',f('compute_s')),
         ('Modele N puissance 4','scaling_cfl',lambda r:statistics.mean(float(a['compute_s']) for a in groups[(sizes[0],'scaling_cfl')])*(float(r['elements_per_axis'])/float(groups[(sizes[0],'scaling_cfl')][0]['elements_per_axis']))**4)])
    validation=args.input.with_name(args.input.stem+'_validation.csv')
    if validation.exists():
        with validation.open(newline='') as stream: valid=list(csv.DictReader(stream))
        for data in ('snapshots','sismos'):
            if selected is not None and data+'_validation' not in selected: continue
            series=[]
            for metric,label in [('max','Maximum'),('mean','Moyenne')]:
                values=[next(float(r['max_error_norm']) for r in valid if r['size']==size and r['data']==data and r['statistic']==metric) for size in sizes]
                series.append((label,values))
            name=data+'_validation.png'
            charts.plot_lines(name,'Ecart numerique ad-hoc / in-situ — '+data+' — '+profile,'Erreur normalisee',range(1,len(sizes)+1),series,x_tick_labels=sizes,x_label='Configuration')
            plotted.append(name)
        if any(r['data']=='fft' for r in valid) and (selected is None or 'fft_validation' in selected):
            series=[]
            for metric,label in [('amplitude','Amplitude'),('frequency_hz','Frequences')]:
                series.append((label,[next(float(r['max_error_norm']) for r in valid if r['size']==size and r['data']=='fft' and r['statistic']==metric) for size in sizes]))
            charts.plot_lines('fft_validation.png','Ecart NumPy CPU / cuFFT GPU','Erreur normalisee',range(1,len(sizes)+1),series,x_tick_labels=sizes,x_label='Configuration')
            plotted.append('fft_validation.png')
    if selected is not None and not plotted:
        parser.error('Aucune figure demandee ne correspond aux modes mesures')
    manifest = args.output/'figures.json'
    sources = {}
    if selected is not None and manifest.exists():
        previous = json.loads(manifest.read_text())
        sources.update(previous.get('sources', {name: previous.get('source') for name in previous.get('figures', [])}))
    sources.update({name: str(args.input) for name in plotted})
    manifest.write_text(json.dumps(dict(source=str(args.input),profile=profile,figures=sorted(sources),sources=sources),indent=2)+'\n')
    print(f'{len(plotted)} figures : {args.output}')


if __name__=='__main__': main()
