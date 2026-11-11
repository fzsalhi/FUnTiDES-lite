import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import numpy as np


def fit_power(x, y):
    x, y = np.asarray(x,dtype=float), np.asarray(y,dtype=float)
    mask = (x>0)&(y>0)
    if mask.sum()<2: return None
    return float(np.polyfit(np.log(x[mask]),np.log(y[mask]),1)[0])


def model(raw, target_nodes=19_000_000_000):
    with Path(raw).open(newline="") as stream: rows=list(csv.DictReader(stream))
    sizes = sorted({r['size'] for r in rows},key=lambda s:next(int(r['elements_per_axis']) for r in rows if r['size']==s))
    points=[]
    for size in sizes:
        cfl=[r for r in rows if r['size']==size and r['mode']=='scaling_cfl']
        adhoc=[r for r in rows if r['size']==size and r['mode']=='snapshot_adhoc']
        if not cfl or not adhoc: continue
        med=lambda data,key:statistics.median(float(r[key]) for r in data)
        points.append(dict(size=size,elements=med(cfl,'elements_per_axis'),nodes=med(cfl,'nodes'),
            dt_s=med(cfl,'dt_s'),steps=med(cfl,'steps'),compute_s=med(cfl,'compute_s'),
            one_pressure_field_bytes=4*med(cfl,'nodes'),output_bytes=med(adhoc,'snapshot_bytes'),
            snapshot_count=med(adhoc,'snapshot_count')))
    result=dict(theory=dict(snapshot_exponent=3,cfl_steps_exponent=1,cfl_compute_exponent=4,
                output_fixed_step_interval_exponent=4,output_fixed_physical_interval_exponent=3,slice_exponent=2),
                measurements=points,target_nodes=target_nodes,pressure_bytes=4*target_nodes,
                coordinates_once_bytes=12*target_nodes,pressure_buffers_bytes=8*target_nodes,
                slice_pressure_bytes=4*target_nodes**(2/3),
                projection_note="Modele a ordre fixe ; champ et coordonnees float32. Les debits buffered ne sont pas des debits physiques du disque.")
    if len(points)>=2:
        x=[p['elements'] for p in points]
        result['measured_exponents']={k:fit_power(x,[p[k] for p in points]) for k in ['one_pressure_field_bytes','steps','compute_s']}
        largest=points[-1]
        order=next(int(r['order']) for r in rows if 'order' in r) if 'order' in rows[0] else 2
        target_elements=(target_nodes**(1/3)-1)/order
        predicted_steps=math.ceil(largest['steps']*target_elements/largest['elements'])
        meta=json.loads(Path(raw).with_suffix('.json').read_text())
        interval=meta['config']['snapshot_interval']
        time_max=meta['config']['common']['timemax']
        tau=interval*meta['config']['common']['dt']
        k_steps=math.ceil(predicted_steps/interval)
        k_physical=math.ceil(time_max/tau)
        result.update(target_elements=target_elements,predicted_cfl_steps=predicted_steps,
            fixed_step_interval=interval,fixed_physical_interval_s=tau,
            predicted_snapshots_fixed_steps=k_steps,predicted_snapshots_fixed_physical=k_physical,
            bytes_fixed_step_interval=target_nodes*(12+4*k_steps),bytes_fixed_physical_interval=target_nodes*(12+4*k_physical))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    Path(args.out).write_text(json.dumps(model(args.raw),indent=2,allow_nan=False)+'\n')
    print(args.out)


if __name__=='__main__': main()
