#!/usr/bin/env python3
"""Analyse AD-HOC des snapshots (lecture fichiers -> stats, histogramme, front d'onde).

Sorties dans --out :
  snapshots_stats.csv   stats par snapshot (min,max,mean,var,std,energie,quantiles,argmax)
  histograms.npz        edges, counts_per_snapshot, counts_global
  wavefront.json        rayon du front vs temps + vitesse estimée (régression linéaire)
  timings_snapshots.json  temps lecture / calcul (à comparer à l'in-situ)

Exemple :
  python analyse_snapshots.py --snap-dir run/snapshots --coords run/coords.bin \
      --source 500 500 500 --dt 0.001 --out results --tag S
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from io_utils import Timer, dump_json, list_snapshots, load_coords, load_snapshot, load_snapshot_and_coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", required=True)
    ap.add_argument("--pattern", default="snapshot_*")
    ap.add_argument("--coords", default=None, help="optionnel si les CSV de snapshots contiennent x,y,z,p")
    ap.add_argument("--col", default=None, help="colonne de pression dans les CSV (nom ou indice ; défaut = dernière)")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--source", type=float, nargs=3, default=None, help="x y z (défaut: centre du domaine)")
    ap.add_argument("--dt", type=float, required=True, help="pas de temps de la simulation (le temps = n° de pas * dt)")
    ap.add_argument("--nbins", type=int, default=100)
    ap.add_argument("--front-frac", type=float, default=0.01, help="seuil du front : |p| > frac * max|p| du snapshot")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="", help="étiquette de config (S, M, L...)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = Timer()

    snaps = list_snapshots(a.snap_dir, a.pattern)
    with T("read_coords"):
        if a.coords:
            coords = load_coords(a.coords, a.dtype)
        else:
            _, coords = load_snapshot_and_coords(snaps[0][1], a.dtype, a.col)
            if coords is None:
                raise SystemExit("Pas de coordonnées : donne --coords ou un CSV de snapshot avec colonnes x,y,z,p")
    src = np.array(a.source) if a.source else 0.5 * (coords.min(0) + coords.max(0))
    dist = np.linalg.norm(coords - src, axis=1)
    r_dom = min((coords.max(0) - src).min(), (src - coords.min(0)).min())  # distance à la frontière la + proche

    # ---- passe 1 : stats + bornes globales ----
    rows, gmin, gmax = [], np.inf, -np.inf
    for step, path in snaps:
        with T("read_pass1"):
            p = load_snapshot(path, a.dtype, a.col)
        assert p.size == coords.shape[0], f"{path}: {p.size} valeurs != {coords.shape[0]} noeuds"
        with T("compute_pass1"):
            p64 = p.astype(np.float64)
            ap_ = np.abs(p64)
            imax = int(ap_.argmax())
            mx = ap_[imax]
            mask = ap_ > a.front_frac * mx if mx > 0 else np.zeros(p.size, bool)
            q50, q99 = np.quantile(ap_, [0.5, 0.99])
            rows.append(dict(
                step=step, t=step * a.dt, min=p64.min(), max=p64.max(), mean=p64.mean(),
                var=p64.var(), std=p64.std(), energy=float(np.dot(p64, p64)),
                absq50=q50, absq99=q99, argmax_x=coords[imax, 0], argmax_y=coords[imax, 1],
                argmax_z=coords[imax, 2], r_peak=dist[imax],
                r_front=dist[mask].max() if mask.any() else 0.0))
            gmin, gmax = min(gmin, rows[-1]["min"]), max(gmax, rows[-1]["max"])

    # ---- passe 2 : histogramme (les bornes doivent être connues -> 2e lecture) ----
    edges = np.linspace(gmin, gmax, a.nbins + 1)
    counts = np.zeros((len(snaps), a.nbins), dtype=np.int64)
    for i, (_, path) in enumerate(snaps):
        with T("read_pass2"):
            p = load_snapshot(path, a.dtype, a.col)
        with T("compute_pass2"):
            counts[i], _ = np.histogram(p, bins=edges)
    np.savez(out / "histograms.npz", edges=edges, counts_per_snapshot=counts,
             counts_global=counts.sum(0)/len(snaps), t=[r["t"] for r in rows])

    # ---- front d'onde -> vitesse (on évite les réflexions sur les bords) ----
    with T("wavefront_fit"):
        t = np.array([r["t"] for r in rows])
        rf = np.array([r["r_front"] for r in rows])
        sel = (rf > 0) & (rf < 0.8 * r_dom)
        vel = {"n_points_fit": int(sel.sum())}
        if sel.sum() >= 3:
            slope, icpt = np.polyfit(t[sel], rf[sel], 1)
            pred = slope * t[sel] + icpt
            ss_res, ss_tot = ((rf[sel] - pred) ** 2).sum(), ((rf[sel] - rf[sel].mean()) ** 2).sum()
            vel.update(velocity=slope, intercept=icpt, r2=1 - ss_res / ss_tot if ss_tot > 0 else 1.0)
    dump_json(out / "wavefront.json", dict(source=src.tolist(), r_domain=r_dom, front_frac=a.front_frac,
                                           t=t.tolist(), r_front=rf.tolist(), fit=vel))

    with open(out / "snapshots_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tot_read = sum(v for k, v in T.total.items() if k.startswith("read"))
    tot_comp = sum(v for k, v in T.total.items() if not k.startswith("read"))
    dump_json(out / "timings_snapshots.json", dict(
        tag=a.tag, n_snapshots=len(snaps), n_nodes=int(coords.shape[0]),
        bytes_read=sum(Path(p).stat().st_size for _, p in snaps) * 2,
        read_s=tot_read, compute_s=tot_comp, total_s=tot_read + tot_comp, detail=T.summary()))
    print(f"[snapshots] {len(snaps)} snapshots, lecture {tot_read:.3f}s, calcul {tot_comp:.3f}s")
    if "velocity" in vel:
        print(f"[snapshots] vitesse du front ≈ {vel['velocity']:.2f} (unités/s), R²={vel['r2']:.4f}")


if __name__ == "__main__":
    main()