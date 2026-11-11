#!/usr/bin/env python3
"""Analyse AD-HOC des sismogrammes : stats, FFT, temps d'arrivée, vitesse de propagation.

Exemple :
  python analyse_sismos.py --sismos run/sismos.csv --receivers run/receivers.txt \
      --source 500 500 500 --out results --tag S
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from io_utils import Timer, dump_json, load_receivers, load_sismos


def first_arrival(t, s, frac):
    """Premier instant où |s| dépasse frac*max|s| (interpolation linéaire)."""
    a = np.abs(s)
    thr = frac * a.max()
    idx = int(np.argmax(a > thr))
    if idx == 0:
        return t[0]
    return t[idx - 1] + (thr - a[idx - 1]) / (a[idx] - a[idx - 1]) * (t[idx] - t[idx - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sismos", required=True)
    ap.add_argument("--dt", type=float, default=None, help="requis si le fichier ne contient pas le temps")
    ap.add_argument("--receivers", required=True, help="fichier x y z par ligne (même ordre que les sismos)")
    ap.add_argument("--source", type=float, nargs=3, default=None)
    ap.add_argument("--pick-frac", type=float, default=0.05, help="seuil de pointé (fraction du max|s|)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = Timer()

    with T("read"):
        t, data = load_sismos(a.sismos, a.dt)
        rec = load_receivers(a.receivers)
    nrec, nt = data.shape
    assert rec.shape[0] == nrec, f"{rec.shape[0]} récepteurs dans le fichier de positions, {nrec} sismos"
    dt = float(t[1] - t[0])
    source = (
        np.array(a.source)
        if a.source is not None
        else 0.5 * (rec.min(axis=0) + rec.max(axis=0))
    )
    dist = np.linalg.norm(rec - source, axis=1)
    stats_rows, arr_rows, freqs = [], [], np.fft.rfftfreq(nt, dt)
    spectra = np.zeros((nrec, freqs.size))
    with T("compute"):
        for i in range(nrec):
            s = data[i].astype(np.float64)
            amp = np.abs(np.fft.rfft(s - s.mean())) * dt
            spectra[i] = amp
            ipk = int(np.abs(s).argmax())
            stats_rows.append(dict(
                receiver=i, distance=dist[i], min=s.min(), max=s.max(), mean=s.mean(), var=s.var(),
                rms=np.sqrt(np.mean(s ** 2)), energy=float(np.dot(s, s)) * dt, t_peak=t[ipk],
                f_dominant=freqs[int(amp.argmax())],
                f_centroid=float((freqs * amp).sum() / amp.sum()) if amp.sum() > 0 else 0.0))
            arr_rows.append(dict(receiver=i, distance=dist[i], t_arrival=first_arrival(t, s, a.pick_frac)))

        # vitesse : t_arr = t0 + d / v  (t0 = retard de la source, ex. ricker décalé)
        d = np.array([r["distance"] for r in arr_rows])
        ta = np.array([r["t_arrival"] for r in arr_rows])
        vel = {"pick_frac": a.pick_frac, "n_receivers": int(nrec)}
        if nrec >= 2 and np.ptp(d) > 0:
            slope, t0 = np.polyfit(d, ta, 1)
            pred = slope * d + t0
            ss_tot = ((ta - ta.mean()) ** 2).sum()
            vel.update(velocity=1.0 / slope, t0=t0, r2=1 - ((ta - pred) ** 2).sum() / ss_tot if ss_tot > 0 else 1.0)

    for fn, rows in (("sismos_stats.csv", stats_rows), ("sismos_arrivals.csv", arr_rows)):
        with open(out / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    np.savez(out / "sismos_spectra.npz", freqs=freqs, spectra=spectra, t=t, data=data, distance=dist)
    dump_json(out / "velocity_sismos.json", vel)
    dump_json(out / "timings_sismos.json", dict(
        tag=a.tag, n_receivers=nrec, n_samples=nt, bytes_read=Path(a.sismos).stat().st_size if Path(a.sismos).is_file() else None,
        read_s=T.total["read"], compute_s=T.total["compute"], total_s=T.total["read"] + T.total["compute"]))
    print(f"[sismos] {nrec} récepteurs x {nt} échantillons, lecture {T.total['read']:.4f}s, calcul {T.total['compute']:.4f}s")
    if "velocity" in vel:
        print(f"[sismos] vitesse ≈ {vel['velocity']:.2f} (unités/s), t0={vel['t0']:.4f}s, R²={vel['r2']:.4f}")


if __name__ == "__main__":
    main()