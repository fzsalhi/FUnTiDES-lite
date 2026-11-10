#!/usr/bin/env python3
"""Compare les resultats ad-hoc sur snapshots reconstruits a ceux des originaux.

Exemple :
  python compare_quant.py --ref results --runs results_f16 results_int16 results_int8
Chaque dossier doit contenir snapshots_stats.csv et wavefront.json
(produits par analyze_snapshots.py / analyse_snapshots_cpm.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(d):
    d = Path(d)
    s = np.genfromtxt(d / "snapshots_stats.csv", delimiter=",", names=True)
    w = json.loads((d / "wavefront.json").read_text())
    return s, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="dossier de resultats sur les donnees originales")
    ap.add_argument("--runs", nargs="+", required=True, help="dossiers de resultats sur donnees reconstruites")
    a = ap.parse_args()

    s0, w0 = load(a.ref)
    amax = max(np.abs(s0["max"]).max(), np.abs(s0["min"]).max())
    e0 = s0["energy"]
    v0 = w0["fit"].get("velocity")
    print(f"ref = {a.ref} : v_front = {v0}, max|p| = {amax:.4g}\n")
    hdr = f"{'run':<16}{'err max/min':>13}{'err energie':>13}{'err r_front':>13}{'v_front':>11}{'dv/v (%)':>10}{'R2':>9}"
    print(hdr)
    print("-" * len(hdr))
    for d in a.runs:
        s, w = load(d)
        n = min(len(s0), len(s))
        e_max = max(np.abs(s["max"][:n] - s0["max"][:n]).max(),
                    np.abs(s["min"][:n] - s0["min"][:n]).max()) / amax
        ok = e0[:n] > 1e-12 * e0.max()
        e_en = (np.abs(s["energy"][:n] - e0[:n])[ok] / e0[:n][ok]).max() if ok.any() else float("nan")
        e_rf = np.abs(s["r_front"][:n] - s0["r_front"][:n]).max()
        v = w["fit"].get("velocity")
        dv = 100 * (v - v0) / v0 if (v is not None and v0) else float("nan")
        r2 = w["fit"].get("r2", float("nan"))
        print(f"{Path(d).name:<16}{e_max:>13.3e}{e_en:>13.3e}{e_rf:>13.3g}{v if v is not None else float('nan'):>11.2f}{dv:>10.3f}{r2:>9.4f}")
    print("\nerr max/min : erreur max sur max(p) et min(p), normalisee par max|p|")
    print("err energie : erreur relative max sur sum p^2 (snapshots non nuls)")
    print("err r_front : ecart max (unites du maillage) du rayon du front")


if __name__ == "__main__":
    main()