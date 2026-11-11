#!/usr/bin/env python3
"""Figures de l'analyse AD-HOC (partie A) à partir des sorties de analyze_snapshots.py / analyze_sismos.py.

Un ou plusieurs dossiers de résultats (ex. S, M, L) : les courbes sont superposées,
les temps sont comparés dans un diagramme en barres (lecture vs calcul).

Figures produites dans --out :
  snapshots_evolution.png   énergie, max/min, r_front en fonction du temps
  histogram.png             histogramme global (échelle log)
  wavefront.png             r_front(t) + droite de régression -> vitesse
  sismos_traces.png         sismogrammes triés par distance à la source
  sismos_spectra.png        spectres d'amplitude (|FFT|) de quelques récepteurs
  sismos_arrivals.png       t_arrivée = f(distance) + régression -> vitesse
  sismos_freq_vs_dist.png   fréquence dominante / centroïde vs distance
  timings_adhoc.png         temps lecture / calcul (snapshots et sismos)

Exemple :
  python plot_results.py --results results_S results_M results_L --out figures
  python plot_results.py --results results            (un seul run)
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    return np.genfromtxt(path, delimiter=",", names=True)


def load_run(d):
    """Charge tous les fichiers disponibles d'un dossier de résultats (les absents valent None)."""
    d = Path(d)
    run = dict(
        dir=d,
        snap=read_csv(d / "snapshots_stats.csv"),
        wavefront=read_json(d / "wavefront.json"),
        t_snap=read_json(d / "timings_snapshots.json"),
        sis_stats=read_csv(d / "sismos_stats.csv"),
        arrivals=read_csv(d / "sismos_arrivals.csv"),
        vel_sis=read_json(d / "velocity_sismos.json"),
        t_sis=read_json(d / "timings_sismos.json"),
        hist=None,
        spec=None,
    )
    if (d / "histograms.npz").exists():
        run["hist"] = dict(np.load(d / "histograms.npz"))
    if (d / "sismos_spectra.npz").exists():
        run["spec"] = dict(np.load(d / "sismos_spectra.npz"))
    tag = (run["t_snap"] or {}).get("tag") or (run["t_sis"] or {}).get("tag") or d.name
    run["label"] = tag or d.name
    return run


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print(f"[figures] {name}")


# ----------------------------------------------------------------------------- snapshots
def fig_evolution(runs, out):
    runs = [r for r in runs if r["snap"] is not None]
    if not runs:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in runs:
        s = r["snap"]
        ax[0].semilogy(s["t"], np.maximum(s["energy"], 1e-300), label=r["label"])
        ax[1].plot(s["t"], s["max"], label=f"max {r['label']}")
        ax[1].plot(s["t"], s["min"], "--", label=f"min {r['label']}")
        ax[2].plot(s["t"], s["r_front"], label=r["label"])
    ax[0].set(title="Énergie  Σp²", xlabel="t (s)", ylabel="énergie")
    ax[1].set(title="Min / max du champ", xlabel="t (s)", ylabel="p")
    ax[2].set(title="Rayon du front d'onde", xlabel="t (s)", ylabel="r_front")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    save(fig, out, "snapshots_evolution.png")


def fig_histogram(runs, out):
    runs = [r for r in runs if r["hist"] is not None]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in runs:
        e, c = r["hist"]["edges"], r["hist"]["counts_global"]
        ax.stairs(np.maximum(c, 0.5), e, label=r["label"])
    ax.set(yscale="log", xlabel="p", ylabel="effectif (tous snapshots)", title="Histogramme global de la pression")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, out, "histogram.png")


def fig_wavefront(runs, out):
    runs = [r for r in runs if r["wavefront"] is not None]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, r in enumerate(runs):
        w = r["wavefront"]
        t, rf, fit = np.array(w["t"]), np.array(w["r_front"]), w["fit"]
        c = f"C{k}"
        ax.plot(t, rf, "o", ms=3, color=c, label=f"{r['label']} : r_front")
        if "velocity" in fit:
            tt = np.array([t[rf > 0].min(), t[rf > 0].max()])
            ax.plot(tt, fit["velocity"] * tt + fit["intercept"], "-", color=c,
                    label=f"{r['label']} : v = {fit['velocity']:.1f}  (R²={fit['r2']:.4f})")
        ax.axhline(0.8 * w["r_domain"], color=c, ls=":", lw=0.8)
    ax.set(xlabel="t (s)", ylabel="r_front", title="Front d'onde et vitesse estimée (pointillés : limite 0.8·r_domaine)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, out, "wavefront.png")


# ----------------------------------------------------------------------------- sismos
def fig_traces(runs, out, max_traces=30):
    runs = [r for r in runs if r["spec"] is not None]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6), squeeze=False)
    for ax, r in zip(axes[0], runs):
        sp = r["spec"]
        t, data, dist = sp["t"], sp["data"], sp["distance"]
        order = np.argsort(dist)
        order = order[np.linspace(0, len(order) - 1, min(max_traces, len(order))).astype(int)]
        scale = np.abs(data).max() or 1.0
        spacing = np.ptp(dist[order]) / max(len(order), 1) if np.ptp(dist[order]) > 0 else 1.0
        for i in order:
            ax.plot(t, dist[i] + data[i] / scale * spacing * 1.5, "k", lw=0.6)
        ax.set(title=f"Sismogrammes ({r['label']})", xlabel="t (s)", ylabel="distance à la source")
    save(fig, out, "sismos_traces.png")


def fig_spectra(runs, out, n_show=4):
    runs = [r for r in runs if r["spec"] is not None]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.5), squeeze=False)
    for ax, r in zip(axes[0], runs):
        sp = r["spec"]
        f, S, dist = sp["freqs"], sp["spectra"], sp["distance"]
        order = np.argsort(dist)
        for i in order[np.linspace(0, len(order) - 1, min(n_show, len(order))).astype(int)]:
            ax.plot(f, S[i], label=f"d = {dist[i]:.0f}")
        f_pk = f[int(S.mean(0).argmax())]
        ax.set(title=f"Spectres d'amplitude ({r['label']})", xlabel="f (Hz)", ylabel="|FFT|·dt",
               xlim=(0, max(4 * f_pk, f[1] * 5)))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, out, "sismos_spectra.png")


def fig_arrivals(runs, out):
    runs = [r for r in runs if r["arrivals"] is not None]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, r in enumerate(runs):
        a, v, c = r["arrivals"], r["vel_sis"] or {}, f"C{k}"
        ax.plot(a["distance"], a["t_arrival"], "o", ms=4, color=c, label=f"{r['label']} : pointés")
        if "velocity" in v:
            dd = np.array([a["distance"].min(), a["distance"].max()])
            ax.plot(dd, v["t0"] + dd / v["velocity"], "-", color=c,
                    label=f"{r['label']} : v = {v['velocity']:.1f}, t0 = {v['t0']:.3f}  (R²={v['r2']:.4f})")
    ax.set(xlabel="distance à la source", ylabel="t d'arrivée (s)", title="Temps d'arrivée et vitesse de propagation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, out, "sismos_arrivals.png")


def fig_freq(runs, out):
    runs = [r for r in runs if r["sis_stats"] is not None]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, r in enumerate(runs):
        s = r["sis_stats"]
        ax.plot(s["distance"], s["f_dominant"], "o", ms=4, color=f"C{k}", label=f"{r['label']} : f dominante")
        ax.plot(s["distance"], s["f_centroid"], "x", ms=4, color=f"C{k}", label=f"{r['label']} : centroïde")
    ax.set(xlabel="distance à la source", ylabel="f (Hz)", title="Contenu fréquentiel vs distance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, out, "sismos_freq_vs_dist.png")


# ----------------------------------------------------------------------------- timings
def fig_timings(runs, out):
    snap = [r for r in runs if r["t_snap"]]
    sis = [r for r in runs if r["t_sis"]]
    if not snap and not sis:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, group, key, title in ((ax[0], snap, "t_snap", "Analyse ad-hoc des snapshots"),
                                 (ax[1], sis, "t_sis", "Analyse ad-hoc des sismos")):
        if not group:
            a.set_visible(False)
            continue
        x = np.arange(len(group))
        rd = np.array([r[key]["read_s"] for r in group])
        cp = np.array([r[key]["compute_s"] for r in group])
        a.bar(x, rd, label="lecture fichiers")
        a.bar(x, cp, bottom=rd, label="calcul")
        a.set_xticks(x)
        a.set_xticklabels([r["label"] for r in group])
        a.set(title=title, ylabel="temps (s)")
        for xi, (r_, c_) in enumerate(zip(rd, cp)):
            a.text(xi, r_ + c_, f"{r_ + c_:.3g}s", ha="center", va="bottom", fontsize=8)
        a.legend()
        a.grid(alpha=0.3, axis="y")
    save(fig, out, "timings_adhoc.png")


def print_summary(runs):
    print("\n=== Résumé ===")
    for r in runs:
        line = [f"{r['label']:>6}"]
        if r["wavefront"] and "velocity" in r["wavefront"]["fit"]:
            line.append(f"v(front)={r['wavefront']['fit']['velocity']:.2f}")
        if r["vel_sis"] and "velocity" in r["vel_sis"]:
            line.append(f"v(sismos)={r['vel_sis']['velocity']:.2f}")
        if r["t_snap"]:
            line.append(f"snap: lect={r['t_snap']['read_s']:.2f}s calc={r['t_snap']['compute_s']:.2f}s "
                        f"({r['t_snap']['bytes_read'] / 1e6:.0f} Mo lus)")
        if r["t_sis"]:
            line.append(f"sismos: total={r['t_sis']['total_s']:.3f}s")
        print("  ".join(line))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", default=["results"], help="dossiers de résultats (un par config S/M/L)")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(d) for d in a.results]

    for fn in (fig_evolution, fig_histogram, fig_wavefront, fig_traces, fig_spectra,
               fig_arrivals, fig_freq, fig_timings):
        fn(runs, out)
    print_summary(runs)


if __name__ == "__main__":
    main()