"""Utilitaires communs : chargement des sorties du TP1 + chronométrage.

Formats supportés par défaut :
  - snapshots : un fichier par snapshot (.csv avec ou sans en-tête, .bin brut, .npy), 1 valeur par noeud.
                En CSV : 1 colonne (p) ou plusieurs (x,y,z,p) -> p = dernière colonne (ou --col),
                et les coordonnées sont lues dans les 3 premières colonnes si --coords est absent.
  - coords    : (n_noeuds, 3) en .csv, .bin (xyz entrelacé) ou .npy
  - sismos    : CSV (colonne 0 = temps, colonnes suivantes = récepteurs),
                .npy (nrec, nt) avec --dt, ou dossier avec un fichier par récepteur
  - récepteurs: texte "x y z" par ligne
"""
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class Timer:
    """with T("nom"): ...  -> cumule les durées par nom (perf_counter)."""

    def __init__(self):
        self.total, self.count = {}, {}

    @contextmanager
    def __call__(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.total[name] = self.total.get(name, 0.0) + dt
            self.count[name] = self.count.get(name, 0) + 1

    def summary(self):
        return {k: {"total_s": v, "count": self.count[k]} for k, v in self.total.items()}


def natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", Path(p).name)]


def _read_csv(path):
    """CSV/TXT numérique -> (tableau, noms de colonnes ou None). Détecte l'en-tête et le séparateur."""
    with open(path) as f:
        first = f.readline().strip()
    delim = ";" if ";" in first else ("," if "," in first else None)
    toks = first.split(delim) if delim else first.split()
    names = None
    try:
        [float(t) for t in toks]
    except ValueError:
        names = [t.strip() for t in toks]
    try:  # pandas (parser C) est ~10x plus rapide que numpy sur de gros CSV ; repli sur numpy sinon
        import pandas as pd
        a = pd.read_csv(path, sep=delim or r"\s+", header=0 if names else None, dtype=np.float64).to_numpy()
        if a.shape[1] == 1:
            a = a[:, 0]
    except ImportError:
        a = np.genfromtxt(path, delimiter=delim, skip_header=1 if names else 0, ndmin=1)
    return a, names


def _load_array(path, dtype):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix in (".bin", ".raw", ".dat"):
        return np.fromfile(p, dtype=dtype)
    return _read_csv(p)[0]


def list_snapshots(snap_dir, pattern="snapshot_*"):
    """Retourne [(numéro_de_pas, chemin)] triés ; le pas est le dernier entier du nom."""
    files = sorted(Path(snap_dir).glob(pattern), key=natural_key)
    out = []
    for f in files:
        nums = re.findall(r"\d+", f.stem)
        out.append((int(nums[-1]) if nums else len(out), f))
    if not out:
        raise FileNotFoundError(f"Aucun snapshot '{pattern}' dans {snap_dir}")
    return out


def load_snapshot_and_coords(path, dtype="float32", col=None):
    """Retourne (p[n], coords[n,3] ou None). CSV multi-colonnes : p = colonne `col` (nom ou indice) sinon la dernière."""
    p = Path(path)
    if p.suffix not in (".csv", ".txt"):
        return _load_array(p, dtype).ravel(), None
    a, names = _read_csv(p)
    if a.ndim == 1:
        return a, None
    if col is None:
        j = a.shape[1] - 1
    elif str(col).lstrip("-").isdigit():
        j = int(col)
    else:
        j = names.index(col)
    return a[:, j], (a[:, :3] if a.shape[1] >= 4 else None)


def load_snapshot(path, dtype="float32", col=None):
    return load_snapshot_and_coords(path, dtype, col)[0]


def load_coords(path, dtype="float32"):
    return _load_array(path, dtype).reshape(-1, 3)

def load_receivers(path):
    """Lit x y z par ligne (séparateur espace, virgule ou point-virgule ; '#' = commentaire ; en-tête ignoré)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].replace(",", " ").replace(";", " ").strip()
            if not line:
                continue
            try:
                rows.append([float(t) for t in line.split()])
            except ValueError:
                continue  # ligne d'en-tête
    return np.array(rows, ndmin=2)[:, :3]

def load_sismos(path, dt=None):
    """Retourne (t[nt], data[nrec, nt])."""
    p = Path(path)
    if p.is_dir():
        traces, t = [], None
        for f in sorted((x for x in p.iterdir() if x.is_file()), key=natural_key):
            a = np.atleast_2d(_load_array(f, "float32"))
            if a.shape[0] == 1:
                a = a.T
            if a.shape[1] >= 2:
                t, v = a[:, 0], a[:, 1]
            else:
                v = a[:, 0]
            traces.append(v)
        data = np.array(traces)
        if t is None:
            t = np.arange(data.shape[1]) * dt
        return t, data
    if p.suffix == ".npy":
        data = np.load(p)
        return np.arange(data.shape[1]) * dt, data
    a, _ = _read_csv(p)
    return a[:, 0], a[:, 1:].T


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)