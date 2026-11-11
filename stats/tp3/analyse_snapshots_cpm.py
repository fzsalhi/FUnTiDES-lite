#!/usr/bin/env python3
"""Lance analyze_snapshots.py (ad-hoc, TP2) sur des snapshots .cpm reconstruits.

Principe : on remplace seulement la lecture des snapshots (list_snapshots,
load_snapshot, load_snapshot_and_coords de io_utils) par une lecture .cpm.
analyze_snapshots.py n'est PAS modifie : memes statistiques, meme histogramme,
meme front d'onde, meme vitesse -> comparaison directe avec les resultats d'origine.

Les .cpm ne contiennent que la pression. Les coordonnees sont reprises du
snapshot ORIGINAL (--orig-dir), ou de --coords si tu le donnes (comme avant).
Hypothese : meme ordre des noeuds dans le .cpm et dans le snapshot original.

Exemple :
  python analyse_snapshots_cpm.py --cpm-dir run_int8/snapshots \
      --orig-dir results/snapshots --dt 0.00298228 --tag S_int8 --out results_int8
Toutes les autres options (--dt, --source, --nbins, --front-frac, --out, --tag,
--coords, ...) sont celles de analyze_snapshots.py.
"""
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tp2" / "benchmarks" / "archives"))
import io_utils
from cpm_reader import HEADER_SIZE, read_cpm, read_header


def main():
    # allow_abbrev=False : sinon "--dt" est pris pour une abreviation de "--dtype"
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--cpm-dir", required=True, help="dossier contenant les .cpm")
    pre.add_argument("--orig-dir", required=True, help="dossier des snapshots originaux (pour les coordonnees)")
    pre.add_argument("--orig-pattern", default="snapshot_*")
    pre.add_argument("--dtype", default="float32")
    pre.add_argument("--col", default=None)
    known, rest = pre.parse_known_args()
    rest += ["--snap-dir", known.cpm_dir, "--dtype", known.dtype]
    if known.col:
        rest += ["--col", known.col]

    orig_lsac = io_utils.load_snapshot_and_coords
    orig_snaps = io_utils.list_snapshots(known.orig_dir, known.orig_pattern)

    def list_cpm(snap_dir, pattern="*"):
        out = []
        for f in sorted(Path(snap_dir).glob("*.cpm")):
            with open(f, "rb") as fh:
                h = read_header(fh.read(HEADER_SIZE))
            out.append((int(h["step"]), str(f)))
        if not out:
            raise SystemExit(f"aucun .cpm dans {snap_dir}")
        return sorted(out)

    def load_cpm(path, dtype="float32", col=None):
        return read_cpm(path, dtype_out=np.float32)

    def load_cpm_and_coords(path, dtype="float32", col=None):
        _, coords = orig_lsac(orig_snaps[0][1], known.dtype, known.col)
        return read_cpm(path, dtype_out=np.float32), coords

    # a patcher AVANT l'import : analyze_snapshots fait "from io_utils import ..."
    io_utils.list_snapshots = list_cpm
    io_utils.load_snapshot = load_cpm
    io_utils.load_snapshot_and_coords = load_cpm_and_coords

    try:
        mod = importlib.import_module("analyze_snapshots")
    except ModuleNotFoundError:
        mod = importlib.import_module("analyse_snapshots")

    sys.argv = [sys.argv[0]] + rest
    mod.main()


if __name__ == "__main__":
    main()