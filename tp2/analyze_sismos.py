#!/usr/bin/env python3
"""Statistiques ad-hoc des sismogrammes binaires, sur tous les pas de temps."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from io_utils import load_sismos, pressure_statistics, dump_json


def analyze(path, output, with_fft=False):
    start = perf_counter()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    t, signals = load_sismos(path)
    meta_path = Path(path).with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    dt = meta.get("dt_s", float(t[1] - t[0]) if len(t) > 1 else 0)
    read_s = perf_counter() - start
    compute_start = perf_counter()
    rows = []
    for receiver, signal in enumerate(signals):
        stats = pressure_statistics(signal)
        rows.append(dict(receiver=receiver, **stats))
    compute_s = perf_counter() - compute_start
    fft_prepare_s = fft_compute_s = 0.0
    if with_fft:
        if signals.shape[1] < 2:
            raise ValueError("La FFT necessite au moins deux echantillons")
        fft_start = perf_counter()
        centered = (signals.astype(np.float64) - np.mean(signals, axis=1, dtype=np.float64)[:, None]).astype(np.float32)
        fft_prepare_s = perf_counter() - fft_start
        fft_start = perf_counter()
        amplitudes = np.abs(np.fft.rfft(centered, axis=1)) * dt
        frequencies = np.fft.rfftfreq(signals.shape[1], dt)
        fft_compute_s = perf_counter() - fft_start
        compute_s += fft_prepare_s + fft_compute_s
    write_start = perf_counter()
    with (out / "sismos_stats.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    if with_fft:
        with (out / "fft.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["receiver", "frequency_hz", "amplitude"])
            for receiver, spectrum in enumerate(amplitudes):
                writer.writerows((receiver, float(f), float(a)) for f, a in zip(frequencies, spectrum))
    timing = dict(read_s=read_s, compute_s=compute_s, render_s=0.0, write_s=perf_counter()-write_start,
                  fft_prepare_s=fft_prepare_s, fft_compute_s=fft_compute_s,
                  n_receivers=len(signals), n_samples=signals.shape[1], bytes_read=Path(path).stat().st_size + (meta_path.stat().st_size if meta_path.exists() else 0),
                  total_s=perf_counter()-start)
    dump_json(out / "timings_sismos.json", timing)
    return rows, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sismos", required=True)
    parser.add_argument("--out", default="tp2/results")
    parser.add_argument("--fft", action="store_true")
    args = parser.parse_args()
    _, timing = analyze(args.sismos, args.out, with_fft=args.fft)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
