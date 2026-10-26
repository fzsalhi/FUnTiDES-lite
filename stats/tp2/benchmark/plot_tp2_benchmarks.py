import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def unique_path(folder, stem):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{stem}.png"
    if not p.exists():
        return p
    i = 1
    while True:
        p = folder / f"{stem}_{i:03d}.png"
        if not p.exists():
            return p
        i += 1


def load_csv(path):
    if not path or not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def row_for(rows, size, mode):
    return next(
        (
            r for r in rows
            if r["size"] == size
            and r["mode"] == mode
        ),
        None
    )


def value(row, key):
    if row is None:
        return 0.0
    return float(row.get(key, 0) or 0)


def savefig(folder, stem):
    p = unique_path(folder, stem)
    plt.tight_layout()
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(p)


def plot_workflow(rows, folder):
    sizes = ["S", "M", "L"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    snapshot_modes = [
        ("snapshot_adhoc", "Ad-hoc"),
        ("snapshot_insitu_cpu", "In-situ")
    ]

    sismo_modes = [
        ("sismo_adhoc", "Ad-hoc NumPy"),
        ("sismo_insitu_cufft", "In-situ cuFFT")
    ]

    x = np.arange(len(sizes))
    width = 0.32

    for j, (mode, label) in enumerate(snapshot_modes):
        vals = [
            value(row_for(rows, size, mode), "workflow_s_mean")
            for size in sizes
        ]
        axes[0].bar(
            x + (j - 0.5) * width,
            vals,
            width=width,
            label=label
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sizes)
    axes[0].set_ylabel("Temps workflow (s)")
    axes[0].set_title("Snapshots")
    axes[0].legend()

    for j, (mode, label) in enumerate(sismo_modes):
        vals = [
            value(row_for(rows, size, mode), "workflow_s_mean")
            for size in sizes
        ]
        axes[1].bar(
            x + (j - 0.5) * width,
            vals,
            width=width,
            label=label
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sizes)
    axes[1].set_ylabel("Temps workflow (s)")
    axes[1].set_title("Sismogrammes")
    axes[1].legend()

    fig.suptitle("Comparaison ad-hoc et in-situ")
    savefig(folder, "tp2_bench_workflow_overview")


def plot_threads(rows, folder):
    if not rows:
        return

    labels = {
        "baseline": "Sans analyse",
        "snapshot_adhoc": "Snapshots ad-hoc",
        "snapshot_insitu_cpu": "Snapshots in-situ",
        "sismo_adhoc": "Sismos ad-hoc",
        "sismo_insitu_cufft": "Sismos in-situ cuFFT"
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    modes = [
        "baseline",
        "snapshot_adhoc",
        "snapshot_insitu_cpu",
        "sismo_adhoc",
        "sismo_insitu_cufft"
    ]

    all_threads = sorted(
        set(int(r["threads"]) for r in rows)
    )

    for mode in modes:
        items = sorted(
            [r for r in rows if r["mode"] == mode],
            key=lambda r: int(r["threads"])
        )

        if not items:
            continue

        axes[0].plot(
            [int(r["threads"]) for r in items],
            [float(r["workflow_s_mean"]) for r in items],
            marker="o",
            label=labels[mode]
        )

        axes[1].plot(
            [int(r["threads"]) for r in items],
            [float(r["speedup"]) for r in items],
            marker="o",
            label=labels[mode]
        )

    axes[0].set_xticks(all_threads)
    axes[0].set_xlabel("Nombre de threads OpenMP")
    axes[0].set_ylabel("Temps total workflow (s)")
    axes[0].set_title("Temps total")
    axes[0].legend()

    axes[1].plot(
        all_threads,
        all_threads,
        linestyle="--",
        label="Speedup idéal"
    )
    axes[1].set_xticks(all_threads)
    axes[1].set_xlabel("Nombre de threads OpenMP")
    axes[1].set_ylabel("Speedup T1/Tp")
    axes[1].set_title("Scalabilité OpenMP")
    axes[1].legend()

    size = rows[0]["size"]
    fig.suptitle(
        f"Influence du nombre de threads — cas {size}"
    )

    savefig(folder, "tp2_bench_threads_overview")


def plot_storage(rows, slices, folder):
    sizes = ["S", "M", "L"]
    fig, axes = plt.subplots(
        1,
        2 if slices else 1,
        figsize=(13 if slices else 7, 5)
    )

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    modes = [
        ("snapshot_adhoc", "Snapshots bruts"),
        ("snapshot_insitu_cpu", "Résultats in-situ"),
        ("sismo_adhoc", "Sismos bruts"),
        ("sismo_insitu_cufft", "FFT in-situ")
    ]

    x = np.arange(len(sizes))
    width = 0.18

    for j, (mode, label) in enumerate(modes):
        vals = [
            value(
                row_for(rows, size, mode),
                "bytes_written_mean"
            ) / (1024 * 1024)
            for size in sizes
        ]
        axes[0].bar(
            x + (j - 1.5) * width,
            vals,
            width=width,
            label=label
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sizes)
    axes[0].set_ylabel("Volume écrit (MiB)")
    axes[0].set_title("Sorties ad-hoc et in-situ")
    axes[0].legend()

    if slices:
        sx = [r["size"] for r in slices]
        snap = [
            float(r["snapshot_bytes"]) / (1024 * 1024)
            for r in slices
        ]
        sli = [
            float(r["slice_bytes"]) / (1024 * 1024)
            for r in slices
        ]
        x2 = np.arange(len(sx))

        axes[1].bar(
            x2 - 0.18,
            snap,
            width=0.36,
            label="Snapshot"
        )
        axes[1].bar(
            x2 + 0.18,
            sli,
            width=0.36,
            label="Slice"
        )

        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(sx)
        axes[1].set_ylabel("Volume écrit (MiB)")
        axes[1].set_title("Snapshots complets vs slices")
        axes[1].legend()

    fig.suptitle("Volume de données écrit sur disque")
    savefig(folder, "tp2_bench_storage_overview")


def plot_precision(precision, folder):
    if not precision:
        return

    groups = defaultdict(list)

    for r in precision:
        groups[r["size"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for size, items in groups.items():
        items = sorted(
            items,
            key=lambda r: int(r["interval"])
        )

        intervals = [
            int(r["interval"])
            for r in items
        ]

        max_error = [
            100.0 * float(r["max_rel_error"])
            for r in items
        ]

        mean_error = [
            100.0 * float(r["mean_rel_error"])
            for r in items
        ]

        axes[0].plot(
            intervals,
            max_error,
            marker="o",
            label=size
        )

        axes[1].plot(
            intervals,
            mean_error,
            marker="o",
            label=size
        )

    axes[0].set_xlabel("Intervalle entre snapshots")
    axes[0].set_ylabel("Erreur relative (%)")
    axes[0].set_title("Maximum global")
    axes[0].legend()

    axes[1].set_xlabel("Intervalle entre snapshots")
    axes[1].set_ylabel("Erreur relative (%)")
    axes[1].set_title("Moyenne temporelle")
    axes[1].legend()

    fig.suptitle("Précision en fonction de la fréquence de sortie")
    savefig(folder, "tp2_bench_precision_overview")


def plot_cudamemcpy(rows, folder):
    sizes = ["S", "M", "L"]
    h2d = []
    d2h = []

    for size in sizes:
        r = row_for(rows, size, "sismo_insitu_cufft")
        h2d.append(value(r, "cufft_h2d_gbs_mean"))
        d2h.append(value(r, "cufft_d2h_gbs_mean"))

    x = np.arange(len(sizes))
    width = 0.34

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        h2d,
        width=width,
        label="Host → Device"
    )

    plt.bar(
        x + width / 2,
        d2h,
        width=width,
        label="Device → Host"
    )

    plt.xticks(x, sizes)
    plt.ylabel("Débit effectif (GB/s)")
    plt.xlabel("Taille du cas")
    plt.title("Débit cudaMemcpy du workflow cuFFT")
    plt.legend()

    savefig(folder, "tp2_bench_cudamemcpy_throughput")


def plot_scalability(scale, folder):
    if not scale:
        return

    values = {
        r["metric"]: float(r["value"])
        for r in scale
    }

    labels = [
        "Volume snapshot\nmesuré",
        "Volume snapshot\nthéorique",
        "Calcul\nmesuré",
        "Calcul total\nthéorique"
    ]

    vals = [
        values.get(
            "measured_snapshot_volume_exponent_vs_N",
            0
        ),
        values.get(
            "theory_snapshot_volume_exponent_vs_N",
            3
        ),
        values.get(
            "measured_compute_exponent_vs_N",
            0
        ),
        values.get(
            "theory_compute_total_exponent_vs_N",
            4
        )
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.ylabel("Exposant en N")
    plt.title("Scalabilité mesurée et modèle théorique")

    savefig(folder, "tp2_bench_scalability_overview")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--threads")
    p.add_argument("--precision")
    p.add_argument("--slices")
    p.add_argument("--scalability")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    rows = load_csv(a.summary)
    thread_rows = load_csv(a.threads)
    precision = load_csv(a.precision)
    slices = load_csv(a.slices)
    scale = load_csv(a.scalability)

    plot_workflow(rows, a.out_dir)
    plot_threads(thread_rows, a.out_dir)
    plot_storage(rows, slices, a.out_dir)
    plot_precision(precision, a.out_dir)
    plot_cudamemcpy(rows, a.out_dir)
    plot_scalability(scale, a.out_dir)


if __name__ == "__main__":
    main()