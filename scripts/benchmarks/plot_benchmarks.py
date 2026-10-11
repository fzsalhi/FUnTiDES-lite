import csv
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results" / "benchmarks.csv"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_recent_batch():
    if not RESULTS_FILE.is_file():
        raise FileNotFoundError(f"Resultats introuvables : {RESULTS_FILE}")
    with RESULTS_FILE.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Le fichier des benchmarks est vide")
    batch_id = rows[-1].get("batch_id", "")
    return [row for row in rows if row.get("batch_id") == batch_id] if batch_id else [rows[-1]]


def average(rows, field):
    values = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
            if math.isfinite(value) and value >= 0:
                values.append(value)
        except (TypeError, ValueError):
            continue
    return statistics.mean(values) if values else 0.0


def gnuplot_string(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def fallback_svg(path, title, unit, values):
    width = max(1100, 180 * len(values) + 170)
    height = 700
    left, right, top, bottom = 110, 50, 95, 220
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max((value for _, value in values), default=0) * 1.2 or 1.0
    spacing = plot_width / len(values)
    bars = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="48" text-anchor="middle" font-size="23" font-family="sans-serif">{escape(title)}</text>',
        f'<text x="{left}" y="75" font-size="14" font-family="sans-serif">{escape(unit)}</text>',
    ]
    for index in range(6):
        value = maximum * index / 5
        y = top + plot_height * (1 - index / 5)
        bars.append(f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" stroke="#dddddd"/>')
        bars.append(f'<text x="{left - 10}" y="{y + 5}" text-anchor="end" font-size="13" font-family="sans-serif">{value:.3g}</text>')
    for index, (label, value) in enumerate(values):
        center = left + spacing * (index + 0.5)
        height_bar = value / maximum * plot_height
        bars.append(f'<rect x="{center - 38}" y="{top + plot_height - height_bar}" width="76" height="{height_bar}" fill="#4e80ad"/>')
        label_y = top + plot_height + 23
        bars.append(f'<text x="{center}" y="{label_y}" transform="rotate(-30 {center} {label_y})" text-anchor="end" font-size="13" font-family="sans-serif">{escape(label)}</text>')
    bars.append("</svg>")
    path.write_text("\n".join(bars), encoding="utf-8")
    print(f"Graphique SVG : {path}")


def plot_bars(filename, title, unit, values, x_label):
    data = "$values << EOD\n" + "\n".join(
        f"{index} {value:.12g}" for index, (_, value) in enumerate(values, start=1)
    ) + "\nEOD\n"
    xtics = ", ".join(
        f"{gnuplot_string(label)} {index}"
        for index, (label, _) in enumerate(values, start=1)
    )
    png = PLOTS_DIR / filename
    png.unlink(missing_ok=True)
    script = f"""
set terminal pngcairo size 1500,850 font "Sans,12"
set output {gnuplot_string(str(png))}
set title {gnuplot_string(title)}
set ylabel {gnuplot_string(unit)}
set xlabel {gnuplot_string(x_label)}
set xrange [0.5:{len(values) + 0.5}]
set yrange [0:*]
set boxwidth 0.65 absolute
set style fill solid 0.85 border -1
set grid ytics
set key off
set xtics ({xtics}) rotate by -25
plot $values using 1:2 with boxes
unset output
"""
    try:
        result = subprocess.run(
            ["gnuplot"], input=data + script, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0 and png.is_file() and png.stat().st_size > 0:
            print(f"Graphique PNG : {png}")
            return
        print(f"Gnuplot indisponible (code {result.returncode}), utilisation de SVG.")
    except FileNotFoundError:
        print("Gnuplot introuvable, utilisation de SVG.")
    png.unlink(missing_ok=True)
    fallback_svg(png.with_suffix(".svg"), title, unit, values)



def fallback_grouped_svg(path, title, unit, categories, groups, series):
    width = max(1200, len(categories) * 240 + 250)
    height = 760
    left, right, top, bottom = 110, 55, 125, 190
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(
        (value for values in series.values() for value in values if value is not None),
        default=0.0
    )
    maximum = maximum * 1.2 if maximum > 0 else 1.0
    colors = ["#3f78a9", "#dd9140", "#3c9a70", "#8858ac", "#c04f68", "#698a35"]
    group_width = plot_width / len(categories)
    bar_width = min(43, group_width * 0.75 / len(groups))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="44" font-family="sans-serif" text-anchor="middle" font-size="22">{escape(title)}</text>',
        f'<text x="{left}" y="82" font-family="sans-serif" font-size="14">{escape(unit)}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_height * (1 - tick / 5)
        parts.append(f'<line x1="{left}" y1="{y:.3f}" x2="{width-right}" y2="{y:.3f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-9}" y="{y+5:.3f}" font-family="sans-serif" font-size="13" text-anchor="end">{value:.3g}</text>')
    for index, label in enumerate(categories):
        center = left + (index + 0.5) * group_width
        y_label = top + plot_height + 27
        parts.append(f'<text x="{center:.3f}" y="{y_label}" transform="rotate(-23 {center:.3f} {y_label})" font-family="sans-serif" font-size="13" text-anchor="end">{escape(label)}</text>')
        for series_index, group in enumerate(groups):
            value = series[group][index]
            if value is None:
                continue
            x = center + (series_index - (len(groups)-1)/2) * bar_width - bar_width * 0.43
            bar_height = value / maximum * plot_height
            y = top + plot_height - bar_height
            parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{bar_width*0.86:.3f}" height="{bar_height:.3f}" fill="{colors[series_index%len(colors)]}"/>')
    legend_width = min(200, (width-left-right)/len(groups))
    for series_index, group in enumerate(groups):
        x = left + series_index * legend_width
        parts.append(f'<rect x="{x:.3f}" y="96" width="16" height="13" fill="{colors[series_index%len(colors)]}"/>')
        parts.append(f'<text x="{x+24:.3f}" y="108" font-family="sans-serif" font-size="13">{escape(str(group))} thread(s)</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Graphique SVG : {path}")


def plot_grouped_phases(filename, title, categories, groups, series):
    if not categories or not groups:
        return
    width = 0.78 / len(groups)
    blocks = []
    plots = []
    colors = ["#3f78a9", "#dd9140", "#3c9a70", "#8858ac", "#c04f68", "#698a35"]
    for index, group in enumerate(groups):
        name = f"phase_{index}"
        pairs = []
        for phase_index, value in enumerate(series[group], start=1):
            if value is None:
                continue
            center = phase_index + (index - (len(groups)-1)/2) * width
            pairs.append(f"{center:.10g} {value:.12g}")
        blocks.append(f"${name} << EOD\n" + "\n".join(pairs) + "\nEOD\n")
        plots.append(f"${name} using 1:2 with boxes lc rgb {gnuplot_string(colors[index%len(colors)])} title {gnuplot_string(str(group) + ' thread(s)')}")
    xtics = ", ".join(
        f"{gnuplot_string(label)} {index}"
        for index, label in enumerate(categories, start=1)
    )
    png = PLOTS_DIR / filename
    png.unlink(missing_ok=True)
    script = f"""
set terminal pngcairo size 1700,900 font "Sans,12"
set output {gnuplot_string(str(png))}
set title {gnuplot_string(title)}
set ylabel "Temps moyen (secondes)"
set xlabel "Etapes"
set xrange [0.45:{len(categories) + 0.55}]
set yrange [0:*]
set boxwidth {width*0.85:.12g} absolute
set style fill solid 0.85 border -1
set grid ytics
set key outside top horizontal
set xtics ({xtics}) rotate by -20
plot {', '.join(plots)}
unset output
"""
    try:
        result = subprocess.run(
            ["gnuplot"], input="\n".join(blocks) + script,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0 and png.is_file() and png.stat().st_size > 0:
            print(f"Graphique PNG : {png}")
            return
        print(f"Gnuplot indisponible (code {result.returncode}), utilisation de SVG.")
    except FileNotFoundError:
        print("Gnuplot introuvable, utilisation de SVG.")
    png.unlink(missing_ok=True)
    fallback_grouped_svg(png.with_suffix(".svg"), title, "Temps moyen (s)", categories, groups, series)

def main():
    rows = load_recent_batch()
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["threads"])].append(row)
    thread_counts = sorted(groups)
    latest_threads = int(rows[-1]["threads"])
    latest = groups[latest_threads]
    repetitions = len(latest)
    categories = [
        "Initialisation",
        "Propagation / calcul",
        "Sorties FUnTiDES",
        "Visualisation snapshots",
        "Visualisation sismos",
    ]
    fields = ["initial_s", "compute_s", "output_s", "snapshots_viz_s", "sismos_viz_s"]
    phase_series = {
        threads: [average(groups[threads], field) for field in fields]
        for threads in thread_counts
    }
    plot_grouped_phases(
        "phase_times.png",
        "Temps moyen par etape selon le nombre de threads",
        categories,
        thread_counts,
        phase_series,
    )
    plot_grouped_phases(
        "simulation_phases.png",
        "Etapes internes de FUnTiDES selon le nombre de threads",
        categories[:3],
        thread_counts,
        {threads: values[:3] for threads, values in phase_series.items()},
    )
    plot_bars(
        "data_sizes.png",
        "Taille des fichiers traites - derniere serie",
        "Taille (MiB)",
        [
            ("Snapshots CSV", average(latest, "snapshot_csv_mib")),
            ("Sismogrammes CSV", average(latest, "sismo_csv_mib")),
            ("Images snapshots", average(latest, "snapshot_png_mib")),
            ("Images sismos", average(latest, "sismo_png_mib")),
        ],
        "Type de donnees",
    )
    plot_bars(
        "workflow_time.png",
        "Temps moyen du workflow selon le nombre de threads",
        "Temps (secondes)",
        [
            (f"{threads} thread(s)", average(groups[threads], "workflow_wall_s"))
            for threads in thread_counts
        ],
        "Nombre de threads FUnTiDES",
    )
    plot_bars(
        "simulation_time.png",
        "Temps moyen de la simulation selon le nombre de threads",
        "Temps (secondes)",
        [
            (f"{threads} thread(s)", average(groups[threads], "simulation_wall_s"))
            for threads in thread_counts
        ],
        "Nombre de threads FUnTiDES",
    )
    if any(row.get("data_status") != "generated_this_run" for row in rows):
        print("Attention : donnees de test ou CSV preexistants ; simulation et visualisation non couplees.")


if __name__ == "__main__":
    main()