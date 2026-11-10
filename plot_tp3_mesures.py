#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Matplotlib manquant : python3 -m pip install matplotlib") from exc

MODES = ["baseline", "P3_cache", "P6_cache", "P3_fsync", "P6_fsync"]
NOMS = {
    "baseline": "Sans visu",
    "P3_cache": "P3 sans fsync",
    "P6_cache": "P6 sans fsync",
    "P3_fsync": "P3 avec fsync",
    "P6_fsync": "P6 avec fsync",
    "P3_P6_paired": "P3 et P6",
}


def lire_csv(chemin):
    if not chemin.is_file():
        print(f"Absent : {chemin.name} (graphe correspondant ignore)")
        return []
    with chemin.open(newline="", encoding="utf-8-sig") as fichier:
        return list(csv.DictReader(fichier))


def nombre(valeur):
    try:
        return float(valeur)
    except (ValueError, TypeError):
        return None


def mediane(valeurs):
    propres = [v for v in valeurs if v is not None]
    return statistics.median(propres) if propres else None


def afficher_ou_sauver(fig, destination, afficher):
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    print(f"Cree : {destination}")
    if afficher:
        plt.show()
    plt.close(fig)


def configurer(ax, titre, ylabel):
    ax.set_title(titre, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def ordonner_cas(lignes):
    cas = []
    for ligne in lignes:
        nom = ligne.get("case", "")
        if nom and nom not in cas:
            cas.append(nom)
    return cas


def mesure_runs(lignes):
    groupes = defaultdict(list)
    for ligne in lignes:
        if str(ligne.get("warmup", "0")) != "0":
            continue
        if str(ligne.get("return_code", "0")) != "0":
            continue
        temps = nombre(ligne.get("wall_ms"))
        if temps is not None:
            groupes[(ligne.get("case"), ligne.get("mode"))].append(temps)
    return groupes


def graphe_simulation(runs, synthese, dossier, afficher):
    groupes = mesure_runs(runs)
    cas = ordonner_cas(runs or synthese)
    if not cas:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    centres = list(range(len(MODES)))
    largeur = min(0.8 / len(cas), 0.34)
    trace = False
    for i, cas_nom in enumerate(cas):
        x = [k + (i - (len(cas) - 1) / 2) * largeur for k in centres]
        valeurs = []
        bas = []
        haut = []
        for mode in MODES:
            repetitions = groupes.get((cas_nom, mode), [])
            if repetitions:
                med = statistics.median(repetitions)
                valeurs.append(med)
                bas.append(med - min(repetitions))
                haut.append(max(repetitions) - med)
            else:
                ligne = next((r for r in synthese if r.get("case") == cas_nom and r.get("mode") == mode), None)
                med = nombre(ligne.get("wall_median_ms")) if ligne else None
                valeurs.append(med if med is not None else float("nan"))
                minimum = nombre(ligne.get("wall_min_ms")) if ligne else None
                maximum = nombre(ligne.get("wall_max_ms")) if ligne else None
                bas.append(max(0.0, med - minimum) if med is not None and minimum is not None else 0)
                haut.append(max(0.0, maximum - med) if med is not None and maximum is not None else 0)
        ax.bar(x, valeurs, largeur, label=cas_nom, yerr=[bas, haut], capsize=3)
        trace = True
    if not trace:
        plt.close(fig)
        return
    ax.set_xticks(centres, [NOMS[m] for m in MODES], rotation=12)
    configurer(ax, "Duree totale : sans visualisation vs P3 / P6", "Temps total median (ms)")
    ax.legend(title="Maillage")
    afficher_ou_sauver(fig, dossier / "01_simulation_sans_avec_visu.png", afficher)


def graphe_surcout(runs, synthese, dossier, afficher):
    groupes = mesure_runs(runs)
    cas = ordonner_cas(runs or synthese)
    modes = MODES[1:]
    if not cas:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    largeur = min(0.8 / len(cas), 0.34)
    trace = False
    for i, cas_nom in enumerate(cas):
        base = mediane(groupes.get((cas_nom, "baseline"), []))
        valeurs = []
        for mode in modes:
            mesure = mediane(groupes.get((cas_nom, mode), []))
            if base and mesure is not None:
                valeurs.append(100 * (mesure / base - 1))
            else:
                ligne = next((r for r in synthese if r.get("case") == cas_nom and r.get("mode") == mode), None)
                v = nombre(ligne.get("overhead_pct")) if ligne else None
                valeurs.append(v if v is not None else float("nan"))
        x = [k + (i - (len(cas) - 1) / 2) * largeur for k in range(len(modes))]
        ax.bar(x, valeurs, largeur, label=cas_nom)
        trace = True
    if not trace:
        plt.close(fig)
        return
    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(len(modes)), [NOMS[m] for m in modes], rotation=10)
    configurer(ax, "Surcout de la visualisation par rapport a la simulation seule", "Surcout median (%)")
    ax.legend(title="Maillage")
    afficher_ou_sauver(fig, dossier / "02_surcout_visualisation.png", afficher)


def graphe_temps_ecriture(synthese, dossier, afficher):
    cas = ordonner_cas(synthese)
    for cas_nom in cas:
        lignes = {r["mode"]: r for r in synthese if r.get("case") == cas_nom}
        modes = [m for m in MODES[1:] if m in lignes]
        if not modes:
            continue
        fig, ax = plt.subplots(figsize=(10.5, 6))
        x = range(len(modes))
        bas = [0.0] * len(modes)
        for cle, nom in [("encode_ms_median", "Encodage RGB -> PPM"),
                         ("write_ms_median", "Ouverture + ecriture"),
                         ("fsync_ms_median", "fsync")]:
            hauteurs = [nombre(lignes[m].get(cle)) or 0.0 for m in modes]
            ax.bar(x, hauteurs, bottom=bas, label=nom)
            bas = [a + b for a, b in zip(bas, hauteurs)]
        ax.set_xticks(list(x), [NOMS[m] for m in modes], rotation=10)
        configurer(ax, f"Temps P3 / P6, avec et sans fsync : {cas_nom}", "Temps median par image (ms)")
        ax.legend()
        afficher_ou_sauver(fig, dossier / f"03_temps_ecriture_{nettoyer(cas_nom)}.png", afficher)


def graphe_tailles(paires, synthese, dossier, afficher):
    cas = ordonner_cas(paires or synthese)
    etiquettes = []
    p3 = []
    p6 = []
    for cas_nom in cas:
        lignes = [r for r in paires if r.get("case") == cas_nom]
        if lignes:
            taille_p3 = mediane([nombre(r.get("P3_bytes")) for r in lignes])
            taille_p6 = mediane([nombre(r.get("P6_bytes")) for r in lignes])
        else:
            a = next((r for r in synthese if r.get("case") == cas_nom and r.get("mode") == "P3_cache"), None)
            b = next((r for r in synthese if r.get("case") == cas_nom and r.get("mode") == "P6_cache"), None)
            taille_p3 = nombre(a.get("size_bytes_median")) if a else None
            taille_p6 = nombre(b.get("size_bytes_median")) if b else None
        if taille_p3 is not None and taille_p6 is not None:
            etiquettes.append(cas_nom)
            p3.append(taille_p3 / 1024)
            p6.append(taille_p6 / 1024)
    if not etiquettes:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = range(len(etiquettes))
    ax.bar([x - 0.2 for x in indices], p3, 0.4, label="P3 (texte)")
    ax.bar([x + 0.2 for x in indices], p6, 0.4, label="P6 (binaire)")
    ax.set_xticks(list(indices), etiquettes)
    configurer(ax, "Tailles des images PPM : P3 vs P6", "Taille mediane (Kio)")
    ax.legend()
    afficher_ou_sauver(fig, dossier / "04_tailles_P3_P6.png", afficher)


def graphe_pipeline(synthese, dossier, afficher):
    cas = ordonner_cas(synthese)
    for cas_nom in cas:
        lignes = {r["mode"]: r for r in synthese if r.get("case") == cas_nom}
        modes = [m for m in MODES[1:] if m in lignes]
        if not modes:
            continue
        fig, ax = plt.subplots(figsize=(10.5, 6))
        x = range(len(modes))
        bas = [0.0] * len(modes)
        for cle, nom in [("extract_ms_median", "Extraction"),
                         ("color_ms_median", "Coloration"),
                         ("output_total_ms_median", "Preparation + ecriture PPM")]:
            hauteurs = [nombre(lignes[m].get(cle)) or 0.0 for m in modes]
            ax.bar(x, hauteurs, bottom=bas, label=nom)
            bas = [a + b for a, b in zip(bas, hauteurs)]
        ax.set_xticks(list(x), [NOMS[m] for m in modes], rotation=10)
        configurer(ax, f"Etapes de visualisation : {cas_nom}", "Temps median par image (ms)")
        ax.legend()
        afficher_ou_sauver(fig, dossier / f"05_pipeline_{nettoyer(cas_nom)}.png", afficher)


def graphe_evolution(images, dossier, afficher):
    groupes = defaultdict(list)
    for ligne in images:
        if str(ligne.get("warmup", "0")) != "0":
            continue
        cas_nom, mode = ligne.get("case"), ligne.get("mode")
        if mode not in MODES[1:]:
            continue
        pas = nombre(ligne.get("step"))
        duree = nombre(ligne.get("output_total_ms"))
        if pas is not None and duree is not None:
            groupes[(cas_nom, mode, int(pas))].append(duree)
    cas = sorted({cle[0] for cle in groupes})
    for cas_nom in cas:
        fig, ax = plt.subplots(figsize=(11, 6))
        for mode in MODES[1:]:
            pas = sorted(k[2] for k in groupes if k[0] == cas_nom and k[1] == mode)
            if pas:
                temps = [mediane(groupes[(cas_nom, mode, t)]) for t in pas]
                ax.plot(pas, temps, marker="o", markersize=3, label=NOMS[mode])
        ax.set_xlabel("Pas de temps")
        configurer(ax, f"Temps d'ecriture PPM selon le pas : {cas_nom}", "Temps total de sortie median (ms)")
        ax.legend()
        afficher_ou_sauver(fig, dossier / f"06_evolution_ecriture_{nettoyer(cas_nom)}.png", afficher)


def nettoyer(valeur):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in valeur)


def main():
    parser = argparse.ArgumentParser(description="Graphiques TP3 a partir des fichiers CSV du benchmark")
    parser.add_argument("--input", default="tp3_bench_ppm", help="Dossier contenant runs.csv, synthese.csv, images.csv")
    parser.add_argument("--output", default=None, help="Dossier de sortie des PNG")
    parser.add_argument("--show", action="store_true", help="Afficher aussi les graphiques a l'ecran")
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve() if args.output else source / "graphiques"
    if not source.is_dir():
        parser.error(f"Dossier introuvable : {source}")
    runs = lire_csv(source / "runs.csv")
    synthese = lire_csv(source / "synthese.csv")
    images = lire_csv(source / "images.csv")
    paires = lire_csv(source / "tailles_P3_P6_meme_image.csv")
    if not runs and not synthese:
        parser.error("Aucun resultat exploitable : il faut runs.csv ou synthese.csv")
    destination.mkdir(parents=True, exist_ok=True)
    graphe_simulation(runs, synthese, destination, args.show)
    graphe_surcout(runs, synthese, destination, args.show)
    graphe_temps_ecriture(synthese, destination, args.show)
    graphe_tailles(paires, synthese, destination, args.show)
    graphe_pipeline(synthese, destination, args.show)
    graphe_evolution(images, destination, args.show)
    print(f"\nGraphiques enregistres dans : {destination}")


if __name__ == "__main__":
    main()