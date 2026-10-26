import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rows = []
    fields = None

    for path in a.inputs:
        if not Path(path).exists():
            continue
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            if fields is None:
                fields = r.fieldnames
            for row in r:
                rows.append(row)

    if not rows or fields is None:
        raise RuntimeError("No rows to combine.")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(a.out)


if __name__ == "__main__":
    main()
