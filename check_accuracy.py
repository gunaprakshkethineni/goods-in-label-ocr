# scores the ocr output against the ground truth the generator wrote.
# also reruns everything with the opencv step switched off so i can show it actually helps.

import csv
import os
import sys

from ocr_reader import read_all
from textutil import edit_distance

TRUTH_FILE = "data/labels_truth.csv"
INV_FILE = "output/inventory.csv"
FLAG_FILE = "output/flagged.csv"

FIELDS = ["part_no", "serial", "lot_no", "qty"]


def load_truth():
    with open(TRUTH_FILE, newline="") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def score(rows, truth):
    dist = 0
    chars = 0
    exact = {f: 0 for f in FIELDS}
    n = 0

    for r in rows:
        t = truth.get(r["filename"])
        if not t:
            continue
        n += 1
        for f in FIELDS:
            got = r.get(f, "") or ""
            want = t[f]
            dist += edit_distance(got, want)
            chars += len(want)
            if got == want:
                exact[f] += 1

    acc = 1.0 - (dist / chars) if chars else 0
    return acc, exact, n


def report(name, rows, truth):
    acc, exact, n = score(rows, truth)
    print()
    print("===", name, "===")
    print("labels scored:      ", n)
    print("character accuracy:  %.1f%%" % (acc * 100))
    print("exact field matches:")
    for f in FIELDS:
        print("   %-10s %2d/%2d  (%.0f%%)" % (f, exact[f], n, 100.0 * exact[f] / n))
    return acc


# how good the flagging itself is. a flag on a label i deliberately broke is a real catch,
# a flag on a label that was fine is a false alarm and costs somebody time for nothing
def report_flags(truth):
    if not os.path.exists(FLAG_FILE):
        print("\n(no flagged.csv yet, run validate.py first)")
        return

    with open(FLAG_FILE, newline="") as f:
        flagged = {r["filename"] for r in csv.DictReader(f)}

    real = {n for n, t in truth.items() if t.get("defects")}
    caught = real & flagged
    missed = real - flagged
    false_alarms = flagged - real

    print()
    print("=== flagging quality ===")
    print("labels i deliberately broke:  ", len(real))
    print("  of those, flagged:          ", len(caught))
    print("  of those, missed:           ", len(missed))
    print("clean labels flagged anyway:  ", len(false_alarms), "(ocr was not confident enough)")
    if flagged:
        print("so %.0f%% of the flags were real defects" % (100.0 * len(caught) / len(flagged)))


def main():
    truth = load_truth()

    with open(INV_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    with_pre = report("with opencv preprocessing", rows, truth)
    report_flags(truth)

    if "quick" in sys.argv:
        return

    print()
    print("now rerunning with preprocessing off, takes a minute...")
    raw_rows = read_all(use_preprocess=False, quiet=True)
    without_pre = report("no preprocessing (raw image straight to tesseract)", raw_rows, truth)

    print()
    print("preprocessing is worth %.1f percentage points" % ((with_pre - without_pre) * 100))


if __name__ == "__main__":
    main()
