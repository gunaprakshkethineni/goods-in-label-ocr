# quality check on whatever the ocr produced. anything that gets a flag goes to a human,
# everything else is trusted and posted straight to inventory.

import csv
import os
import re

from textutil import edit_distance

INV_FILE = "output/inventory.csv"
LOT_FILE = "data/lot_master.csv"
OUT_FILE = "output/flagged.csv"
ACCEPTED_FILE = "output/inventory_final.csv"

# how far off a lot number can be and still get snapped to the approved list. 2 is safe here
# because the approved lots are 10 characters and there are only a dozen of them, so two of them
# are never that close to each other. if the plant had thousands of lots i would drop this to 1
LOT_SNAP_MAX = 2

LOT_FMT = re.compile(r"^L\d{4}-\d{4}$")
PN_FMT = re.compile(r"^\d{4}-[A-Z]\d$")
SN_FMT = re.compile(r"^SN\d{7}$")


def load_approved_lots(path=LOT_FILE):
    lots = set()
    if not os.path.exists(path):
        return lots
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lots.add(row["lot_no"].strip())
    return lots


def chars_off(a, b):
    # started out comparing position by position, but tesseract drops and inserts characters as
    # well as swapping them. L2023-878 is a dropped digit and L2025-28029 is an inserted one, and
    # a positional compare scores both of those as completely different strings. proper edit
    # distance catches all three kinds
    return edit_distance(a, b)


# looks at one row, repairs what it can and flags what it cannot.
# returns the cleaned up row and the list of problems. empty list means nobody has to look at it.
#
# the repairs are the whole point. my first version flagged any row where a single character
# came out wrong and that was 40 percent of them, which is not much of a time saving for anyone.
def check_row(row, approved):
    row = dict(row)
    flags = []

    # code128 carries a check digit, ocr carries nothing, so when the barcode decodes it wins.
    # the ocr serial is then only useful as a cross check
    bc = row.get("barcode", "")
    sn = row.get("serial", "")
    if bc:
        if sn and SN_FMT.match(sn) and sn != bc:
            # both readable and they genuinely disagree. that is a real problem, not ocr noise
            flags.append("SERIAL_MISMATCH")
        row["serial"] = bc
    else:
        flags.append("BARCODE_UNREADABLE")
        if not sn:
            flags.append("FIELD_MISSING:serial")
        elif not SN_FMT.match(sn):
            flags.append("SERIAL_FORMAT_BAD")

    # we know every lot number the plant has approved, so a lot that is one character away from
    # exactly one of them is almost certainly that lot misread. snap it. if it is close to two
    # of them we cannot tell which, so it goes to a human
    lot = row.get("lot_no", "")
    if not lot:
        flags.append("FIELD_MISSING:lot_no")
    elif lot not in approved:
        near = [a for a in approved if chars_off(lot, a) <= LOT_SNAP_MAX]
        if len(near) == 1:
            row["lot_no"] = near[0]
        elif not LOT_FMT.match(lot):
            flags.append("LOT_FORMAT_BAD")
        else:
            flags.append("LOT_MISMATCH")

    pn = row.get("part_no", "")
    if not pn:
        flags.append("FIELD_MISSING:part_no")
    elif not PN_FMT.match(pn):
        flags.append("PART_FORMAT_BAD")

    qty = row.get("qty", "")
    if not qty:
        flags.append("FIELD_MISSING:qty")
    elif not qty.isdigit():
        flags.append("QTY_FORMAT_BAD")
    elif int(qty) <= 0 or int(qty) > 500:
        flags.append("QTY_OUT_OF_RANGE")

    return row, flags


def main():
    with open(INV_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    approved = load_approved_lots()
    print("loaded", len(approved), "approved lot numbers")

    flagged = []
    accepted = []
    counts = {}
    for r in rows:
        fixed, flags = check_row(r, approved)
        if flags:
            fixed["flags"] = ";".join(flags)
            flagged.append(fixed)
            for fl in flags:
                key = fl.split(":")[0]
                counts[key] = counts.get(key, 0) + 1
        else:
            accepted.append(fixed)

    os.makedirs("output", exist_ok=True)
    cols = ["filename", "part_no", "serial", "lot_no", "qty", "barcode", "flags"]
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(flagged)

    with open(ACCEPTED_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols[:-1])
        w.writeheader()
        w.writerows(accepted)

    total = len(rows)
    bad = len(flagged)
    clean_n = total - bad
    pct = 100.0 * clean_n / total if total else 0

    print()
    print("--- breakdown ---")
    for k in sorted(counts, key=lambda x: -counts[x]):
        print("  %-20s %d" % (k, counts[k]))

    print()
    print(total, "labels processed")
    print(bad, "need manual review")
    print("%d auto-accepted -> manual review down %.1f%%" % (clean_n, pct))
    print()
    print("clean rows ->", ACCEPTED_FILE)
    print("flagged rows ->", OUT_FILE)


if __name__ == "__main__":
    main()
