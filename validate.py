# the quality gate. takes what the reader got off a crate label and decides whether that crate is
# allowed onto the build, or whether somebody has to come and look at it.
#
# reading the label is the easy half. this is the half that matters, because a reader that is
# right 96 percent of the time is no use if you still have to check all 60 rows to find the wrong
# ones. everything here exists to answer "can i trust this reading, and is this crate right for
# this blade".

import csv
import datetime
import os
import re
import sys

from textutil import edit_distance

INV_FILE = "output/inventory.csv"
LOT_FILE = "data/material_master.csv"
SPEC_FILE = "data/blade_spec.csv"
COMPAT_FILE = "data/compatibility.csv"
OUT_FILE = "output/flagged.csv"
ACCEPTED_FILE = "output/inventory_final.csv"

# the batch of crates in data/labels all arrived for this build. the app lets you pick
DEFAULT_BUILD = "BLADE-402"

MAT_FMT = re.compile(r"^[A-Z]{3}-[A-Z]{2}-\d{4}$")
LOT_FMT = re.compile(r"^L\d{4}-\d{4}$")
DATE_FMT = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# how far off a code can be and still get snapped to the master list. the catalogues are small
# and the codes are ten characters, so two of them are never this close to each other. against a
# master of thousands of lots i would drop this to 1
SNAP_MAX = 2

COLUMNS = ["filename", "material", "lot_no", "expiry", "qty", "barcode"]


def load_rules():
    lots = {}
    if os.path.exists(LOT_FILE):
        with open(LOT_FILE, newline="") as f:
            for r in csv.DictReader(f):
                lots[r["lot_no"]] = {"material": r["material"],
                                     "supplier": r["supplier"],
                                     "released": r["released"].strip().upper() == "Y",
                                     "expiry": r["expiry"]}

    spec = {}
    if os.path.exists(SPEC_FILE):
        with open(SPEC_FILE, newline="") as f:
            for r in csv.DictReader(f):
                spec.setdefault(r["work_order"], set()).add(r["material"])

    compat = {}
    if os.path.exists(COMPAT_FILE):
        with open(COMPAT_FILE, newline="") as f:
            for r in csv.DictReader(f):
                compat[r["resin"]] = r["hardener"]

    materials = sorted({v["material"] for v in lots.values()})
    return {"lots": lots, "spec": spec, "compat": compat, "materials": materials}


# nearest entry in a known list, but only if there is a single clear winner. if two are equally
# close we cannot tell which was meant, and guessing a lot number is exactly the kind of quiet
# mistake this whole thing exists to prevent
def snap(value, known):
    if not value or value in known:
        return value
    scored = sorted((edit_distance(value, k), k) for k in known)
    if not scored or scored[0][0] > SNAP_MAX:
        return value
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return value
    return scored[0][1]


def today():
    return datetime.date.today().isoformat()


def class_of(material):
    return material.split("-")[0] if material else ""


# checks one crate against the build it is meant to be going onto.
# issued is what has already been accepted onto that build, which is what makes the compatibility
# check possible: whether this hardener is right depends on which resin is already out there.
def check_crate(row, rules, work_order=DEFAULT_BUILD, issued=()):
    row = dict(row)
    flags = []
    lots = rules["lots"]

    # --- the lot number, which is the field everything else hangs off -------------------------
    # code128 carries a check digit and the ocr carries nothing, so when the barcode decodes it
    # overrules the text. the printed lot is then only useful as a cross check
    bc = row.get("barcode", "")
    ocr_lot = row.get("lot_no", "")
    if bc:
        if ocr_lot and LOT_FMT.match(ocr_lot) and ocr_lot != bc:
            flags.append("LOT_MISMATCH_ON_LABEL")
        row["lot_no"] = bc
    else:
        flags.append("BARCODE_UNREADABLE")
        row["lot_no"] = snap(ocr_lot, lots)
        if not row["lot_no"]:
            flags.append("FIELD_MISSING:lot_no")
        elif not LOT_FMT.match(row["lot_no"]):
            flags.append("LOT_FORMAT_BAD")

    lot = row.get("lot_no", "")
    rec = lots.get(lot)

    # --- the material code --------------------------------------------------------------------
    row["material"] = snap(row.get("material", ""), rules["materials"])
    material = row["material"]
    if not material:
        flags.append("FIELD_MISSING:material")
    elif not MAT_FMT.match(material):
        flags.append("MATERIAL_FORMAT_BAD")

    # --- is this lot one we have actually booked in -------------------------------------------
    if lot and LOT_FMT.match(lot) and rec is None:
        flags.append("LOT_NOT_IN_MASTER")

    if rec:
        # the label says one material but the master says that lot is something else. either the
        # label is on the wrong drum or somebody relabelled it, and neither is ok
        if material and MAT_FMT.match(material) and rec["material"] != material:
            flags.append("LABEL_MATERIAL_MISMATCH")
            row["material"] = rec["material"]
            material = rec["material"]

        if not rec["released"]:
            flags.append("LOT_NOT_RELEASED")

        # the master holds the real shelf life, so use it and repair whatever the ocr made of the
        # printed date
        row["expiry"] = rec["expiry"]
        if rec["expiry"] < today():
            flags.append("EXPIRED")
    else:
        exp = row.get("expiry", "")
        if exp and DATE_FMT.match(exp) and exp < today():
            flags.append("EXPIRED")

    # --- is this material even called for on this blade ---------------------------------------
    allowed = rules["spec"].get(work_order, set())
    if allowed and material and MAT_FMT.match(material) and material not in allowed:
        flags.append("NOT_ON_BUILD_SPEC")

    # --- and does it go with what is already out on that build --------------------------------
    # the plant is qualified for two epoxy systems and a build may draw either, it just must not
    # mix them. this is the failure that passes every check on its own and still cracks the blade
    # three years later, because nothing is wrong with either drum, only with the pair
    compat = rules["compat"]
    for other in issued:
        om = other.get("material", "")
        pair = None
        if class_of(material) == "HRD" and class_of(om) == "RES":
            pair = (om, material)
        elif class_of(material) == "RES" and class_of(om) == "HRD":
            pair = (material, om)
        if pair and compat.get(pair[0]) and compat[pair[0]] != pair[1]:
            flags.append("INCOMPATIBLE_WITH_ISSUED:" + other.get("lot_no", om))
            break

    # --- quantity ------------------------------------------------------------------------------
    qty = row.get("qty", "")
    if not qty:
        flags.append("FIELD_MISSING:qty")
    elif not qty.isdigit():
        flags.append("QTY_FORMAT_BAD")
    elif int(qty) <= 0 or int(qty) > 500:
        flags.append("QTY_OUT_OF_RANGE")

    return row, flags


def main():
    work_order = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BUILD

    with open(INV_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    rules = load_rules()
    print("build %s, %d lots in the material master, %d materials on spec"
          % (work_order, len(rules["lots"]), len(rules["spec"].get(work_order, set()))))

    flagged = []
    accepted = []
    issued = []
    counts = {}
    for r in rows:
        fixed, flags = check_crate(r, rules, work_order, issued)
        if flags:
            fixed["flags"] = ";".join(flags)
            flagged.append(fixed)
            for fl in flags:
                key = fl.split(":")[0]
                counts[key] = counts.get(key, 0) + 1
        else:
            accepted.append(fixed)
            # only what actually passed counts as out on the build
            issued.append({"material": fixed.get("material", ""), "lot_no": fixed.get("lot_no", "")})

    os.makedirs("output", exist_ok=True)
    cols = COLUMNS + ["flags"]
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(flagged)

    with open(ACCEPTED_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(accepted)

    total = len(rows)
    bad = len(flagged)
    clean_n = total - bad
    pct = 100.0 * clean_n / total if total else 0

    print()
    print("--- why crates were held ---")
    for k in sorted(counts, key=lambda x: -counts[x]):
        print("  %-26s %d" % (k, counts[k]))

    print()
    print(total, "crates booked in")
    print(bad, "held for a person")
    print("%d went straight onto the build -> manual checking down %.1f%%" % (clean_n, pct))
    print()
    print("cleared ->", ACCEPTED_FILE)
    print("held    ->", OUT_FILE)


if __name__ == "__main__":
    main()
