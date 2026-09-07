# this is my quality gate. it decides whether a crate can go on the run or whether somebody has
# to come and look at it.
#
# i think this is the half that actually matters. reading the label was the easy part, because a
# reader that is right 99% of the time is no use to anybody if you still have to check all 60
# rows to find the wrong ones.
#
# the catalogue and the allergens i check against are real, from open food facts.

import csv
import datetime
import os
import re
import sys

from textutil import edit_distance

INV_FILE = "output/inventory.csv"
MASTER_FILE = "data/product_master.csv"
RUN_FILE = "data/run_spec.csv"
OUT_FILE = "output/flagged.csv"
ACCEPTED_FILE = "output/inventory_final.csv"

DEFAULT_BUILD = "RUN-OAT-COOKIE"

SSCC_FMT = re.compile(r"^\d{18}$")
CODE_FMT = re.compile(r"^\d{13}$")
LOT_FMT = re.compile(r"^L\d{4}-\d{4}$")
DATE_FMT = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# how far off i let a code be and still match it to the catalogue. if i had millions of
# products instead of a few hundred i would drop this to 1
SNAP_MAX = 2

COLUMNS = ["filename", "sscc", "code", "lot_no", "bbe", "qty", "barcode"]

# i check against the fourteen allergens that labelling law actually makes you declare.
# open food facts is crowd sourced so the tags turn up in whatever language the person spoke,
# and some of them are not allergens at all. i had one product declaring Farine, which is just
# french for flour. checking against the real list means an untranslated tag cannot sneak in and
# become an allergen of its own
ALLERGENS = {"gluten", "crustaceans", "eggs", "fish", "peanuts", "soybeans", "milk",
             "nuts", "celery", "mustard", "sesame-seeds", "sulphites", "lupin", "molluscs"}

ALLERGEN_FIX = {
    "avoine": "gluten", "ble": "gluten", "wheat": "gluten", "weizen": "gluten",
    "oats": "gluten", "barley": "gluten", "rye": "gluten",
    "lait": "milk", "leche": "milk", "latte": "milk", "melk": "milk", "milch": "milk",
    "fruits-a-coque": "nuts", "noix": "nuts", "nueces": "nuts", "tree-nuts": "nuts",
    "almonds": "nuts", "hazelnuts": "nuts", "walnuts": "nuts", "cashew-nuts": "nuts",
    "arachides": "peanuts", "cacahuetes": "peanuts",
    "soja": "soybeans", "soy": "soybeans", "soya": "soybeans",
    "oeufs": "eggs", "huevo": "eggs", "egg": "eggs",
    "sesame": "sesame-seeds", "sesamo": "sesame-seeds", "sesamolja": "sesame-seeds",
    "sesamfro": "sesame-seeds", "sesame-seed": "sesame-seeds",
    "sulphur-dioxide-and-sulphites": "sulphites", "sulfites": "sulphites",
    "sulfitos": "sulphites", "sulfites-and-sulfur-dioxide": "sulphites",
    "apio": "celery", "celeri": "celery",
    "moutarde": "mustard", "mostaza": "mustard",
    "poisson": "fish", "pescado": "fish",
    "crustaces": "crustaceans", "molluscs-and-products-thereof": "molluscs",
}

# i ended up with three outcomes rather than pass or fail, because they go to different people.
#   ACCEPTED  books itself in, nobody looks at it
#   HELD      a person has to deal with it. the goods are usually fine
#   REJECTED  not allowed on this run, back to the supplier
#
# the rule i would defend if anybody asked me about this project: you cannot reject a crate on a
# reading you do not trust. so i send anything the reader was unsure about to HELD even when it
# also looks wrong, because the wrongness might just be the misreading talking
UNSURE = {"BARCODE_UNREADABLE", "BATCH_BARCODE_UNREADABLE", "SERIAL_BARCODE_UNREADABLE",
          "CODE_MISMATCH_ON_LABEL", "LOT_MISMATCH_ON_LABEL", "BBE_MISMATCH_ON_LABEL",
          "SERIAL_MISMATCH_ON_LABEL", "CODE_FORMAT_BAD", "LOT_FORMAT_BAD", "BBE_FORMAT_BAD",
          "SERIAL_FORMAT_BAD", "FIELD_MISSING", "QTY_FORMAT_BAD", "QTY_OUT_OF_RANGE",
          "DUPLICATE_CRATE", "ALLERGEN_CHANGEOVER_CHECK"}

NONCONFORMING = {"CODE_NOT_IN_CATALOGUE", "EXPIRED", "NOT_ON_RUN_SPEC", "UNDECLARED_ALLERGEN"}


def base(flag):
    return flag.split(":")[0]


def outcome(flags):
    if any(base(f) in UNSURE for f in flags):
        return "HELD"
    if any(base(f) in NONCONFORMING for f in flags):
        return "REJECTED"
    return "ACCEPTED"


def clean_allergens(raw):
    out = set()
    for a in (raw or "").split("|"):
        a = ALLERGEN_FIX.get(a.strip().lower(), a.strip().lower())
        if a in ALLERGENS:
            out.add(a)
    return out


def load_rules():
    products = {}
    if os.path.exists(MASTER_FILE):
        with open(MASTER_FILE, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                products[r["code"]] = {"product_name": r["product_name"],
                                       "brand": r["brand"],
                                       "pack_size": r["pack_size"],
                                       "category": r["category"],
                                       "allergens": clean_allergens(r["allergens"])}

    runs = {}
    if os.path.exists(RUN_FILE):
        with open(RUN_FILE, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                runs[r["run_id"]] = {
                    "declares": {a for a in r["declares"].split("|") if a},
                    "categories": {c for c in r["categories"].split("|") if c},
                }

    return {"products": products, "runs": runs, "codes": sorted(products)}


# nearest entry in a known list, but only if there is a single clear winner. if two are equally
# close we cannot tell which was meant, and guessing a product code is exactly the kind of quiet
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


# i only want to shout when the print and the barcode are genuinely different numbers. a
# character or two out is just the ocr struggling and the barcode has already fixed it.
# i had this as a flat threshold of two, which was fine for a 10 character batch code, and then
# i added the 18 digit serial and it flagged eight crates. so now it scales with the length
def really_disagrees(printed, scanned):
    if not printed or not scanned:
        return False
    allowed = max(2, round(0.25 * len(scanned)))
    return edit_distance(printed, scanned) >= allowed


# issued is what is already on the run. the changeover check needs it
def check_crate(row, rules, run_id=DEFAULT_BUILD, issued=()):
    row = dict(row)
    flags = []
    products = rules["products"]
    run = rules["runs"].get(run_id, {"declares": set(), "categories": set()})

    # --- which crate is this ---
    # i added the serial because it is the only thing on the label that identifies THIS crate.
    # the product, batch and date are identical across a whole delivery, so before i had it
    # there was no way for me to tell a second crate from the same one scanned twice
    sscc_bc = row.get("sscc_bc", "")
    if sscc_bc:
        if row.get("sscc") and really_disagrees(row["sscc"], sscc_bc):
            flags.append("SERIAL_MISMATCH_ON_LABEL")
        row["sscc"] = sscc_bc
    else:
        flags.append("SERIAL_BARCODE_UNREADABLE")
        if not row.get("sscc"):
            flags.append("FIELD_MISSING:sscc")
        elif not SSCC_FMT.match(row["sscc"]):
            flags.append("SERIAL_FORMAT_BAD")

    if row.get("sscc") and any(row["sscc"] == s.get("sscc") for s in issued):
        flags.append("DUPLICATE_CRATE")     # booked twice, somebody has to void one of them

    # --- the product code ---
    # i let the barcode overrule the printed digits, because EAN-13 carries a check digit and
    # the ocr carries nothing. that turns the printed text into a cross check instead
    bc = row.get("barcode", "")
    printed = row.get("code", "")
    if bc:
        if CODE_FMT.match(printed or "") and really_disagrees(printed, bc):
            flags.append("CODE_MISMATCH_ON_LABEL")
        row["code"] = bc
    else:
        flags.append("BARCODE_UNREADABLE")
        row["code"] = snap(printed, products)
        if not row["code"]:
            flags.append("FIELD_MISSING:code")
        elif not CODE_FMT.match(row["code"]):
            flags.append("CODE_FORMAT_BAD")

    code = row.get("code", "")
    rec = products.get(code)

    # --- do we even know this product ---
    if code and CODE_FMT.match(code) and rec is None:
        flags.append("CODE_NOT_IN_CATALOGUE")

    row["product_name"] = rec["product_name"] if rec else ""
    row["category"] = rec["category"] if rec else ""
    row["allergens"] = "|".join(sorted(rec["allergens"])) if rec else ""

    # --- batch and date ---
    # i put both of these in the second barcode, and it wins for the same reason. a batch code
    # is unique to one delivery so i have no master to check it against, which means without
    # that barcode a misread digit would be completely undetectable
    lot_bc = row.get("lot_bc", "")
    bbe_bc = row.get("bbe_bc", "")

    if lot_bc:
        if LOT_FMT.match(row.get("lot_no") or "") and really_disagrees(row["lot_no"], lot_bc):
            flags.append("LOT_MISMATCH_ON_LABEL")
        row["lot_no"] = lot_bc
    if bbe_bc:
        if DATE_FMT.match(row.get("bbe") or "") and really_disagrees(row["bbe"], bbe_bc):
            flags.append("BBE_MISMATCH_ON_LABEL")
        row["bbe"] = bbe_bc

    if not (lot_bc and bbe_bc):
        flags.append("BATCH_BARCODE_UNREADABLE")

    bbe = row.get("bbe", "")
    if not bbe:
        flags.append("FIELD_MISSING:bbe")
    elif not DATE_FMT.match(bbe):
        flags.append("BBE_FORMAT_BAD")
    elif bbe < today():
        flags.append("EXPIRED")

    if rec:
        # --- is this ingredient even used on this run ---------------------------------------
        if run["categories"] and rec["category"] not in run["categories"]:
            flags.append("NOT_ON_RUN_SPEC:" + rec["category"])

        # --- this is the one i built the whole thing around ---
        # an undeclared allergen is the biggest cause of food recalls, and there is nothing to
        # see on the crate at all. the ingredient is perfectly good, it is just not allowed in
        # this particular product
        extra = rec["allergens"] - run["declares"]
        if extra:
            flags.append("UNDECLARED_ALLERGEN:" + ",".join(sorted(extra)))

        # i flag the first of each allergen onto a run. it is declared so it is not a recall,
        # but it is a changeover point and i think a supervisor should see it rather than have
        # it slide past
        already = set()
        for other in issued:
            already |= clean_allergens(other.get("allergens", ""))
        new_here = (rec["allergens"] & run["declares"]) - already
        if new_here and issued:
            flags.append("ALLERGEN_CHANGEOVER_CHECK:" + ",".join(sorted(new_here)))

    # --- the rest ---
    lot = row.get("lot_no", "")
    if not lot:
        flags.append("FIELD_MISSING:lot_no")
    elif not LOT_FMT.match(lot):
        flags.append("LOT_FORMAT_BAD")

    qty = row.get("qty", "")
    if not qty:
        flags.append("FIELD_MISSING:qty")
    elif not qty.isdigit():
        flags.append("QTY_FORMAT_BAD")
    elif int(qty) <= 0 or int(qty) > 500:
        flags.append("QTY_OUT_OF_RANGE")

    return row, flags


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BUILD

    with open(INV_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rules = load_rules()
    run = rules["runs"].get(run_id, {})
    print("run %s, %d real products in the catalogue" % (run_id, len(rules["products"])))
    print("declared allergens: %s" % (", ".join(sorted(run.get("declares", []))) or "none"))

    flagged, accepted, issued = [], [], []
    counts = {}
    tally = {"ACCEPTED": 0, "HELD": 0, "REJECTED": 0}

    for r in rows:
        fixed, flags = check_crate(r, rules, run_id, issued)
        status = outcome(flags)
        fixed["status"] = status
        fixed["flags"] = ";".join(flags)
        tally[status] += 1

        if status == "ACCEPTED":
            accepted.append(fixed)
        else:
            flagged.append(fixed)
            for fl in flags:
                counts[base(fl)] = counts.get(base(fl), 0) + 1

        # a rejected crate goes back to the supplier, but an accepted or held one is physically
        # at the line, and for changeover purposes that is what counts. getting this wrong is
        # why the changeover check fired sixteen times instead of twice: held crates never made
        # it into the list, so every gluten crate after one looked like the first
        if status != "REJECTED":
            issued.append({"sscc": fixed.get("sscc", ""), "code": fixed.get("code", ""),
                           "allergens": fixed.get("allergens", "")})
            for fl in flags:
                counts[base(fl)] = counts.get(base(fl), 0) + 1

    os.makedirs("output", exist_ok=True)
    extra = ["product_name", "category", "allergens"]
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + extra + ["status", "flags"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(flagged)

    with open(ACCEPTED_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + extra, extrasaction="ignore")
        w.writeheader()
        w.writerows(accepted)

    total = len(rows)
    touched = tally["HELD"] + tally["REJECTED"]
    pct = 100.0 * tally["ACCEPTED"] / total if total else 0

    print()
    print("--- why crates did not go straight through ---")
    for k in sorted(counts, key=lambda x: -counts[x]):
        where = "held" if k in UNSURE else "rejected"
        print("  %-26s %d   (%s)" % (k, counts[k], where))

    print()
    print("%d crates booked in for %s" % (total, run_id))
    print("  ACCEPTED %3d   booked straight in, nobody looks at them" % tally["ACCEPTED"])
    print("  HELD     %3d   reader was not sure, somebody re-reads the crate" % tally["HELD"])
    print("  REJECTED %3d   reading was solid and it is not allowed on this run" % tally["REJECTED"])
    print()
    print("%d of %d needed a person -> manual checking down %.1f%%" % (touched, total, pct))
    print()
    print("accepted ->", ACCEPTED_FILE)
    print("the rest ->", OUT_FILE)


if __name__ == "__main__":
    main()
