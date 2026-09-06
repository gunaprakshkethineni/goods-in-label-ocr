# makes fake material crate labels so i have images with known ground truth to test the reader on.
# i looked for a public dataset of real goods-in labels with the lot numbers written down
# alongside and there is not one, nobody publishes their material traceability records, so
# generating them was the only way to actually measure accuracy instead of eyeballing it.
#
# the crates are the stuff that goes into a wind turbine blade: epoxy resin, amine hardener,
# carbon and glass fabric, adhesive, studs. supplier names are invented.

import argparse
import csv
import datetime
import io
import os
import random

import barcode
import numpy as np
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W = 600
# 400 not 380. the quantity line used to sit about 17px above the barcode and tesseract kept
# swallowing it into the barcode block and returning nothing for it at all. real labels have a
# gap there for exactly this reason
H = 400

OUT_DIR = "data/labels"
TRUTH_FILE = "data/labels_truth.csv"
LOT_FILE = "data/material_master.csv"
SPEC_FILE = "data/blade_spec.csv"
COMPAT_FILE = "data/compatibility.csv"

SUPPLIERS = ["VESTRA POLYMERS", "CARBOLINE FIBRES", "AXIOM RESINS",
             "TENSA COMPOSITES", "HELIOS ADHESIVES", "KESTREL FASTENERS"]

# every code is 3 letters, 2 letters, 4 digits so one mask fits all of them.
# the first three letters say what class of material it is, which is what the compatibility
# check keys off
MATERIALS = [
    ("RES-EP-2400", "epoxy resin, system A"),
    ("RES-EP-2600", "epoxy resin, system B"),
    ("HRD-AM-1150", "amine hardener for system A"),
    ("HRD-AM-1180", "amine hardener for system B"),
    ("FAB-CF-0600", "carbon fabric 600gsm"),
    ("FAB-GF-1200", "glass fabric 1200gsm"),
    ("ADH-MA-0320", "methacrylate adhesive"),
    ("FST-ST-0880", "M24 steel stud"),
]

# which hardener actually cures which resin. mixing across these two systems is the failure that
# looks fine on the shop floor and cracks three years later offshore
COMPATIBLE = {"RES-EP-2400": "HRD-AM-1150",
              "RES-EP-2600": "HRD-AM-1180"}

# the plant is qualified for both resin systems so a build may draw either one, it just must not
# mix them. the fabric is what actually differs between these two blades
BLADE_SPEC = {
    "BLADE-402": ["RES-EP-2400", "RES-EP-2600", "HRD-AM-1150", "HRD-AM-1180",
                  "FAB-CF-0600", "ADH-MA-0320", "FST-ST-0880"],
    "BLADE-518": ["RES-EP-2400", "RES-EP-2600", "HRD-AM-1150", "HRD-AM-1180",
                  "FAB-GF-1200", "ADH-MA-0320", "FST-ST-0880"],
}

# the crates in this batch are all arriving for this build
BUILD = "BLADE-402"

# what a normal delivery for that build looks like. resin and hardener from system A, carbon
# fabric, plus the adhesive and studs
NORMAL = ["RES-EP-2400", "HRD-AM-1150", "FAB-CF-0600", "ADH-MA-0320", "FST-ST-0880"]

HEAD_FONTS = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]
MONO_FONTS = ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf", "DejaVuSansMono.ttf"]
FIELD_SIZE = 23
TEXT_INK = 66         # 0 is fresh black ink, higher is a faded thermal print

# how beaten up the labels get. i tuned these by running the decode rate and the ocr accuracy
# after every change, they are not arbitrary
BARCODE_MW = 0.4
BLUR = (0.5, 0.9)
NOISE = (3, 7)
SALT = (0.000, 0.002)
JPEG = (58, 78)

# how often a crate turns up with something actually wrong with it. these are meant to be what a
# line running properly looks like, not a plant in crisis
HARD_BARCODE_RATE = 0.03      # label scuffed, barcode will not scan
UNKNOWN_LOT_RATE = 0.02       # lot not in the material master at all
WRONG_MATERIAL_RATE = 0.02    # material not called for on this blade
EXPIRED_RATE = 0.02           # past its shelf life
NOT_RELEASED_RATE = 0.02      # quality have not signed the lot off yet
INCOMPATIBLE_RATE = 0.02      # hardener from the other resin system


def load_font(size, mono=False):
    # windows font paths. falls back to the pillow default so this still runs on someone elses laptop
    names = MONO_FONTS if mono else HEAD_FONTS
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    print("warning: no truetype font found, labels will look bad")
    return ImageFont.load_default()


def make_lot():
    return "L%d-%04d" % (random.choice([2024, 2025, 2026]), random.randint(1, 9999))


def make_barcode(lot):
    code128 = barcode.get_barcode_class("code128")
    obj = code128(lot, writer=ImageWriter())
    buf = io.BytesIO()
    # 0.4 / 200dpi comes out about 418px wide which drops onto the canvas at its natural size.
    # i was rendering it small and scaling up before, that softened the bar edges and zbar
    # could not read a single one
    obj.write(buf, options={"module_width": BARCODE_MW, "module_height": 11.0,
                            "quiet_zone": 2.0, "write_text": False, "dpi": 200})
    buf.seek(0)
    return Image.open(buf).convert("L")


# draws the clean crate label before any damage is added
def draw_label(supplier, material, lot_no, expiry, qty):
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    head = load_font(27)
    body = load_font(FIELD_SIZE, mono=True)
    small = load_font(15, mono=True)

    d.text((22, 16), supplier, font=head, fill=0)
    d.line([(22, 54), (W - 22, 54)], fill=0, width=2)

    # the field lines print in grey, thermal label printers fade and this is the bit that
    # actually makes the ocr work for its money. header and barcode stay solid black
    d.text((26, 70), "MAT: " + material, font=body, fill=TEXT_INK)
    d.text((26, 104), "LOT: " + lot_no, font=body, fill=TEXT_INK)
    d.text((26, 138), "EXP: " + expiry, font=body, fill=TEXT_INK)
    d.text((26, 172), "QTY: " + str(qty), font=body, fill=TEXT_INK)

    # the barcode carries the lot number, not the material. the lot is the field that matters for
    # traceability and it is the one you least want to get wrong
    bc = make_barcode(lot_no)
    img.paste(bc, (26, 232))

    d.text((26, 352), "GOODS IN - BLADE MATERIALS", font=small, fill=0)
    d.rectangle([(2, 2), (W - 3, H - 3)], outline=0, width=2)
    return img


def add_lighting(img):
    # simulates a shadow falling across the label. this is the one that broke plain otsu thresholding
    a = np.asarray(img).astype(np.float32)
    gx = np.linspace(random.uniform(0.55, 0.85), 1.0, W)
    gy = np.linspace(1.0, random.uniform(0.7, 1.0), H)
    grad = np.outer(gy, gx)
    if random.random() < 0.5:
        grad = np.fliplr(grad)
    a = a * grad
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def add_gauss_noise(img, sigma):
    a = np.asarray(img).astype(np.float32)
    a = a + np.random.normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def add_salt_pepper(img, amount):
    a = np.asarray(img).copy()
    n = int(a.size * amount)
    ys = np.random.randint(0, a.shape[0], n)
    xs = np.random.randint(0, a.shape[1], n)
    a[ys[:n // 2], xs[:n // 2]] = 0
    a[ys[n // 2:], xs[n // 2:]] = 255
    return Image.fromarray(a)


def jpeg_squash(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("L")


# beats up a clean label so the reader actually has to work for it
def degrade(img, hard_barcode):
    if hard_barcode:
        # smudge just the barcode strip. these are the unreadable-barcode cases validate.py catches
        box = (20, 225, 450, 340)
        strip = img.crop(box).filter(ImageFilter.GaussianBlur(random.uniform(2.4, 3.6)))
        strip = add_gauss_noise(strip, 22)
        img.paste(strip, box)

    img = img.rotate(random.uniform(-4, 4), resample=Image.BICUBIC, fillcolor=255)
    img = add_lighting(img)

    # i swept these against zbar before settling on them. sensor noise past sigma 12 and jpeg below
    # q40 kill the barcode on literally every label, and speckle is much worse than it looks because
    # it lands after the blur, so the dots sit on top of already soft bars and bridge them.
    # keeping the damage in the ranges the barcode survives is the only way the lot cross check has
    # anything to work with
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(*BLUR)))
    img = add_gauss_noise(img, random.uniform(*NOISE))
    img = add_salt_pepper(img, random.uniform(*SALT))
    img = jpeg_squash(img, random.randint(*JPEG))
    return img


def iso(days_from_now):
    return (datetime.date.today() + datetime.timedelta(days=days_from_now)).isoformat()


# builds the material master. every material gets two lots that are fine, one that is out of
# date and one quality have not released yet. they have to exist for every material or the
# expired and not-released checks never get exercised, which is exactly the mistake i made first
# time round: both rules sat there looking correct and never once fired
def build_master():
    rows = []
    for code, _desc in MATERIALS:
        for kind in ["good", "good", "expired", "held"]:
            rows.append({"lot_no": make_lot(),
                         "material": code,
                         "supplier": random.choice(SUPPLIERS),
                         "released": "N" if kind == "held" else "Y",
                         "expiry": iso(-random.randint(10, 400)) if kind == "expired"
                                   else iso(random.randint(120, 900))})
    return rows


def pick_lot(master, material, kind="good"):
    pool = [r for r in master if r["material"] == material]
    if kind == "expired":
        want = [r for r in pool if r["expiry"] < iso(0)]
    elif kind == "held":
        want = [r for r in pool if r["released"] != "Y"]
    else:
        want = [r for r in pool if r["released"] == "Y" and r["expiry"] >= iso(0)]
    return random.choice(want) if want else random.choice(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # wipe whatever is in there first. i changed the label format once and left the old images
    # behind, so the reader picked up 120 labels for a 60 label truth file and every stale one
    # came back as a fistful of missing fields
    for old in os.listdir(OUT_DIR):
        if old.endswith(".png"):
            os.remove(os.path.join(OUT_DIR, old))

    master = build_master()
    by_lot = {r["lot_no"]: r for r in master}

    # decide up front which crates are bad and what is wrong with them, rather than rolling dice
    # per crate. rounding the rate up to at least one means every rule in validate.py actually
    # gets exercised by the batch. i had it random first and one run came out with no incompatible
    # hardener at all, so the most important check in the whole thing sat there never firing
    plan = []
    for name, rate in [("wrong_material", WRONG_MATERIAL_RATE),
                       ("incompatible", INCOMPATIBLE_RATE),
                       ("expired", EXPIRED_RATE),
                       ("not_released", NOT_RELEASED_RATE),
                       ("unknown_lot", UNKNOWN_LOT_RATE),
                       ("hard_barcode", HARD_BARCODE_RATE)]:
        plan += [name] * max(1, round(rate * args.n))
    plan += [""] * (args.n - len(plan))
    random.shuffle(plan)

    # a system B hardener is only incompatible with a system A resin that is already out on the
    # build, so it must not be the first thing through the door
    for i in range(min(10, len(plan))):
        if plan[i] == "incompatible":
            j = random.randrange(10, len(plan))
            plan[i], plan[j] = plan[j], plan[i]

    rows = []
    for i in range(args.n):
        defect = plan[i]
        defects = [defect] if defect else []

        material = random.choice(NORMAL)
        kind = "good"
        hard_barcode = False

        if defect == "wrong_material":
            material = "FAB-GF-1200"      # glass fabric turning up for a carbon blade
        elif defect == "incompatible":
            material = "HRD-AM-1180"      # the hardener off the other resin system
        elif defect == "expired":
            kind = "expired"
        elif defect == "not_released":
            kind = "held"
        elif defect == "hard_barcode":
            hard_barcode = True

        rec = pick_lot(master, material, kind)
        lot_no = rec["lot_no"]
        expiry = rec["expiry"]
        supplier = rec["supplier"]

        if defect == "unknown_lot":
            # a lot number nobody has ever booked in
            lot_no = make_lot()
            while lot_no in by_lot:
                lot_no = make_lot()

        qty = random.randint(1, 500)

        img = draw_label(supplier, material, lot_no, expiry, qty)
        img = degrade(img, hard_barcode)

        # write down what is wrong with this crate. validate.py never sees this, it is only so i
        # can check afterwards whether the flags it raised were real problems or the ocr having
        # a bad day
        fname = "crate_%03d.png" % i
        img.save(os.path.join(OUT_DIR, fname))
        rows.append([fname, supplier, material, lot_no, expiry, qty, "|".join(defects)])

        if (i + 1) % 10 == 0:
            print("made", i + 1, "labels")

    with open(TRUTH_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "supplier", "material", "lot_no", "expiry", "qty", "defects"])
        w.writerows(rows)

    with open(LOT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lot_no", "material", "supplier", "released", "expiry"])
        w.writeheader()
        w.writerows(master)

    with open(SPEC_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["work_order", "material"])
        for wo, mats in BLADE_SPEC.items():
            for m in mats:
                w.writerow([wo, m])

    with open(COMPAT_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["resin", "hardener"])
        for r, h in COMPATIBLE.items():
            w.writerow([r, h])

    print("done ->", TRUTH_FILE)
    print("       ", LOT_FILE, "(%d lots)" % len(master))
    print("       ", SPEC_FILE, "and", COMPAT_FILE)


if __name__ == "__main__":
    main()
