# i generate the crate labels here, built out of real products.
#
# real: the EAN-13 codes, brands, pack sizes and allergens, all off open food facts.
# not real: the picture, the batch number and the best before date. those get printed on the
# crate as it is packed and are in no public database anywhere, and i could not measure accuracy
# at all without knowing the right answer. so i test the reading on labels i make, and i do the
# checking against a real catalogue.

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
# i made it this tall to fit three barcodes. i also had to put a gap above them, because the
# quantity line sat 17px above the first one and tesseract kept swallowing it into the barcode
# block. real labels have that gap and now i know why
H = 590

OUT_DIR = "data/labels"
MASTER_FILE = "data/product_master.csv"
TRUTH_FILE = "data/labels_truth.csv"
RUN_FILE = "data/run_spec.csv"

# these are the runs i made up for the factory to be making. each pack declares an allergen
# list, and anything turning up with an allergen outside that list is an undeclared allergen,
# which is the biggest cause of food recalls
RUNS = {
    "RUN-OAT-COOKIE": {
        "declares": ["gluten", "milk"],
        "categories": ["flours", "sugars", "vegetable-oils", "chocolates",
                       "dried-fruits", "breakfast-cereals"],
    },
    "RUN-FREE-FROM": {
        "declares": [],
        "categories": ["flours", "sugars", "vegetable-oils", "dried-fruits", "juices"],
    },
}

# the crates in this batch are all arriving for this run
BUILD = "RUN-OAT-COOKIE"

HEAD_FONTS = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]
MONO_FONTS = ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf", "DejaVuSansMono.ttf"]
FIELD_SIZE = 23
TEXT_INK = 66         # 0 is fresh black ink, higher is a faded thermal print

# this is how beaten up i make the labels. i tuned every one of these against the decode rate
# and the accuracy rather than just picking numbers
BARCODE_MW = 0.45
BLUR = (0.5, 0.9)
NOISE = (3, 7)
SALT = (0.000, 0.002)
JPEG = (58, 78)

# how often a crate turns up with something actually wrong with it
HARD_BARCODE_RATE = 0.03      # label scuffed, barcodes will not scan
UNKNOWN_CODE_RATE = 0.02      # a code that is not in the catalogue
WRONG_CATEGORY_RATE = 0.02    # an ingredient this run does not use
EXPIRED_RATE = 0.02           # past its best before
ALLERGEN_RATE = 0.03          # brings in an allergen the run has not declared
DUPLICATE_RATE = 0.02         # the same physical crate booked in twice

# the first part of an SSCC is the extension digit and the company prefix, which are the same for
# every crate this site ships. only the serial reference on the end changes
SSCC_PREFIX = "05012345"

# the same tidying validate.py does, imported from there so the generator and the checker can
# never disagree about what counts as an allergen
from validate import clean_allergens as tidy_allergens


def clean_allergens(raw):
    return sorted(tidy_allergens(raw))


def load_master():
    with open(MASTER_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["allergen_set"] = clean_allergens(r["allergens"])
    return rows


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


# GS1 mod 10, same idea as the EAN check digit but over 17 digits instead of 12. weights alternate
# 3 and 1 starting from the rightmost data digit
def gs1_check_digit(body):
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return str((10 - total % 10) % 10)


# i added a Serial Shipping Container Code, 18 digits, unique to one physical crate, because
# that is what real pallet labels carry. it is the only thing that says THIS crate rather than
# what is inside it
def make_sscc(serial):
    body = SSCC_PREFIX + "%09d" % serial
    return body + gs1_check_digit(body)


def make_barcode(code):
    # real EAN-13, same as the actual product carries. the check digit on the end is why the
    # barcode gets to overrule the printed text later
    ean = barcode.get_barcode_class("ean13")
    obj = ean(code, writer=ImageWriter())
    buf = io.BytesIO()
    obj.write(buf, options={"module_width": BARCODE_MW, "module_height": 11.0,
                            "quiet_zone": 3.0, "write_text": False, "dpi": 200})
    buf.seek(0)
    return Image.open(buf).convert("L")


# GS1 identifiers: 17 is best before as YYMMDD, 10 is the batch. 17 is fixed length so it goes
# first and 10 is variable so it goes last, then you need no separator between them.
# all digits on purpose, code128 packs digit pairs so 18 digits is about as wide as 9 letters
def gs1_payload(lot_no, bbe):
    yy, mm, dd = bbe[2:4], bbe[5:7], bbe[8:10]
    digits = lot_no[1:5] + lot_no[6:10]        # L2025-1234 -> 20251234
    return "17" + yy + mm + dd + "10" + digits


def make_code128(payload, mw, height):
    obj = barcode.get_barcode_class("code128")(payload, writer=ImageWriter())
    buf = io.BytesIO()
    obj.write(buf, options={"module_width": mw, "module_height": height,
                            "quiet_zone": 2.0, "write_text": False, "dpi": 200})
    buf.seek(0)
    return Image.open(buf).convert("L")


def make_gs1_barcode(lot_no, bbe):
    return make_code128(gs1_payload(lot_no, bbe), 0.36, 8.0)


# the serial gets its own barcode, identifier 00, same as a real pallet label
def make_sscc_barcode(sscc):
    return make_code128("00" + sscc, 0.30, 7.0)


# draws the clean crate label before any damage is added
def draw_label(brand, sscc, code, lot_no, bbe, qty):
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    head = load_font(27)
    body = load_font(FIELD_SIZE, mono=True)
    small = load_font(15, mono=True)

    d.text((22, 16), brand.upper()[:26], font=head, fill=0)
    d.line([(22, 54), (W - 22, 54)], fill=0, width=2)

    # i print these grey rather than black, because thermal printers fade. this is the setting
    # that makes the reader work for its money. i leave the header and barcodes solid
    d.text((26, 70), "SSC: " + sscc, font=body, fill=TEXT_INK)
    d.text((26, 104), "PRD: " + code, font=body, fill=TEXT_INK)
    d.text((26, 138), "LOT: " + lot_no, font=body, fill=TEXT_INK)
    d.text((26, 172), "BBE: " + bbe, font=body, fill=TEXT_INK)
    d.text((26, 206), "QTY: " + str(qty), font=body, fill=TEXT_INK)

    img.paste(make_barcode(code), (26, 266))
    img.paste(make_gs1_barcode(lot_no, bbe), (26, 382))
    img.paste(make_sscc_barcode(sscc), (26, 474))

    d.text((26, 556), "GOODS IN - CRATE LABEL", font=small, fill=0)
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
    return Image.fromarray(np.clip(a * grad, 0, 255).astype(np.uint8))


def add_gauss_noise(img, sigma):
    a = np.asarray(img).astype(np.float32) + np.random.normal(0, sigma, (img.size[1], img.size[0]))
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
        # smudge all three. scuffing one is a much easier problem than a crate dragged along a wall
        box = (20, 258, 480, 550)
        strip = img.crop(box).filter(ImageFilter.GaussianBlur(random.uniform(2.4, 3.6)))
        strip = add_gauss_noise(strip, 22)
        img.paste(strip, box)

    img = img.rotate(random.uniform(-4, 4), resample=Image.BICUBIC, fillcolor=255)
    img = add_lighting(img)

    # i swept all of these against zbar before settling on them. noise past sigma 12 and jpeg
    # below q40 killed the barcode on every single label, and speckle turned out worse than it
    # looks because it lands after the blur and bridges the bars. so i put the difficulty into
    # the ink instead, where zbar cannot see it
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(*BLUR)))
    img = add_gauss_noise(img, random.uniform(*NOISE))
    img = add_salt_pepper(img, random.uniform(*SALT))
    img = jpeg_squash(img, random.randint(*JPEG))
    return img


def iso(days_from_now):
    return (datetime.date.today() + datetime.timedelta(days=days_from_now)).isoformat()


def fake_code(master):
    # a 13 digit code that passes the EAN check digit but is not in the catalogue
    while True:
        body = "".join(random.choice("0123456789") for _ in range(12))
        total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(body))
        code = body + str((10 - total % 10) % 10)
        if code not in master:
            return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(MASTER_FILE):
        print("no product master yet. run: python fetch_products.py")
        return

    products = load_master()
    by_code = {p["code"]: p for p in products}
    run = RUNS[BUILD]
    declared = set(run["declares"])

    # an ingredient this run can legitimately use: right category, and no allergen beyond what
    # the finished pack already declares
    ok_pool = [p for p in products
               if p["category"] in run["categories"] and set(p["allergen_set"]) <= declared]
    off_spec = [p for p in products if p["category"] not in run["categories"]]
    allergenic = [p for p in products
                  if p["category"] in run["categories"] and set(p["allergen_set"]) - declared]

    print("catalogue: %d real products, %d usable on %s, %d with an undeclared allergen"
          % (len(products), len(ok_pool), BUILD, len(allergenic)))
    if not ok_pool:
        print("no usable products, check data/product_master.csv")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    # i wipe the folder first. i changed the label format once and left the old images sitting
    # there, and the reader picked up 120 labels against a 60 row truth file
    for old in os.listdir(OUT_DIR):
        if old.endswith(".png"):
            os.remove(os.path.join(OUT_DIR, old))

    # i decide up front which crates are bad instead of rolling dice per crate, and i round each
    # rate up to at least one so every rule gets exercised. when i was rolling dice, one run came
    # out with no undeclared allergen in it at all and my most important check just sat there
    plan = []
    for name, rate in [("hard_barcode", HARD_BARCODE_RATE),
                       ("unknown_code", UNKNOWN_CODE_RATE),
                       ("wrong_category", WRONG_CATEGORY_RATE),
                       ("expired", EXPIRED_RATE),
                       ("undeclared_allergen", ALLERGEN_RATE),
                       ("duplicate_crate", DUPLICATE_RATE)]:
        plan += [name] * max(1, round(rate * args.n))
    plan += [""] * (args.n - len(plan))
    random.shuffle(plan)

    # a duplicate has to copy a crate that already went past, so it cannot be the first one in
    for i in range(min(5, len(plan))):
        if plan[i] == "duplicate_crate":
            j = random.randrange(5, len(plan))
            plan[i], plan[j] = plan[j], plan[i]

    rows = []
    seen = []
    for i in range(args.n):
        defect = plan[i]
        defects = [defect] if defect else []

        if defect == "wrong_category" and off_spec:
            p = random.choice(off_spec)
        elif defect == "undeclared_allergen" and allergenic:
            p = random.choice(allergenic)
        else:
            p = random.choice(ok_pool)

        code = p["code"]
        if defect == "unknown_code":
            code = fake_code(by_code)

        bbe = iso(-random.randint(5, 300)) if defect == "expired" else iso(random.randint(60, 700))
        lot_no = make_lot()
        qty = random.randint(1, 500)
        sscc = make_sscc(1000 + i)

        if defect == "duplicate_crate" and seen:
            # copy everything. it is not another crate, it is the same one scanned twice
            prev = random.choice(seen)
            sscc, code, lot_no, bbe, qty = prev["sscc"], prev["code"], prev["lot_no"], prev["bbe"], prev["qty"]
            p = {"brand": prev["brand"]}

        img = draw_label(p["brand"], sscc, code, lot_no, bbe, qty)
        img = degrade(img, defect == "hard_barcode")

        fname = "crate_%03d.png" % i
        img.save(os.path.join(OUT_DIR, fname))
        rows.append([fname, p["brand"], sscc, code, lot_no, bbe, qty, "|".join(defects)])
        if not defects:
            seen.append({"sscc": sscc, "code": code, "lot_no": lot_no, "bbe": bbe,
                         "qty": qty, "brand": p["brand"]})

        if (i + 1) % 10 == 0:
            print("made", i + 1, "labels")

    with open(TRUTH_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "brand", "sscc", "code", "lot_no", "bbe", "qty", "defects"])
        w.writerows(rows)

    with open(RUN_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "declares", "categories"])
        for rid, spec in RUNS.items():
            w.writerow([rid, "|".join(spec["declares"]), "|".join(spec["categories"])])

    print("done ->", TRUTH_FILE, "and", RUN_FILE)


if __name__ == "__main__":
    main()
