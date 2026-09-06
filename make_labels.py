# makes fake component labels so i have images with known ground truth to test the ocr against.
# i looked for a public dataset of real factory labels with serial/lot annotations and there isnt one,
# so generating them was the only way to actually measure accuracy instead of eyeballing it.

import argparse
import csv
import io
import os
import random

import numpy as np
import barcode
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W = 600
H = 380

OUT_DIR = "data/labels"
TRUTH_FILE = "data/labels_truth.csv"
LOT_FILE = "data/lot_master.csv"

VENDORS = ["ACME COMPONENTS", "NORDEX PARTS", "TITAN BEARINGS",
           "VOLTA ELECTRIC", "KRAMER TOOLING", "PRIME CASTINGS"]

# how beaten up the labels get. i tuned these by running the decode rate and the ocr accuracy
# after every change, they are not arbitrary
BARCODE_MW = 0.4
BLUR = (0.5, 0.9)
NOISE = (3, 7)
SALT = (0.000, 0.002)
JPEG = (58, 78)
# how often a label comes off the line actually defective. i had these at 5 and 8 percent to
# start with and that is not a factory, that is a factory with a serious problem. at those rates
# 13 of 60 labels were genuinely broken, so even a perfect reader would have to send 22 percent
# of them to a human and the most you could ever save is 78 percent. 3 percent each is closer to
# what a line that is running properly looks like
BAD_LOT_RATE = 0.03
HARD_BARCODE_RATE = 0.03


HEAD_FONTS = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]
MONO_FONTS = ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf", "DejaVuSansMono.ttf"]
FIELD_SIZE = 23
TEXT_INK = 66         # 0 is fresh black ink, higher is a faded thermal print


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


def make_part_no():
    return "%04d-%s%d" % (random.randint(1000, 9999),
                          random.choice("ABCDEFGHJKLMNPRSTUVWXYZ"),
                          random.randint(1, 9))


def make_serial():
    return "SN%07d" % random.randint(1000000, 9999999)


def make_lot():
    return "L%d-%04d" % (random.choice([2023, 2024, 2025]), random.randint(1, 9999))


def make_barcode(serial):
    code128 = barcode.get_barcode_class("code128")
    obj = code128(serial, writer=ImageWriter())
    buf = io.BytesIO()
    # 0.4 / 200dpi comes out about 418px wide which drops onto the canvas at its natural size.
    # i was rendering it small and scaling up before, that softened the bar edges and zbar
    # could not read a single one
    obj.write(buf, options={"module_width": BARCODE_MW, "module_height": 11.0,
                            "quiet_zone": 2.0, "write_text": False, "dpi": 200})
    buf.seek(0)
    return Image.open(buf).convert("L")


# draws the clean label before any damage is added
def draw_label(vendor, part_no, serial, lot_no, qty):
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    head = load_font(27)
    body = load_font(FIELD_SIZE, mono=True)
    small = load_font(15, mono=True)

    d.text((22, 16), vendor, font=head, fill=0)
    d.line([(22, 54), (W - 22, 54)], fill=0, width=2)

    # the field lines print in grey, thermal label printers fade and this is the bit that
    # actually makes the ocr work for its money. header and barcode stay solid black
    d.text((26, 70), "PN:  " + part_no, font=body, fill=TEXT_INK)
    d.text((26, 104), "SN:  " + serial, font=body, fill=TEXT_INK)
    d.text((26, 138), "LOT: " + lot_no, font=body, fill=TEXT_INK)
    d.text((26, 172), "QTY: " + str(qty), font=body, fill=TEXT_INK)

    bc = make_barcode(serial)
    img.paste(bc, (26, 212))

    d.text((26, 330), "MADE IN INDIA", font=small, fill=0)
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


# beats up a clean label so the ocr actually has to work for it
def degrade(img, hard_barcode):
    if hard_barcode:
        # smudge just the barcode strip. these are the unreadable-barcode cases validate.py catches
        box = (20, 205, 450, 320)
        strip = img.crop(box).filter(ImageFilter.GaussianBlur(random.uniform(2.4, 3.6)))
        strip = add_gauss_noise(strip, 22)
        img.paste(strip, box)

    img = img.rotate(random.uniform(-4, 4), resample=Image.BICUBIC, fillcolor=255)
    img = add_lighting(img)

    # i swept these against zbar before settling on them. sensor noise past sigma 12 and jpeg below
    # q40 kill the barcode on literally every label, and speckle is much worse than it looks because
    # it lands after the blur, so the dots sit on top of already soft bars and bridge them.
    # keeping the damage in the ranges the barcode survives is the only way the mismatch check has
    # anything to work with
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(*BLUR)))
    img = add_gauss_noise(img, random.uniform(*NOISE))
    img = add_salt_pepper(img, random.uniform(*SALT))
    img = jpeg_squash(img, random.randint(*JPEG))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # the approved lots, like what the erp system would hold. a few labels get a lot outside this
    # list on purpose so the mismatch check has real cases and not only ocr mistakes to find
    approved = [make_lot() for _ in range(12)]

    rows = []
    for i in range(args.n):
        vendor = random.choice(VENDORS)
        part_no = make_part_no()
        serial = make_serial()
        qty = random.randint(1, 500)

        bad_lot = random.random() < BAD_LOT_RATE
        lot_no = make_lot() if bad_lot else random.choice(approved)

        hard_barcode = random.random() < HARD_BARCODE_RATE

        img = draw_label(vendor, part_no, serial, lot_no, qty)
        img = degrade(img, hard_barcode)

        # write down what i deliberately broke on this label. validate.py never sees this, it is
        # only so i can check afterwards whether the flags it raised were real problems or just
        # the ocr having a bad day
        defects = []
        if bad_lot:
            defects.append("bad_lot")
        if hard_barcode:
            defects.append("hard_barcode")

        fname = "label_%03d.png" % i
        img.save(os.path.join(OUT_DIR, fname))
        rows.append([fname, vendor, part_no, serial, lot_no, qty, "|".join(defects)])

        if (i + 1) % 10 == 0:
            print("made", i + 1, "labels")

    with open(TRUTH_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "vendor", "part_no", "serial", "lot_no", "qty", "defects"])
        w.writerows(rows)

    with open(LOT_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lot_no"])
        for lot in approved:
            w.writerow([lot])

    print("done ->", TRUTH_FILE, "and", LOT_FILE)


if __name__ == "__main__":
    main()
