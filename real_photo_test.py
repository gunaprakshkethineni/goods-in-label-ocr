# runs the reader over real photographs instead of labels i drew myself.
#
#   python real_photo_test.py --n 30
#
# photos off open food facts, taken by whoever with whatever phone: bad light, glare, packaging
# bent round a jar. none of it is synthetic and none of it is kind.
#
# it cannot test the crate label parser, and that is worth being clear about. a jar of nutella
# has no SSC or BBE printed on it in a fixed layout. it tests the two things underneath: whether
# the barcode decoder can be trusted on a real photo, and whether the opencv cleanup plus
# tesseract can get the brand name off one.
#
# cached in data/real_photos. they are other people's photos under CC-BY-SA, so not committed.

import argparse
import csv
import json
import os
import random
import time
import urllib.request

import cv2
import numpy as np
import pytesseract
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol

from preprocess import clean

TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS):
    pytesseract.pytesseract.tesseract_cmd = TESS

MASTER = "data/product_master.csv"
CACHE = "data/real_photos"
HDR = {"User-Agent": "factory-ocr-student-project/1.0 (coursework)"}

# a phone photo is far bigger than anything this was tuned on
WORK_WIDTH = 1200
MAX_BYTES = 6_000_000


def image_folder(code):
    c = code.zfill(13)
    return "https://images.openfoodfacts.org/images/products/%s/%s/%s/%s/" % (
        c[:3], c[3:6], c[6:9], c[9:])


def get(url, timeout=45):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=timeout).read()


def photo_ids(code):
    url = "https://world.openfoodfacts.org/api/v2/product/%s.json?fields=images" % code
    meta = json.loads(get(url, 25))
    return [k for k in meta.get("product", {}).get("images", {}) if k.isdigit()]


def fetch_photos(code, want=3):
    os.makedirs(CACHE, exist_ok=True)
    have = sorted(f for f in os.listdir(CACHE) if f.startswith(code + "_"))
    if have:
        return [os.path.join(CACHE, f) for f in have]

    try:
        ids = photo_ids(code)
    except Exception:
        return []

    saved = []
    for k in ids[:want]:
        try:
            raw = get(image_folder(code) + k + ".jpg")
        except Exception:
            continue
        if len(raw) > MAX_BYTES:
            continue
        path = os.path.join(CACHE, "%s_%s.jpg" % (code, k))
        with open(path, "wb") as f:
            f.write(raw)
        saved.append(path)
        time.sleep(0.4)
    return saved


def squash(text):
    return "".join(c for c in text.upper() if c.isalnum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="how many products to look at")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    if not os.path.exists(MASTER):
        print("no product master. run: python fetch_products.py")
        return

    with open(MASTER, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["brand"] and r["brand"] != "UNKNOWN"]

    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))

    photos = 0
    decoded = 0
    decoded_right = 0
    brand_read = 0
    brand_tried = 0

    for r in sample:
        code = r["code"]
        paths = fetch_photos(code)
        if not paths:
            continue

        # the first word of the brand, long enough not to match by accident
        brand = squash(r["brand"].split(",")[0])
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            photos += 1

            h, w = img.shape
            if w > WORK_WIDTH:
                img = cv2.resize(img, (WORK_WIDTH, int(h * WORK_WIDTH / w)),
                                 interpolation=cv2.INTER_AREA)

            found = pyzbar.decode(img, symbols=[ZBarSymbol.EAN13, ZBarSymbol.CODE128])
            if found:
                decoded += 1
                got = found[0].data.decode("utf-8")
                if got == code:
                    decoded_right += 1
                else:
                    print("  !! decoded %s on a photo of %s" % (got, code))

            if len(brand) >= 4:
                brand_tried += 1
                text = squash(pytesseract.image_to_string(clean(img)))
                if brand in text:
                    brand_read += 1

        print("%-14s %-26s %d photo(s)" % (code, r["product_name"][:26], len(paths)))

    print()
    print("=== real photographs, not labels i made ===")
    print("photos looked at:        ", photos)
    print()
    print("barcode decoded in:      ", decoded)
    print("  and it was correct:    ", decoded_right)
    print("  and it was wrong:      ", decoded - decoded_right)
    print("most of these are front-of-pack shots with no barcode in frame at all, so the number")
    print("that matters is the wrong one, not the rate")
    print()
    print("brand name read off the photo: %d of %d" % (brand_read, brand_tried))
    print("that is the opencv cleanup and tesseract against real packaging: curved jars, glare,")
    print("stylised type, half the shot in shadow. nothing like the flat printed label the")
    print("pipeline was built for")


if __name__ == "__main__":
    main()
