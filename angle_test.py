# what an off-axis camera costs, and how much of it flattening gets back.
#
#   python angle_test.py
#
# a camera over a belt is never square to the crate, so the label arrives as a trapezoid. deskew
# cannot fix that, the label is not rotated, it is tilted away. this warps the labels the way an
# angled camera would see them and reads them three ways: square on, tilted, and flattened.

import argparse
import csv
import glob
import os
import random

import cv2
import numpy as np

import ocr_reader as OR
from preprocess import clean, deperspective
from textutil import edit_distance

FIELDS = ["sscc", "code", "lot_no", "bbe", "qty"]


# tilt the label as if the camera were off to one side and above it
def tilt(img, strength):
    h, w = img.shape
    # dark behind it, like a conveyor. mid grey first time and the corner finder gave up on
    # eighteen labels out of twenty five
    pad = int(max(h, w) * 0.12)
    canvas = np.full((h + 2 * pad, w + 2 * pad), 45, np.uint8)
    canvas[pad:pad + h, pad:pad + w] = img

    src = np.array([[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]], np.float32)
    dx, dy = w * strength, h * strength
    dst = np.array([
        [pad + random.uniform(0, dx), pad + random.uniform(0, dy)],
        [pad + w - random.uniform(0, dx), pad + random.uniform(0, dy)],
        [pad + w - random.uniform(0, dx * 0.4), pad + h - random.uniform(0, dy)],
        [pad + random.uniform(0, dx * 0.4), pad + h - random.uniform(0, dy)],
    ], np.float32)

    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(canvas, m, (w + 2 * pad, h + 2 * pad),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=120)


LABEL_W, LABEL_H = 600, 590


def read(img, flatten):
    if flatten:
        fixed = deperspective(img)
        # only resize when the corners were actually found. resizing regardless was my own bug:
        # on the ones it could not flatten it squashed the padded canvas into a label shaped box
        # and made them worse than leaving them tilted
        if fixed.shape != img.shape:
            img = cv2.resize(fixed, (LABEL_W, LABEL_H), interpolation=cv2.INTER_CUBIC)

    cleaned = clean(img, crop_to=OR.TEXT_BOTTOM)
    import pytesseract
    text = pytesseract.image_to_string(cleaned, config=OR.TESS_CFG)
    return {"sscc": OR.fix_by_mask(OR.grab("SSC", text), OR.MASKS["sscc"]),
            "code": OR.fix_by_mask(OR.grab("PRD", text), OR.MASKS["code"]),
            "lot_no": OR.fix_by_mask(OR.grab("LOT", text), OR.MASKS["lot_no"]),
            "bbe": OR.fix_by_mask(OR.grab("BBE", text), OR.MASKS["bbe"]),
            "qty": OR.grab("QTY", text)}


def score(got, t):
    d = c = 0
    hits = 0
    for k in FIELDS:
        v = got.get(k) or ""
        d += edit_distance(v, t[k])
        c += len(t[k])
        if v == t[k]:
            hits += 1
    return d, c, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--strength", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    truth = {r["filename"]: r for r in
             csv.DictReader(open("data/labels_truth.csv", encoding="utf-8"))}
    files = sorted(glob.glob("data/labels/*.png"))[:args.n]
    if not files:
        print("no labels. run: python make_labels.py --n 60 --seed 7")
        return

    ways = {"square on": [0, 0, 0], "tilted": [0, 0, 0], "tilted, flattened": [0, 0, 0]}

    random.seed(args.seed)
    np.random.seed(args.seed)
    for f in files:
        flat = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        t = truth[os.path.basename(f)]
        angled = tilt(flat, args.strength)

        for name, img, flatten in [("square on", flat, False),
                                   ("tilted", angled, False),
                                   ("tilted, flattened", angled, True)]:
            d, c, hits = score(read(img, flatten), t)
            ways[name][0] += d
            ways[name][1] += c
            ways[name][2] += hits

    print()
    print("%d labels, tilt strength %.2f" % (len(files), args.strength))
    print()
    print("%-20s %-10s %s" % ("", "chars", "fields exactly right"))
    for name in ["square on", "tilted", "tilted, flattened"]:
        d, c, hits = ways[name]
        print("%-20s %5.1f%%     %d of %d" % (name, 100 * (1 - d / c), hits, len(files) * 5))


if __name__ == "__main__":
    main()
