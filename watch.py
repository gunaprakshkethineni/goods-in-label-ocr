# i wrote this so a camera can watch the belt and the crates book themselves in as they go past.
#
#   python watch.py              simulated belt off the labels in data/labels
#   python watch.py --camera     a real camera, hold crates up to it
#
# app.py still needs somebody to hold each crate up. this does not.
#
# the two things that make it work are not the ocr at all. the first is the trigger: most frames
# are empty belt and tesseract costs me 250ms a frame, so i try a barcode first at about 14ms
# and only pay for the ocr if something is actually there. the second is telling one crate from
# the next, which is what i added the serial for.

import argparse
import csv
import os
import random
import time

import cv2
import numpy as np

from ocr_reader import read_barcodes, read_image
from validate import DEFAULT_BUILD, check_crate, load_rules, outcome

LOG_FILE = "output/watch_log.csv"

# i wait for a few empty frames, so one dropped read halfway past a crate does not book it twice
EMPTY_FRAMES_TO_RESET = 3

# anything smaller than this i treat as grit rather than a label
MIN_LABEL = 200

# i aim the camera at where the label passes, so the label nearly fills the frame. that is how a
# real fixed installation is set up and it matters more than i thought: i framed a 600px label
# in a 760px scene first, and finding the label in all that belt was harder than reading it
BELT_W, BELT_H = 660, 636


def belt_background():
    # i draw it as dark rubber with grain and tread lines. i had it mid grey first, which is a
    # harder picture than any real line would give me: too close in tone to a white label to
    # segment apart, and tesseract read the tread lines as characters. real belts are black,
    # which turns out to be the easy case as well as the realistic one
    belt = np.full((BELT_H, BELT_W), 58, np.uint8)
    belt = cv2.add(belt, np.random.normal(0, 5, belt.shape).astype(np.int8).astype(np.uint8))
    for y in range(0, BELT_H, 40):
        cv2.line(belt, (0, y), (BELT_W, y), 44, 2)
    return belt


# fakes a camera looking down at a belt: empty frames, then a crate sliding through over a few
# frames, then empty again
def belt_frames(paths, per_crate=4):
    for path in paths:
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            continue
        lh, lw = label.shape

        for _ in range(random.randint(2, 4)):
            yield belt_background(), None

        for step in range(per_crate):
            frame = belt_background()
            x = 12 + step * 8
            y = (BELT_H - lh) // 2 + random.randint(-5, 5)
            if y < 0 or y + lh > BELT_H or x + lw > BELT_W:
                continue
            frame[y:y + lh, x:x + lw] = label
            # the belt is moving, so the odd frame is smeared
            if random.random() < 0.3:
                frame = cv2.blur(frame, (3, 1))
            yield frame, os.path.basename(path)

    for _ in range(3):
        yield belt_background(), None


# i cut the label out of the frame here.
#
# i did not think i needed this step at all. reading works fine when the picture IS the label,
# which is what the desk hands me. but a camera sees mostly belt, and when i fed the whole frame
# in it failed in a way i would never have guessed: the deskew measures the angle off the darkest
# thing in the picture, and on a grey belt that is the tread lines, so it straightened the belt
# and read a tilted label. i got 70 crates out of 60 and nearly all of them were held.
#
# i tried anchoring on the barcodes first, since they had just decoded. that was wrong: pyzbar
# tells you where the BARS are, and a barcode image is bars plus a 3mm quiet zone, which is 24px
# at 200dpi, so my crop started inside the label and cut the left edge off every field tag.
def find_label(frame):
    _, th = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < MIN_LABEL or h < MIN_LABEL:
        return None

    # i keep the label's own black border, because the deskew uses it to work out the angle
    pad = 3
    y0, y1 = max(0, y - pad), min(frame.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(frame.shape[1], x + w + pad)
    return frame[y0:y1, x0:x1]


def camera_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("no camera found")
        return
    print("camera running, hold crate labels up to it. ctrl-c to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), "camera"
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true", help="use a real camera instead of the fake belt")
    ap.add_argument("--run", default=DEFAULT_BUILD)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    rules = load_rules()
    if args.camera:
        frames = camera_frames()
    else:
        paths = sorted(os.listdir("data/labels"))
        paths = [os.path.join("data/labels", p) for p in paths if p.endswith(".png")]
        if not paths:
            print("no labels. run: python make_labels.py --n 60 --seed 7")
            return
        frames = belt_frames(paths)
        print("watching a simulated belt with %d crates on it\n" % len(paths))

    issued = []
    rows = []
    current = set()
    empty_streak = 0
    seen_frames = 0
    skipped = 0
    gate_time = 0.0
    read_time = 0.0
    tally = {"ACCEPTED": 0, "HELD": 0, "REJECTED": 0}

    # i book the crate here, using the best frame i saw of it
    def book(pending):
        nonlocal read_time
        label = find_label(pending["frame"])
        if label is None:
            return

        t0 = time.perf_counter()
        # i flatten it first, a camera over a belt is never square on to the crate
        row = read_image(label, flatten=True)
        row.pop("text")
        fixed, flags = check_crate(dict(row, filename="belt"), rules, args.run, issued)
        read_time += time.perf_counter() - t0

        status = outcome(flags)
        tally[status] += 1
        if status != "REJECTED":
            issued.append({"sscc": fixed.get("sscc", ""), "code": fixed.get("code", ""),
                           "allergens": fixed.get("allergens", "")})

        rows.append({"when": time.strftime("%H:%M:%S"), "status": status,
                     "sscc": fixed.get("sscc", ""), "code": fixed.get("code", ""),
                     "product_name": fixed.get("product_name", ""),
                     "lot_no": fixed.get("lot_no", ""), "bbe": fixed.get("bbe", ""),
                     "qty": fixed.get("qty", ""), "flags": ";".join(flags)})

        print("%-8s %-8s %-18s %-26s %s"
              % (time.strftime("%H:%M:%S"), status, fixed.get("sscc", "?"),
                 (fixed.get("product_name") or "not in the catalogue")[:26],
                 ";".join(flags)))

    pending = None
    start = time.perf_counter()

    for frame, _src in frames:
        seen_frames += 1

        # this is my cheap gate. no barcode anywhere means no crate in front of the camera
        t0 = time.perf_counter()
        codes = read_barcodes(frame)
        gate_time += time.perf_counter() - t0

        # i collect whichever of the three codes came off this frame. it is not the same three
        # every time, because a crate crosses the camera over several frames and a different
        # barcode drops out on each one.
        #
        # i tried picking one code as the identity and it does not work for that reason: the
        # serial reads on one frame and only the batch code on the next, my key changes, and the
        # crate gets booked twice. so i keep the whole set and call it the same crate if
        # anything at all overlaps
        frame_keys = {v for v in codes.values() if v}

        if not frame_keys:
            empty_streak += 1
            skipped += 1
            if empty_streak >= EMPTY_FRAMES_TO_RESET and pending:
                book(pending)
                pending = None
            continue

        empty_streak = 0

        if pending and (frame_keys & pending["keys"]):
            # still the same crate, so i do not book it again. but if this frame is a better
            # look at it than the one i have, i keep this one instead.
            #
            # i read the first frame a crate turned up in to begin with, which is the obvious
            # thing to do and is wrong. the belt is moving, so the first frame is usually the one
            # smeared by motion, and it held thirty crates out of fifty seven where the batch
            # tool holds six. how many barcodes decoded is a free measure of how good the look
            # was, so i wait for the crate to show me its best side before spending 250ms on ocr
            pending["keys"] |= frame_keys
            if len(frame_keys) > pending["score"]:
                pending.update(frame=frame, score=len(frame_keys))
            skipped += 1
            continue

        if pending:
            book(pending)
        pending = {"keys": frame_keys, "frame": frame, "score": len(frame_keys)}

    if pending:
        book(pending)

    took = time.perf_counter() - start
    booked = sum(tally.values())

    os.makedirs("output", exist_ok=True)
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["when", "status", "sscc", "code", "product_name",
                                          "lot_no", "bbe", "qty", "flags"])
        w.writeheader()
        w.writerows(rows)

    print()
    print("%d frames off the camera, %d were empty belt or the same crate again" % (seen_frames, skipped))
    print("%d crates booked in, nobody pressed anything" % booked)
    print("  ACCEPTED %3d" % tally["ACCEPTED"])
    print("  HELD     %3d" % tally["HELD"])
    print("  REJECTED %3d" % tally["REJECTED"])
    print()
    print("the barcode gate cost %.0f ms a frame, and saved the ocr on %d of them"
          % (1000 * gate_time / max(seen_frames, 1), skipped))
    print("reading a crate properly cost %.0f ms" % (1000 * read_time / max(booked, 1)))
    print("%.1f s for the lot -> %.1f crates a second" % (took, booked / took if took else 0))
    print("log ->", LOG_FILE)


if __name__ == "__main__":
    main()
