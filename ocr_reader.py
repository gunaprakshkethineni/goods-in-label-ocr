# this is the main one. i read the crate labels here and pull the fields out of them.

import csv
import glob
import os
import sys

import cv2
import pytesseract
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol

from preprocess import clean, deperspective
from textutil import edit_distance

# i have to point pytesseract at the exe, it cannot find it on windows by itself
TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS):
    pytesseract.pytesseract.tesseract_cmd = TESS

IN_DIR = "data/labels"
OUT_FILE = "output/inventory.csv"

# i settled on psm 6, which treats it as one block of text. i tried psm 3 and it sliced the
# barcodes into junk lines, and psm 11 scored 20%. i added the whitelist to stop it inventing
# punctuation
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:- "
TESS_CFG = "--oem 3 --psm 6 -c tessedit_char_whitelist=" + WHITELIST

# these are the swaps i kept seeing tesseract make on my labels
DIGIT_FIX = {"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
             "S": "5", "B": "8", "Z": "2", "G": "6", "T": "7"}
LETTER_FIX = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
JUNK_FIX = {"$": "S", "§": "S", "£": "L", "!": "1", "|": "1", "¢": "C", "©": "C"}

TAGS = ["SSC", "PRD", "LOT", "BBE", "QTY"]

# i wrote the shape of each field out so i can repair characters later. a = letter, d = digit
MASKS = {"sscc": "d" * 18,            # serial shipping container code
         "code": "ddddddddddddd",     # EAN-13
         "lot_no": "adddd-dddd",
         "bbe": "dddd-dd-dd"}

# everything printed is above this line on my labels, below it is all barcodes
TEXT_BOTTOM = 320

LABEL_SIZE = (600, 590)


def fix_by_mask(s, mask):
    # i only swap a character where the format tells me what belongs in that slot. i did it
    # blind at first and made things worse, it kept turning the S of SSC into a 5
    if s is None or len(s) != len(mask):
        return s
    out = []
    for ch, m in zip(s, mask):
        if m == "d":
            out.append(DIGIT_FIX.get(ch, ch))
        elif m == "a":
            ch = ch.upper()
            out.append(LETTER_FIX.get(ch, ch))
        else:
            out.append(m)
    return "".join(out)


def clean_val(rest):
    rest = rest.lstrip(": \t.-")
    rest = "".join(JUNK_FIX.get(c, c) for c in rest)
    val = "".join(c for c in rest if c.isalnum() or c == "-")
    return val.upper() or None


def split_head(line):
    # i accept a space as well as a colon, because tesseract drops the colon often enough that
    # "QTY 149" has to work too
    if ":" in line:
        head, rest = line.split(":", 1)
        return head.strip().upper(), rest
    parts = line.strip().split(None, 1)
    if len(parts) == 2:
        return parts[0].upper(), parts[1]
    return "", ""


def best_tag(head):
    # i make it demand a clear winner. SSC and PRD are both three letters, so something like
    # 5SC is no closer to one than the other, and i would rather find nothing than write a
    # product code into the serial column
    d = sorted((edit_distance(head, t), t) for t in TAGS)
    if len(d) > 1 and d[0][0] == d[1][0]:
        return None, 99
    return d[0][1], d[0][0]


# i find the line for a tag and give back whatever comes after the colon.
#
# i wrote this as a regex first and it kept losing whole fields on me: if tesseract reads the L
# of L2026-6851 as a 1 the pattern just does not match, and my batch column came back empty on
# half the rows. going line by line is uglier but i actually get something back.
def grab(tag, text):
    lines = text.splitlines()

    for line in lines:
        head, rest = split_head(line)
        if head == tag:
            return clean_val(rest)

    # i allow the tag itself to be one character out, because tesseract mangles those too and i
    # was losing whole fields to QTY coming back as OTY. it must never steal a line that
    # already belongs to another field though
    for line in lines:
        head, rest = split_head(line)
        if not head or head in TAGS:
            continue
        best, dist = best_tag(head)
        if best == tag and dist <= 1:
            return clean_val(rest)

    for line in lines:
        i = line.upper().find(tag)
        if i != -1:
            return clean_val(line[i + len(tag):])

    # i gave quantity a last resort, because it is the only field with no barcode behind it: a
    # short line of plain digits. one crate came back as just "77", another as "O:151".
    # i added the length cap after my first version read the 13 digit product code as the
    # quantity and booked in a crate of three trillion boxes
    if tag == "QTY":
        for line in lines:
            s = line.strip()
            if s.isdigit() and len(s) <= 4:
                return s
        for line in lines:
            head, rest = split_head(line)
            if head and best_tag(head)[0] not in (None, "QTY"):
                continue
            val = rest.strip()
            if val.isdigit() and len(val) <= 4:
                return val
    return None


# i tell zbar what to expect, otherwise it also tries pdf417 and spams my console with warnings
SYMBOLS = [ZBarSymbol.EAN13, ZBarSymbol.CODE128]


def read_barcodes(img):
    # there are three barcodes on my labels. the EAN-13 is the product, and i tell the two
    # code128s apart by the GS1 identifier they start with: 00 is the crate serial, 17 is the
    # date followed by the batch
    out = {"ean": "", "gs1": "", "sscc": ""}
    for attempt in [img, cv2.equalizeHist(img), cv2.medianBlur(img, 3)]:
        for r in pyzbar.decode(attempt, symbols=SYMBOLS):
            data = r.data.decode("utf-8")
            if r.type == "EAN13":
                out["ean"] = out["ean"] or data
            elif r.type == "CODE128":
                if data.startswith("00") and len(data) == 20:
                    out["sscc"] = out["sscc"] or data[2:]
                elif data.startswith("17"):
                    out["gs1"] = out["gs1"] or data
        if all(out.values()):
            break
        # i retry with a bit of contrast help, one of them often will not show up otherwise
    return out


def parse_gs1(payload):
    # i encoded it as 17YYMMDD then 10<batch digits>, so i unpick it the same way here
    if not payload or not payload.isdigit() or len(payload) < 18:
        return "", ""
    if payload[:2] != "17" or payload[8:10] != "10":
        return "", ""
    bbe = "20%s-%s-%s" % (payload[2:4], payload[4:6], payload[6:8])
    digits = payload[10:]
    lot = "L%s-%s" % (digits[:4], digits[4:8])
    return lot, bbe


def flatten_label(img):
    fixed = deperspective(img)
    if fixed.shape == img.shape:
        return img          # it could not find the corners, so i leave it alone
    # i put it back to the size a label should be. without that my TEXT_BOTTOM crop is a pixel
    # count against an image that comes out a different size every time
    return cv2.resize(fixed, LABEL_SIZE, interpolation=cv2.INTER_CUBIC)


# i take an image already in memory here, because the webcam frames never touch the disk
def read_image(raw, use_preprocess=True, flatten=False):
    if flatten:
        raw = flatten_label(raw)

    img = clean(raw, crop_to=TEXT_BOTTOM) if use_preprocess else raw
    text = pytesseract.image_to_string(img, config=TESS_CFG)

    sscc = fix_by_mask(grab("SSC", text), MASKS["sscc"])
    code = fix_by_mask(grab("PRD", text), MASKS["code"])
    lot_no = fix_by_mask(grab("LOT", text), MASKS["lot_no"])
    bbe = fix_by_mask(grab("BBE", text), MASKS["bbe"])

    qty = grab("QTY", text)
    if qty:
        qty = "".join(DIGIT_FIX.get(c, c) for c in qty)

    # i always read the barcodes off the original, never the cleaned one. it wrecks the thin bars
    bc = read_barcodes(raw)
    lot_bc, bbe_bc = parse_gs1(bc["gs1"])

    return {"sscc": sscc or "",
            "code": code or "",
            "lot_no": lot_no or "",
            "bbe": bbe or "",
            "qty": qty or "",
            "barcode": bc["ean"],
            "sscc_bc": bc["sscc"],
            "lot_bc": lot_bc,
            "bbe_bc": bbe_bc,
            "text": text}


def read_label(path, use_preprocess=True):
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None

    row = read_image(raw, use_preprocess)
    row.pop("text")
    row["filename"] = os.path.basename(path)
    return row


def read_all(use_preprocess=True, quiet=False):
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.png")))
    rows = []
    for i, f in enumerate(files):
        r = read_label(f, use_preprocess)
        if r:
            rows.append(r)
        if not quiet and (i + 1) % 10 == 0:
            print("read", i + 1, "of", len(files))
    return rows


COLUMNS = ["filename", "sscc", "code", "lot_no", "bbe", "qty",
           "barcode", "sscc_bc", "lot_bc", "bbe_bc"]


if __name__ == "__main__":
    use_pre = "nopre" not in sys.argv
    if not use_pre:
        print("running WITHOUT preprocessing")

    rows = read_all(use_pre)
    os.makedirs("output", exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", len(rows), "rows to", OUT_FILE)
