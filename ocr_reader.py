# reads the labels and pulls out the fields. this is the main script.

import csv
import glob
import os
import re
import sys

import cv2
import pytesseract
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol

from preprocess import clean

# pytesseract wont find the exe on windows unless you point it at the install folder
TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS):
    pytesseract.pytesseract.tesseract_cmd = TESS

IN_DIR = "data/labels"
OUT_FILE = "output/inventory.csv"

# psm 6 = treat it as one block of text. psm 3 kept slicing the barcode strip into junk lines
# and psm 11 was a disaster, it scored 20 percent. the whitelist stops tesseract inventing
# punctuation, it was reading the S in SN as a dollar sign constantly
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:- "
TESS_CFG = "--oem 3 --psm 6 -c tessedit_char_whitelist=" + WHITELIST

# tesseract mixes these up constantly on the noisy labels
DIGIT_FIX = {"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
             "S": "5", "B": "8", "Z": "2", "G": "6", "T": "7"}
LETTER_FIX = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}

# punctuation tesseract throws in where a letter belongs. it loves turning S into $
JUNK_FIX = {"$": "S", "§": "S", "£": "L", "!": "1", "|": "1", "¢": "C", "©": "C"}


# only swap a character when the field format says what belongs in that slot.
# doing it blindly made things worse, eg it was turning the S in SN into a 5
def fix_by_mask(s, mask):
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


TAGS = ["PN", "SN", "LOT", "QTY"]


def clean_val(rest):
    rest = rest.lstrip(": \t.-")
    rest = "".join(JUNK_FIX.get(c, c) for c in rest)
    val = "".join(c for c in rest if c.isalnum() or c == "-")
    return val.upper() or None


def tag_ok(head, tag):
    return len(head) == len(tag) and sum(1 for a, b in zip(head, tag) if a != b) <= 1


# find the line starting with a tag like SN and give back what comes after the colon.
# i used a regex for this at first and it kept dropping whole fields. if tesseract read the S in
# SN9332820 as a dollar sign the pattern just did not match and i lost the serial completely,
# which is why the serial column was empty on most rows. going line by line is uglier but it
# actually gets something back to work with.
#
# the second pass is because tesseract misreads the tag itself, QTY comes out as OTY and LOT as
# 1OT often enough that i was losing whole fields to it. but PN and SN are only one character
# apart, so a fuzzy pass has to refuse any line that already belongs to another field or the
# part number happily eats the serial line
def grab(tag, text):
    lines = text.splitlines()

    for line in lines:
        if ":" in line:
            head, rest = line.split(":", 1)
            if head.strip().upper() == tag:
                return clean_val(rest)

    for line in lines:
        if ":" in line:
            head, rest = line.split(":", 1)
            h = head.strip().upper()
            if h in TAGS and h != tag:
                continue
            if tag_ok(h, tag):
                return clean_val(rest)

    # last resort, the colon went missing entirely
    for line in lines:
        i = line.upper().find(tag)
        if i != -1:
            return clean_val(line[i + len(tag):])
    return None


def read_barcode(img):
    # telling zbar it is code128 only. left to itself it also tries pdf417 and spams the console
    # with assertion warnings on the noisy labels
    found = pyzbar.decode(img, symbols=[ZBarSymbol.CODE128])
    if found:
        return found[0].data.decode("utf-8")
    # sometimes it only picks it up after a bit of contrast help, or with the speckle knocked off
    for fixed in [cv2.equalizeHist(img), cv2.medianBlur(img, 3)]:
        found = pyzbar.decode(fixed, symbols=[ZBarSymbol.CODE128])
        if found:
            return found[0].data.decode("utf-8")
    return None


def read_label(path, use_preprocess=True):
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None

    img = clean(raw) if use_preprocess else raw
    text = pytesseract.image_to_string(img, config=TESS_CFG)

    part_no = fix_by_mask(grab("PN", text), "dddd-ad")
    serial = fix_by_mask(grab("SN", text), "aaddddddd")
    lot_no = fix_by_mask(grab("LOT", text), "adddd-dddd")

    qty = grab("QTY", text)
    if qty:
        qty = "".join(DIGIT_FIX.get(c, c) for c in qty)

    # barcode always comes off the original. thresholding wrecks the thin bars
    bc = read_barcode(raw)

    return {"filename": os.path.basename(path),
            "part_no": part_no or "",
            "serial": serial or "",
            "lot_no": lot_no or "",
            "qty": qty or "",
            "barcode": bc or ""}


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


if __name__ == "__main__":
    use_pre = "nopre" not in sys.argv
    if not use_pre:
        print("running WITHOUT preprocessing")

    rows = read_all(use_pre)
    os.makedirs("output", exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "part_no", "serial", "lot_no", "qty", "barcode"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", len(rows), "rows to", OUT_FILE)
