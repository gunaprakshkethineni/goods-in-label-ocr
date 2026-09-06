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
from textutil import edit_distance

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


TAGS = ["MAT", "LOT", "EXP", "QTY"]

# what shape each field is meant to be, so a character can be repaired when the format says
# what belongs in that slot. a is a letter, d is a digit, anything else is a literal
MASKS = {"material": "aaa-aa-dddd",
         "lot_no": "adddd-dddd",
         "expiry": "dddd-dd-dd"}


def clean_val(rest):
    rest = rest.lstrip(": \t.-")
    rest = "".join(JUNK_FIX.get(c, c) for c in rest)
    val = "".join(c for c in rest if c.isalnum() or c == "-")
    return val.upper() or None


# pull the bit before the colon off a line. tesseract loses the colon often enough that i also
# accept a plain space, so "QTY 149" is read the same as "QTY: 149"
def split_head(line):
    if ":" in line:
        head, rest = line.split(":", 1)
        return head.strip().upper(), rest
    parts = line.strip().split(None, 1)
    if len(parts) == 2:
        return parts[0].upper(), parts[1]
    return "", ""


# which of the four tags is this misread head closest to. it has to be a clear winner, because
# PN and SN are one character apart and a misread 5N is exactly as close to one as the other.
# if it is a tie i would rather find nothing and flag the row than guess and put a serial number
# in the part number column
def best_tag(head):
    d = sorted((edit_distance(head, t), t) for t in TAGS)
    if len(d) > 1 and d[0][0] == d[1][0]:
        return None, 99
    return d[0][1], d[0][0]


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
        head, rest = split_head(line)
        if head == tag:
            return clean_val(rest)

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

    # quantity is the only field on the label that is a bare number, so if the tag went missing
    # altogether then a line that is nothing but digits is almost certainly it. one crate came
    # back as just "77", another as "O:151" with the tag mangled past recognising. the date and
    # the lot both carry dashes so neither can be mistaken for a quantity here
    if tag == "QTY":
        for line in lines:
            if line.strip().isdigit():
                return line.strip()
        for line in lines:
            if ":" in line:
                val = line.split(":", 1)[1].strip()
                if val.isdigit():
                    return val
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


# the actual reading, on an image that is already in memory. app.py needs this because the
# frames coming off a webcam never touch the disk
def read_image(raw, use_preprocess=True):
    img = clean(raw) if use_preprocess else raw
    text = pytesseract.image_to_string(img, config=TESS_CFG)

    material = fix_by_mask(grab("MAT", text), MASKS["material"])
    lot_no = fix_by_mask(grab("LOT", text), MASKS["lot_no"])
    expiry = fix_by_mask(grab("EXP", text), MASKS["expiry"])

    qty = grab("QTY", text)
    if qty:
        qty = "".join(DIGIT_FIX.get(c, c) for c in qty)

    # barcode always comes off the original. thresholding wrecks the thin bars
    bc = read_barcode(raw)

    return {"material": material or "",
            "lot_no": lot_no or "",
            "expiry": expiry or "",
            "qty": qty or "",
            "barcode": bc or "",
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


COLUMNS = ["filename", "material", "lot_no", "expiry", "qty", "barcode"]


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
