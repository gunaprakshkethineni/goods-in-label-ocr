# pulls the real product catalogue off open food facts into data/product_master.csv.
# this is the master data everything gets checked against: real EAN-13 codes, brands, pack sizes
# and allergens. open data under the ODbL.
#
# run it once, everything after that works offline.
#
#   python fetch_products.py

import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

OUT_FILE = "data/product_master.csv"
API = "https://world.openfoodfacts.org/cgi/search.pl"

# they ask you to identify yourself rather than pretend to be a browser, which is fair
HEADERS = {"User-Agent": "factory-ocr-student-project/1.0 (goods-in label reading coursework)"}

# ingredients a biscuit line would actually book in. picked to give a spread of allergens,
# because a catalogue where everything contains milk would not exercise the checks
CATEGORIES = ["breakfast-cereals", "chocolates", "biscuits", "nuts", "flours",
              "yogurts", "vegetable-oils", "sugars", "dried-fruits", "juices"]

PER_CATEGORY = 60

# open food facts is mostly european, so without this you get a catalogue full of Cassonade pure
# canne. the country tag helps but only so far, it means sold there, not named in english.
# the actual fix is that they keep a name per language, so ask for product_name_en and then drop
# anything still carrying accents
COUNTRY = "united-kingdom"


def is_english(s):
    return bool(s) and s.isascii()


# a code that fails the check digit is a typo in the database, not a barcode. it will not render
def ean13_ok(code):
    if len(code) != 13 or not code.isdigit():
        return False
    total = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(code[:12]))
    return (10 - total % 10) % 10 == int(code[12])


# it is a free service and gets hammered, so a 503 is normal. back off and ask again
def fetch(category, page_size, tries=4):
    for attempt in range(tries):
        try:
            return fetch_once(category, page_size)
        except Exception as e:
            if attempt == tries - 1:
                raise
            wait = 4 * (attempt + 1)
            print("      %s, waiting %ds" % (type(e).__name__, wait))
            time.sleep(wait)


def fetch_once(category, page_size):
    q = urllib.parse.urlencode({
        "action": "process",
        "tagtype_0": "categories",
        "tag_contains_0": "contains",
        "tag_0": category,
        "tagtype_1": "countries",
        "tag_contains_1": "contains",
        "tag_1": COUNTRY,
        "page_size": page_size,
        "json": 1,
        "fields": ("code,product_name,product_name_en,lang,brands,quantity,"
                   "allergens_tags,categories_tags"),
    })
    req = urllib.request.Request(API + "?" + q, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r).get("products", [])


def tidy(s, limit=40):
    s = " ".join((s or "").split())
    return s[:limit]


def main():
    os.makedirs("data", exist_ok=True)

    # top up what is cached rather than starting over, so a rate limited run can just be re-run
    rows = {}
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["code"]] = r
        print("already had", len(rows), "products cached")

    for cat in CATEGORIES:
        try:
            products = fetch(cat, PER_CATEGORY)
        except Exception as e:
            print("  %-20s failed (%s), skipping" % (cat, type(e).__name__))
            time.sleep(3)
            continue

        kept = 0
        for p in products:
            code = (p.get("code") or "").strip()
            # the english name if there is one, otherwise whatever the default is, and then only
            # keep it if it actually reads as english
            name = tidy(p.get("product_name_en") or p.get("product_name"))
            brand = tidy(p.get("brands"), 24)
            if not ean13_ok(code) or code in rows:
                continue
            if not is_english(name) or not is_english(brand):
                continue

            # allergens come back like en:milk, en:nuts. strip the language prefix
            allergens = sorted({a.split(":")[-1] for a in (p.get("allergens_tags") or []) if a})

            rows[code] = {
                "code": code,
                "product_name": name,
                "brand": brand or "UNKNOWN",
                "pack_size": tidy(p.get("quantity"), 16),
                "category": cat,
                "allergens": "|".join(allergens),
            }
            kept += 1

        print("  %-20s %d products" % (cat, kept))
        time.sleep(2)   # be polite, it is a free service run by a non profit

    if not rows:
        print("got nothing back. open food facts rate limits, try again in a minute")
        sys.exit(1)

    cols = ["code", "product_name", "brand", "pack_size", "category", "allergens"]
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for code in sorted(rows):
            w.writerow(rows[code])

    with_allergens = sum(1 for r in rows.values() if r["allergens"])
    print()
    print("wrote", len(rows), "real products to", OUT_FILE)
    print(with_allergens, "of them declare at least one allergen")


if __name__ == "__main__":
    main()
