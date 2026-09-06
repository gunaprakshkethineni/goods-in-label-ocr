# builds an html page showing what the tool read off every label and what it decided to flag.
# this is the one to open if you want to see it working rather than read numbers in a terminal.
#
#   python demo.py
#   then open output/report.html

import base64
import csv
import glob
import io
import os
import time
import webbrowser

import cv2
from PIL import Image

from ocr_reader import read_label
from preprocess import clean
from validate import NONCONFORMING, DEFAULT_BUILD, check_crate, load_rules, outcome

IN_DIR = "data/labels"
TRUTH_FILE = "data/labels_truth.csv"
OUT_FILE = "output/report.html"

FIELDS = ["material", "lot_no", "expiry", "qty"]
NICE = {"material": "Material", "lot_no": "Lot no", "expiry": "Expires", "qty": "Qty"}


# shrink the picture and turn it into a base64 string so the html file works on its own with no
# folder of images next to it. makes it easy to email the report to somebody
def embed(img, width=340):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("L")
    w, h = img.size
    img = img.resize((width, int(h * width / w)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_truth():
    if not os.path.exists(TRUTH_FILE):
        return {}
    with open(TRUTH_FILE, newline="") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def card(row, flags, truth, thumb):
    status = outcome(flags)
    cls = status.lower()

    body = ""
    for k in FIELDS:
        got = row.get(k, "") or "-"
        want = truth.get(k) if truth else None
        if want is None:
            mark = ""
        elif got == want:
            mark = "<span class='tick'>ok</span>"
        else:
            mark = "<span class='cross'>expected %s</span>" % want
        body += "<tr><th>%s</th><td class='val'>%s</td><td>%s</td></tr>" % (NICE[k], got, mark)

    bc = row.get("barcode", "")
    body += "<tr><th>Barcode</th><td class='val'>%s</td><td>%s</td></tr>" % (
        bc if bc else "-", "" if bc else "<span class='cross'>could not decode</span>")

    tags = "".join("<span class='flag %s'>%s</span>"
                   % ("stop" if f.split(":")[0] in NONCONFORMING else "warn", f) for f in flags)
    real = ""
    if truth and truth.get("defects"):
        real = "<div class='real'>this crate really was bad: %s</div>" % truth["defects"]

    return """
    <div class='card %s'>
      <div class='head'><span class='name'>%s</span><span class='badge %s'>%s</span></div>
      <img src='%s'>
      <table>%s</table>
      <div class='flags'>%s</div>
      %s
    </div>""" % (cls, row["filename"], cls, status, thumb, body, tags, real)


def main():
    truth = load_truth()
    rules = load_rules()
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.png")))
    if not files:
        print("no labels in", IN_DIR, "- run: python make_labels.py --n 60 --seed 7")
        return

    print("reading", len(files), "crates for", DEFAULT_BUILD, "...")
    start = time.perf_counter()

    cards = []
    tally = {"ACCEPTED": 0, "HELD": 0, "REJECTED": 0}
    issued = []
    for i, path in enumerate(files):
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        row = read_label(path)
        fixed, flags = check_crate(row, rules, DEFAULT_BUILD, issued)
        tally[outcome(flags)] += 1
        if not flags:
            # only what cleared is actually out on the build, so only that counts for the
            # compatibility check on later crates
            issued.append({"material": fixed.get("material", ""), "lot_no": fixed.get("lot_no", "")})
        cards.append((outcome(flags), card(fixed, flags, truth.get(row["filename"], {}), embed(raw))))
        if (i + 1) % 10 == 0:
            print("  ", i + 1, "of", len(files))

    took = time.perf_counter() - start

    # rejected first, then held, then the boring ones. the top of the page should be the crates
    # that would have gone into a blade if nobody was looking
    order = {"REJECTED": 0, "HELD": 1, "ACCEPTED": 2}
    cards.sort(key=lambda c: order[c[0]])

    sample = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    before = embed(sample, 420)
    after = embed(clean(sample), 420)

    total = len(files)
    html = PAGE % {
        "build": DEFAULT_BUILD,
        "total": total,
        "accepted": tally["ACCEPTED"],
        "held": tally["HELD"],
        "rejected": tally["REJECTED"],
        "pct": 100.0 * tally["ACCEPTED"] / total,
        "secs": took,
        "per": 1000 * took / total,
        "before": before,
        "after": after,
        "cards": "".join(c[1] for c in cards),
    }

    os.makedirs("output", exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print()
    print("%d crates -> %d accepted, %d held, %d rejected"
          % (total, tally["ACCEPTED"], tally["HELD"], tally["REJECTED"]))
    print("took %.1f s" % took)
    print("report ->", os.path.abspath(OUT_FILE))
    webbrowser.open("file:///" + os.path.abspath(OUT_FILE).replace("\\", "/"))


PAGE = """<!doctype html>
<html><head><meta charset='utf-8'><title>Label reader report</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1c1e21}
 header{background:#1f2933;color:#fff;padding:26px 32px}
 header h1{margin:0 0 4px;font-size:21px}
 header p{margin:0;color:#9aa5b1;font-size:13px}
 .stats{display:flex;gap:14px;flex-wrap:wrap;padding:20px 32px;background:#fff;
        border-bottom:1px solid #dfe3e8}
 .stat{background:#f7f8fa;border:1px solid #e3e7ec;border-radius:8px;padding:12px 18px;min-width:120px}
 .stat b{display:block;font-size:23px;margin-bottom:2px}
 .stat span{font-size:12px;color:#6b7480}
 .pp{padding:22px 32px;background:#fff;border-bottom:1px solid #dfe3e8}
 .pp h2,.grid-title{font-size:15px;margin:0 0 12px}
 .pp .pair{display:flex;gap:26px;flex-wrap:wrap}
 .pp figure{margin:0}
 .pp figcaption{font-size:12px;color:#6b7480;margin-top:6px}
 .pp img{border:1px solid #d9dee4;border-radius:6px;display:block}
 .wrap{padding:22px 32px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(370px,1fr));gap:16px}
 .card{background:#fff;border:1px solid #dfe3e8;border-radius:9px;padding:14px;
       border-left:4px solid #2e9e5b}
 .card.held{border-left-color:#d99400}
 .card.rejected{border-left-color:#d4472e}
 .stat b.a{color:#1d7541} .stat b.h{color:#8a5a00} .stat b.r{color:#a8331c}
 .head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
 .name{font-family:Consolas,monospace;font-size:12px;color:#6b7480}
 .badge{font-size:11px;font-weight:700;padding:3px 9px;border-radius:11px;letter-spacing:.3px}
 .badge.accepted{background:#e3f5ea;color:#1d7541}
 .badge.held{background:#fdf3e0;color:#8a5a00}
 .badge.rejected{background:#fbe6e1;color:#a8331c}
 .card img{width:100%%;border:1px solid #e3e7ec;border-radius:5px;display:block}
 table{width:100%%;border-collapse:collapse;margin-top:10px;font-size:13px}
 th{text-align:left;font-weight:600;color:#6b7480;padding:3px 0;width:74px;font-size:12px}
 td{padding:3px 0}
 .val{font-family:Consolas,monospace}
 .tick{color:#2e9e5b;font-size:11px}
 .cross{color:#d4472e;font-size:11px}
 .flags{margin-top:9px}
 .flag{display:inline-block;font-size:10.5px;font-family:Consolas,monospace;
       padding:2px 7px;border-radius:4px;margin:2px 3px 0 0}
 .flag.stop{background:#fbe6e1;color:#a8331c}
 .flag.warn{background:#fdf3e0;color:#8a5a00}
 .real{margin-top:7px;font-size:11.5px;color:#8a6d1f;background:#fdf6e3;
       border-radius:4px;padding:5px 8px}
</style></head><body>
<header>
  <h1>Material intake &middot; %(build)s</h1>
  <p>Every crate booked in for this blade, what came off the label, and what happened to it.
     Rejected first, then held, then the ones that went through on their own.</p>
</header>
<div class='stats'>
  <div class='stat'><b>%(total)d</b><span>crates booked in</span></div>
  <div class='stat'><b class='a'>%(accepted)d</b><span>accepted</span></div>
  <div class='stat'><b class='h'>%(held)d</b><span>held, re-read it</span></div>
  <div class='stat'><b class='r'>%(rejected)d</b><span>rejected, wrong material</span></div>
  <div class='stat'><b>%(pct).0f%%</b><span>less checking</span></div>
  <div class='stat'><b>%(per).0f ms</b><span>per crate</span></div>
</div>
<div class='pp'>
  <h2>What the OpenCV step does</h2>
  <div class='pair'>
    <figure><img src='%(before)s'><figcaption>as photographed</figcaption></figure>
    <figure><img src='%(after)s'><figcaption>denoised, deskewed, upscaled</figcaption></figure>
  </div>
</div>
<div class='wrap'>
  <div class='grid-title'>Every crate</div>
  <div class='grid'>%(cards)s</div>
</div>
</body></html>
"""


if __name__ == "__main__":
    main()
