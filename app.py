# the goods-in desk. pick the blade you are building, then either drop crate photos in or hold
# labels up to the camera. every crate lands in a running log as accepted, held or rejected.
#
#   python app.py
#   then go to http://localhost:5000

import base64
import csv
import io
import os
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from ocr_reader import read_image
from preprocess import clean
from validate import DEFAULT_BUILD, check_crate, load_rules, outcome

app = Flask(__name__)

# the labels the generator makes are 600 wide. a phone photo is 3000+ and tesseract crawls on
# something that big, so everything gets brought down to roughly the size the pipeline was
# tuned on before it goes anywhere near the ocr
WORK_WIDTH = 900

# how long the same crate can sit in front of the camera before it counts as a new one. auto mode
# fires every second or so and you do not want twenty rows for one drum
SAME_CRATE_SECS = 8

BATCH_FILE = "output/inventory.csv"
EXPORT_FILE = "output/session_scans.csv"

rules = load_rules()

# the log lives in memory, so restarting clears it. that is fine for a desk that gets emptied at
# the end of a shift, and Save CSV is there for anything worth keeping
scans = []

# which blade the crates on this desk are going onto. it decides which materials are allowed at
# all, and whatever has already been accepted onto it decides which ones still are
build = {"wo": DEFAULT_BUILD}


# what is physically out on the build. only accepted crates count, a rejected drum never left
# the loading bay
def issued_now():
    return [{"material": s["material"], "lot_no": s["lot_no"]}
            for s in scans if s["status"] == "ACCEPTED"]


def to_gray(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    if w > WORK_WIDTH:
        img = cv2.resize(img, (WORK_WIDTH, int(h * WORK_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return img


def as_data_url(img, width=460):
    pil = Image.fromarray(img)
    w, h = pil.size
    pil = pil.resize((width, int(h * width / w)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# what identifies a physical crate. the barcode if we got it, otherwise the printed lot
def key_for(row):
    return row.get("barcode") or row.get("lot_no") or ""


# put one reading into the log.
#
# live is False when we are pulling a batch file in. that matters, because both the rules below
# only make sense for somebody standing at the desk waving drums at a camera. a batch file is 60
# separate crates that happen to be read quickly, and a delivery genuinely does arrive as several
# drums off one lot, so applying either rule to it just invents problems
def log_scan(row, flags, source, live=True):
    key = key_for(row)
    now = time.time()

    if live:
        # same crate still sitting in front of the lens, auto scan fires about once a second
        if scans and key and scans[-1]["key"] == key and now - scans[-1]["t"] < SAME_CRATE_SECS:
            return None, flags

        # a lot that has already been through this session. it might be the next drum of a split
        # delivery, or it might be the same drum booked in twice and the stock count now wrong.
        # the label carries no per-crate serial so there is no way to tell from the picture, and
        # a guess either way is worse than asking
        if key and any(s["key"] == key for s in scans):
            flags = flags + ["LOT_ALREADY_BOOKED_IN"]

    entry = {"n": len(scans) + 1,
             "t": now,
             "when": time.strftime("%H:%M:%S"),
             "key": key,
             "source": source,
             "status": outcome(flags),
             "material": row.get("material", ""),
             "lot_no": row.get("lot_no", ""),
             "expiry": row.get("expiry", ""),
             "qty": row.get("qty", ""),
             "barcode": row.get("barcode", ""),
             "flags": flags}
    scans.append(entry)
    return entry, flags


def summary():
    tally = {"ACCEPTED": 0, "HELD": 0, "REJECTED": 0}
    for s in scans:
        tally[s["status"]] = tally.get(s["status"], 0) + 1
    total = len(scans)
    return {"total": total,
            "accepted": tally["ACCEPTED"],
            "held": tally["HELD"],
            "rejected": tally["REJECTED"],
            "pct": round(100.0 * tally["ACCEPTED"] / total) if total else 0,
            "build": build["wo"],
            "builds": sorted(rules["spec"]) or [DEFAULT_BUILD],
            "rows": [{k: s[k] for k in ["n", "when", "source", "status", "material",
                                        "lot_no", "expiry", "qty", "flags"]}
                     for s in reversed(scans)]}


# pulls whatever the batch run already read into the log, so the 60 crates from ocr_reader.py sit
# alongside anything scanned by hand. runs once at startup so the desk is not empty when you open
# it, and the button re-runs it
def import_batch():
    if not os.path.exists(BATCH_FILE):
        return 0
    added = 0
    with open(BATCH_FILE, newline="") as f:
        for row in csv.DictReader(f):
            fixed, flags = check_crate(row, rules, build["wo"], issued_now())
            entry, _ = log_scan(fixed, flags, row.get("filename", "batch"), live=False)
            if entry:
                added += 1
    return added


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/scan", methods=["POST"])
def scan():
    blob = request.files.get("image")
    if blob is None:
        return jsonify({"error": "no image"}), 400

    img = to_gray(blob.read())
    if img is None:
        return jsonify({"error": "could not read that file"}), 400

    source = request.form.get("source", "upload")
    start = time.perf_counter()
    row = read_image(img)
    raw_text = row.pop("text")
    fixed, flags = check_crate(dict(row, filename=source), rules, build["wo"], issued_now())
    took = time.perf_counter() - start

    entry, flags = log_scan(fixed, flags, source)

    out = {
        "status": outcome(flags),
        "logged": entry is not None,
        "fields": {k: fixed.get(k, "") for k in ["material", "lot_no", "expiry", "qty"]},
        "barcode": fixed.get("barcode", ""),
        "ocr_lot": row.get("lot_no", ""),
        "flags": flags,
        "ms": round(took * 1000),
        "text": raw_text.strip(),
        "summary": summary(),
    }
    # the live camera fires constantly, no point shipping pictures back every time
    if request.form.get("pics") == "1":
        out["before"] = as_data_url(img)
        out["after"] = as_data_url(clean(img))
    return jsonify(out)


@app.route("/history")
def history():
    return jsonify(summary())


@app.route("/clear", methods=["POST"])
def clear():
    scans.clear()
    return jsonify(summary())


@app.route("/build", methods=["POST"])
def set_build():
    wo = request.form.get("work_order", "")
    if wo in rules["spec"]:
        build["wo"] = wo
    return jsonify(summary())


@app.route("/import_batch", methods=["POST"])
def import_batch_route():
    if not os.path.exists(BATCH_FILE):
        return jsonify({"error": "no output/inventory.csv yet, run: python ocr_reader.py"}), 400
    return jsonify(dict(summary(), added=import_batch()))


@app.route("/export", methods=["POST"])
def export():
    if not scans:
        return jsonify({"error": "nothing booked in yet"}), 400

    os.makedirs("output", exist_ok=True)
    cols = ["n", "when", "build", "source", "status", "material", "lot_no", "expiry", "qty",
            "barcode", "flags"]
    with open(EXPORT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in scans:
            w.writerow({"n": s["n"], "when": s["when"], "build": build["wo"],
                        "source": s["source"], "status": s["status"], "material": s["material"],
                        "lot_no": s["lot_no"], "expiry": s["expiry"], "qty": s["qty"],
                        "barcode": s["barcode"], "flags": ";".join(s["flags"])})
    return jsonify({"file": os.path.abspath(EXPORT_FILE), "rows": len(scans)})


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Material intake</title>
<style>
 *{box-sizing:border-box}
 body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1c1e21}
 header{background:#1f2933;color:#fff;padding:20px 30px}
 header h1{margin:0 0 3px;font-size:20px}
 header p{margin:0;color:#9aa5b1;font-size:13px;max-width:760px}
 .wo{margin-top:13px;font-size:13px;color:#9aa5b1}
 .wo select{font-size:14px;padding:5px 9px;border-radius:5px;border:1px solid #3b4653;
            background:#2b3542;color:#fff}
 #wospec{margin-left:12px;font-size:12px;font-family:Consolas,monospace;color:#7b8794}
 .tabs{display:flex;gap:6px;padding:14px 30px 0;background:#fff;border-bottom:1px solid #dfe3e8}
 .tab{padding:9px 18px;border:1px solid #dfe3e8;border-bottom:none;border-radius:7px 7px 0 0;
      background:#f4f5f7;cursor:pointer;font-size:14px}
 .tab.on{background:#fff;font-weight:600;position:relative;top:1px}
 .wrap{display:flex;gap:20px;padding:22px 30px 0;flex-wrap:wrap;align-items:flex-start}
 .col{flex:1 1 380px;min-width:340px}
 .box{background:#fff;border:1px solid #dfe3e8;border-radius:9px;padding:16px;margin-bottom:20px}
 .box h2{margin:0 0 12px;font-size:14px}
 #drop{border:2px dashed #c3cad3;border-radius:8px;padding:32px 16px;text-align:center;
       color:#6b7480;cursor:pointer;font-size:14px}
 #drop.over{border-color:#2e6fd4;background:#f0f6ff;color:#2e6fd4}
 video{width:100%;border-radius:7px;background:#000;display:block}
 .row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
 button{padding:9px 16px;border:0;border-radius:6px;background:#2e6fd4;color:#fff;
        font-size:14px;cursor:pointer}
 button.grey{background:#66707c}
 button:disabled{background:#b9c0c8;cursor:default}
 .verdict{padding:13px 15px;border-radius:8px;font-weight:700;margin-bottom:13px;font-size:15px}
 .verdict .sub{display:block;font-weight:400;font-size:12.5px;margin-top:3px;opacity:.85}
 .ACCEPTED{background:#e3f5ea;color:#1d7541}
 .HELD{background:#fdf3e0;color:#8a5a00}
 .REJECTED{background:#fbe6e1;color:#a8331c}
 .IDLE{background:#eef0f3;color:#6b7480}
 table{width:100%;border-collapse:collapse;font-size:14px}
 .fields th{text-align:left;color:#6b7480;font-weight:600;padding:5px 0;width:92px;font-size:12.5px}
 .fields td{padding:5px 0;font-family:Consolas,monospace}
 .flag{display:inline-block;background:#eef0f3;color:#48515b;font-size:11px;
       font-family:Consolas,monospace;padding:3px 8px;border-radius:4px;margin:3px 4px 0 0}
 .flag.stop{background:#fbe6e1;color:#a8331c}
 .flag.warn{background:#fdf3e0;color:#8a5a00}
 .meta{margin-top:11px;font-size:12px;color:#6b7480}
 details{margin-top:11px}
 summary{font-size:12.5px;color:#6b7480;cursor:pointer}
 pre{background:#f7f8fa;border:1px solid #e3e7ec;border-radius:6px;padding:9px;
     font-size:11.5px;white-space:pre-wrap;max-height:170px;overflow:auto}
 .pics{display:flex;gap:11px;margin-top:12px;flex-wrap:wrap}
 .pics figure{margin:0;flex:1 1 190px}
 .pics img{width:100%;border:1px solid #e3e7ec;border-radius:6px;display:block}
 .pics figcaption{font-size:11.5px;color:#6b7480;margin-top:4px}
 .hint{font-size:12.5px;color:#6b7480;margin-top:9px;line-height:1.5}
 .logwrap{padding:0 30px 30px}
 .stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
 .stat{border:1px solid #e3e7ec;border-radius:8px;padding:10px 16px;min-width:112px;
       background:#f7f8fa;cursor:pointer}
 .stat.sel{outline:2px solid #2e6fd4;outline-offset:-2px}
 .stat b{display:block;font-size:21px}
 .stat span{font-size:11.5px;color:#6b7480}
 .stat.a b{color:#1d7541} .stat.h b{color:#8a5a00} .stat.r b{color:#a8331c}
 .log{overflow-x:auto}
 .log table{min-width:820px}
 .log th{text-align:left;font-size:11.5px;color:#6b7480;border-bottom:1px solid #dfe3e8;
         padding:7px 10px 7px 0;white-space:nowrap}
 .log td{padding:7px 10px 7px 0;border-bottom:1px solid #f0f2f4;font-size:13px;
         font-family:Consolas,monospace;white-space:nowrap}
 .pill{font-size:10.5px;font-weight:700;padding:2px 9px;border-radius:10px}
 .empty{color:#6b7480;font-size:13px;padding:14px 0}
 code{background:#f0f2f4;padding:1px 5px;border-radius:3px;font-size:12px}
</style></head><body>
<header>
  <h1>Material intake</h1>
  <p>Book crates onto a blade. Reads the label, checks the lot against the material master and the
     build spec, and sorts every crate into accepted, held or rejected.</p>
  <div class="wo">building <select id="wo"></select><span id="wospec"></span></div>
</header>

<div class="tabs">
  <div class="tab on" id="tab-up">Upload a photo</div>
  <div class="tab" id="tab-cam">Live camera</div>
</div>

<div class="wrap">
  <div class="col">
    <div class="box" id="pane-up">
      <h2>Upload</h2>
      <div id="drop">Drop crate label images here, or click to pick them</div>
      <input type="file" id="file" accept="image/*" multiple hidden>
      <div class="hint">Several at once is fine. <code>data/labels/</code> is full of them.
        Changing the blade at the top changes what counts as acceptable, so the same crate can be
        accepted on one build and rejected on another.</div>
    </div>

    <div class="box" id="pane-cam" style="display:none">
      <h2>Camera</h2>
      <video id="vid" autoplay playsinline muted></video>
      <div class="row">
        <button id="start">Start camera</button>
        <button id="grab" class="grey" disabled>Scan once</button>
        <button id="auto" class="grey" disabled>Auto scan: off</button>
      </div>
      <div class="hint">Hold the label flat on so it fills the frame. The same crate sitting in
        front of the lens only gets logged once.</div>
    </div>
  </div>

  <div class="col">
    <div class="box">
      <h2>Last crate</h2>
      <div class="verdict IDLE" id="verdict">nothing booked in yet</div>
      <table class="fields">
        <tr><th>Material</th><td id="f_mat">-</td></tr>
        <tr><th>Lot no</th><td id="f_lot">-</td></tr>
        <tr><th>Expires</th><td id="f_exp">-</td></tr>
        <tr><th>Qty</th><td id="f_qty">-</td></tr>
        <tr><th>Barcode</th><td id="f_bc">-</td></tr>
      </table>
      <div id="flags"></div>
      <div class="meta" id="meta"></div>
      <div class="pics" id="pics"></div>
      <details><summary>what tesseract actually returned</summary><pre id="raw"></pre></details>
    </div>
  </div>
</div>

<div class="logwrap">
  <div class="box">
    <h2>This shift</h2>
    <div class="stats">
      <div class="stat sel" data-f="ALL"><b id="s_total">0</b><span>booked in</span></div>
      <div class="stat a" data-f="ACCEPTED"><b id="s_acc">0</b><span>accepted</span></div>
      <div class="stat h" data-f="HELD"><b id="s_held">0</b><span>held, re-read it</span></div>
      <div class="stat r" data-f="REJECTED"><b id="s_rej">0</b><span>rejected, wrong material</span></div>
      <div class="stat" data-f="ALL"><b id="s_pct">0%</b><span>went through clean</span></div>
    </div>
    <div class="row" style="margin:0 0 14px">
      <button id="imp" class="grey">Re-add the batch run</button>
      <button id="exp" class="grey">Save CSV</button>
      <button id="clr" class="grey">Clear</button>
      <span class="meta" id="logmsg" style="align-self:center"></span>
    </div>
    <div class="log" id="log"><div class="empty">nothing yet</div></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const STOP = ["LOT_NOT_IN_MASTER","LOT_NOT_RELEASED","EXPIRED","LABEL_MATERIAL_MISMATCH",
              "NOT_ON_BUILD_SPEC","INCOMPATIBLE_WITH_ISSUED"];
const WORDS = {ACCEPTED: "ACCEPTED - booked onto the build",
               HELD: "HELD - somebody needs to re-read this drum",
               REJECTED: "REJECTED - do not put this on the build"};
const WHY = {ACCEPTED: "every check passed, nobody has to look at it",
             HELD: "the reader was not sure what it was looking at, so it is not making the call",
             REJECTED: "the reading was solid and the material is wrong for this blade"};
let timer = null, filter = "ALL", last = null;

function chip(f) {
  const cls = STOP.includes(f.split(":")[0]) ? "flag stop" : "flag warn";
  return "<span class='" + cls + "'>" + f + "</span>";
}

function drawLog(s) {
  last = s;
  $("s_total").textContent = s.total;
  $("s_acc").textContent   = s.accepted;
  $("s_held").textContent  = s.held;
  $("s_rej").textContent   = s.rejected;
  $("s_pct").textContent   = s.pct + "%";

  const sel = $("wo");
  if (sel.options.length !== s.builds.length)
    sel.innerHTML = s.builds.map(b => "<option>" + b + "</option>").join("");
  sel.value = s.build;

  const rows = s.rows.filter(r => filter === "ALL" || r.status === filter);
  if (!rows.length) {
    $("log").innerHTML = "<div class='empty'>nothing " +
      (filter === "ALL" ? "yet" : "in this bucket") + "</div>";
    return;
  }
  let h = "<table><tr><th>#</th><th>time</th><th>source</th><th>status</th><th>material</th>" +
          "<th>lot</th><th>expires</th><th>qty</th><th>why</th></tr>";
  for (const r of rows) {
    h += "<tr><td>" + r.n + "</td><td>" + r.when + "</td><td>" + r.source + "</td><td>" +
         "<span class='pill " + r.status + "'>" + r.status + "</span></td><td>" +
         (r.material || "-") + "</td><td>" + (r.lot_no || "-") + "</td><td>" +
         (r.expiry || "-") + "</td><td>" + (r.qty || "-") + "</td><td>" +
         r.flags.map(chip).join("") + "</td></tr>";
  }
  $("log").innerHTML = h + "</table>";
}

function show(r) {
  const v = $("verdict");
  v.className = "verdict " + r.status;
  v.innerHTML = WORDS[r.status] + "<span class='sub'>" + WHY[r.status] + "</span>";
  $("f_mat").textContent = r.fields.material || "-";
  $("f_lot").textContent = r.fields.lot_no   || "-";
  $("f_exp").textContent = r.fields.expiry   || "-";
  $("f_qty").textContent = r.fields.qty      || "-";
  $("f_bc").textContent  = r.barcode         || "could not decode";
  $("flags").innerHTML = r.flags.map(chip).join("");
  let m = "read in " + r.ms + " ms";
  if (!r.logged) m += " &middot; same crate as the last one, not logged again";
  if (r.barcode && r.ocr_lot && r.barcode !== r.ocr_lot)
    m += " &middot; the printed lot read as " + r.ocr_lot + ", the barcode overruled it";
  $("meta").innerHTML = m;
  $("raw").textContent = r.text || "(nothing)";
  if (r.before) {
    $("pics").innerHTML =
      "<figure><img src='" + r.before + "'><figcaption>as sent</figcaption></figure>" +
      "<figure><img src='" + r.after + "'><figcaption>after the OpenCV step</figcaption></figure>";
  }
  if (r.summary) drawLog(r.summary);
}

async function send(blob, pics, source) {
  const fd = new FormData();
  fd.append("image", blob, "frame.jpg");
  fd.append("pics", pics ? "1" : "0");
  fd.append("source", source || "upload");
  try {
    show(await (await fetch("/scan", {method: "POST", body: fd})).json());
  } catch (e) {
    $("verdict").className = "verdict REJECTED";
    $("verdict").textContent = "could not reach the server";
  }
}

async function sendMany(files) {
  for (let i = 0; i < files.length; i++)
    // one at a time on purpose, firing them all at once just queues them behind the ocr anyway
    await send(files[i], files.length === 1, files[i].name);
}

$("drop").onclick = () => $("file").click();
$("file").onchange = e => { if (e.target.files.length) sendMany([...e.target.files]); };
$("drop").ondragover = e => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = e => {
  e.preventDefault(); $("drop").classList.remove("over");
  if (e.dataTransfer.files.length) sendMany([...e.dataTransfer.files]);
};

function frame() {
  const v = $("vid");
  if (!v.videoWidth) return null;
  const c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  return new Promise(r => c.toBlob(r, "image/jpeg", 0.9));
}

$("start").onclick = async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({video: {facingMode: "environment",
                                                                 width: {ideal: 1280}}});
    $("vid").srcObject = s;
    $("grab").disabled = false; $("auto").disabled = false;
    $("start").textContent = "Camera on"; $("start").disabled = true;
  } catch (e) { alert("no camera available: " + e.message); }
};

$("grab").onclick = async () => { const b = await frame(); if (b) send(b, true, "camera"); };

$("auto").onclick = () => {
  if (timer) {
    clearInterval(timer); timer = null;
    $("auto").textContent = "Auto scan: off";
  } else {
    // roughly a scan a second. any faster and the requests just queue up behind the ocr
    timer = setInterval(async () => { const b = await frame(); if (b) send(b, false, "camera"); }, 1100);
    $("auto").textContent = "Auto scan: ON";
  }
};

for (const el of document.querySelectorAll(".stat")) {
  el.onclick = () => {
    filter = el.dataset.f;
    for (const o of document.querySelectorAll(".stat")) o.classList.remove("sel");
    el.classList.add("sel");
    if (last) drawLog(last);
  };
}

$("imp").onclick = async () => {
  $("logmsg").textContent = "reading the batch results...";
  const r = await (await fetch("/import_batch", {method: "POST"})).json();
  if (r.error) { $("logmsg").textContent = r.error; return; }
  $("logmsg").textContent = "added " + r.added + " crates from the batch run";
  drawLog(r);
};

$("exp").onclick = async () => {
  const r = await (await fetch("/export", {method: "POST"})).json();
  $("logmsg").textContent = r.error ? r.error : ("saved " + r.rows + " rows to " + r.file);
};

$("clr").onclick = async () => {
  drawLog(await (await fetch("/clear", {method: "POST"})).json());
  $("logmsg").textContent = "";
};

$("wo").onchange = async () => {
  const fd = new FormData();
  fd.append("work_order", $("wo").value);
  drawLog(await (await fetch("/build", {method: "POST", body: fd})).json());
  $("logmsg").textContent = "now building " + $("wo").value +
                            ", anything already in the log was judged against the old one";
};

fetch("/history").then(r => r.json()).then(drawLog);
</script>
</body></html>
"""


if __name__ == "__main__":
    n = import_batch()
    if n:
        print("loaded", n, "crates from the last batch run so the desk is not empty")
    print("open http://localhost:5000")
    app.run(debug=False, port=5000)
