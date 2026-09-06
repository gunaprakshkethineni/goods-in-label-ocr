# the goods-in desk. point a camera at a label or drop a photo in, and it keeps a running log of
# everything scanned this shift. same reading code as the batch tool, this just wraps it and
# remembers what has been through.
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
from validate import DEFAULT_BUILD, check_crate, load_rules

app = Flask(__name__)

# the labels the generator makes are 600 wide. a phone photo is 3000+ and tesseract crawls on
# something that big, so everything gets brought down to roughly the size the pipeline was
# tuned on before it goes anywhere near the ocr
WORK_WIDTH = 900

# how long the same label can sit in front of the camera before it counts as a new scan. auto
# mode fires every second or so and you do not want twenty rows for one part
SAME_LABEL_SECS = 8

EXPORT_FILE = "output/session_scans.csv"

rules = load_rules()

# the log lives in memory, so restarting the server clears it. that is fine for a desk that gets
# emptied at the end of a shift, and the Save CSV button is there for anything worth keeping
scans = []

# which blade the crates on this desk are going onto. it decides which materials are allowed at
# all, and everything already cleared onto it decides which ones still are
build = {"wo": DEFAULT_BUILD}


# what is physically out on the build, which is only the crates that cleared
def issued_now():
    return [{"material": s["material"], "lot_no": s["lot_no"]} for s in scans if not s["flags"]]


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


# put one reading into the log, unless it is the same label still sitting in front of the lens
def log_scan(row, flags, source):
    key = key_for(row)
    now = time.time()

    if scans and key and scans[-1]["key"] == key and now - scans[-1]["t"] < SAME_LABEL_SECS:
        return None, flags

    # the same lot arriving twice as two separate crates is not automatically wrong, a delivery
    # can be split over several drums. but it is worth putting in front of somebody, because the
    # other explanation is that the same crate got booked in twice and the stock count is now out
    if key and any(s["key"] == key for s in scans):
        flags = flags + ["LOT_ALREADY_BOOKED_IN"]

    entry = {"n": len(scans) + 1,
             "t": now,
             "when": time.strftime("%H:%M:%S"),
             "key": key,
             "source": source,
             "material": row.get("material", ""),
             "lot_no": row.get("lot_no", ""),
             "expiry": row.get("expiry", ""),
             "qty": row.get("qty", ""),
             "barcode": row.get("barcode", ""),
             "flags": flags}
    scans.append(entry)
    return entry, flags


def summary():
    bad = sum(1 for s in scans if s["flags"])
    total = len(scans)
    return {"total": total,
            "passed": total - bad,
            "flagged": bad,
            "pct": round(100.0 * (total - bad) / total) if total else 0,
            "build": build["wo"],
            "builds": sorted(rules["spec"]) or [DEFAULT_BUILD],
            "rows": [{k: s[k] for k in
                      ["n", "when", "source", "material", "lot_no", "expiry", "qty", "flags"]}
                     for s in reversed(scans)]}


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
        "ok": not flags,
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


# pulls in whatever the batch run already read, so the log shows the 60 labels from
# ocr_reader.py alongside anything scanned by hand
@app.route("/import_batch", methods=["POST"])
def import_batch():
    path = "output/inventory.csv"
    if not os.path.exists(path):
        return jsonify({"error": "no output/inventory.csv yet, run: python ocr_reader.py"}), 400

    added = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fixed, flags = check_crate(row, rules, build["wo"], issued_now())
            entry, _ = log_scan(fixed, flags, row.get("filename", "batch"))
            if entry:
                added += 1
    return jsonify(dict(summary(), added=added))


@app.route("/export", methods=["POST"])
def export():
    if not scans:
        return jsonify({"error": "nothing scanned yet"}), 400

    os.makedirs("output", exist_ok=True)
    cols = ["n", "when", "build", "source", "material", "lot_no", "expiry", "qty", "barcode",
            "status", "flags"]
    with open(EXPORT_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in scans:
            w.writerow({"n": s["n"], "when": s["when"], "build": build["wo"],
                        "source": s["source"], "material": s["material"], "lot_no": s["lot_no"],
                        "expiry": s["expiry"], "qty": s["qty"], "barcode": s["barcode"],
                        "status": "HELD" if s["flags"] else "CLEARED",
                        "flags": ";".join(s["flags"])})
    return jsonify({"file": os.path.abspath(EXPORT_FILE), "rows": len(scans)})


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Label scanner</title>
<style>
 *{box-sizing:border-box}
 body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1c1e21}
 header{background:#1f2933;color:#fff;padding:22px 30px}
 header h1{margin:0 0 3px;font-size:20px}
 header p{margin:0;color:#9aa5b1;font-size:13px}
 .wo{margin-top:12px;font-size:13px;color:#9aa5b1}
 .wo select{font-size:14px;padding:4px 8px;border-radius:5px;border:1px solid #3b4653;
            background:#2b3542;color:#fff}
 #wospec{margin-left:10px;font-size:12px}
 .tabs{display:flex;gap:6px;padding:14px 30px 0;background:#fff;border-bottom:1px solid #dfe3e8}
 .tab{padding:9px 18px;border:1px solid #dfe3e8;border-bottom:none;border-radius:7px 7px 0 0;
      background:#f4f5f7;cursor:pointer;font-size:14px}
 .tab.on{background:#fff;font-weight:600;position:relative;top:1px}
 .wrap{display:flex;gap:20px;padding:22px 30px 0;flex-wrap:wrap;align-items:flex-start}
 .col{flex:1 1 380px;min-width:340px}
 .box{background:#fff;border:1px solid #dfe3e8;border-radius:9px;padding:16px;margin-bottom:20px}
 .box h2{margin:0 0 12px;font-size:14px}
 #drop{border:2px dashed #c3cad3;border-radius:8px;padding:34px 16px;text-align:center;
       color:#6b7480;cursor:pointer;font-size:14px}
 #drop.over{border-color:#2e6fd4;background:#f0f6ff;color:#2e6fd4}
 video{width:100%;border-radius:7px;background:#000;display:block}
 .row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
 button{padding:9px 16px;border:0;border-radius:6px;background:#2e6fd4;color:#fff;
        font-size:14px;cursor:pointer}
 button.grey{background:#66707c}
 button:disabled{background:#b9c0c8;cursor:default}
 .verdict{padding:12px 15px;border-radius:8px;font-weight:700;margin-bottom:13px;font-size:15px}
 .verdict.ok{background:#e3f5ea;color:#1d7541}
 .verdict.bad{background:#fbe6e1;color:#a8331c}
 .verdict.idle{background:#eef0f3;color:#6b7480}
 table{width:100%;border-collapse:collapse;font-size:14px}
 .fields th{text-align:left;color:#6b7480;font-weight:600;padding:5px 0;width:96px;font-size:12.5px}
 .fields td{padding:5px 0;font-family:Consolas,monospace}
 .flag{display:inline-block;background:#fbe6e1;color:#a8331c;font-size:11px;
       font-family:Consolas,monospace;padding:3px 8px;border-radius:4px;margin:3px 4px 0 0}
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
 .stat{background:#f7f8fa;border:1px solid #e3e7ec;border-radius:8px;padding:10px 16px;min-width:104px}
 .stat b{display:block;font-size:21px}
 .stat span{font-size:11.5px;color:#6b7480}
 .log{overflow-x:auto}
 .log table{min-width:760px}
 .log th{text-align:left;font-size:11.5px;color:#6b7480;border-bottom:1px solid #dfe3e8;
         padding:7px 9px 7px 0;white-space:nowrap}
 .log td{padding:7px 9px 7px 0;border-bottom:1px solid #f0f2f4;font-size:13px;
         font-family:Consolas,monospace;white-space:nowrap}
 .pill{font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:10px}
 .pill.ok{background:#e3f5ea;color:#1d7541}
 .pill.bad{background:#fbe6e1;color:#a8331c}
 .empty{color:#6b7480;font-size:13px;padding:14px 0}
 code{background:#f0f2f4;padding:1px 5px;border-radius:3px;font-size:12px}
</style></head><body>
<header>
  <h1>Material intake</h1>
  <p>Book crates onto a blade. Reads the label, checks the lot against the material master and the
     build spec, and holds anything it is not sure about.</p>
  <div class="wo">
    building <select id="wo"></select>
    <span id="wospec"></span>
  </div>
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
      <div class="hint">You can pick several at once. <code>data/labels/</code> is full of them.
        Switching the blade at the top changes what counts as acceptable, so the same crate can
        clear on one build and be held on another.</div>
    </div>

    <div class="box" id="pane-cam" style="display:none">
      <h2>Camera</h2>
      <video id="vid" autoplay playsinline muted></video>
      <div class="row">
        <button id="start">Start camera</button>
        <button id="grab" class="grey" disabled>Scan once</button>
        <button id="auto" class="grey" disabled>Auto scan: off</button>
      </div>
      <div class="hint">
        Hold the label so it fills the frame, flat on and in decent light. The same label sitting
        in front of the lens only gets logged once.
      </div>
    </div>
  </div>

  <div class="col">
    <div class="box">
      <h2>Last read</h2>
      <div class="verdict idle" id="verdict">nothing scanned yet</div>
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
    <h2>Scanned this session</h2>
    <div class="stats">
      <div class="stat"><b id="s_total">0</b><span>crates booked in</span></div>
      <div class="stat"><b id="s_pass">0</b><span>cleared to the build</span></div>
      <div class="stat"><b id="s_flag">0</b><span>held for a person</span></div>
      <div class="stat"><b id="s_pct">0%</b><span>went through clean</span></div>
    </div>
    <div class="row" style="margin:0 0 14px">
      <button id="imp" class="grey">Add the 60 batch crates</button>
      <button id="exp" class="grey">Save CSV</button>
      <button id="clr" class="grey">Clear</button>
      <span class="meta" id="logmsg" style="align-self:center"></span>
    </div>
    <div class="log" id="log"><div class="empty">nothing yet</div></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let timer = null;

function drawLog(s) {
  $("s_total").textContent = s.total;
  $("s_pass").textContent  = s.passed;
  $("s_flag").textContent  = s.flagged;
  $("s_pct").textContent   = s.pct + "%";

  const sel = $("wo");
  if (sel.options.length !== s.builds.length) {
    sel.innerHTML = s.builds.map(b => "<option>" + b + "</option>").join("");
  }
  sel.value = s.build;

  if (!s.rows.length) { $("log").innerHTML = "<div class='empty'>nothing yet</div>"; return; }
  let h = "<table><tr><th>#</th><th>time</th><th>source</th><th>material</th><th>lot</th>" +
          "<th>expires</th><th>qty</th><th>status</th></tr>";
  for (const r of s.rows) {
    const ok = r.flags.length === 0;
    h += "<tr><td>" + r.n + "</td><td>" + r.when + "</td><td>" + r.source + "</td><td>" +
         (r.material || "-") + "</td><td>" + (r.lot_no || "-") + "</td><td>" +
         (r.expiry || "-") + "</td><td>" + (r.qty || "-") + "</td><td>" +
         (ok ? "<span class='pill ok'>CLEARED</span>"
             : "<span class='pill bad'>HELD</span> " +
               r.flags.map(f => "<span class='flag'>" + f + "</span>").join("")) +
         "</td></tr>";
  }
  $("log").innerHTML = h + "</table>";
}

function show(r) {
  const v = $("verdict");
  v.className = "verdict " + (r.ok ? "ok" : "bad");
  v.textContent = r.ok ? "CLEARED - booked onto the build" : "HELD - a person has to look at this";
  $("f_mat").textContent = r.fields.material || "-";
  $("f_lot").textContent = r.fields.lot_no   || "-";
  $("f_exp").textContent = r.fields.expiry   || "-";
  $("f_qty").textContent = r.fields.qty      || "-";
  $("f_bc").textContent  = r.barcode         || "could not decode";
  $("flags").innerHTML = r.flags.map(f => "<span class='flag'>" + f + "</span>").join("");
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
    const res = await fetch("/scan", {method: "POST", body: fd});
    show(await res.json());
  } catch (e) {
    $("verdict").className = "verdict bad";
    $("verdict").textContent = "could not reach the server";
  }
}

async function sendMany(files) {
  for (let i = 0; i < files.length; i++) {
    // one at a time on purpose, firing them all at once just queues them behind the ocr anyway
    await send(files[i], files.length === 1, files[i].name);
  }
}

$("drop").onclick = () => $("file").click();
$("file").onchange = e => { if (e.target.files.length) sendMany([...e.target.files]); };
$("drop").ondragover = e => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = e => {
  e.preventDefault();
  $("drop").classList.remove("over");
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
    $("grab").disabled = false;
    $("auto").disabled = false;
    $("start").textContent = "Camera on";
    $("start").disabled = true;
  } catch (e) {
    alert("no camera available: " + e.message);
  }
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

$("imp").onclick = async () => {
  $("logmsg").textContent = "reading the batch results...";
  const r = await (await fetch("/import_batch", {method: "POST"})).json();
  if (r.error) { $("logmsg").textContent = r.error; return; }
  $("logmsg").textContent = "added " + r.added + " labels from the batch run";
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
  $("logmsg").textContent = "now building " + $("wo").value;
};

$("tab-up").onclick = () => {
  $("tab-up").classList.add("on"); $("tab-cam").classList.remove("on");
  $("pane-up").style.display = ""; $("pane-cam").style.display = "none";
};
$("tab-cam").onclick = () => {
  $("tab-cam").classList.add("on"); $("tab-up").classList.remove("on");
  $("pane-cam").style.display = ""; $("pane-up").style.display = "none";
};

fetch("/history").then(r => r.json()).then(drawLog);
</script>
</body></html>
"""


if __name__ == "__main__":
    print("open http://localhost:5000")
    app.run(debug=False, port=5000)
