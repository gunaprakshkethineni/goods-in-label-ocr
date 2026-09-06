# little web app so you can actually point a camera at a label, or drop a photo in, and see what
# comes back. same reading code as everything else, this just wraps it.
#
#   python app.py
#   then go to http://localhost:5000

import base64
import io
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from ocr_reader import read_image
from preprocess import clean
from validate import check_row, load_approved_lots

app = Flask(__name__)

# the labels the generator makes are 600 wide. a phone photo is 3000+ and tesseract crawls on
# something that big, so everything gets brought down to roughly the size the pipeline was
# tuned on before it goes anywhere near the ocr
WORK_WIDTH = 900

approved = load_approved_lots()


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

    start = time.perf_counter()
    row = read_image(img)
    raw_text = row.pop("text")
    fixed, flags = check_row(dict(row, filename="upload"), approved)
    took = time.perf_counter() - start

    # the live camera fires constantly, no point shipping pictures back every time
    want_pics = request.form.get("pics") == "1"

    out = {
        "ok": not flags,
        "fields": {k: fixed.get(k, "") for k in ["part_no", "serial", "lot_no", "qty"]},
        "barcode": fixed.get("barcode", ""),
        "ocr_serial": row.get("serial", ""),
        "flags": flags,
        "ms": round(took * 1000),
        "text": raw_text.strip(),
    }
    if want_pics:
        out["before"] = as_data_url(img)
        out["after"] = as_data_url(clean(img))
    return jsonify(out)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Label scanner</title>
<style>
 *{box-sizing:border-box}
 body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1c1e21}
 header{background:#1f2933;color:#fff;padding:22px 30px}
 header h1{margin:0 0 3px;font-size:20px}
 header p{margin:0;color:#9aa5b1;font-size:13px}
 .tabs{display:flex;gap:6px;padding:14px 30px 0;background:#fff;border-bottom:1px solid #dfe3e8}
 .tab{padding:9px 18px;border:1px solid #dfe3e8;border-bottom:none;border-radius:7px 7px 0 0;
      background:#f4f5f7;cursor:pointer;font-size:14px}
 .tab.on{background:#fff;font-weight:600;position:relative;top:1px}
 .wrap{display:flex;gap:20px;padding:22px 30px;flex-wrap:wrap;align-items:flex-start}
 .col{flex:1 1 380px;min-width:340px}
 .box{background:#fff;border:1px solid #dfe3e8;border-radius:9px;padding:16px}
 .box h2{margin:0 0 12px;font-size:14px}
 #drop{border:2px dashed #c3cad3;border-radius:8px;padding:34px 16px;text-align:center;
       color:#6b7480;cursor:pointer;font-size:14px}
 #drop.over{border-color:#2e6fd4;background:#f0f6ff;color:#2e6fd4}
 video,#shot{width:100%;border-radius:7px;background:#000;display:block}
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
 th{text-align:left;color:#6b7480;font-weight:600;padding:5px 0;width:96px;font-size:12.5px}
 td{padding:5px 0;font-family:Consolas,monospace}
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
</style></head><body>
<header>
  <h1>Label scanner</h1>
  <p>Drop in a photo or point a camera at a label. Same reader and same checks as the batch tool.</p>
</header>

<div class="tabs">
  <div class="tab on" id="tab-up">Upload a photo</div>
  <div class="tab" id="tab-cam">Live camera</div>
</div>

<div class="wrap">
  <div class="col">
    <div class="box" id="pane-up">
      <h2>Upload</h2>
      <div id="drop">Drop a label image here, or click to pick one</div>
      <input type="file" id="file" accept="image/*" hidden>
      <div class="hint">Anything the generator made works. <code>data/labels/</code> is full of them.</div>
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
        Hold the label so it fills the frame, flat on and in decent light. Printing one of the
        generated labels, or just showing it on a phone screen, scans best.
      </div>
    </div>
  </div>

  <div class="col">
    <div class="box">
      <h2>Result</h2>
      <div class="verdict idle" id="verdict">nothing scanned yet</div>
      <table>
        <tr><th>Part no</th><td id="f_part">-</td></tr>
        <tr><th>Serial</th><td id="f_serial">-</td></tr>
        <tr><th>Lot no</th><td id="f_lot">-</td></tr>
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

<script>
const $ = id => document.getElementById(id);
let timer = null;

function show(r) {
  const v = $("verdict");
  v.className = "verdict " + (r.ok ? "ok" : "bad");
  v.textContent = r.ok ? "PASS - goes straight to inventory" : "NEEDS CHECK - sent to a person";
  $("f_part").textContent   = r.fields.part_no || "-";
  $("f_serial").textContent = r.fields.serial  || "-";
  $("f_lot").textContent    = r.fields.lot_no  || "-";
  $("f_qty").textContent    = r.fields.qty     || "-";
  $("f_bc").textContent     = r.barcode        || "could not decode";
  $("flags").innerHTML = r.flags.map(f => "<span class='flag'>" + f + "</span>").join("");
  let m = "read in " + r.ms + " ms";
  if (r.barcode && r.ocr_serial && r.barcode !== r.ocr_serial)
    m += " &middot; OCR read the serial as " + r.ocr_serial + ", the barcode overruled it";
  $("meta").innerHTML = m;
  $("raw").textContent = r.text || "(nothing)";
  if (r.before) {
    $("pics").innerHTML =
      "<figure><img src='" + r.before + "'><figcaption>as sent</figcaption></figure>" +
      "<figure><img src='" + r.after + "'><figcaption>after the OpenCV step</figcaption></figure>";
  }
}

async function send(blob, pics) {
  const fd = new FormData();
  fd.append("image", blob, "frame.jpg");
  fd.append("pics", pics ? "1" : "0");
  try {
    const res = await fetch("/scan", {method: "POST", body: fd});
    show(await res.json());
  } catch (e) {
    $("verdict").className = "verdict bad";
    $("verdict").textContent = "could not reach the server";
  }
}

// upload side
$("drop").onclick = () => $("file").click();
$("file").onchange = e => { if (e.target.files[0]) send(e.target.files[0], true); };
$("drop").ondragover = e => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = e => {
  e.preventDefault();
  $("drop").classList.remove("over");
  if (e.dataTransfer.files[0]) send(e.dataTransfer.files[0], true);
};

// camera side
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

$("grab").onclick = async () => { const b = await frame(); if (b) send(b, true); };

$("auto").onclick = () => {
  if (timer) {
    clearInterval(timer); timer = null;
    $("auto").textContent = "Auto scan: off";
  } else {
    // roughly a scan a second. any faster and the requests just queue up behind the ocr
    timer = setInterval(async () => { const b = await frame(); if (b) send(b, false); }, 1100);
    $("auto").textContent = "Auto scan: ON";
  }
};

// tabs
$("tab-up").onclick = () => {
  $("tab-up").classList.add("on"); $("tab-cam").classList.remove("on");
  $("pane-up").style.display = ""; $("pane-cam").style.display = "none";
};
$("tab-cam").onclick = () => {
  $("tab-cam").classList.add("on"); $("tab-up").classList.remove("on");
  $("pane-cam").style.display = ""; $("pane-up").style.display = "none";
};
</script>
</body></html>
"""


if __name__ == "__main__":
    print("open http://localhost:5000")
    app.run(debug=False, port=5000)
