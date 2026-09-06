# Material intake verification for composite blade manufacturing

Reads the label on an incoming material crate, checks the lot against the material master and the
build it is meant for, and holds anything it is not sure about before the material can reach the
mould.

## The problem it solves

A wind turbine blade is 60-odd metres of epoxy, fabric and adhesive, and every kilogram of it
arrives in a batch with a lot number on the side. Those lot numbers are the entire traceability
chain. If a blade fails offshore three years from now, the only way to work out which other
blades are suspect is to know exactly which lots went into it.

The way that usually gets recorded is somebody with a clipboard reading tiny print off a drum and
typing it into the system. It is slow, and a typo in a lot number is invisible until it matters.

Worse, some mistakes are not typos at all. The plant here is qualified for two epoxy systems.
Either can be used on a blade. But system A resin cured with system B hardener produces a blade
that looks perfect on the shop floor and cracks under storm loading years later. Nothing is wrong
with either drum. Only with the pair. No amount of reading the label carefully catches that,
because you have to know what else is already out on that build.

So the job is not "read text from a picture". Any OCR library does that in one line. The job is
deciding **which readings you can trust and which crates are actually allowed on this blade** -
because a reader that is right 99% of the time is no use if a person still has to check all 60
rows to find the wrong ones.

## What it actually checks

Reading the label is the easy half. Once it has the fields it cross-examines them:

| Check | What it catches |
|---|---|
| `BARCODE_UNREADABLE` | zbar got nothing off the crate |
| `LOT_MISMATCH_ON_LABEL` | printed lot and barcode disagree |
| `LOT_NOT_IN_MASTER` | a lot nobody ever booked in |
| `LOT_NOT_RELEASED` | quality have not signed that lot off yet |
| `EXPIRED` | past its shelf life |
| `LABEL_MATERIAL_MISMATCH` | the drum is labelled as one material, the master says that lot is another |
| `NOT_ON_BUILD_SPEC` | glass fabric turning up for a carbon blade |
| `INCOMPATIBLE_WITH_ISSUED` | system B hardener, when system A resin is already out on this build |
| `MATERIAL_FORMAT_BAD`, `LOT_FORMAT_BAD`, `FIELD_MISSING`, `QTY_OUT_OF_RANGE` | the reader was not confident |

The last one is the point. When the reader is unsure it says so instead of guessing, and that
crate goes to a person. Everything else books itself in.

## Results on my machine

60 crates for BLADE-402, Python 3.12, Tesseract 5.4.0, i5 laptop, `--seed 7`.

| | |
|---|---|
| character accuracy, straight off the OCR | **98.9%** |
| same thing with the OpenCV step turned off | 45.6% |
| after the barcode and master data repairs | **99.8%** |
| 10 crates, start to finish | **2.40 s** (240 ms each) |
| crates needing a person | **8 of 60, so 86.7% less checking** |
| real defects caught | **7 of 7, none let through** |
| good crates held anyway | 1 |

The OpenCV preprocessing is worth **53.3 percentage points** on its own. That is the number I care
about most, because it is the difference between the tool being useful and being noise.

Two numbers are worth separating. Straight off the OCR the material code is only right on 49 of
60, because Tesseract drops or invents a digit at the end of `HRD-AM-1150` constantly. After the
master data repairs it is right on 60 of 60. The reader on its own is not good enough. The reader
plus a barcode with a check digit plus a catalogue to match against is.

Everything is seeded, so a rerun gives the same numbers.

## Setup

Tesseract is a separate program, pip does not install it:

```
winget install UB-Mannheim.TesseractOCR
```

Then:

```
pip install -r requirements.txt
```

If Tesseract lands somewhere other than `C:\Program Files\Tesseract-OCR`, change the path at the
top of `ocr_reader.py`.

## Booking crates in

```
python app.py
```

Then http://localhost:5000. Pick the blade you are building at the top, then either drop crate
photos in or turn the camera on and hold labels up to it. You get the material, lot, expiry and
quantity, whether it cleared or was held, and why.

Underneath is the running log of the shift, with counters and a Save CSV button. **Add the 60
batch crates** drops the whole batch run into the same log.

The thing worth demoing is the blade selector. Scan a carbon fabric crate against BLADE-402 and
it clears. Switch to BLADE-518, which is the glass layup, scan the same crate, and it is held on
`NOT_ON_BUILD_SPEC`. Same crate, same label, different answer, because the question is not "is
this a valid lot" but "is this right for what we are building".

## Seeing the whole batch

```
python make_labels.py --n 60 --seed 7
python demo.py
```

Builds `output/report.html` and opens it: every crate as a picture next to what was read off it,
what it should have been, and why it was held. Held crates come first. The images are baked into
the file so you can send it to somebody.

## Running the pieces

```
python make_labels.py --n 60 --seed 7    # generate the crates and the master data
python ocr_reader.py                     # read them -> output/inventory.csv
python validate.py                       # check them -> inventory_final.csv + flagged.csv
python check_accuracy.py                 # score it against the ground truth
```

`python validate.py BLADE-518` checks the same crates against the other blade.
`python batch_test.py` does the timing run, `python -m pytest` runs the tests (26 of them), and
`python preprocess.py data/labels/crate_003.png` dumps each stage of the cleanup to
`output/debug/`.

## Where the images come from

I generate them. There is no public dataset of goods-in labels with the lot numbers written down
alongside, because material traceability records are commercially sensitive and nobody publishes
theirs. Without ground truth there is no way to put a number on accuracy at all.

`make_labels.py` draws a crate label, then attacks it: rotates it a few degrees, lays a shadow
across it, blurs it, adds sensor noise and speckle, and saves it as a low quality JPEG. The field
text prints in grey rather than black because thermal label printers fade, and that one setting
is what makes the reader work for its money.

It also writes the master data the checks run against: 32 lots across 8 materials, two blade
specs, and the resin/hardener compatibility table. Supplier names are invented.

About one crate in eight gets something genuinely wrong with it, and the generator records what,
so afterwards I can tell a real catch from a false alarm. The defect plan rounds every rate up to
at least one, so every rule gets exercised on every run.

## The OpenCV part

`preprocess.py`, in order:

1. grayscale
2. median blur, for the speckle
3. `fastNlMeansDenoising`, for the sensor grain
4. deskew, using `minAreaRect` on the text pixels
5. 2x upscale with cubic interpolation

Then Tesseract with `--psm 6` and a character whitelist.

`searchWindowSize` on the denoiser defaults to 21 and costs about a second per ten crates on its
own. Dropping it to 9 made the whole thing a third faster and scored slightly better, because a
big search window averages over half the label and starts softening the digits.

## The two repairs

`validate.py` fixes two things rather than flagging them, and both are the difference between a
useful tool and one that shouts constantly.

**The barcode wins.** Code128 carries a check digit and OCR carries nothing, so when the barcode
decodes it overrules the printed lot. The text is then only a cross check: if both are readable
and they disagree, that is worth a person's time.

**Codes get matched to the catalogue.** A material code within two edits of exactly one entry in
the master gets snapped to it, which is what turns `HRD-AM-115` back into `HRD-AM-1150`. If two
entries are equally close it refuses and lets the crate be held, because guessing a lot number is
exactly the quiet mistake this whole thing exists to prevent.

Without those two, the tool held 40% of crates and saved nobody any time.

## Things that did not work

**Thresholding it myself.** I had an adaptive threshold and a morphology open in the pipeline and
both made it worse, about 8 points worse. Tesseract binarises internally and is better at it than
my fixed block size was.

**`BORDER_REPLICATE` on the deskew.** Rotating with replicated borders smears the black frame line
of the label into big black wedges in the corners. The picture still looks perfectly readable to a
human and Tesseract returns `F M110` for the whole label. Eight of sixty came back empty.

**`minAreaRect` angles.** My OpenCV hands back angles in (-90, 0], so a label tilted 1.3 degrees
arrives as -88.7. I was checking `abs(angle) > 10` and bailing out, which quietly turned the
deskew off on nearly every label. Folding the angle into -45..45 first took accuracy from 84.8%
to 96.9% in one go.

**Regexes for the fields.** `LOT[:\s]*([A-Z0-9]+)` looks fine until the L reads as a 1 or the
colon disappears. The tags themselves get mangled too: one run gave `OTY 149`, another `O:151`,
and one crate lost the tag entirely and left the line as just `77`. Going line by line, accepting
a space instead of a colon and allowing the tag to be one edit out handles most of it, with a
digits-only fallback for the quantity since it is the only bare number on the label.

**Putting the quantity too close to the barcode.** It sat about 17px above it and Tesseract kept
swallowing the whole line into the barcode block and returning nothing for it. Moving the barcode
down 20px fixed four false alarms at a stroke. Real labels have a gap there and now I know why.

**Making the barcode bigger.** I assumed thicker bars would survive more damage. They decoded
worse. The real problem was that the speckle lands after the blur, so the dots sit on soft bars
and bridge them. Sensor noise past sigma 12 and JPEG below quality 40 kill the barcode on every
label, so the difficulty had to go somewhere the barcode does not care about, which is why the
text is faded rather than noisy.

**Leaving the old labels in the folder.** I changed the label format and the reader picked up 120
images against a 60 row truth file. Every stale one came back as a fistful of missing fields and
it looked like the parser had broken. The generator wipes the folder now.

**Rules that never fired.** The expired and not-released checks looked correct and sat there doing
nothing for two runs, because the generator only ever picked good lots. Then the incompatible
hardener stopped appearing at all when a random seed shifted. A rule you have never seen fire is
not a rule you have tested, so the defect plan now guarantees at least one of every kind.

## Limitations

The crates are synthetic, so 98.9% is 98.9% on my generator and not a promise about a photograph
of a real drum in a real loading bay. Glare off shrink wrap, dust, perspective and creased labels
are all things I do not simulate.

The layout is fixed, one label format, and the field parser assumes it. Point it at a different
supplier's label and the barcode will still decode but the fields will come back empty.
Supporting a new format means adding its layout; the checking and flagging logic is unchanged.

The lot snapping is safe here because there are 32 lots and they are ten characters long, so no
two are close to each other. Against a master of thousands I would drop `SNAP_MAX` to 1.

`INCOMPATIBLE_WITH_ISSUED` only knows about what has been booked in during this session. A real
one would ask the MES what has already been issued to that work order.
