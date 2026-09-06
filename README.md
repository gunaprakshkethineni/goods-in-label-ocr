# Factory Inventory Tracking and Quality Control (OCR)

Reads component labels from photos, pulls out the part number, serial number, lot number and
quantity, and decides which ones a person actually needs to look at.

The idea is a goods-in desk. Somebody photographs a tray of parts, the tool reads every label,
writes the good ones straight to the inventory file, and puts the doubtful ones in a separate
file for a human. That second file is the whole point, because checking 12 labels is a lot
quicker than checking 60.

## Results on my machine

60 labels, Python 3.12, Tesseract 5.4.0, i5 laptop.

| | |
|---|---|
| character accuracy | **97.1%** |
| same thing with the OpenCV step turned off | 59.3% |
| 10 labels, start to finish | **3.48 s** (348 ms each) |
| labels needing manual review | **12 of 60, so 80% less checking** |
| injected defects the validator caught | **7 of 7, none missed** |
| false alarms | 5 |

The OpenCV preprocessing is worth 37.8 percentage points on its own. That is the number I care
about most, because it is the difference between the tool being useful and being noise.

Exact field matches: part number 56/60, lot 53/60, quantity 55/60, serial 44/60. The serial
looks bad until you remember the barcode also carries it, and Code128 has a check digit while
OCR has nothing, so when the barcode decodes it overrules the text. In the final inventory the
serial is right on every row where the barcode read.

## Setup

Tesseract is a separate program, pip does not install it:

```
winget install UB-Mannheim.TesseractOCR
```

Then:

```
pip install -r requirements.txt
```

If Tesseract ends up somewhere other than `C:\Program Files\Tesseract-OCR`, change the path at
the top of `ocr_reader.py`.

## Running it

```
python make_labels.py --n 60 --seed 7
python ocr_reader.py
python validate.py
python check_accuracy.py
```

`make_labels.py` builds the test images, `ocr_reader.py` reads them into
`output/inventory.csv`, `validate.py` splits that into `output/inventory_final.csv` and
`output/flagged.csv`, and `check_accuracy.py` scores the whole thing against the ground truth.

`python batch_test.py` does the timing run and `python -m pytest` runs the tests.

## Where the images come from

I generate them. I looked for a public dataset of real component labels with the serial and lot
numbers written down alongside, and there is not one. Kaggle has barcode photos but no text
ground truth, and nobody publishes their factory labels because the numbers on them are
commercially sensitive. Without ground truth I cannot put a number on accuracy at all, so I
build the labels myself and keep the answers.

`make_labels.py` draws a label, then attacks it: rotates it a few degrees, lays a shadow
gradient across it, blurs it, adds sensor noise and speckle, and saves it as a low quality JPEG.
The field text prints in grey rather than black because thermal label printers fade, and that
one setting is what makes the OCR work for its money.

It also breaks about 3% of the barcodes on purpose and puts a lot number that is not on the
approved list on another 3%. Those are the defects `validate.py` is supposed to find, and
because the generator writes down which labels it broke, I can check afterwards whether the
flags were real or not.

## The OpenCV part

`preprocess.py`, in order:

1. grayscale
2. median blur, for the speckle
3. `fastNlMeansDenoising`, for the sensor grain
4. deskew, using `minAreaRect` on the text pixels
5. 3x upscale with cubic interpolation

Then Tesseract with `--psm 6` and a character whitelist.

## What validate.py checks

| flag | when |
|---|---|
| `BARCODE_UNREADABLE` | zbar got nothing |
| `SERIAL_MISMATCH` | barcode and OCR serial both readable and they disagree |
| `LOT_MISMATCH` | lot is not on the approved list and is not close to one |
| `LOT_FORMAT_BAD` | lot does not look like `L2024-0917` |
| `FIELD_MISSING` | nothing came back for a field |
| `PART_FORMAT_BAD` | part number does not look like `4471-B2` |
| `QTY_OUT_OF_RANGE` | quantity is 0 or over 500 |

It repairs two things instead of flagging them. If the barcode decoded it wins over the OCR
serial, and if a lot number is within two edits of exactly one approved lot it gets snapped to
it. Without those repairs the tool flagged 40% of rows and saved almost nobody any time.

## Things that did not work

**Thresholding it myself.** I had an adaptive threshold and a morphology open in the pipeline
and both made it worse, about 8 points worse. Tesseract binarises internally and it is better at
it than my fixed block size was. The denoised greyscale goes straight through now.

**`BORDER_REPLICATE` on the deskew.** This one took me ages. Rotating with replicated borders
smears the black frame line of the label into big black wedges in the corners. The picture still
looks completely readable to a human, and Tesseract returns `F M110` for the entire label. Eight
of sixty labels were coming back totally empty because of it.

**`minAreaRect` angles.** My OpenCV hands back angles in (-90, 0], so a label tilted by 1.3
degrees arrives as -88.7. I was checking `abs(angle) > 10` and bailing out, which quietly turned
the deskew off on nearly every label. Folding the angle into -45..45 first took the accuracy from
84.8% to 96.9% in one go.

**Regexes for the fields.** `SN[:\s]*([A-Z0-9]+)` looks fine until Tesseract reads the S of
SN9332820 as a dollar sign, and then the pattern does not match and the serial is gone
completely. Same for the tags themselves, QTY comes out as OTY often enough to matter. Going
line by line and allowing the tag to be one character out fixed it.

**Making the barcode bigger.** I assumed thicker bars would survive more damage and they did
not, wider bars decoded worse. The real problem was that the speckle is applied after the blur,
so the dots sit on top of already soft bars and bridge them together. Sensor noise past sigma 12
and JPEG below quality 40 kill the barcode on literally every label, so the difficulty had to go
somewhere the barcode does not care about, which is why the text is faded rather than noisy.

## Limitations

The labels are synthetic, so 97.1% is 97.1% on my generator and not a promise about a real
photograph. Real ones would bring perspective, glare, creases and dirt that I do not simulate.
The layout is also fixed, one vendor format, and the field regexes assume it.

The lot snapping is safe here because there are only a dozen approved lots and they are ten
characters long, so no two are close to each other. With thousands of lots I would drop
`LOT_SNAP_MAX` to 1, otherwise it will eventually snap something to the wrong lot.
