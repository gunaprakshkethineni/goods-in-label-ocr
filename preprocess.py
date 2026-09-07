# this is the opencv cleanup i run before tesseract sees anything.
# i got the order of these steps wrong twice before it stuck.

import os
import sys

import cv2
import numpy as np

DEBUG_DIR = "output/debug"


def deskew(img):
    # i added this because pyzbar kept missing the tilted labels
    inv = cv2.bitwise_not(img)
    _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(th)
    if pts is None:
        return img

    angle = cv2.minAreaRect(pts)[-1]

    # i had to fold the angle into -45..45 first. my opencv gives them back in (-90, 0], so a
    # label only 1.3 degrees off arrives as -88.7. before i worked that out my "too far off to
    # trust" check below was throwing away nearly every real measurement i had
    if angle > 45:
        angle = angle - 90
    elif angle < -45:
        angle = angle + 90

    if abs(angle) > 10:
        return img          # i decided anything past this is the border, not the text

    h, w = img.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    # i fill the corners with white here. i used BORDER_REPLICATE first, which smeared the black
    # border of the label into big wedges, and tesseract then gave me nothing back at all for the
    # whole label. it took me ages to spot because the picture still looks fine to me
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def order_corners(pts):
    # i sort them tl, tr, br, bl. x+y is smallest top left and biggest bottom right, and x-y
    # separates the other two
    out = np.zeros((4, 2), np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out


def side(a, b):
    return float(np.hypot(*(a - b)))


# i wrote this to flatten a label shot from an angle.
# deskew only turns the picture, and i found that is not enough: a label photographed from the
# side is a trapezoid, the far edge really is shorter, and turning it will never fix that.
# it is the same idea a scanner app uses on a photo of a page.
def deperspective(img, min_skew=0.04):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * img.size:
        return img

    # i take the hull first so a nick in the edge cannot add a corner. i also tried one fixed
    # tolerance and it only found the corners on half of them, so now i try a few
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx
            break
    if quad is None:
        return img

    tl, tr, br, bl = order_corners(quad.reshape(4, 2).astype(np.float32))
    top, bottom = side(tl, tr), side(bl, br)
    left, right = side(tl, bl), side(tr, br)
    if min(top, bottom, left, right) < 40:
        return img

    # i leave it alone if the sides already match, because warping a square label just blurs it
    skew = max(abs(top - bottom) / max(top, bottom), abs(left - right) / max(left, right))
    if skew < min_skew:
        return img

    w, h = int(max(top, bottom)), int(max(left, right))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def clean(img, debug=False, crop_to=None, flatten=False):
    steps = []

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append(("1_gray", img))

    if flatten:
        # i only do this for camera pictures. the batch tool reads files that are already flat
        img = deperspective(img)
        steps.append(("1b_flattened", img))

    # i put the median blur first. i had the gaussian first and the salt and pepper survived it,
    # then the denoiser smeared the dots into grey blobs that looked like strokes
    img = cv2.medianBlur(img, 3)
    steps.append(("2_median", img))

    # i dropped searchWindowSize from the default 21 to 9. i expected to trade accuracy for speed
    # and got both: a third faster and slightly better, because a big window averages over half
    # the label and softens the digits
    img = cv2.fastNlMeansDenoising(img, None, h=11, templateWindowSize=7, searchWindowSize=9)
    steps.append(("3_denoise", img))

    img = deskew(img)
    steps.append(("4_deskew", img))

    if crop_to:
        # i crop to the text here, the rest of the label is barcodes. it has to come after the
        # deskew, because the deskew needs the whole label to work the angle out
        img = img[:crop_to, :]
        steps.append(("5_crop", img))

    # i tried an adaptive threshold and a morphology open here and both made it about 8 points
    # worse, so i took them out. tesseract binarises internally and does it better than i did.
    # i also tried 3x instead of 2x, which scores a bit higher but pushes ten crates past four
    # seconds
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    steps.append(("6_upscale", img))

    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        for name, s in steps:
            cv2.imwrite(os.path.join(DEBUG_DIR, name + ".png"), s)
        print("wrote", len(steps), "debug images to", DEBUG_DIR)

    return img


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/labels/crate_000.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("cant read", path)
        sys.exit(1)
    out = clean(img, debug=True)
    cv2.imwrite("output/debug/0_original.png", img)
    print("original", img.shape, "-> cleaned", out.shape)
