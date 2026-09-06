# opencv cleanup that runs before tesseract sees anything.
# the order here matters a lot, i got the ordering wrong twice before this stuck.

import os
import sys

import cv2
import numpy as np

DEBUG_DIR = "output/debug"


def deskew(img):
    # find the angle of the text block and rotate it back level.
    # pyzbar was failing on tilted labels until i put this in
    inv = cv2.bitwise_not(img)
    _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(th)
    if pts is None:
        return img

    angle = cv2.minAreaRect(pts)[-1]

    # minAreaRect does not agree with itself across opencv versions, mine hands back angles in
    # (-90, 0] so a label that is only 1.3 degrees off comes through as -88.7. i was checking
    # abs(angle) > 10 and bailing out, which quietly turned the deskew off on nearly every label.
    # fold it into -45..45 first and it means what you expect
    if angle > 45:
        angle = angle - 90
    elif angle < -45:
        angle = angle + 90

    # anything bigger than this is probably minAreaRect latching onto the border, dont trust it
    if abs(angle) > 10:
        return img

    h, w = img.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    # fill the corners with white, NOT replicate. replicate smears the black border line of the
    # label into big black wedges and tesseract then returns basically nothing for the whole image.
    # cost me a while to find because the cleaned picture still looks fine to a human
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def clean(img, debug=False):
    steps = []

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append(("1_gray", img))

    # median first because the salt and pepper dots survive gaussian blur and then the denoiser
    # just smears them into grey blobs that look like strokes
    img = cv2.medianBlur(img, 3)
    steps.append(("2_median", img))

    # searchWindowSize 21 is the usual default and it is slow, about a second for ten labels on
    # its own. dropping it to 9 made the whole thing about a third faster AND scored slightly
    # better, which i did not expect. a big search window averages over half the label and starts
    # softening the digits
    img = cv2.fastNlMeansDenoising(img, None, h=11, templateWindowSize=7, searchWindowSize=9)
    steps.append(("3_denoise", img))

    img = deskew(img)
    steps.append(("4_deskew", img))

    # i had an adaptive threshold and a morphology open here and both of them made it WORSE.
    # thresholding myself cost about 8 points, tesseract does its own binarisation internally
    # and it is better at it than my fixed block size was. so the denoised greyscale goes
    # straight through now
    # 3x scores about a point and a half higher but pushes a batch of ten past four seconds,
    # and tesseract time goes up with the pixel count. 2x keeps it comfortably inside
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    steps.append(("5_upscale", img))

    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        for name, s in steps:
            cv2.imwrite(os.path.join(DEBUG_DIR, name + ".png"), s)
        print("wrote", len(steps), "debug images to", DEBUG_DIR)

    return img


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/labels/label_000.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("cant read", path)
        sys.exit(1)
    out = clean(img, debug=True)
    cv2.imwrite("output/debug/0_original.png", img)
    print("original", img.shape, "-> cleaned", out.shape)
