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
    if angle > 45:
        angle = angle - 90

    # anything bigger than this is probably minAreaRect latching onto the border, dont trust it
    if abs(angle) > 10:
        return img

    h, w = img.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def clean(img, debug=False):
    steps = []

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append(("1_gray", img))

    # median first because the salt and pepper dots survive gaussian blur and then the denoiser
    # just smears them into grey blobs that look like strokes
    img = cv2.medianBlur(img, 3)
    steps.append(("2_median", img))

    img = cv2.fastNlMeansDenoising(img, None, h=11, templateWindowSize=7, searchWindowSize=21)
    steps.append(("3_denoise", img))

    img = deskew(img)
    steps.append(("4_deskew", img))

    # adaptive not otsu. otsu kept blowing out the shadowed corner into solid black
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 15)
    steps.append(("5_thresh", img))

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    steps.append(("6_open", img))

    # tesseract is noticeably better on bigger text, this was worth about 3 percent on its own
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    steps.append(("7_upscale", img))

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
