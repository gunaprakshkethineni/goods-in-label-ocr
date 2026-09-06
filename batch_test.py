# times a batch of 10 labels end to end, which is roughly one tray on the line.

import glob
import time

from ocr_reader import read_label
from validate import check_row, load_approved_lots

BATCH = 10


def run_once(files, approved):
    t0 = time.perf_counter()
    for f in files:
        row = read_label(f)
        check_row(row, approved)
    return time.perf_counter() - t0


def main():
    files = sorted(glob.glob("data/labels/*.png"))[:BATCH]
    approved = load_approved_lots()
    print("timing", len(files), "labels\n")

    times = []
    for i in range(3):
        t = run_once(files, approved)
        times.append(t)
        # first pass is always the slowest, tesseract is still loading its language data
        print("run %d: %.2f s  (%.0f ms per label)" % (i + 1, t, 1000 * t / len(files)))

    avg = sum(times) / len(times)
    print()
    print("average: %.2f s for %d labels" % (avg, len(files)))
    print("best:    %.2f s" % min(times))
    print("that is %.0f ms per label" % (1000 * avg / len(files)))


if __name__ == "__main__":
    main()
