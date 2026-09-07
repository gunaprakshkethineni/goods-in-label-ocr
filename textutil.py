# levenshtein distance. i wrote it out rather than pip installing a package for 12 lines.
# i use it in check_accuracy.py to score the ocr, and in validate.py to match a misread code
# against the catalogue.

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]
