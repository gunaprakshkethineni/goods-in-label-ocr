# levenshtein distance. wrote it out instead of pip installing a package for 12 lines of code.
# check_accuracy.py uses it to score the ocr, validate.py uses it to match a misread lot number
# against the approved list.

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]
