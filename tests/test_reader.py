import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_reader import grab, fix_by_mask

# these are all real lines i pulled out of tesseract while testing, not made up ones
GOOD = "KRAMER TOOLING\nPN:6140-L6\nSN:SN9332820\nLOT:L2024-0792\nQTY:297\nMADE IN INDIA"


def test_reads_a_clean_label():
    assert grab("PN", GOOD) == "6140-L6"
    assert grab("SN", GOOD) == "SN9332820"
    assert grab("LOT", GOOD) == "L2024-0792"
    assert grab("QTY", GOOD) == "297"


def test_dollar_sign_instead_of_S():
    assert grab("SN", "SN: $N9332820") == "SN9332820"


def test_misread_tag_and_no_colon():
    # this line came out of label_033
    assert grab("QTY", "PN 2630-24\nOTY 149") == "149"


def test_tag_missing_completely():
    # label_008 lost the QTY tag and left the number on its own
    assert grab("QTY", "PN:1417-C4\nSN:SN7312081\nLOT:L2025-3518\n77") == "77"


def test_part_number_does_not_steal_the_serial_line():
    # PN and SN are one character apart so this is the one that would quietly ruin the data
    assert grab("PN", "SN: SN1740991\nLOT:L2024-2472") is None


def test_missing_field_gives_nothing():
    assert grab("QTY", "PN:6140-L6\nSN:SN9332820") is None


def test_mask_repairs_letters_and_digits_in_the_right_places():
    assert fix_by_mask("5N883O471", "aaddddddd") == "SN8830471"
    assert fix_by_mask("SN8830471", "aaddddddd") == "SN8830471"


def test_mask_leaves_wrong_length_alone():
    # if the length is off we cannot tell which slot is which, so dont guess
    assert fix_by_mask("SN88304719", "aaddddddd") == "SN88304719"
