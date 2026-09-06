import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_reader import MASKS, fix_by_mask, grab

# these are all real lines i pulled out of tesseract while testing, not made up ones
GOOD = ("VESTRA POLYMERS\nMAT:RES-EP-2400\nLOT:L2025-9549\n"
        "EXP:2028-06-06\nQTY:67\nGOODS IN - BLADE MATERIALS")


def test_reads_a_clean_crate_label():
    assert grab("MAT", GOOD) == "RES-EP-2400"
    assert grab("LOT", GOOD) == "L2025-9549"
    assert grab("EXP", GOOD) == "2028-06-06"
    assert grab("QTY", GOOD) == "67"


def test_dollar_sign_instead_of_S():
    assert grab("LOT", "LOT: L2025-9549") == "L2025-9549"
    assert grab("MAT", "MAT: RE$-EP-2400") == "RES-EP-2400"


def test_misread_tag_and_no_colon():
    # crate_033 came back like this, the Q read as an O and the colon was gone
    assert grab("QTY", "MAT 2630-24\nOTY 149") == "149"


def test_tag_missing_completely():
    # one crate lost the tag and left the number sitting on its own
    assert grab("QTY", "MAT:RES-EP-2400\nLOT:L2025-3518\n77") == "77"


def test_tag_mangled_past_recognising():
    # crate_048 came back as O:151. quantity is the only bare number on the label, and the lot
    # and the date both carry dashes, so a plain number after a colon can only be the quantity
    assert grab("QTY", "LOT:L2025-5573\nEXP:2027-12-28\nO:151") == "151"


def test_material_tag_does_not_steal_another_line():
    assert grab("MAT", "LOT: L2025-9549\nEXP: 2028-06-06") is None


def test_missing_field_gives_nothing():
    assert grab("QTY", "MAT:RES-EP-2400\nLOT:L2025-9549") is None


def test_mask_repairs_letters_and_digits_in_the_right_places():
    assert fix_by_mask("RE5-EP-24O0", MASKS["material"]) == "RES-EP-2400"
    assert fix_by_mask("L2O25-9S49", MASKS["lot_no"]) == "L2025-9549"
    assert fix_by_mask("2O28-O6-O6", MASKS["expiry"]) == "2028-06-06"


def test_mask_leaves_wrong_length_alone():
    # a dropped digit means we cannot tell which slot is which any more, so do not guess.
    # validate.py picks these up instead by matching against the material catalogue
    assert fix_by_mask("HRD-AM-115", MASKS["material"]) == "HRD-AM-115"
