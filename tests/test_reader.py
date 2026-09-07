import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_reader import MASKS, fix_by_mask, grab, parse_gs1

# these are all real lines i pulled out of tesseract while testing, not made up ones
GOOD = ("DADDY\nPRD:3165433724019\nLOT:L2026-6851\n"
        "BBE:2027-03-05\nQTY:85\nGOODS IN - CRATE LABEL")


def test_reads_a_clean_crate_label():
    assert grab("PRD", GOOD) == "3165433724019"
    assert grab("LOT", GOOD) == "L2026-6851"
    assert grab("BBE", GOOD) == "2027-03-05"
    assert grab("QTY", GOOD) == "85"


def test_dollar_sign_instead_of_S():
    assert grab("LOT", "LOT: L2026-6851") == "L2026-6851"


def test_misread_tag_and_no_colon():
    assert grab("QTY", "PRD 3165433724019\nOTY 149") == "149"


def test_tag_missing_completely():
    assert grab("QTY", "PRD:3165433724019\nLOT:L2026-6851\n77") == "77"


def test_the_product_code_is_not_mistaken_for_the_quantity():
    # this one actually happened. the fallback took the digits after any colon, and the product
    # code is thirteen digits after a colon, so a crate of 85 got booked in as 3165433724019
    assert grab("QTY", "PRD:3165433724019\nLOT:L2026-6851") is None


def test_a_long_bare_number_is_not_a_quantity_either():
    assert grab("QTY", "PRD:3165433724019\n3165433724019") is None


def test_missing_field_gives_nothing():
    assert grab("BBE", "PRD:3165433724019\nLOT:L2026-6851") is None


def test_mask_repairs_letters_and_digits_in_the_right_places():
    assert fix_by_mask("31654337240I9", MASKS["code"]) == "3165433724019"
    assert fix_by_mask("L2O26-685I", MASKS["lot_no"]) == "L2026-6851"
    assert fix_by_mask("2O27-O3-O5", MASKS["bbe"]) == "2027-03-05"


def test_mask_leaves_wrong_length_alone():
    # a dropped digit means we cannot tell which slot is which any more, so do not guess
    assert fix_by_mask("316543372401", MASKS["code"]) == "316543372401"


def test_gs1_batch_barcode_unpacks():
    # 17 is best before as YYMMDD, 10 is the batch. 17 is fixed length so it goes first and 10
    # is variable so it goes last, which is why no separator is needed between them
    lot, bbe = parse_gs1("172703051020266851")
    assert bbe == "2027-03-05"
    assert lot == "L2026-6851"


def test_gs1_rubbish_is_ignored_rather_than_guessed():
    assert parse_gs1("") == ("", "")
    assert parse_gs1("not digits") == ("", "")
    assert parse_gs1("9927030510202668") == ("", "")   # wrong application identifiers
