import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validate import check_row, chars_off

APPROVED = {"L2024-0917", "L2023-3944"}


def good_row():
    return {"filename": "x.png", "part_no": "4471-B2", "serial": "SN8830471",
            "lot_no": "L2024-0917", "qty": "25", "barcode": "SN8830471"}


def flags_for(r):
    return check_row(r, APPROVED)[1]


def test_clean_row_has_no_flags():
    assert flags_for(good_row()) == []


def test_missing_barcode_is_flagged():
    r = good_row()
    r["barcode"] = ""
    assert "BARCODE_UNREADABLE" in flags_for(r)


def test_barcode_not_matching_serial():
    r = good_row()
    r["barcode"] = "SN0000001"
    assert "SERIAL_MISMATCH" in flags_for(r)


def test_barcode_wins_over_a_misread_serial():
    # ocr got the serial wrong but the barcode decoded, so it should be repaired and not flagged
    r = good_row()
    r["serial"] = "5N883O471"
    fixed, flags = check_row(r, APPROVED)
    assert fixed["serial"] == "SN8830471"
    assert flags == []


def test_lot_not_on_approved_list():
    r = good_row()
    r["lot_no"] = "L2025-1234"
    assert "LOT_MISMATCH" in flags_for(r)


def test_lot_one_character_off_is_snapped():
    r = good_row()
    r["lot_no"] = "L2024-0917".replace("9", "8")   # L2024-0817, one digit out
    fixed, flags = check_row(r, APPROVED)
    assert fixed["lot_no"] == "L2024-0917"
    assert flags == []


def test_lot_with_broken_format():
    r = good_row()
    r["lot_no"] = "LZOZ4-O9I7"
    flags = flags_for(r)
    assert "LOT_FORMAT_BAD" in flags
    # a badly formatted lot shouldnt also get reported as a mismatch, thats the same problem twice
    assert "LOT_MISMATCH" not in flags


def test_missing_serial_with_no_barcode():
    r = good_row()
    r["serial"] = ""
    r["barcode"] = ""
    assert "FIELD_MISSING:serial" in flags_for(r)


def test_qty_out_of_range():
    r = good_row()
    r["qty"] = "9999"
    assert "QTY_OUT_OF_RANGE" in flags_for(r)


def test_chars_off():
    assert chars_off("L2024-0917", "L2024-0918") == 1
    assert chars_off("L2024-0917", "L2024-0917") == 0
    assert chars_off("L2024-0917", "L2024-0928") == 2
    assert chars_off("L2024-0917", "L2024-091") == 1     # dropped a digit
    assert chars_off("L2024-0917", "L2024-09177") == 1   # gained one


def test_lot_with_a_dropped_digit_is_snapped():
    r = good_row()
    r["lot_no"] = "L2024-091"
    fixed, flags = check_row(r, APPROVED)
    assert fixed["lot_no"] == "L2024-0917"
    assert flags == []


def test_lot_close_to_two_approved_lots_is_not_snapped():
    # cannot tell which one it was meant to be, so a human gets it
    approved = {"L2024-0917", "L2024-0918"}
    r = good_row()
    r["lot_no"] = "L2024-0919"
    fixed, flags = check_row(r, approved)
    assert fixed["lot_no"] == "L2024-0919"
    assert "LOT_MISMATCH" in flags
