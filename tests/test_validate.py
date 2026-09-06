import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validate import check_crate, snap

# a small stand-in for data/material_master.csv and friends
RULES = {
    "lots": {
        "L2024-0917": {"material": "RES-EP-2400", "supplier": "VESTRA POLYMERS",
                       "released": True, "expiry": "2099-01-01"},
        "L2024-0918": {"material": "HRD-AM-1150", "supplier": "AXIOM RESINS",
                       "released": True, "expiry": "2099-01-01"},
        "L2024-0919": {"material": "HRD-AM-1180", "supplier": "AXIOM RESINS",
                       "released": True, "expiry": "2099-01-01"},
        "L2023-1111": {"material": "RES-EP-2400", "supplier": "VESTRA POLYMERS",
                       "released": True, "expiry": "2020-01-01"},
        "L2023-2222": {"material": "RES-EP-2400", "supplier": "VESTRA POLYMERS",
                       "released": False, "expiry": "2099-01-01"},
        "L2024-7777": {"material": "FAB-GF-1200", "supplier": "CARBOLINE FIBRES",
                       "released": True, "expiry": "2099-01-01"},
    },
    "spec": {"BLADE-402": {"RES-EP-2400", "RES-EP-2600", "HRD-AM-1150", "HRD-AM-1180",
                           "FAB-CF-0600", "ADH-MA-0320", "FST-ST-0880"}},
    "compat": {"RES-EP-2400": "HRD-AM-1150", "RES-EP-2600": "HRD-AM-1180"},
    "materials": ["ADH-MA-0320", "FAB-CF-0600", "FAB-GF-1200", "FST-ST-0880",
                  "HRD-AM-1150", "HRD-AM-1180", "RES-EP-2400", "RES-EP-2600"],
}


def crate(**kw):
    row = {"filename": "x.png", "material": "RES-EP-2400", "lot_no": "L2024-0917",
           "expiry": "2099-01-01", "qty": "200", "barcode": "L2024-0917"}
    row.update(kw)
    return row


def flags_for(row, issued=()):
    return check_crate(row, RULES, "BLADE-402", issued)[1]


def test_a_good_crate_goes_straight_through():
    assert flags_for(crate()) == []


def test_unreadable_barcode_is_held():
    assert "BARCODE_UNREADABLE" in flags_for(crate(barcode=""))


def test_barcode_overrules_a_misread_lot():
    # the printed lot came out wrong but the barcode decoded, so it is repaired, not held
    row, flags = check_crate(crate(lot_no="L2O24-O917"), RULES, "BLADE-402")
    assert row["lot_no"] == "L2024-0917"
    assert flags == []


def test_printed_lot_disagreeing_with_the_barcode_is_reported():
    assert "LOT_MISMATCH_ON_LABEL" in flags_for(crate(lot_no="L2024-0918"))


def test_lot_nobody_has_booked_in():
    assert "LOT_NOT_IN_MASTER" in flags_for(crate(lot_no="L2024-5555", barcode="L2024-5555"))


def test_lot_quality_have_not_released():
    assert "LOT_NOT_RELEASED" in flags_for(
        crate(lot_no="L2023-2222", barcode="L2023-2222"))


def test_out_of_date_lot():
    assert "EXPIRED" in flags_for(crate(lot_no="L2023-1111", barcode="L2023-1111"))


def test_material_not_called_for_on_this_blade():
    # glass fabric turning up for a carbon blade
    assert "NOT_ON_BUILD_SPEC" in flags_for(
        crate(material="FAB-GF-1200", lot_no="L2024-7777", barcode="L2024-7777"))


def test_the_label_says_one_thing_and_the_master_says_another():
    # that lot is a hardener, whatever the drum is labelled
    flags = flags_for(crate(material="RES-EP-2400", lot_no="L2024-0918",
                            barcode="L2024-0918"))
    assert "LABEL_MATERIAL_MISMATCH" in flags


def test_hardener_from_the_wrong_resin_system_is_caught():
    # system A resin is already out on this build, so the system B hardener must not go near it.
    # nothing is wrong with either drum on its own, only with the pair
    issued = [{"material": "RES-EP-2400", "lot_no": "L2024-0917"}]
    flags = flags_for(crate(material="HRD-AM-1180", lot_no="L2024-0919",
                            barcode="L2024-0919"), issued)
    assert any(f.startswith("INCOMPATIBLE_WITH_ISSUED") for f in flags)


def test_the_matching_hardener_is_fine():
    issued = [{"material": "RES-EP-2400", "lot_no": "L2024-0917"}]
    assert flags_for(crate(material="HRD-AM-1150", lot_no="L2024-0918",
                           barcode="L2024-0918"), issued) == []


def test_a_hardener_on_its_own_is_fine():
    # with no resin issued yet there is nothing to be incompatible with
    assert flags_for(crate(material="HRD-AM-1180", lot_no="L2024-0919",
                           barcode="L2024-0919")) == []


def test_silly_quantity():
    assert "QTY_OUT_OF_RANGE" in flags_for(crate(qty="9999"))


def test_missing_quantity():
    assert "FIELD_MISSING:qty" in flags_for(crate(qty=""))


def test_snap_repairs_a_dropped_digit():
    # HRD-AM-115 for HRD-AM-1150 is the single most common thing the ocr does to a material code
    assert snap("HRD-AM-115", RULES["materials"]) == "HRD-AM-1150"


def test_snap_refuses_when_two_are_equally_close():
    # cannot tell which was meant, so leave it alone and let the row get held
    assert snap("HRD-AM-11X0", ["HRD-AM-1150", "HRD-AM-1180"]) == "HRD-AM-11X0"


def test_snap_leaves_something_far_off_alone():
    assert snap("ZZZ-ZZ-9999", RULES["materials"]) == "ZZZ-ZZ-9999"
