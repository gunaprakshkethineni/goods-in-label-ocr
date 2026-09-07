import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validate import check_crate, outcome, really_disagrees, snap

# a small stand-in for the real data/product_master.csv and data/run_spec.csv.
# the codes are real EAN-13s off real products
RULES = {
    "products": {
        "3165433724019": {"product_name": "Sucre en poudre", "brand": "Daddy",
                          "pack_size": "1 kg", "category": "sugars", "allergens": set()},
        "3017624010701": {"product_name": "Nutella", "brand": "Ferrero",
                          "pack_size": "400 g", "category": "chocolates",
                          "allergens": {"milk", "nuts", "soybeans"}},
        "5000159407236": {"product_name": "Oat flour", "brand": "Anon",
                          "pack_size": "1 kg", "category": "flours",
                          "allergens": {"gluten"}},
        "4047247992978": {"product_name": "Whole milk yogurt", "brand": "Aldi",
                          "pack_size": "500 g", "category": "yogurts",
                          "allergens": {"milk"}},
    },
    "runs": {
        "RUN-OAT-COOKIE": {"declares": {"gluten", "milk"},
                           "categories": {"flours", "sugars", "chocolates"}},
    },
    "codes": ["3017624010701", "3165433724019", "4047247992978", "5000159407236"],
}


SSCC = "050123450000010004"


def crate(**kw):
    row = {"filename": "x.png", "sscc": SSCC, "code": "3165433724019",
           "lot_no": "L2025-1234", "bbe": "2099-01-01", "qty": "120",
           "barcode": "3165433724019", "sscc_bc": SSCC,
           "lot_bc": "L2025-1234", "bbe_bc": "2099-01-01"}
    row.update(kw)
    return row


def flags_for(row, issued=()):
    return check_crate(row, RULES, "RUN-OAT-COOKIE", issued)[1]


def test_a_good_crate_goes_straight_through():
    assert flags_for(crate()) == []


def test_unreadable_product_barcode_is_held():
    assert "BARCODE_UNREADABLE" in flags_for(crate(barcode=""))


def test_unreadable_batch_barcode_is_held():
    assert "BATCH_BARCODE_UNREADABLE" in flags_for(crate(lot_bc="", bbe_bc=""))


def test_the_barcodes_overrule_a_misread_label():
    # every printed field came out slightly wrong, both barcodes decoded, so all three get
    # repaired and nobody has to look at the crate
    row, flags = check_crate(crate(code="3165433724O19", lot_no="L2O25-1234",
                                   bbe="2O99-01-01"), RULES, "RUN-OAT-COOKIE")
    assert row["code"] == "3165433724019"
    assert row["lot_no"] == "L2025-1234"
    assert row["bbe"] == "2099-01-01"
    assert flags == []


def test_one_character_out_is_not_a_mismatch():
    # the barcode already corrected it, there is nothing for a person to do
    assert not really_disagrees("3165433724019", "3165433724010")


def test_the_tolerance_scales_with_the_length_of_the_field():
    # two digits out of an eighteen digit serial is the ocr struggling, not a different crate.
    # the same two digits out of a short field would be
    assert not really_disagrees("050123450000010084", "050123450000010004")
    assert really_disagrees("2027-10-29", "2026-01-30")


def test_the_crate_serial_is_read_and_kept():
    row, flags = check_crate(crate(), RULES, "RUN-OAT-COOKIE")
    assert row["sscc"] == SSCC
    assert flags == []


def test_serial_barcode_failing_is_held():
    assert "SERIAL_BARCODE_UNREADABLE" in flags_for(crate(sscc_bc=""))


def test_the_same_crate_booked_in_twice():
    # this is what the serial is for. the batch, the product and the date would all be identical
    # on a second crate off the same delivery, so nothing else on the label could tell them apart
    issued = [{"sscc": SSCC, "code": "3165433724019", "allergens": ""}]
    assert "DUPLICATE_CRATE" in flags_for(crate(), issued)


def test_a_different_crate_off_the_same_batch_is_fine():
    issued = [{"sscc": "050123450000019999", "code": "3165433724019", "allergens": ""}]
    assert flags_for(crate(), issued) == []


def test_a_completely_different_number_is_a_mismatch():
    assert really_disagrees("3165433724019", "3017624010701")
    assert "CODE_MISMATCH_ON_LABEL" in flags_for(crate(code="3017624010701"))


def test_a_code_nobody_has_in_the_catalogue():
    assert "CODE_NOT_IN_CATALOGUE" in flags_for(
        crate(code="9999999999993", barcode="9999999999993"))


def test_out_of_date_stock():
    assert "EXPIRED" in flags_for(crate(bbe="2020-01-01", bbe_bc="2020-01-01"))


def test_an_ingredient_this_run_does_not_use():
    # yogurt turning up for a biscuit run
    flags = flags_for(crate(code="4047247992978", barcode="4047247992978"))
    assert any(f.startswith("NOT_ON_RUN_SPEC") for f in flags)


def test_an_allergen_the_finished_pack_does_not_declare():
    # nutella is a perfectly good product, it just brings nuts and soya into a run whose label
    # only declares gluten and milk. that is an undeclared allergen and a recall waiting to happen
    flags = flags_for(crate(code="3017624010701", barcode="3017624010701"))
    hit = [f for f in flags if f.startswith("UNDECLARED_ALLERGEN")]
    assert hit and "nuts" in hit[0] and "soybeans" in hit[0]


def test_a_declared_allergen_is_fine():
    assert flags_for(crate(code="5000159407236", barcode="5000159407236")) == []


def test_first_of_its_kind_on_the_run_asks_for_a_changeover_check():
    # gluten is declared so it is not a recall risk, but it is the first gluten onto this line
    # and a supervisor should see that rather than it sliding past
    issued = [{"code": "3165433724019", "allergens": ""}]
    flags = flags_for(crate(code="5000159407236", barcode="5000159407236"), issued)
    assert any(f.startswith("ALLERGEN_CHANGEOVER_CHECK") for f in flags)


def test_not_asked_again_once_that_allergen_is_already_on_the_line():
    issued = [{"code": "5000159407236", "allergens": "gluten"}]
    flags = flags_for(crate(code="5000159407236", barcode="5000159407236"), issued)
    assert not any(f.startswith("ALLERGEN_CHANGEOVER_CHECK") for f in flags)


def test_silly_case_count():
    assert "QTY_OUT_OF_RANGE" in flags_for(crate(qty="9999"))


def test_snap_repairs_a_dropped_digit():
    assert snap("316543372401", RULES["codes"]) == "3165433724019"


def test_snap_refuses_when_two_are_equally_close():
    assert snap("1111111111111", ["1111111111112", "1111111111113"]) == "1111111111111"


def test_a_clean_crate_is_accepted():
    assert outcome([]) == "ACCEPTED"


def test_a_reading_we_do_not_trust_is_held():
    assert outcome(["BARCODE_UNREADABLE"]) == "HELD"


def test_a_solid_reading_of_the_wrong_thing_is_rejected():
    assert outcome(["UNDECLARED_ALLERGEN:nuts"]) == "REJECTED"
    assert outcome(["EXPIRED"]) == "REJECTED"


def test_you_cannot_reject_on_a_reading_you_do_not_trust():
    # the code looks unknown, but the barcode did not decode so it came off the ocr and the ocr
    # might simply have got it wrong. that is a person's call, not a rejection
    assert outcome(["BARCODE_UNREADABLE", "CODE_NOT_IN_CATALOGUE"]) == "HELD"
