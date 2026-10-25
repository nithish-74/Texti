import unittest

from texti.privacy.patterns import (
    detect,
    luhn_valid,
    required_tier,
    verhoeff_check_digit,
    verhoeff_valid,
)
from texti.privacy.redact import redact
from tests import fixtures as fx


def kinds(text):
    return [f.kind for f in detect(text)]


class ChecksumTests(unittest.TestCase):
    def test_verhoeff_known_vectors(self):
        # Standard Verhoeff test vectors.
        self.assertTrue(verhoeff_valid("2363"))
        self.assertTrue(verhoeff_valid("12345" + verhoeff_check_digit("12345")))
        self.assertFalse(verhoeff_valid("2364"))

    def test_verhoeff_detects_every_single_digit_error(self):
        good = fx.AADHAAR
        self.assertTrue(verhoeff_valid(good))
        for i in range(len(good)):
            for d in "0123456789":
                if d != good[i]:
                    self.assertFalse(verhoeff_valid(good[:i] + d + good[i + 1 :]), (i, d))

    def test_luhn(self):
        self.assertTrue(luhn_valid("4111111111111111"))
        self.assertFalse(luhn_valid("4111111111111112"))
        self.assertTrue(luhn_valid(fx.CARD))


class DetectorTests(unittest.TestCase):
    def test_valid_aadhaar_plain_and_spaced(self):
        self.assertEqual(kinds(f"id {fx.AADHAAR}"), ["aadhaar"])
        self.assertEqual(kinds(f"id {fx.AADHAAR_SPACED}"), ["aadhaar"])
        self.assertEqual(kinds(f"id {fx.AADHAAR[:4]}-{fx.AADHAAR[4:8]}-{fx.AADHAAR[8:]}"), ["aadhaar"])

    def test_invalid_aadhaar_checksum_ignored(self):
        bad = fx.AADHAAR[:-1] + str((int(fx.AADHAAR[-1]) + 1) % 10)
        self.assertNotIn("aadhaar", kinds(f"number {bad}"))

    def test_aadhaar_cannot_start_with_0_or_1(self):
        for lead in "01":
            number = lead + fx.AADHAAR[1:]
            self.assertNotIn("aadhaar", kinds(number))

    def test_pan_valid_and_invalid(self):
        self.assertEqual(kinds(f"PAN {fx.PAN}"), ["pan"])
        self.assertEqual(kinds("PAN ABCXE1234F"), [])  # 4th letter X is not a holder type
        self.assertEqual(kinds("PAN ABCPE12345"), [])  # last char must be a letter
        self.assertEqual(kinds("PAN abcpe1234f"), [])  # PAN is uppercase

    def test_card_with_luhn(self):
        self.assertEqual(kinds(f"card {fx.CARD_SPACED}"), ["card"])
        self.assertEqual(kinds(f"card {fx.CARD}"), ["card"])
        self.assertEqual(kinds("order id 4111111111111112"), [])

    def test_ifsc_passport_phone_email(self):
        self.assertEqual(kinds(f"branch {fx.IFSC}"), ["ifsc"])
        self.assertEqual(kinds("noise OOOO0OOOOOO noise"), [])  # repetitive OCR garble rejected
        self.assertEqual(kinds("noise CCCC0000000 noise"), [])
        self.assertEqual(kinds(f"passport {fx.PASSPORT}"), ["passport"])
        self.assertEqual(kinds(f"call {fx.PHONE}"), ["phone"])
        self.assertEqual(kinds("call +91 98765 43210"), ["phone"])
        self.assertEqual(kinds(f"mail {fx.EMAIL}"), ["email"])


    def test_bank_account_needs_context(self):
        self.assertEqual(kinds(f"A/c No: {fx.ACCOUNT}"), ["bank_account"])
        self.assertEqual(kinds(f"account number {fx.ACCOUNT}"), ["bank_account"])
        self.assertEqual(kinds(f"serial {fx.ACCOUNT}"), [])

    def test_otp_and_secrets(self):
        self.assertEqual(kinds("Your OTP is 482913. Do not share."), ["otp"])
        self.assertEqual(kinds("password: hunter2!"), ["secret"])
        self.assertEqual(kinds("key AKIAABCDEFGHIJKLMNOP"), ["secret"])

    def test_plain_lecture_text_has_no_findings(self):
        self.assertEqual(detect(fx.LECTURE_TEXT), [])

    def test_overlaps_resolved_to_most_specific(self):
        # A 12-digit Aadhaar must not also be reported as a phone number.
        self.assertEqual(kinds(f"{fx.AADHAAR}"), ["aadhaar"])

    def test_required_tier(self):
        self.assertEqual(required_tier(detect(fx.LECTURE_TEXT)), 1)
        self.assertEqual(required_tier(detect(f"mail {fx.EMAIL}")), 1)
        self.assertEqual(required_tier(detect(f"card {fx.CARD}")), 2)
        self.assertEqual(required_tier(detect(f"id {fx.AADHAAR}")), 3)

    def test_finding_repr_hides_text(self):
        (f,) = detect(f"id {fx.AADHAAR}")
        self.assertNotIn(fx.AADHAAR, repr(f))
        self.assertNotIn(fx.AADHAAR, str([f]))


class RedactionTests(unittest.TestCase):
    def test_aadhaar_keeps_last_four(self):
        out = redact(f"Aadhaar: {fx.AADHAAR_SPACED}", tier=3)
        self.assertIn(f"XXXX XXXX {fx.AADHAAR[-4:]}", out)
        self.assertNotIn(fx.AADHAAR, out)
        self.assertNotIn(fx.AADHAAR_SPACED, out)

    def test_card_keeps_last_four(self):
        out = redact(f"card {fx.CARD_SPACED}", tier=1)  # cards are masked at every tier
        self.assertIn(f"XXXX XXXX XXXX {fx.CARD[-4:]}", out)
        self.assertNotIn(fx.CARD_SPACED, out)

    def test_pan_passport_account_masked_at_every_tier(self):
        text = f"{fx.PAN} {fx.PASSPORT} A/c No: {fx.ACCOUNT}"
        for tier in (1, 2, 3):
            out = redact(text, tier)
            self.assertNotIn(fx.PAN, out)
            self.assertNotIn(fx.PASSPORT, out)
            self.assertNotIn(fx.ACCOUNT, out)
            self.assertIn(fx.ACCOUNT[-4:], out)

    def test_email_phone_masked_only_from_tier_two(self):
        text = f"{fx.EMAIL} {fx.PHONE}"
        self.assertEqual(redact(text, 1), text)
        out = redact(text, 2)
        self.assertNotIn(fx.EMAIL, out)
        self.assertNotIn(fx.PHONE, out)

    def test_otp_and_password_masked(self):
        out = redact("OTP is 482913 and password: hunter2!", tier=1)
        self.assertNotIn("482913", out)
        self.assertNotIn("hunter2", out)

    def test_surrounding_text_is_preserved(self):
        out = redact(f"Name: Asha. Aadhaar {fx.AADHAAR}. End.", tier=3)
        self.assertTrue(out.startswith("Name: Asha. Aadhaar "))
        self.assertTrue(out.endswith(". End."))

    def test_unchanged_when_nothing_found(self):
        self.assertEqual(redact(fx.LECTURE_TEXT, 3), fx.LECTURE_TEXT)


if __name__ == "__main__":
    unittest.main()
