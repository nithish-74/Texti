import unittest

from texti.privacy.classifier import (
    PathRule,
    classify,
    classify_chunk,
    document_base_tier,
    parse_tier,
    rule_tier,
)
from tests import fixtures as fx


class ClassifierTests(unittest.TestCase):
    def test_fake_bank_statement_is_tier3(self):
        self.assertEqual(classify(fx.BANK_STATEMENT_TEXT).tier, 3)

    def test_fake_id_card_is_tier3(self):
        d = classify(fx.ID_CARD_TEXT)
        self.assertEqual(d.tier, 3)
        self.assertTrue(any("aadhaar" in r or "detected" in r for r in d.reasons))

    def test_lecture_slides_are_tier1(self):
        d = classify(fx.LECTURE_TEXT)
        self.assertEqual(d.tier, 1)

    def test_unknown_plain_text_defaults_to_tier2(self):
        self.assertEqual(classify(fx.PRIVATE_NOTE_TEXT).tier, 2)

    def test_single_cue_is_not_enough_for_tier1(self):
        self.assertEqual(classify("Today's lecture was fun.").tier, 2)

    def test_folder_rule_sets_base_tier(self):
        rules = [PathRule("Finance", 3), PathRule("College", 1)]
        self.assertEqual(classify("plain words here", "College/notes.png", rules).tier, 1)
        self.assertEqual(classify("plain words here", "Finance/x.png", rules).tier, 3)

    def test_longest_prefix_wins(self):
        rules = [PathRule("College", 1), PathRule("College/Private", 3)]
        self.assertEqual(rule_tier("College/Private/a.png", rules), 3)
        self.assertEqual(rule_tier("College/Slides/a.png", rules), 1)
        self.assertEqual(rule_tier("collegeX/a.png", rules), None)  # not a path-segment match
        self.assertEqual(rule_tier("college\\private\\a.png", rules), 3)

    def test_later_rule_wins_a_tie(self):
        self.assertEqual(rule_tier("College/a.png", [PathRule("College", 2), PathRule("college", 1)]), 1)

    def test_content_can_only_raise_a_folder_rule(self):
        rules = [PathRule("College", 1)]
        d = classify(fx.ID_CARD_TEXT, "College/scan.png", rules)
        self.assertEqual(d.tier, 3)

    def test_folder_rule_is_not_lowered_by_cues(self):
        rules = [PathRule("Stuff", 2)]
        self.assertEqual(classify(fx.LECTURE_TEXT, "Stuff/a.pdf", rules).tier, 2)

    def test_private_phrase_applies_whole_document(self):
        base, _ = document_base_tier("Patient name: A. Diagnosis: flu", "Notes/a.pdf", [PathRule("Notes", 1)])
        self.assertEqual(base, 3)

    def test_chunk_level_raise_keeps_other_chunks_low(self):
        base, _ = document_base_tier(fx.LECTURE_TEXT)
        self.assertEqual(base, 1)
        self.assertEqual(classify_chunk(fx.LECTURE_TEXT, base).tier, 1)
        self.assertEqual(classify_chunk(f"scan: {fx.AADHAAR}", base).tier, 3)

    def test_phone_in_lecture_does_not_raise_tier(self):
        text = fx.LECTURE_TEXT + f"\nOffice hours: call {fx.PHONE}"
        self.assertEqual(classify(text).tier, 1)

    def test_parse_tier(self):
        self.assertEqual(parse_tier("3"), 3)
        self.assertEqual(parse_tier(2), 2)
        self.assertIsNone(parse_tier("4"))
        self.assertIsNone(parse_tier(""))
        self.assertIsNone(parse_tier(None))


if __name__ == "__main__":
    unittest.main()


class PagePhraseTests(unittest.TestCase):
    def test_identity_wording_makes_only_that_page_private(self):
        base, _ = document_base_tier(fx.LECTURE_TEXT + "\nScanned ID page: Aadhaar card")
        self.assertEqual(base, 1, "an ID word must not make the whole document private")
        self.assertEqual(classify_chunk("Government ID: Aadhaar card, number unreadable", base).tier, 3)
        self.assertEqual(classify_chunk(fx.LECTURE_TEXT, base).tier, 1)

    def test_document_type_phrases_still_apply_to_the_whole_file(self):
        base, _ = document_base_tier("Page 2 of the Bank Statement: nothing numeric here")
        self.assertEqual(base, 3)

    def test_generic_and_synthetic_id_card_wording_escalates_to_tier3(self):
        base, _ = document_base_tier(fx.LECTURE_TEXT)
        self.assertEqual(classify_chunk("DEMO ID CARD\nID Number: TEST-0000-1234\nName: DEMO PERSON", base).tier, 3)
        self.assertEqual(classify_chunk("Student ID card\nCollege of Engineering", base).tier, 3)

