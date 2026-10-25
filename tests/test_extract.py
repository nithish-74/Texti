import unittest

from texti.extract import extract_file, is_supported
from texti.extract.ocr import OcrError, Segment, check_languages, installed_languages, ocr_image
from texti.extract.pdf import extract_pages
from tests.helpers import Env, text_image


class FakePage:
    def __init__(self, text, scanned_text="OCR TEXT FROM SCAN"):
        self._text, self.rendered = text, False

    def get_text(self):
        return self._text

    def render_image(self):
        self.rendered = True
        return "image"


class PdfPageLogicTests(unittest.TestCase):
    def fake_ocr(self, image, lang, offset):
        return Segment(f"scanned page {offset} text", offset, 0.8, lang)

    def test_text_pages_are_used_directly_with_full_confidence(self):
        page = FakePage("A real text layer with plenty of characters.")
        segs, n = extract_pages([page], ocr_fn=self.fake_ocr)
        self.assertEqual((n, len(segs)), (1, 1))
        self.assertEqual(segs[0].confidence, 1.0)
        self.assertFalse(page.rendered, "OCR must not run when the page has a text layer")

    def test_scanned_pages_go_to_ocr(self):
        segs, n = extract_pages([FakePage("")], ocr_fn=self.fake_ocr)
        self.assertEqual(segs[0].text, "scanned page 1 text")
        self.assertEqual(segs[0].confidence, 0.8)

    def test_mixed_document_keeps_page_numbers(self):
        pages = [FakePage("Page one has a real text layer here."), FakePage("  "),
                 FakePage("Page three also has a real text layer.")]
        segs, n = extract_pages(pages, ocr_fn=self.fake_ocr)
        self.assertEqual(n, 3)
        self.assertEqual([s.offset for s in segs], [1, 2, 3])
        self.assertEqual([s.confidence for s in segs], [1.0, 0.8, 1.0])

    def test_blank_scanned_page_is_dropped_but_counted(self):
        blank = lambda image, lang, offset: Segment("", offset, 0.0, lang)
        segs, n = extract_pages([FakePage("")], ocr_fn=blank)
        self.assertEqual((segs, n), ([], 1))

    def test_tiny_text_layer_counts_as_scanned(self):
        segs, _ = extract_pages([FakePage("12")], ocr_fn=self.fake_ocr)
        self.assertEqual(segs[0].text, "scanned page 1 text")


class OcrTests(unittest.TestCase):
    def setUp(self):
        self.env = Env()

    def tearDown(self):
        self.env.close()

    def test_reads_clean_text_with_high_confidence(self):
        p = self.env.write("a.png", text_image(["Lecture 3: Binary Search Trees", "Insert and delete nodes."]))
        seg = ocr_image(p)
        self.assertIn("Binary Search Trees", seg.text)
        self.assertIn("Insert and delete nodes", seg.text)
        self.assertGreater(seg.confidence, 0.7)
        self.assertIn("\n", seg.text)  # lines are kept

    def test_blank_image_gives_empty_text_and_zero_confidence(self):
        seg = ocr_image(self.env.write("b.png", text_image([""])))
        self.assertEqual((seg.text.strip(), seg.confidence), ("", 0.0))

    def test_missing_language_pack_gives_a_helpful_error(self):
        with self.assertRaises(OcrError) as cm:
            check_languages("eng+zzz")
        self.assertIn("zzz", str(cm.exception))
        self.assertIn("eng", str(cm.exception))  # lists what is installed

    def test_installed_languages_include_english(self):
        self.assertIn("eng", installed_languages())

    def test_extract_file_dispatch(self):
        p = self.env.write("c.png", text_image(["Hello extraction world"]))
        r = extract_file(p)
        self.assertEqual(r.pages, 1)
        self.assertIn("extraction", r.segments[0].text)
        with self.assertRaises(ValueError):
            extract_file(self.env.write("d.docx", b"x"))

    def test_supported_extensions(self):
        for name in ("a.PNG", "b.jpeg", "c.pdf", "d.tiff", "e.webp", "f.wav", "g.mp3", "h.mp4", "i.mkv"):
            self.assertTrue(is_supported(name), name)
        for name in ("a.txt", "b.docx", "c", "d.exe"):
            self.assertFalse(is_supported(name), name)



if __name__ == "__main__":
    unittest.main()
