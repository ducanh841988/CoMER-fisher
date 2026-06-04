"""Tests for LG -> SRT -> LaTeX preprocessing pipeline."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.lg_srt import lg_to_srt, parse_lg_text
from preprocessing.lg_to_latex import convert_lg_file, latex_from_lg, latex_from_srt
from preprocessing.srt_to_latex import srt_to_latex

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestLgSrt(unittest.TestCase):
    def test_parse_fraction_tree(self):
        text = (FIXTURES / "form_001_E1.lg").read_text(encoding="utf-8")
        tree = parse_lg_text(text)

        self.assertEqual(tree.header, "IUD, form_001_Eq1")
        self.assertEqual(tree.roots, ["_1"])
        self.assertEqual(len(tree.nodes), 6)
        self.assertEqual(tree.nodes["_1"].label, "-")
        self.assertEqual(tree.nodes["_1"].children["Above"], ["3_1"])
        self.assertEqual(tree.nodes["_1"].children["Below"], ["8_1"])
        self.assertEqual(tree.nodes["z_1"].children["Sup"], ["2_1"])

    def test_normalize_relation_aliases(self):
        text = """
O, a_1, a, 1.0, O
O, b_1, b, 1.0, OR
R, a_1, b_1, R, 1.0
"""
        tree = parse_lg_text(text)
        self.assertIn("Right", tree.nodes["a_1"].children)
        self.assertEqual(tree.nodes["a_1"].children["Right"], ["b_1"])


class TestSrtToLatex(unittest.TestCase):
    def test_fraction_with_superscript(self):
        tree = lg_to_srt(FIXTURES / "form_001_E1.lg")
        latex = srt_to_latex(tree)
        self.assertEqual(latex, r"\frac { 3 + z ^ { 2 } } { 8 }")

    def test_sqrt_with_subscript_and_superscript(self):
        tree = lg_to_srt(FIXTURES / "form_001_E2.lg")
        latex = srt_to_latex(tree)
        self.assertEqual(latex, r"y = \sqrt { y _ { i } y ^ { i } }")

    def test_subscript_only(self):
        tree = lg_to_srt(FIXTURES / "subscript.lg")
        latex = srt_to_latex(tree)
        self.assertEqual(latex, r"B _ { f }")

    def test_commat_label_is_expanded(self):
        text = """
O, a_1, P, 1.0, O
O, b_1, VCOMMAT, 1.0, OR
R, a_1, b_1, Right, 1.0
"""
        latex = srt_to_latex(parse_lg_text(text))
        self.assertEqual(latex, "P V,")

    def test_float_label_split_into_digits(self):
        text = """
O, n_1, 918.082, 1.0, O
"""
        latex = srt_to_latex(parse_lg_text(text))
        self.assertEqual(latex, "9 1 8 . 0 8 2")

    def test_integer_label_split_into_digits(self):
        text = """
O, n_1, 919, 1.0, O
"""
        latex = srt_to_latex(parse_lg_text(text))
        self.assertEqual(latex, "9 1 9")


class TestLgToLatex(unittest.TestCase):
    def test_latex_from_lg_fixture(self):
        latex, err = latex_from_lg(FIXTURES / "form_001_E1.lg")
        self.assertEqual(err, "")
        self.assertIsNotNone(latex)
        self.assertEqual(latex, r"\frac { 3 + z ^ { 2 } } { 8 }")

    def test_latex_from_srt_matches_end_to_end(self):
        tree = lg_to_srt(FIXTURES / "form_001_E2.lg")
        self.assertEqual(
            latex_from_srt(tree),
            srt_to_latex(tree),
        )

    def test_convert_lg_file_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.tex"
            ok, latex = convert_lg_file(FIXTURES / "subscript.lg", out)
            self.assertTrue(ok)
            self.assertEqual(latex, r"B _ { f }")
            self.assertEqual(out.read_text(encoding="utf-8").strip(), r"B _ { f }")

    def test_missing_file_returns_error(self):
        latex, err = latex_from_lg(FIXTURES / "does_not_exist.lg")
        self.assertIsNone(latex)
        self.assertNotEqual(err, "")


if __name__ == "__main__":
    unittest.main()
