"""Command-line parsing into RLConfig."""

import sys
import unittest
from unittest import mock

from crypto_rl.cli import build_parser


def _parse(*argv: str):
    with mock.patch.object(sys, "argv", ["main.py", *argv]):
        return build_parser()


class FinalEvalSplitTest(unittest.TestCase):
    def test_defaults_to_test(self):
        self.assertEqual(_parse().final_eval_split, "test")

    def test_val_for_diagnostic_runs(self):
        config = _parse("--final-eval-split", "val", "--fee-rate", "0")
        self.assertEqual(config.final_eval_split, "val")
        self.assertEqual(config.fee_rate, 0.0)

    def test_rejects_other_splits(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            _parse("--final-eval-split", "train")


if __name__ == "__main__":
    unittest.main()
