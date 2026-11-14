from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch
from predictml.demo import examples_dir, export_examples
from predictml.ui.launcher import launch

ROOT=Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_packaged_examples_match_canonical_files(self):
        for source in (ROOT/'examples').rglob('*'):
            if source.is_file():
                copied=examples_dir()/source.relative_to(ROOT/'examples')
                self.assertTrue(copied.is_file(),str(copied))
                self.assertEqual(hashlib.sha256(source.read_bytes()).digest(),hashlib.sha256(copied.read_bytes()).digest())

    def test_export_demo_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination=Path(temporary)/'demo'
            export_examples(destination)
            self.assertTrue((destination/'raw/notes.csv').is_file())
            self.assertTrue((destination/'tabular/housing.xlsx').is_file())
            with self.assertRaisesRegex(ValueError,'empty'):
                export_examples(destination)
