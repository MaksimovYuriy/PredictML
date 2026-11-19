from pathlib import Path
import tempfile
import unittest
from predictml.demo import export_examples


class ReleaseTests(unittest.TestCase):
    def test_export_demo_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination=Path(temporary)/'demo'
            export_examples(destination)
            self.assertTrue((destination/'raw/notes.csv').is_file())
            self.assertTrue((destination/'tabular/housing.xlsx').is_file())
            with self.assertRaisesRegex(ValueError,'empty'):
                export_examples(destination)
