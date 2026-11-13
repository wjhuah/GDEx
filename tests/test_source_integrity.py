import copy
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.integrity import alignment_issue


class SourceIntegrityTests(unittest.TestCase):
    def test_recorded_daniel_translation_mismatch(self):
        rows = json.loads((Path(__file__).parent / 'fixtures/source_alignment_cases.json').read_text())
        self.assertTrue(rows)
        for r in rows:
            c = r['candidate']
            self.assertEqual(alignment_issue(c), 'source_translation_chapter_mismatch')
            corrected = copy.deepcopy(c)
            corrected['supplemental_source']['sha256'] = 'new-source-version'
            self.assertIsNone(alignment_issue(corrected))


if __name__ == '__main__':
    unittest.main()
