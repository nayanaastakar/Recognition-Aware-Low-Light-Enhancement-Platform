import os
import sys
import unittest
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evaluation.annotation_indexer import AnnotationIndexer
from evaluation.exdark_parser import parse_annotation_content
from evaluation.detailed_evaluator import compute_detailed_detection_metrics

class TestAnnotationIndexer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.indexer = AnnotationIndexer(PROJECT_ROOT)

    def test_indexer_loaded(self):
        self.assertGreater(self.indexer.total_indexed, 7000, "Indexer should contain at least 7000 ExDark annotations.")

    def test_exact_matches(self):
        samples = [
            ("2015_00001.png", "2015_00001.png.txt"),
            ("2015_00005.jpg", "2015_00005.jpg.txt"),
            ("2015_00026.jpg", "2015_00026.jpg.txt"),
            ("2015_02446.jpg", "2015_02446.jpg.txt"),
            ("yellow_bicycle_bench.png", "yellow_bicycle_bench.png.txt")
        ]
        for query_img, expected_txt in samples:
            path, via, warn = self.indexer.lookup(query_img)
            self.assertIsNotNone(path, f"Failed to match {query_img}")
            self.assertEqual(os.path.basename(path).lower(), expected_txt.lower(), f"Mismatch for {query_img}")

    def test_roboflow_and_stem_variants(self):
        rf_name = "2015_00026_jpg.rf.4ed27f2ff81dd07c2a7b271d8bb85ad3.jpg"
        path, via, warn = self.indexer.lookup(rf_name)
        self.assertIsNotNone(path)
        self.assertTrue("2015_00026" in os.path.basename(path))

    def test_missing_annotation_returns_none(self):
        missing_name = "unrelated_random_photo_xyz_9999.png"
        path, via, warn = self.indexer.lookup(missing_name)
        self.assertIsNone(path)
        self.assertIn("No corresponding", warn)

    def test_empty_query(self):
        path, via, warn = self.indexer.lookup("")
        self.assertIsNone(path)

    def test_parsing_from_indexed_path(self):
        path, via, warn = self.indexer.lookup("2015_00005.jpg")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        gts = parse_annotation_content(content, 500, 333, filename=os.path.basename(path))
        self.assertEqual(len(gts), 1)
        self.assertEqual(gts[0]['class'], 'bicycle')
        self.assertEqual(gts[0]['bbox'], [225, 61, 406, 332])

if __name__ == '__main__':
    unittest.main()
