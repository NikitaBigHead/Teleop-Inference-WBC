"""Storage tests for the discarded episode review tool."""

import json
from pathlib import Path
import tempfile
import unittest

from annotate_discarded import Dataset, read_discarded


class DiscardedDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "meta").mkdir()
        self.info_path = self.root / "meta/info.json"
        self.original = b'{\n    "total_episodes": 2,\n    "other": {"keep": true}\n}\n'
        self.info_path.write_bytes(self.original)
        (self.root / "meta/episodes.jsonl").write_text(
            '{"episode_index":2,"tasks":["fist_bump"]}\n'
            '{"episode_index":17,"tasks":["handshake"]}\n', encoding="utf-8"
        )
        for index, chunk in ((2, 0), (17, 1)):
            folder = self.root / f"videos/chunk-{chunk:03d}/observation.images.external_view_camera"
            folder.mkdir(parents=True)
            (folder / f"episode_{index:06d}.mp4").touch()

    def test_mark_reload_unmark_and_preserve_other_metadata(self):
        dataset = Dataset(self.root)
        self.assertEqual(dataset.episodes, [2, 17])
        self.assertEqual(dataset.discarded, set())
        self.assertEqual(self.info_path.read_bytes(), self.original)
        dataset.set_discarded(17, True)
        dataset.set_discarded(2, True)
        self.assertEqual(Dataset(self.root).discarded, {2, 17})
        self.assertEqual(json.loads(self.info_path.read_text())["discarded_episode_indices"], [2, 17])
        dataset.set_discarded(17, False)
        saved = json.loads(self.info_path.read_text())
        self.assertEqual(saved["discarded_episode_indices"], [2])
        self.assertEqual(saved["other"], {"keep": True})
        self.assertEqual(self.info_path.with_name("info.json.bak").read_bytes(), self.original)

    def test_second_window_merges_latest_marks(self):
        first, second = Dataset(self.root), Dataset(self.root)
        first.set_discarded(2, True)
        second.set_discarded(17, True)
        self.assertEqual(Dataset(self.root).discarded, {2, 17})

    def test_invalid_existing_indices_are_rejected_without_write(self):
        for bad in (True, "2", 99):
            with self.subTest(bad=bad):
                self.info_path.write_text(json.dumps({"discarded_episode_indices": [bad]}))
                before = self.info_path.read_bytes()
                with self.assertRaises(ValueError):
                    Dataset(self.root)
                self.assertEqual(self.info_path.read_bytes(), before)

    def test_unknown_episode_is_rejected(self):
        dataset = Dataset(self.root)
        with self.assertRaisesRegex(ValueError, "Unknown episode"):
            dataset.set_discarded(99, True)
        self.assertEqual(self.info_path.read_bytes(), self.original)

    def test_field_must_be_array(self):
        with self.assertRaisesRegex(ValueError, "JSON array"):
            read_discarded({"discarded_episode_indices": "2"}, {2})


if __name__ == "__main__":
    unittest.main()
