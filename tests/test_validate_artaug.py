from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.validate_artaug import (
    sync_approved_to_manifest,
    validate_artaug_images,
)


class ValidateArtAugTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_dir = self.root / "data"
        self.original_dir = self.root / "dataset"
        self.review_path = self.data_dir / "artaug_review.csv"
        for class_name in ("peace", "stop", "thumbs_up"):
            (self.data_dir / "artaug_input" / class_name).mkdir(parents=True)
            (self.data_dir / "artaug_generated" / class_name).mkdir(parents=True)
            (self.original_dir / class_name).mkdir(parents=True)

        self.source = self.data_dir / "artaug_input" / "peace" / "sample.jpeg"
        Image.new("RGB", (20, 20), color=(200, 30, 80)).save(self.source, format="JPEG")
        Image.new("RGB", (20, 20), color=(10, 200, 30)).save(
            self.original_dir / "peace" / "source.jpeg", format="JPEG"
        )
        generated = self.data_dir / "artaug_generated" / "peace" / "sample_artaug.png"
        Image.new("RGB", (24, 24), color=(20, 30, 220)).save(generated, format="PNG")

        with (self.data_dir / "manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "filename",
                    "class",
                    "source",
                    "original_filename",
                    "split",
                    "approval_status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "artaug_input/peace/sample.jpeg",
                    "class": "peace",
                    "source": "original",
                    "original_filename": "sample.jpeg",
                    "split": "train",
                    "approval_status": "approved",
                }
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_image_is_pending_and_not_added_to_manifest(self) -> None:
        rows = validate_artaug_images(self.data_dir, self.original_dir, self.review_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["technical_status"], "passed")
        self.assertEqual(rows[0]["manual_review_status"], "pending")
        self.assertEqual(rows[0]["accepted"], "")
        self.assertEqual(sync_approved_to_manifest(self.data_dir, rows), 0)

        with (self.data_dir / "manifest.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            self.assertEqual(len(list(csv.DictReader(file))), 1)

    def test_exact_original_duplicate_fails_technical_check(self) -> None:
        output = self.data_dir / "artaug_generated" / "peace" / "source_artaug.png"
        Image.new("RGB", (20, 20), color=(10, 200, 30)).save(output, format="PNG")
        with (self.data_dir / "manifest.csv").open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "artaug_input/peace/source.jpeg",
                    "peace",
                    "original",
                    "source.jpeg",
                    "train",
                    "approved",
                ]
            )

        rows = validate_artaug_images(self.data_dir, self.original_dir, self.review_path)
        duplicate_row = next(row for row in rows if row["filename"].endswith("source_artaug.png"))
        self.assertEqual(duplicate_row["technical_status"], "failed")
        self.assertIn("exact image duplicate exists in the original dataset", duplicate_row["notes"])
        self.assertEqual(sync_approved_to_manifest(self.data_dir, rows), 0)

    def test_manifest_sync_requires_explicit_human_approval(self) -> None:
        rows = validate_artaug_images(self.data_dir, self.original_dir, self.review_path)
        rows[0]["manual_review_status"] = "approved"
        rows[0]["accepted"] = "true"

        self.assertEqual(sync_approved_to_manifest(self.data_dir, rows), 1)
        with (self.data_dir / "manifest.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            manifest_rows = list(csv.DictReader(file))
        generated_row = manifest_rows[-1]
        self.assertEqual(generated_row["source"].lower(), "artaug")
        self.assertEqual(generated_row["split"], "train")
        self.assertEqual(generated_row["approval_status"], "approved")
        self.assertEqual(generated_row["original_filename"], "sample.jpeg")


if __name__ == "__main__":
    unittest.main()
