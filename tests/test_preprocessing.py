from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from src.preprocessing import (
    CLASSES,
    IMAGE_SIZE,
    load_dataset,
    load_rgb_array,
    open_oriented_rgb_image,
)


class LoadRgbArrayTests(unittest.TestCase):
    def test_decodes_rgb_and_resizes(self) -> None:
        image = Image.new("L", (20, 10), color=128)
        result = load_rgb_array(image)

        self.assertEqual(result.shape, (IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.allclose(result, 128))

    def test_decodes_image_bytes(self) -> None:
        buffer = BytesIO()
        Image.new("RGB", (8, 12), color=(10, 20, 30)).save(buffer, format="PNG")

        result = load_rgb_array(buffer.getvalue())

        self.assertEqual(result.shape, (IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertTrue(np.allclose(result[0, 0], (10, 20, 30), atol=1))

    def test_rejects_invalid_image_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a supported or valid image"):
            load_rgb_array(b"not an image")

    def test_training_and_app_preprocessing_apply_exif_orientation_consistently(
        self,
    ) -> None:
        image = Image.new("RGB", (40, 20))
        for x in range(40):
            for y in range(20):
                image.putpixel((x, y), (255, 0, 0) if x < 20 else (0, 0, 255))
        exif = image.getexif()
        exif[274] = 6

        with TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            for class_name in CLASSES:
                class_dir = dataset_dir / class_name
                class_dir.mkdir()
                image.save(class_dir / "oriented.jpg", format="JPEG", exif=exif)

            training_images, labels, _ = load_dataset(dataset_dir)
            for index, class_name in enumerate(CLASSES):
                image_path = dataset_dir / class_name / "oriented.jpg"
                with image_path.open("rb") as file:
                    uploaded_bytes = file.read()

                with Image.open(image_path) as raw_image:
                    app_image = open_oriented_rgb_image(raw_image)
                expected_image = load_rgb_array(app_image)

                self.assertEqual(app_image.mode, "RGB")
                self.assertEqual(app_image.size, (20, 40))
                self.assertNotIn(274, app_image.getexif())
                self.assertTrue(np.array_equal(training_images[index], expected_image))
                self.assertTrue(np.array_equal(load_rgb_array(image_path), expected_image))
                self.assertTrue(np.array_equal(load_rgb_array(uploaded_bytes), expected_image))
                self.assertEqual(int(labels[index]), index)

            self.assertEqual(training_images.shape, (len(CLASSES), IMAGE_SIZE, IMAGE_SIZE, 3))


if __name__ == "__main__":
    unittest.main()
