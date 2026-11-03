from __future__ import annotations

import unittest

import numpy as np

from src.explainability import overlay_gradcam


class GradCamOverlayTests(unittest.TestCase):
    def test_resizes_heatmap_and_preserves_input_image_shape(self) -> None:
        image = np.full((40, 60, 3), 180, dtype=np.uint8)
        heatmap = np.zeros((4, 6), dtype=np.float32)
        heatmap[1:3, 2:5] = 1.0

        result = overlay_gradcam(image, heatmap)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.array_equal(result[0, 0], image[0, 0]))
        self.assertFalse(np.array_equal(result[20, 30], image[20, 30]))

    def test_zero_heatmap_returns_original_image_without_tint(self) -> None:
        image = np.full((8, 12, 3), (40, 90, 150), dtype=np.uint8)
        heatmap = np.zeros((2, 3), dtype=np.float32)

        result = overlay_gradcam(image, heatmap)

        self.assertTrue(np.array_equal(result, image))

    def test_rejects_non_rgb_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "RGB image"):
            overlay_gradcam(np.zeros((8, 12), dtype=np.uint8), np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
