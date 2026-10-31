import os
import unittest
import numpy as np
import pandas as pd
import torch

from core.model import build_model, load_models, get_device
from core.tiling import make_tiles, pad_if_needed, TileInstance
from core.pipeline import run_inference
from core.viz import label_mask, instances_dataframe


class TestKidneyVesselSegmentation(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        os.makedirs("weights", exist_ok=True)
        # Ensure at least one test weight file exists
        test_weight_path = "weights/fold0_best.pth"
        if not os.path.exists(test_weight_path):
            m = build_model()
            torch.save(m.state_dict(), test_weight_path)

    def test_1_weights_load_strict(self):
        """(1) all weights load with strict=True without error."""
        models = load_models("weights", device=self.device)
        self.assertGreater(len(models), 0, "Expected at least one model to be loaded.")
        for name, model in models:
            self.assertIsInstance(model, torch.nn.Module)
            self.assertFalse(model.training, f"Model {name} should be in eval mode.")

    def test_2_synthetic_pink_noise_inference(self):
        """(2) synthetic pink noise 512x512 image runs through Fast mode without exception."""
        models = load_models("weights", device=self.device)
        
        # Pink noise (H&E stain simulation)
        np.random.seed(42)
        pink_base = np.array([225, 190, 215], dtype=np.float32)
        noise = np.random.normal(0, 15, (512, 512, 3)).astype(np.float32)
        pink_noise = np.clip(pink_base + noise, 0, 255).astype(np.uint8)

        result = run_inference(
            image=pink_noise,
            models=models,
            mode="Fast",
            mask_thr=0.4,
            device=self.device,
        )
        self.assertEqual(result.image.shape, (512, 512, 3))
        self.assertEqual(result.mode, "Fast")
        self.assertEqual(result.n_tiles, 1)
        self.assertIsInstance(result.instances, list)
        print(f"Test 2 passed: {len(result.instances)} instances detected in pink noise.")

    def test_3_make_tiles_covers_every_pixel(self):
        """(3) make_tiles covers every pixel for several sizes (including 512, 513, 1300x900, 300x300)."""
        test_sizes = [
            (512, 512),
            (513, 512),
            (512, 513),
            (513, 513),
            (1300, 900),
            (900, 1300),
            (300, 300),
            (250, 700),
        ]
        tile_size = 512
        stride = 384

        for h, w in test_sizes:
            dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
            padded, pad_h, pad_w = pad_if_needed(dummy_img, min_size=tile_size)
            curr_h, curr_w = padded.shape[:2]

            tiles = make_tiles(curr_h, curr_w, tile=tile_size, stride=stride)
            self.assertGreater(len(tiles), 0)

            # Check that every pixel of the padded image is covered by at least one tile
            coverage = np.zeros((curr_h, curr_w), dtype=bool)
            for y0, x0 in tiles:
                self.assertGreaterEqual(y0, 0)
                self.assertGreaterEqual(x0, 0)
                self.assertLessEqual(y0 + tile_size, curr_h)
                self.assertLessEqual(x0 + tile_size, curr_w)
                coverage[y0 : y0 + tile_size, x0 : x0 + tile_size] = True

            # All pixels in padded image must be covered
            self.assertTrue(
                coverage.all(),
                f"Tiling did not cover all pixels for input size ({h}, {w}) padded to ({curr_h}, {curr_w})",
            )
            # And original region is fully covered
            self.assertTrue(coverage[:h, :w].all())
            print(f"Test 3 passed for size ({h}, {w}) -> {len(tiles)} tile(s)")

    def test_4_label_mask_ids_match_dataframe(self):
        """(4) label_mask ids match the dataframe."""
        shape = (200, 200, 3)
        # Create test mock instances
        m1 = np.ones((15, 20), dtype=bool)
        m2 = np.ones((25, 25), dtype=bool)
        m3 = np.ones((10, 10), dtype=bool)

        instances = [
            TileInstance(
                id=1,
                score=0.85,
                area=300,
                bbox=(10, 10, 30, 25),
                centroid=(20.0, 17.5),
                mask_crop=m1,
                x0=10,
                y0=10,
            ),
            TileInstance(
                id=2,
                score=0.45,
                area=625,
                bbox=(40, 40, 65, 65),
                centroid=(52.5, 52.5),
                mask_crop=m2,
                x0=40,
                y0=40,
            ),
            TileInstance(
                id=3,
                score=0.15,
                area=100,
                bbox=(100, 100, 110, 110),
                centroid=(105.0, 105.0),
                mask_crop=m3,
                x0=100,
                y0=100,
            ),
        ]

        score_thr = 0.3
        df = instances_dataframe(instances, score_thr=score_thr)
        lmask = label_mask(shape, instances, score_thr=score_thr)

        df_ids = set(df["id"].tolist())
        self.assertEqual(df_ids, {1, 2})  # ID 3 filtered out by score_thr 0.3

        # Non-zero mask labels must exactly equal df_ids
        mask_unique_ids = set(np.unique(lmask)) - {0}
        self.assertEqual(mask_unique_ids, df_ids)
        print("Test 4 passed: label_mask ids match dataframe IDs exactly.")


if __name__ == "__main__":
    unittest.main()
