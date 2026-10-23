"""Focused tests for dataset loading, deterministic splits, and image prediction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from problem_statement_42_solution import collect_dataset, create_splits
from scripts.predict_image import predict_image


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()

        for class_index, class_name in enumerate(("A", "B", "J", "Z")):
            class_directory = self.dataset / class_name
            class_directory.mkdir()
            for image_index in range(5):
                pixels = np.full(
                    (20, 20, 3),
                    (class_index * 40 + image_index) % 255,
                    dtype=np.uint8,
                )
                tf.keras.utils.save_img(
                    class_directory / f"sample_{image_index}.png", pixels
                )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_dataset_loader_excludes_motion_signs(self) -> None:
        paths, labels, class_names = collect_dataset(self.dataset)

        self.assertEqual(class_names, ["A", "B"])
        self.assertEqual(len(paths), 10)
        self.assertEqual(labels.dtype, np.int32)
        self.assertEqual(np.bincount(labels).tolist(), [5, 5])

    def test_splits_are_reproducible_and_disjoint(self) -> None:
        paths, labels, _ = collect_dataset(self.dataset)
        first = create_splits(paths, labels, seed=17)
        second = create_splits(paths, labels, seed=17)

        for first_split, second_split in zip(
            (first.train, first.validation, first.test),
            (second.train, second.validation, second.test),
        ):
            np.testing.assert_array_equal(first_split.paths, second_split.paths)
            np.testing.assert_array_equal(first_split.labels, second_split.labels)

        path_sets = [
            set(first.train.paths),
            set(first.validation.paths),
            set(first.test.paths),
        ]
        self.assertEqual(len(set.union(*path_sets)), len(paths))
        self.assertTrue(path_sets[0].isdisjoint(path_sets[1]))
        self.assertTrue(path_sets[0].isdisjoint(path_sets[2]))
        self.assertTrue(path_sets[1].isdisjoint(path_sets[2]))

    def test_loader_rejects_missing_dataset(self) -> None:
        with self.assertRaises(FileNotFoundError):
            collect_dataset(self.root / "missing")

    def test_loader_rejects_classes_too_small_for_stratified_splits(self) -> None:
        small_dataset = self.root / "small_dataset"
        for class_name in ("C", "D"):
            class_directory = small_dataset / class_name
            class_directory.mkdir(parents=True)
            for image_index in range(3):
                tf.keras.utils.save_img(
                    class_directory / f"sample_{image_index}.png",
                    np.zeros((20, 20, 3), dtype=np.uint8),
                )

        with self.assertRaisesRegex(ValueError, "at least 5 images"):
            collect_dataset(small_dataset)

    def test_predict_image_returns_named_top_predictions(self) -> None:
        image_path = self.root / "query.png"
        tf.keras.utils.save_img(
            image_path, np.full((24, 24, 3), 128, dtype=np.uint8)
        )

        inputs = tf.keras.Input(shape=(16, 16, 3))
        pooled = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        output = tf.keras.layers.Dense(2, activation="softmax")(pooled)
        model = tf.keras.Model(inputs, output)
        model.layers[-1].set_weights(
            [
                np.zeros((3, 2), dtype=np.float32),
                np.array([0.0, 2.0], dtype=np.float32),
            ]
        )
        model_path = self.root / "model.keras"
        model.save(model_path)

        class_names_path = self.root / "class_names.json"
        class_names_path.write_text(json.dumps(["A", "B"]), encoding="utf-8")

        predictions = predict_image(
            image_path,
            model_path,
            class_names_path,
            image_size=16,
            top_k=2,
        )

        self.assertEqual([name for name, _ in predictions], ["B", "A"])
        self.assertAlmostEqual(sum(score for _, score in predictions), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
