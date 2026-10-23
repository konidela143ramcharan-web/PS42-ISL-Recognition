"""Predict the most likely ISL sign for one image using a saved Keras model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def predict_image(
    image_path: str | Path,
    model_path: str | Path,
    class_names_path: str | Path,
    image_size: int = 128,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    image_file = Path(image_path).expanduser()
    model_file = Path(model_path).expanduser()
    classes_file = Path(class_names_path).expanduser()
    for file_path, description in (
        (image_file, "Image"),
        (model_file, "Model"),
        (classes_file, "Class-name file"),
    ):
        if not file_path.is_file():
            raise FileNotFoundError(f"{description} not found: {file_path}")
    if image_size < 1:
        raise ValueError("image_size must be a positive integer.")
    if top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    with classes_file.open(encoding="utf-8") as file:
        class_names = json.load(file)
    if (
        not isinstance(class_names, list)
        or not class_names
        or not all(isinstance(name, str) and name for name in class_names)
    ):
        raise ValueError(f"Expected a non-empty JSON list of class names: {classes_file}")

    image_data = tf.io.read_file(str(image_file))
    image = tf.io.decode_image(image_data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0

    model = tf.keras.models.load_model(model_file, compile=False)
    probabilities = np.asarray(model.predict(image[tf.newaxis, ...], verbose=0))[0]
    if probabilities.ndim != 1 or len(probabilities) != len(class_names):
        raise ValueError(
            "Model output count does not match the class-name list. "
            "Use the class_names.json generated with this model."
        )

    best_indices = np.argsort(probabilities)[::-1][: min(top_k, len(class_names))]
    return [
        (class_names[int(index)], float(probabilities[index]))
        for index in best_indices
    ]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = project_dir / "outputs"
    parser = argparse.ArgumentParser(
        description="Classify one image with a saved PS-42 ISL model."
    )
    parser.add_argument("image", type=Path, help="Path to the image to classify.")
    parser.add_argument(
        "--model",
        type=Path,
        default=output_dir / "best_model.keras",
        help="Saved Keras model (default: outputs/best_model.keras).",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=output_dir / "class_names.json",
        help="Class mapping saved by the training pipeline.",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    predictions = predict_image(
        args.image,
        args.model,
        args.classes,
        image_size=args.image_size,
        top_k=args.top_k,
    )
    for rank, (class_name, confidence) in enumerate(predictions, start=1):
        print(f"{rank}. {class_name}: {confidence:.2%}")


if __name__ == "__main__":
    main()
