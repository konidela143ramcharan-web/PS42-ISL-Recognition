"""Training and evaluation tools for the PS-42 ISL image-classification task."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EXCLUDED_MOTION_SIGNS = {"j", "z"}
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "ISLAN"


@dataclass(frozen=True)
class DatasetSplit:
    paths: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class DatasetSplits:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit


def collect_dataset(
    dataset_path: str | Path,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Read images from one subfolder per sign class; ignore motion signs J and Z."""
    root = Path(dataset_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}. Expected one folder per sign class."
        )

    class_dirs = [
        directory
        for directory in sorted(root.iterdir(), key=lambda item: item.name.casefold())
        if directory.is_dir() and directory.name.casefold() not in EXCLUDED_MOTION_SIGNS
    ]
    class_names: list[str] = []
    image_paths: list[str] = []
    labels: list[int] = []

    for directory in class_dirs:
        files = sorted(
            (
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=lambda item: str(item).casefold(),
        )
        if not files:
            continue
        class_id = len(class_names)
        class_names.append(directory.name)
        image_paths.extend(str(path) for path in files)
        labels.extend([class_id] * len(files))

    if len(class_names) < 2:
        raise ValueError(
            f"Expected image files in at least two class folders under {root}."
        )

    counts = Counter(labels)
    too_small = [
        class_names[class_id]
        for class_id, count in counts.items()
        if count < 5
    ]
    if too_small:
        raise ValueError(
            "Each class needs at least 5 images for stratified 60/20/20 "
            f"train/validation/test splits. Too few images in: {', '.join(too_small)}."
        )

    return image_paths, np.asarray(labels, dtype=np.int32), class_names


def create_splits(
    image_paths: list[str],
    labels: np.ndarray,
    seed: int = 42,
) -> DatasetSplits:
    """Create deterministic, stratified 60/20/20 train/validation/test splits."""
    paths = np.asarray(image_paths)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=0.25,
        random_state=seed,
        stratify=train_val_labels,
    )

    return DatasetSplits(
        train=DatasetSplit(train_paths, train_labels),
        validation=DatasetSplit(validation_paths, validation_labels),
        test=DatasetSplit(test_paths, test_labels),
    )


def save_dataset_overview(
    image_paths: list[str],
    labels: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Save class counts and a representative image from each class."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(labels, minlength=len(class_names))

    with (output_dir / "class_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["class", "image_count"])
        writer.writerows(zip(class_names, counts.tolist()))

    figure, axis = plt.subplots(figsize=(max(8, len(class_names) * 0.45), 4))
    axis.bar(class_names, counts)
    axis.set_title("Images per ISL sign class")
    axis.set_xlabel("Sign class")
    axis.set_ylabel("Image count")
    axis.tick_params(axis="x", labelrotation=90)
    figure.tight_layout()
    figure.savefig(output_dir / "class_counts.png", dpi=160)
    plt.close(figure)

    columns = min(6, len(class_names))
    rows = math.ceil(len(class_names) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.2, rows * 2.5))
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    first_path_by_label: dict[int, str] = {}
    for path, label in zip(image_paths, labels):
        first_path_by_label.setdefault(int(label), path)

    for class_id, axis in enumerate(flat_axes):
        axis.axis("off")
        if class_id >= len(class_names):
            continue
        axis.imshow(plt.imread(first_path_by_label[class_id]))
        axis.set_title(class_names[class_id])
        axis.axis("off")

    figure.suptitle("One sample image per sign class")
    figure.tight_layout()
    figure.savefig(output_dir / "sample_images.png", dpi=160)
    plt.close(figure)


def make_dataset(
    split: DatasetSplit,
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = tf.convert_to_tensor(split.paths, dtype=tf.string)
    labels = tf.convert_to_tensor(split.labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(split.paths), seed=seed, reshuffle_each_iteration=True
        )

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_data = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_data, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_model(image_size: int, num_classes: int) -> tf.keras.Model:
    """Build a compact CNN with conservative, label-preserving augmentations."""
    augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(0.04),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomZoom(0.08),
        ],
        name="training_augmentation",
    )
    model = models.Sequential(
        [
            layers.Input(shape=(image_size, image_size, 3)),
            augmentation,
            layers.Conv2D(32, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.35),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_curves(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(
        axes, ("accuracy", "loss"), ("Accuracy", "Loss")
    ):
        axis.plot(history.history[metric], label="Training")
        axis.plot(history.history[f"val_{metric}"], label="Validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)

    metric_names = list(history.history)
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *metric_names])
        for epoch, values in enumerate(
            zip(*(history.history[name] for name in metric_names)), start=1
        ):
            writer.writerow([epoch, *values])


def evaluate_model(
    model: tf.keras.Model,
    test_split: DatasetSplit,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
    training_seconds: float,
) -> dict[str, object]:
    predictions = model.predict(test_dataset, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    report = classification_report(
        test_split.labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    test_loss = float(
        tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                test_split.labels, predictions
            )
        ).numpy()
    )
    test_accuracy = float(np.mean(predicted_labels == test_split.labels))

    figure, axis = plt.subplots(
        figsize=(max(7, len(class_names) * 0.45), max(6, len(class_names) * 0.4))
    )
    ConfusionMatrixDisplay.from_predictions(
        test_split.labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
        display_labels=class_names,
        xticks_rotation=90,
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Held-out test-set confusion matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    metrics: dict[str, object] = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "training_seconds": round(training_seconds, 2),
        "test_image_count": int(len(test_split.labels)),
        "per_class": {
            name: {
                metric: float(report[name][metric])
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for name in class_names
        },
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    misclassified = np.flatnonzero(test_split.labels != predicted_labels)
    if len(misclassified):
        shown = misclassified[: min(4, len(misclassified))]
        columns = min(2, len(shown))
        rows = math.ceil(len(shown) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4))
        flat_axes = np.asarray(axes, dtype=object).reshape(-1)
        for axis, index in zip(flat_axes, shown):
            axis.imshow(plt.imread(test_split.paths[index]))
            confidence = predictions[index, predicted_labels[index]]
            axis.set_title(
                f"True: {class_names[test_split.labels[index]]}\n"
                f"Predicted: {class_names[predicted_labels[index]]} "
                f"({confidence:.1%})"
            )
            axis.axis("off")
        for axis in flat_axes[len(shown) :]:
            axis.axis("off")
        figure.suptitle("Misclassified held-out test images")
        figure.tight_layout()
        figure.savefig(output_dir / "misclassified_examples.png", dpi=160)
        plt.close(figure)

    return metrics


def build_and_train_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = Path(__file__).resolve().parent / "outputs",
    image_size: int = 128,
    batch_size: int = 16,
    epochs: int = 40,
    seed: int = 42,
) -> dict[str, object]:
    """Run data inspection, training, and one final held-out test evaluation."""
    if image_size < 32 or batch_size < 1 or epochs < 1:
        raise ValueError("image_size must be >= 32, and batch_size/epochs must be positive.")

    tf.keras.utils.set_random_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths, labels, class_names = collect_dataset(dataset_path)
    splits = create_splits(image_paths, labels, seed)
    save_dataset_overview(image_paths, labels, class_names, output_path)
    with (output_path / "class_names.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    train_dataset = make_dataset(
        splits.train, image_size, batch_size, training=True, seed=seed
    )
    validation_dataset = make_dataset(
        splits.validation, image_size, batch_size, training=False, seed=seed
    )
    test_dataset = make_dataset(
        splits.test, image_size, batch_size, training=False, seed=seed
    )

    model = build_model(image_size, len(class_names))
    class_ids = np.unique(splits.train.labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=class_ids, y=splits.train.labels
    )
    class_weights = dict(zip(class_ids.tolist(), weights.tolist()))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_path / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(output_path / "epoch_log.csv"),
    ]

    start_time = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    training_seconds = time.perf_counter() - start_time
    model.save(output_path / "final_model.keras")
    save_training_curves(history, output_path)

    metrics = evaluate_model(
        model,
        splits.test,
        test_dataset,
        class_names,
        output_path,
        training_seconds,
    )
    print(f"Classes: {', '.join(class_names)}")
    print(
        "Split sizes: "
        f"train={len(splits.train.labels)}, "
        f"validation={len(splits.validation.labels)}, "
        f"test={len(splits.test.labels)}"
    )
    print(
        f"Held-out test accuracy: {metrics['test_accuracy']:.4f}; "
        f"macro F1: {metrics['macro_f1']:.4f}"
    )
    print(f"Evaluation artifacts saved to: {output_path.resolve()}")
    return metrics
