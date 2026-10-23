import argparse
from pathlib import Path

from problem_statement_42_solution import DEFAULT_DATASET_PATH, build_and_train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN for static Indian Sign Language signs."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Dataset root containing one image subfolder per sign class.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for the trained model, metrics, and diagnostic plots.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_and_train_model(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
