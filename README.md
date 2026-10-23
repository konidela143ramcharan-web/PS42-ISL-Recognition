# Problem Statement 42 — Indian Sign Language Recognition

This project trains a CNN to classify static Indian Sign Language (ISL) images.
The assignment excludes the motion signs J and Z. The pipeline also avoids
horizontal-flip augmentation by default because flipping can change handedness
or the meaning of a sign.

## Dataset layout

Place the ISLAN dataset under `data\ISLAN`, with one folder per sign class:

```text
data\
└── ISLAN\
    ├── A\
    │   ├── image_001.jpg
    │   └── image_002.jpg
    ├── B\
    │   └── image_003.jpg
    └── ...
```

Supported image types are JPG, JPEG, PNG, and BMP. Class folders named `J` or
`Z` are excluded. Each included class needs at least five images, and the
dataset must be large enough for stratified train, validation, and test splits.

## Setup (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Run

```powershell
py main.py --dataset "C:\path\to\ISLAN" --epochs 40 --seed 42
```

The default dataset folder is `data\ISLAN` relative to this project. Run
`py main.py --help` to see all available options.

## What the pipeline does

- Reports and plots the image count per class and saves one sample image per
  class.
- Creates deterministic, stratified 60/20/20 training, validation, and held-out
  test splits.
- Trains a CNN with small rotation, translation, and zoom augmentations,
  balanced class weights, early stopping, and best-validation-loss checkpointing.
- Evaluates once on the held-out test set, saving accuracy, macro and per-class
  precision/recall/F1, training time, and a confusion matrix.
- Saves training curves, epoch logs, up to four misclassified test examples, and
  Keras model files and the `class_names.json` label mapping.

All generated files go to `outputs\`. The training script does not invent
results when the dataset is missing; provide the dataset before running it.

## Predict one image

After training, classify a single image with the best validation checkpoint:

```powershell
py scripts\predict_image.py "C:\path\to\sign.png"
```

The script prints the top five predicted sign labels and confidence scores.
Use `--model`, `--classes`, `--image-size`, or `--top-k` to override defaults.
The model and class mapping must come from the same training run.

## Run tests

```powershell
py -m unittest discover -s tests -v
```

The tests create temporary sample images and do not require the ISLAN dataset.

## Assignment submission reminder

The assignment additionally requires an executed notebook with outputs, a
two-page report, a GitHub link, an explanation of the augmentation choices,
and a written analysis of at least four misclassified test images. The saved
artifacts support that work but do not replace the report or notebook.
