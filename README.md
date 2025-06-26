# Cell Segmentator

---

## Overview

This repository provides two main scripts to configure and run a cell segmentation workflow:

* **generate\_config.py**: Interactive script to create JSON configuration files for training or prediction.
* **main.py**: Entry point to train, test, or predict using the generated configuration.

---

## Installation

0. **Install uv**:
Follow the official guide at [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)

    **Linux / macOS**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    **Windows**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    ```bash
    uv --version
    ```

1. **Clone the repository**:

   ```bash
   git clone https://git.ai.infran.ru/ilyukhin/model-v
   cd model-v
   ```
2. **Install dependencies**:

   ```bash
   uv sync
   ```

---

## Dataset Structure

Your data directory must follow this hierarchy:

```
path_to_data_folder/
├── images/        # Input images (any supported format)
│   ├── img1.tif
│   ├── img2.png
│   └── …
└── masks/         # Ground-truth instance masks (any supported format)
    ├── mask1.tif
    ├── mask2.jpg
    └── …
```

If your dataset contains multiple classes (e.g., class A and B) and you prefer not to duplicate images, you can organize masks into class-specific subdirectories:

```
path_to_data_folder/
├── images/        # Input images (any supported format)
│   └── img1.bmp
└── masks/
    ├── A/         # Masks for class A (any supported format)
    │   ├── img1_mask.png
    │   └── …
    └── B/         # Masks for class B (any supported format)
        ├── img1_mask.jpeg
        └── …
```

In this case, set the `masks_subdir` field in your dataset configuration to the name of the mask subdirectory (e.g., `"A"` or `"B"`).

**Supported file formats**: Image and mask files can have any of these extensions:
`tif`, `tiff`, `png`, `jpg`, `bmp`, `jpeg`.

**Mask format**: Instance masks should be provided for multi-label segmentation with channel-last ordering, i.e., each mask array must have shape `(H, W, C)`.

---

## generate\_config.py

This script guides you through creating a JSON configuration for either training or prediction.

### Usage

```bash
python generate_config.py
```

1. **Training mode?** Select `y` or `n`.
2. **Model selection**: Choose from available models in the registry.
3. **(If training)**

   * Criterion selection
   * Optimizer selection
   * Scheduler selection
4. Configuration is saved under `config/templates/train/` or `config/templates/predict/` with a unique filename.

Generated config includes sections:

* `model`: Model component and parameters
* `dataset_config`: Paths, training flag, and mask subdirectory (if any)
* `wandb_config`: Weights & Biases integration settings
* *(If training)* `criterion`, `optimizer`, `scheduler`

---

## main.py

Entrypoint to run training, testing, or prediction using a config file.

### Command-line Arguments

```bash
python main.py [-c CONFIG] [-m {train,test,predict}] [--no-save-masks] [--only-masks]
```

* `-c, --config` : Path to JSON config file (default: `config/templates/train/...json`).
* `-m, --mode`   : `train`, `test`, or `predict` (default: `train`).
* `--no-save-masks` : Disable saving predicted masks.
* `--only-masks`    : Save only raw predicted masks (no visual overlays). This flag depends on `--no-save-masks`.

### Workflow

1. **Load config** and verify mode consistency.
2. **Initialize** Weights & Biases if enabled.
3. **Create** `CellSegmentator` and dataloaders with appropriate transforms.
4. **Print** dataset info for the first batch.
5. **Run** training or inference (`.run()`).
6. **Save** model checkpoint and upload to W\&B if in training mode.

---

## Configurable Parameters

A brief overview of the key parameters you can adjust in your JSON config:

### Common Settings (`common`)

* `seed` (int): Random seed for data splitting and reproducibility (default: `0`).
* `device` (str): Compute device to use, e.g., `'cuda:0'` or `'cpu'` (default: `'cuda:0'`).
* `use_amp` (bool): Enable Automatic Mixed Precision for faster training (default: `false`).
* `roi_size` (int): Defines the size of the square Region of Interest (ROI) used for cropping during training. This same size is also applied for the sliding window inference during validation and testing (default: `512`).
* `remove_boundary_objects` (bool): Flag to remove boundary objects when testing (default: `True`).
* `masks_subdir` (str): Name of subdirectory under `masks/` containing the instance masks (default: `""`).
* `predictions_dir` (str): Output directory for saving predicted masks (default: `"."`).
* `pretrained_weights` (str): Path to pretrained model weights (default: `""`).

### Training Settings (`training`)

* `is_split` (bool): Whether your data is already split (`true`) or needs splitting (`false`, default).
* `split` / `pre_split`: Directories for data when pre-split or unsplit.
* `train_size`, `valid_size`, `test_size` (int/float): Size or ratio of your splits (e.g., `0.7`, `0.1`, `0.2`).
* `batch_size` (int): Number of samples per training batch (default: `1`).
* `num_epochs` (int): Total training epochs (default: `100`).
* `val_freq` (int): Frequency (in epochs) to run validation (default: `1`).

### Testing Settings (`testing`)

* `test_dir` (str): Directory containing test data (default: `"."`).
* `test_size` (int/float): Portion or count of data for testing (default: `1.0`).
* `shuffle` (bool): Shuffle test data before evaluation (default: `true`).

> **Batch size note:** Validation, testing, and prediction runs always use a batch size of `1`, regardless of the `batch_size` setting in the training configuration.

---

## Examples

### Generate a training config

```bash
python generate_config.py
# Follow prompts to select model, criterion, optimizer, scheduler
# Output saved to config/templates/train/YourConfig.json
```

### Train a model

```bash
python main.py -c config/templates/train/YourConfig.json -m train
```

### Predict on new data

```bash
python main.py -c config/templates/predict/YourConfig.json -m predict
```

---

## Acknowledgments

This project was developed building upon the following open-source repositories:

* [Cellpose](https://github.com/MouseLand/cellpose) by the MouseLand Lab.
* [MEDIAR](https://github.com/Lee-Gihun/MEDIAR) by Lee Gihun.
