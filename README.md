# BCI Test

EEG motor-imagery classification with EEGNet V4 in PyTorch.

This repository supports two datasets:

- BCI Competition IV 2a
- PhysioNet EEGMMIDB

It also includes a Colab notebook for uploading a dataset zip and training in the cloud.

## Contents

- `bci2a_dataset.py` - local loader and preprocessing for BCI IV 2a
- `eegmmidb_dataset.py` - local loader and preprocessing for EEGMMIDB
- `eegnet_v4.py` - EEGNet V4 model definitions
- `augmentation.py` - EEG augmentation helpers
- `trainer.py` - training engine
- `transfer_learning.py` - transfer learning utilities
- `Config.yaml` - configuration for paths, model settings, and training
- `colab_train_bci.ipynb` - Colab notebook for upload-and-train workflows

## Requirements

Use Python 3.10+ with the following packages:

- `torch`
- `numpy`
- `scipy`
- `mne`
- `pyyaml`
- `scikit-learn`
- `matplotlib`
- `tensorboard`

If you are running on Colab, install the runtime dependencies with:

```bash
pip install mne pyyaml scikit-learn matplotlib
```

PyTorch is usually already available in Colab, but if needed install the CUDA build that matches the runtime.

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the required packages.
4. Update `Config.yaml` with the local paths to your dataset.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy mne pyyaml scikit-learn matplotlib tensorboard
```

## Dataset Layout

### BCI Competition IV 2a

Place the files in a folder like this:

```text
bciv2a_root/
  A01T.gdf
  A01E.gdf
  A01E.mat
  A02T.gdf
  A02E.gdf
  A02E.mat
  ...
  A09E.mat
```

The loader expects the GDF and MAT files in the same root directory.

### EEGMMIDB

Place the PhysioNet folders like this:

```text
eegmmidb_root/
  S001/
    S001R01.edf
    S001R02.edf
    S001R03.edf
    ...
  S002/
  ...
```

## Local Training

The repo is organized around the loaders and model definitions in Python. A typical workflow is:

1. Update the dataset path in `Config.yaml`.
2. Load the config.
3. Build the model and loaders.
4. Train the model and save the best checkpoint.

Example skeleton:

```python
import yaml
from bci2a_dataset import build_bciv2a_loaders
from eegmmidb_dataset import build_eegmmidb_loaders
from eegnet_v4 import build_model

with open("Config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# BCI IV 2a
train_loader, val_loader = build_bciv2a_loaders(cfg, subject=1)
model = build_model(cfg, phase="finetune")

# EEGMMIDB
# train_loader, val_loader = build_eegmmidb_loaders(cfg)
# model = build_model(cfg, phase="pretrain")
```

## Colab Setup

Use `colab_train_bci.ipynb` when you want to upload a zipped dataset and train without setting up a local environment.

Recommended workflow:

1. Open the notebook in Google Colab.
2. Upload a `.zip` file containing your dataset.
3. Set `DATASET_TYPE` to `bciv2a` or `eegmmidb`.
4. Run the notebook cells in order.
5. Download the checkpoint at the end.

The notebook will:

- install the needed packages,
- clone this repository if necessary,
- extract your uploaded archive,
- locate the dataset root automatically,
- build the model and data loaders,
- train the network,
- save a best checkpoint and plot training curves.

## Configuration Notes

`Config.yaml` controls:

- dataset root paths,
- preprocessing settings,
- EEGNet hyperparameters,
- training epochs, batch size, learning rate, and early stopping,
- checkpoint and log output directories.

For BCI IV 2a finetuning, the default model uses:

- `C = 22`
- `T = 500`
- `n_classes = 4`

For EEGMMIDB pretraining, the default model uses:

- `C = 64`
- `T = 480`
- `n_classes = 4`

## Outputs

Training creates the following directories when needed:

- `checkpoints/`
- `logs/`
- `results/`

Best-model checkpoints are written to the configured output path.

## Notes

- The code expects preprocessed EEG epochs with shape `(1, C, T)`.
- BCI IV 2a uses 22 EEG channels and a 500-sample window.
- EEGMMIDB uses 64 EEG channels and a 480-sample window.
- If you want a lighter Colab workflow, the notebook is the recommended entry point.

## License

See [LICENSE](LICENSE) for details.
