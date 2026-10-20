# DIRIAMS (End-to-End & Binned MALDI-TOF Deep Learning)

This repository provides deep learning models (1D-CNN) and evaluation utilities for predicting antimicrobial resistance (AMR) from MALDI-TOF mass spectrometry data (DRIAMS dataset).

This version supports both:
- **Raw MALDI-TOF spectra** (length 20,000 m/z data points).
- **Pre-binned spectra** (e.g. `binned_6000` from Kaggle's `drscarlat/driams` dataset).
- Auto-adaptation of dense layer dimensions (`fc1_nodes`) based on input feature length (2656 for raw 20,000 points; 792 for 6,000 binned features).
- Robust directory path parsing (works with or without trailing slashes, across local and Kaggle environments).

---

## Running on Kaggle

### 1. Clone this repository in Kaggle
```bash
!git clone https://github.com/KhoaLeQuangMinh/DIRIAMS.git
%cd DIRIAMS
```

### 2. Run in a Kaggle Notebook Cell
```python
import pytorch_lightning as pl
from maldi_learn.model import MALDI_1D_CNN, backboneModel
from maldi_learn.analyzer import DataAnalyzer
from maldi_learn.utils import load_all_data

# Set seed
pl.seed_everything(1)

# Training parameters
parameters = { 
    "output_directory": "./results_driams/",
    "task": "multilabel",            # multiclass, multilabel or binary 
    "num_classes": 28, 
    "num_workers": 2, 
    "batch_size": 128, 
    "lr": 0.005, 
    "optimizer": "Adam", 
    "epochs": 50,
    "problem": "antibiotic",
    "cnn1_kernel": 60, 
    "fc1_nodes": 792,                # Auto-adapted to 792 for binned_6000 or 2656 for raw
    "max_pool_kernel": 30, 
    "max_pool_stride": 30, 
    "dropout": 0.3,
    "sampling": "random"             # random, DRIAMS-A, or DRIAMS-B
}

# Prefixes for Kaggle dataset (drscarlat/driams)
prefixes = [
    "/kaggle/input/datasets/drscarlat/driams/DRIAMS-B",
    "/kaggle/input/datasets/drscarlat/driams/DRIAMS-C", 
    "/kaggle/input/datasets/drscarlat/driams/DRIAMS-D", 
    "/kaggle/input/datasets/drscarlat/driams/DRIAMS-A"
]

# 1. Load metadata & spectra
metadata, X = load_all_data(prefixes)

# 2. Setup analyzer and DataModule
analysis = DataAnalyzer(X, metadata, parameters)
analysis.setup()

# 3. Instantiate model
model = MALDI_1D_CNN(
    backbone = backboneModel, 
    classes  = analysis.params["num_classes"], 
    params   = analysis.params, 
    learning_rate = analysis.params["lr"],
    cosine_t_max  = analysis.params["epochs"] * len(analysis.train_dataloader())
)

callbacks = [
    pl.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min"),
    pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3),
    pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
]

# 4. Train the model
analysis.train(model, callbacks)

# 5. Evaluate and save predictions
analysis.evaluate()
```
