# Import modules
from maldi_learn.model import MALDI_1D_CNN, backboneModel
from maldi_learn.analyzer import DataAnalyzer
from maldi_learn.utils import load_all_data
import pytorch_lightning as pl
import json
import sys
import os

def main(args):
    # Set seed
    pl.seed_everything(1)

    # Get parameters
    if len(args) > 1:
        parameters = json.loads(args[1])
    else:
        parameters = {
            "output_directory": "./results_random_end_to_end/",
            "task": "multilabel",
            "num_classes": 28,
            "num_workers": 4,
            "batch_size": 128,
            "lr": 0.005,
            "optimizer": "Adam",
            "epochs": 100,
            "problem": "antibiotic",
            "cnn1_kernel": 60,
            "fc1_nodes": 2656,
            "max_pool_kernel": 30,
            "max_pool_stride": 30,
            "dropout": 0.3,
            "sampling": "random"
        }

    # Detect dataset prefixes
    if len(args) > 2:
        prefixes = json.loads(args[2])
    else:
        kaggle_base = "/kaggle/input/datasets/drscarlat/driams"
        if not os.path.isdir(kaggle_base):
            kaggle_base = "/kaggle/input/driams"

        if os.path.isdir(kaggle_base):
            prefixes = [
                os.path.join(kaggle_base, "DRIAMS-B"),
                os.path.join(kaggle_base, "DRIAMS-C"),
                os.path.join(kaggle_base, "DRIAMS-D"),
                os.path.join(kaggle_base, "DRIAMS-A"),
            ]
        else:
            prefixes = [
                "/faststorage/project/amr_driams/data/DRIAMS-B/",
                "/faststorage/project/amr_driams/data/DRIAMS-C/",
                "/faststorage/project/amr_driams/data/DRIAMS-D/",
                "/faststorage/project/amr_driams/data/DRIAMS-A/"
            ]

    print(f"Loading data from prefixes: {prefixes}")

    # Load data
    metadata, X = load_all_data(prefixes)
    analysis = DataAnalyzer(X, metadata, parameters)
    analysis.setup()

    # Instantiate pytorch lightning model and callbacks
    model = MALDI_1D_CNN(
        backbone = backboneModel,
        classes  = analysis.params["num_classes"],
        params   = analysis.params,
        learning_rate = analysis.params["lr"],
        cosine_t_max  = analysis.params["epochs"] * len(analysis.train_dataloader())
    )

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min"),
        pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    # Train the model
    analysis.train(model, callbacks)

    # Get the predictions
    analysis.evaluate()

    if parameters.get("sampling") != "kfold":
        # Get the attributions
        analysis.get_attributions(n_baselines=500, targets=[9, 7, 4])

if __name__ == "__main__":
    main(sys.argv)
