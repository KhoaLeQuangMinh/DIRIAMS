# make imports
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from captum.attr import DeepLiftShap
from tqdm import tqdm
from maldi_learn.utils import MaldiDataModule, MaldiDataset
from maldi_learn.model import MALDI_1D_CNN, backboneModel
import os
import yaml
from umap import UMAP


class DataAnalyzer(MaldiDataModule):
    """
    A class for analyzing MALDI-TOF mass spectrometry data.

    Args:
        X (torch.Tensor): The input data tensor.
        metadata (list): The metadata associated with the input data.
        params (dict): Additional parameters for the data analyzer.

    Attributes:
        predictions (list): A list to store the model predictions.
        probabilities (list): A list to store the predicted probabilities.
        attributions (dict): A dictionary to store the attributions.
        best_model_path (str): The path to the best model checkpoint.

    Methods:
        train(model, callbacks): Trains the model using the given callbacks.
        evaluate(model): Evaluates the model and stores the predictions and probabilities.
        map_target_to_name(target): Maps the target index to its corresponding name.
        get_attributions(n_baselines, targets): Computes the attributions for the given targets.
        save_results(path): Saves the results to the specified path.
    """

    def __init__(self, X, metadata, params):
        super().__init__(X, metadata, params)
        self.predictions = list()
        self.probabilities = list()
        self.attributions = dict()
        self.best_model_path = None

        if not self.params["output_directory"].endswith("/"):
            self.params["output_directory"] += "/"

        if not os.path.exists(self.params["output_directory"]):
            os.makedirs(self.params["output_directory"])

        # Auto-adjust fc1_nodes based on feature dimension
        if len(self.X.shape) >= 3:
            input_len = self.X.shape[2]
            l1 = input_len - self.params.get("cnn1_kernel", 60) + 1
            l2 = (l1 - self.params.get("max_pool_kernel", 30)) // self.params.get("max_pool_stride", 30) + 1
            expected_fc1 = 4 * l2
            if self.params.get("fc1_nodes") != expected_fc1:
                print(f"Auto-adjusting params['fc1_nodes'] from {self.params.get('fc1_nodes')} to {expected_fc1} for input feature length {input_len}")
                self.params["fc1_nodes"] = expected_fc1


    def save_config(self):
        with open(self.params["output_directory"] + "config.yaml", "w") as f:
            yaml.dump(self.params, f, default_flow_style=False)
        return

    def read_config(self):
        with open(self.params["output_directory"] + "config.yaml", "r") as f:
            self.params = yaml.load(f, Loader=yaml.FullLoader)
        return

    def train(self, model, callbacks):
        """
        Trains the model using the given callbacks.

        Args:
            model: The model to be trained.
            callbacks: The callbacks to be used during training.

        Returns:
            None
        """
        self.setup("fit")
        trainer = pl.Trainer(
            max_epochs=self.params["epochs"], 
            default_root_dir="./", 
            callbacks=callbacks)

        trainer.fit(model = model, datamodule=self)

        
        self.best_model_path = trainer.checkpoint_callback.best_model_path
        self.params["best_model_path"] = self.best_model_path
        self.save_config()
        return
    
    def evaluate(self, transfer_eval = False):
        """
        Evaluates the model and stores the predictions and probabilities.

        Args:
            model: The model to be evaluated.

        Returns:
            None
        """
        
        self.setup("test")
        model = MALDI_1D_CNN.load_from_checkpoint(self.params["best_model_path"], backbone=backboneModel, classes = self.params["num_classes"], params=self.params)
        model = model.to(self.device)

        if transfer_eval:
            extra = "_transfer_"
            test_dataloader = transfer_eval
        else:
            extra = ""
            test_dataloader = self.test_dataloader()
        
        for batch in test_dataloader:
            x, _ = batch
            x = x.to(self.device)
            with torch.inference_mode():
                y_hat = model(x)
            preds = (y_hat>0.5).int()

            self.predictions.append(preds)
            self.probabilities.append(y_hat.sigmoid())
        
        # Concatenate and convert to pandas
        self.predictions   = torch.cat(self.predictions, dim=0).cpu().detach().numpy()
        self.probabilities = torch.cat(self.probabilities, dim=0).cpu().detach().numpy()

        self.save_results(extra=extra)
    
        return

    def map_target_to_name(self, target):
        if self.params["problem"] == "species":
            return self.metadata_test[0].species.unique()[target]
        if self.params["problem"] == "antibiotic":
            return self.metadata_test[0].columns[target]
    
    def get_attributions(self, n_baselines = 500, targets = [0]):
        """
        Computes the attributions for the given targets.

        Args:
            n_baselines (int): The number of baselines to sample.
            targets (list): The list of target indices.

        Returns:
            None
        """
        # Sample baselines
        indices = torch.randperm(len(self.train_dataloader().dataset))[:n_baselines]
        baselines = self.train_dataloader().dataset[indices.tolist()][0].type(torch.float32).to(self.device)

        # Load the best model
        model = MALDI_1D_CNN.load_from_checkpoint(self.params["best_model_path"], backbone=backboneModel, classes = self.params["num_classes"], params=self.params, map_location=self.device)
        model = model.type(torch.float32).to(self.device)
        model.eval()

        # Get attributions
        dl = DeepLiftShap(model)
        
        for target in targets:
            attributions = list()
            for x, y in tqdm(self.test_dataloader()):
                x, y = x.to(self.device), y.to(self.device)
                grads = dl.attribute(
                    x, 
                    target=target, 
                    baselines = baselines
                )
                attr = grads.squeeze(1).cpu().detach().numpy()
                attributions.append(attr)
            
            # Concatenate attributions and save to csv
            self.attributions[self.map_target_to_name(target=target)] = np.concatenate(attributions, axis=0)
            pd.DataFrame(self.attributions[self.map_target_to_name(target=target)]).to_csv(self.params["output_directory"] + str(target) + "_attributions.csv", index=False)
        return
    
    # Make umap of attributions
    def make_umap(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', target=9, seed=42):
        """
        Computes the UMAP of the attributions for the given target.

        Args:
            n_components (int): The number of components for the UMAP.
            n_neighbors (int): The number of neighbors for the UMAP.
            min_dist (float): The minimum distance for the UMAP.
            metric (str): The metric to be used for the UMAP.
            target (int): The target index.
            seed (int): The seed for reproducibility.

        Returns:
            None
        """
        # Set seed for reproducibility
        np.random.seed(seed)

        # Load the attributions
        attributions = pd.read_csv(self.params["output_directory"] + str(target) + "_attributions.csv").values

        # Compute UMAP
        umap = UMAP(
            n_components=n_components, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist, 
            metric=metric,
            random_state=seed
        )
        umap.fit(attributions)

        # Save UMAP
        pd.DataFrame(umap.embedding_).to_csv(self.params["output_directory"] + str(target) + "_umap.csv", index=False)
        return
    
    def save_results(self, extra = ""):
        """
        Saves the results to the specified path.

        Args:
            path (str): The path to save the results.

        Returns:
            None
        """
        # Save predictions
        pd.DataFrame(self.predictions).to_csv(self.params["output_directory"] + extra + "predictions.csv", index=False)
        
        # Save probabilities
        pd.DataFrame(self.probabilities).to_csv(self.params["output_directory"] + extra + "probabilities.csv", index=False)
        
        # Save metadata
        self.metadata_test[0].to_csv(self.params["output_directory"] + extra + "metadata.csv", index=False)
        
        # Save metadata_orig
        self.metadata_test[2].to_csv(self.params["output_directory"] + extra + "metadata_orig.csv", index=False)

        # save train and val metadata
        self.metadata_train[0].to_csv(self.params["output_directory"] + extra + "metadata_train.csv", index=False)
        self.metadata_val[0].to_csv(self.params["output_directory"] + extra + "metadata_val.csv", index=False)

        # save train and val metadata_orig
        self.metadata_train[2].to_csv(self.params["output_directory"] + extra + "metadata_train_orig.csv", index=False)
        self.metadata_val[2].to_csv(self.params["output_directory"] + extra + "metadata_val_orig.csv", index=False)



        
        if self.params["sampling"] != "kfold":
                    # Save the x values of test, train and val
            test_size = len(self.test_dataloader().dataset)
            train_size = len(self.train_dataloader().dataset)
            val_size = len(self.val_dataloader().dataset)

            # Save x values of train test and val
            np.savetxt(self.params["output_directory"] + extra + "test_x.csv", self.test_dataloader().dataset[:test_size][0].numpy().reshape(test_size, -1), delimiter=",")
            np.savetxt(self.params["output_directory"] + extra + "train_x.csv", self.train_dataloader().dataset[:train_size][0].numpy().reshape(train_size, -1), delimiter=",")
            np.savetxt(self.params["output_directory"] + extra + "val_x.csv", self.val_dataloader().dataset[:val_size][0].numpy().reshape(val_size, -1), delimiter=",")

            # save y values of train test and val
            np.savetxt(self.params["output_directory"] + extra + "train_y.csv", self.train_dataloader().dataset[:train_size][1].numpy(), delimiter=",")
            np.savetxt(self.params["output_directory"] + extra + "test_y.csv", self.test_dataloader().dataset[:test_size][1].numpy(), delimiter=",")
            np.savetxt(self.params["output_directory"] + extra + "val_y.csv", self.val_dataloader().dataset[:val_size][1].numpy(), delimiter=",")

        return

