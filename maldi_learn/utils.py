# Imports
import os
import re
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Helper functions for loading data
def get_preprocessed_path(paths):
    total_output = []

    for data_path in paths:
        # Check candidate directories: binned_6000 or raw or preprocessed
        subdirs = [
            os.path.join(data_path, "binned_6000"),
            os.path.join(data_path, "raw"),
            os.path.join(data_path, "preprocessed_raw"),
            os.path.join(data_path, "preprocessed"),
        ]
        found_any = False
        for subdir in subdirs:
            if os.path.isdir(subdir):
                found_any = True
                for root, _, files in os.walk(subdir):
                    for file in files:
                        if file.endswith(".txt") or file.endswith(".csv"):
                            total_output.append(os.path.join(root, file))

        # Fallback: if no recognized subfolder found, walk data_path directly (ignoring 'id' folder)
        if not found_any and os.path.isdir(data_path):
            for root, _, files in os.walk(data_path):
                if os.path.basename(root) == "id" or "/id/" in root:
                    continue
                for file in files:
                    if file.endswith(".txt") or file.endswith(".csv"):
                        total_output.append(os.path.join(root, file))

    return total_output


def get_metadata(paths):
    metadata_files = []
    for prefix in paths:
        metadata_path = os.path.join(prefix, "id")
        if not os.path.isdir(metadata_path):
            if os.path.isdir(prefix):
                metadata_path = prefix

        for root, _, files in os.walk(metadata_path):
            if len(files) == 0:
                continue
            for file in files:
                if "_clean" not in file or not file.endswith(".csv"):
                    continue
                full_file = os.path.join(root, file)
                try:
                    df = pd.read_csv(full_file, low_memory=False)
                    metadata_files.append(df)
                except Exception as e:
                    print(f"Warning: Failed to read metadata file {full_file}: {e}")

    if len(metadata_files) == 0:
        raise ValueError(
            f"No metadata files containing '_clean' found in {paths}. "
            f"Please verify that paths point to DRIAMS site folders (e.g. DRIAMS-A) containing an 'id/' directory."
        )

    return pd.concat(metadata_files, ignore_index=True)


from concurrent.futures import ThreadPoolExecutor

def parse_single_spectrum(path):
    try:
        if "raw" in path and not "raw_end_to_end" in path:
            file = np.loadtxt(path, skiprows=3, delimiter=" ")
            file = file[file[:, 0] > 2000]
            intensity = file[:, 1] ** 0.25
            rng = intensity.max() - intensity.min()
            if rng > 0:
                x = (-1 + 2 * ((intensity - intensity.min()) / rng))[:20000].reshape(1, -1)
            else:
                x = intensity[:20000].reshape(1, -1)
            return x.astype(np.float32)

        # Ultra-fast parser for binned spectra
        with open(path, 'r', errors='ignore') as f:
            text = f.read()

        tokens = text.replace(',', ' ').split()
        try:
            arr = np.array(tokens, dtype=np.float32)
        except ValueError:
            arr = np.fromiter(
                (float(t) for t in tokens if len(t) > 0 and (t[0].isdigit() or t[0] in '+-.')),
                dtype=np.float32
            )

        if len(arr) == 0:
            return None

        # Handle 2-column formats (mass and intensity)
        if len(arr) == 12000:
            arr = arr.reshape(-1, 2)[:, 1]

        rng = arr.max() - arr.min()
        if rng > 0:
            x = (-1 + 2 * ((arr - arr.min()) / rng)).reshape(1, -1)
        else:
            x = arr.reshape(1, -1)

        return x.astype(np.float32)
    except Exception:
        return None


def load_preprocessed(output_paths, max_workers=8):
    paths_list = list(output_paths)
    print(f"Loading {len(paths_list)} spectra using {max_workers} worker threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(parse_single_spectrum, paths_list), total=len(paths_list)))

    X = []
    successfully_loaded = []
    for res in results:
        if res is not None and res.shape[1] > 0:
            X.append(res)
            successfully_loaded.append(True)
        else:
            successfully_loaded.append(False)

    if len(X) == 0:
        raise ValueError(f"Could not load any spectrum arrays from {len(output_paths)} files.")

    X = np.concatenate(X, axis=0)
    return X, successfully_loaded


def serialize_data(metadata, name):
    if not os.path.exists(name):
        # X of shape (sum(success), n_features) and success of shape (n_files,)
        X, successfully_loaded = load_preprocessed(metadata.full_path.to_list())
        metadata = metadata.iloc[successfully_loaded].copy()

        my_object = (metadata, X)
        # Open a file handle in write mode
        with open(name, 'wb') as file:
            pickle.dump(my_object, file)
    else:
        # Open the file handle in read mode
        with open(name, 'rb') as file:
            metadata, X = pickle.load(file)

    return metadata, X


def extract_sampling_site(filename):
    site_match = re.search(r"(DRIAMS-[A-Z])", filename)
    if site_match:
        return site_match.group(1)
    return "Unknown"


def extract_year(filename):
    parent_dir = os.path.basename(os.path.dirname(filename))
    if re.match(r"^\d{4}$", parent_dir):
        return parent_dir
    year_match = re.search(r"/(\d{4})/", filename)
    if year_match:
        return year_match.group(1)
    return "Unknown"


# Parse all data across different years
def load_all_data(prefixes):
    # get all preprocessed paths and metadata
    outputs = get_preprocessed_path(prefixes)
    metadata = get_metadata(prefixes).copy()

    # Map code (spectrum ID without extension) to its full file path
    code_to_path = {}
    for path in outputs:
        code = os.path.splitext(os.path.basename(path))[0]
        code_to_path[code] = path

    # Keep only samples that have corresponding spectra files
    metadata = metadata[metadata["code"].isin(code_to_path.keys())].copy()

    # Filter species: exclude MIX and keep species with > 100 samples
    if "species" in metadata.columns:
        metadata = metadata[~metadata["species"].str.contains("MIX", na=False)].copy()
        species, count = np.unique(metadata["species"].dropna(), return_counts=True)
        accepted_species = species[count > 100]
        metadata = metadata[metadata["species"].isin(accepted_species)].copy()

    metadata["full_path"] = metadata["code"].map(code_to_path)

    # Serialize data
    name = "./datapackage_" + str(metadata.shape[0]) + ".pickle"
    metadata, X = serialize_data(metadata, name)

    # Add extra metadata to the full dataset
    full_paths = metadata.full_path.to_list()
    metadata["sampling_site"] = [extract_sampling_site(f) for f in full_paths]
    metadata["year"] = [extract_year(f) for f in full_paths]

    X = X.reshape(X.shape[0], 1, X.shape[1])  # Shape (n_samples, n_channels, n_features)
    return (metadata, X)


# Generate train test splits for dataloaders
def MaldiDataset(X, metadata, split, params):
    # Not included in the study
    if params["problem"] == "species":
        y = metadata.species.to_numpy()
        metadata = y
        y = (pd.get_dummies(y) * 1).to_numpy()
        y = torch.from_numpy(y).float()
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, random_state=42)
        metadata_train, metadata_test = train_test_split(metadata, random_state=42)
        metadata_train, metadata_val = train_test_split(metadata_train, random_state=42)

    if params["problem"] == "antibiotic":
        minimum_occurences = 1000
        metadata_orig = metadata
        species = metadata.species.to_numpy()

        # Remove duplicated columns + meta info safely
        antibiotic = metadata.drop(
            columns=[
                "code", 'Unnamed: 0.1', "Unnamed: 0", "species",
                "combined_code", 'filename', "full_path", 'genus',
                'sampling_site', 'year', 'laboratory_species'
            ],
            errors="ignore"
        )

        columns = antibiotic.columns.to_numpy()
        columns = np.array([re.sub("_.*|[.].*", "", str(x)) for x in columns])
        unique = np.unique(columns)
        y = ((antibiotic == "R").to_numpy() * 1 + (antibiotic == "I").to_numpy() * 1)

        agents = list()
        for i in unique:
            index = np.array([i == x for x in columns])
            agents.append(y[..., index].max(axis=1, keepdims=True))

        # Redefine cleaned data structures
        y = np.concatenate(agents, axis=1)
        antibiotic = pd.DataFrame(y, columns=unique)

        y_keep = y[:, y.sum(axis=0) > minimum_occurences]
        metadata = antibiotic.iloc[:, y.sum(axis=0) > minimum_occurences]
        y = torch.from_numpy(y_keep).float()

        if params["sampling"] == "kfold":
            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
            Five_folds = torch.utils.data.random_split(dataset, [1 / params["kfold"]] * params["kfold"])
            test_dataset = Five_folds[params["fold"]]
            train_dataset = torch.utils.data.ConcatDataset([Five_folds[i] for i in range(5) if i != params["fold"]])
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

            train_indices, val_indices, test_indices = train_dataset.indices, val_dataset.indices, test_dataset.indices
            metadata_train, metadata_val, metadata_test = metadata.iloc[train_indices], metadata.iloc[val_indices], metadata.iloc[test_indices]
            species_train, species_val, species_test = species[train_indices], species[val_indices], species[test_indices]
            metadata_orig_train, metadata_orig_val, metadata_orig_test = metadata_orig.iloc[train_indices], metadata_orig.iloc[val_indices], metadata_orig.iloc[test_indices]

        if params["sampling"] == "random":
            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

            train_indices, val_indices, test_indices = train_dataset.indices, val_dataset.indices, test_dataset.indices
            metadata_train, metadata_val, metadata_test = metadata.iloc[train_indices], metadata.iloc[val_indices], metadata.iloc[test_indices]
            species_train, species_val, species_test = species[train_indices], species[val_indices], species[test_indices]
            metadata_orig_train, metadata_orig_val, metadata_orig_test = metadata_orig.iloc[train_indices], metadata_orig.iloc[val_indices], metadata_orig.iloc[test_indices]

        if params["sampling"] == "DRIAMS-A":
            train = ((metadata_orig.sampling_site == "DRIAMS-A") & (metadata_orig.year.isin(["2015", "2016", "2017"]))).to_numpy()
            test = ((metadata_orig.sampling_site == "DRIAMS-A") & (metadata_orig.year.isin(["2018"]))).to_numpy()

            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

            train_indices, val_indices = train_dataset.indices, val_dataset.indices
            metadata_train, metadata_val = metadata.iloc[train_indices], metadata.iloc[val_indices]
            species_train, species_val = species[train_indices], species[val_indices]
            metadata_orig_train, metadata_orig_val = metadata_orig.iloc[train_indices], metadata_orig.iloc[val_indices]

            test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
            metadata_test = metadata[test]
            species_test = species[test]
            metadata_orig_test = metadata_orig[test]

        if params["sampling"] == "DRIAMS-B":
            train = ((metadata_orig.sampling_site.isin(["DRIAMS-A", "DRIAMS-C", "DRIAMS-D"]))).to_numpy()
            test = ((metadata_orig.sampling_site.isin(["DRIAMS-B"]))).to_numpy()

            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

            train_indices, val_indices = train_dataset.indices, val_dataset.indices
            metadata_train, metadata_val = metadata.iloc[train_indices], metadata.iloc[val_indices]
            species_train, species_val = species[train_indices], species[val_indices]
            metadata_orig_train, metadata_orig_val = metadata_orig.iloc[train_indices], metadata_orig.iloc[val_indices]

            test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
            metadata_test = metadata[test]
            species_test = species[test]
            metadata_orig_test = metadata_orig[test]

    # Split data
    if split == "train":
        return (metadata_train, species_train, metadata_orig_train), train_dataset
    elif split == "val":
        return (metadata_val, species_val, metadata_orig_val), val_dataset
    elif split == "test":
        return (metadata_test, species_test, metadata_orig_test), test_dataset
    else:
        raise ValueError("Split must be train, val or test")


# Create a pytorch lightning data module
class MaldiDataModule(pl.LightningDataModule):
    def __init__(self, X, metadata, params):
        super().__init__()
        self.params = params
        self.batch_size = params["batch_size"]
        self.num_workers = params["num_workers"]
        self.X = X
        self.metadata = metadata
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.attributions = dict()
        self.best_model_path = None

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=str):
        self.metadata_train, self.train_dataset = MaldiDataset(self.X, self.metadata, "train", self.params)
        self.metadata_val, self.val_dataset = MaldiDataset(self.X, self.metadata, "val", self.params)
        self.metadata_test, self.test_dataset = MaldiDataset(self.X, self.metadata, "test", self.params)

        # get class weights for train y
        if self.params["problem"] == "antibiotic":
            y = self.metadata_train[0].to_numpy()
            class_weights = (y == 0).sum(axis=0) / np.maximum(y.sum(axis=0), 1)
            self.params["class_weights"] = torch.tensor(class_weights).float().to(self.device)
        else:
            self.params["class_weights"] = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)
