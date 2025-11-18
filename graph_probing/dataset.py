import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse


def prepare_data(dataset_filename, ckpt_step, llm_model_name, dataset_path, test_set_ratio, seed, target=["perplexities"], dataset_name="openwebtext", normalize_targets=True):
    data = pd.read_csv(dataset_filename)
    if isinstance(target, str):
        target = [target]
    targets = data[target].to_numpy(dtype=np.float32)
    if normalize_targets:
        targets = (targets - targets.min(axis=0)) / (targets.max(axis=0) - targets.min(axis=0))
    num_sentences = len(targets)

    if ckpt_step == -1:
        path = os.path.join(dataset_path, llm_model_name)
    else:
        path = os.path.join(dataset_path, f"{llm_model_name}_step{ckpt_step}")
    
    path = os.path.join(path, dataset_name)
    
    generator = torch.Generator().manual_seed(seed)
    test_set_size = int(num_sentences * test_set_ratio)
    train_data_split, test_data_split = torch.utils.data.random_split(
        list(range(num_sentences)), [num_sentences - test_set_size, test_set_size], generator=generator)
    return train_data_split, test_data_split, targets, path


def wrap_data(path, network_id, llm_layer, target_value, network_density, from_sparse_data=False):
    if not from_sparse_data:
        network = np.load(os.path.join(path, f"{network_id}/layer_{llm_layer}_corr.npy")).astype(np.float32)
        percentile_threshold = network_density * 100
        threshold = np.percentile(np.abs(network), 100 - percentile_threshold)
        network[np.abs(network) < threshold] = 0
        np.fill_diagonal(network, 1.0)
        llm_brain_network = torch.from_numpy(network)
        edge_index_llm, edge_attr_llm = dense_to_sparse(llm_brain_network)
        num_nodes = llm_brain_network.shape[0]
    else:
        edge_index_llm = np.load(os.path.join(path, f"{network_id}/layer_{llm_layer}_sparse_{network_density}_edge_index.npy"))
        edge_attr_llm = np.load(os.path.join(path, f"{network_id}/layer_{llm_layer}_sparse_{network_density}_edge_attr.npy")).astype(np.float32)
        num_nodes = edge_index_llm.max() + 1
        edge_index_llm = torch.from_numpy(edge_index_llm)
        edge_attr_llm = torch.from_numpy(edge_attr_llm)
    data = Data(
        x=torch.arange(num_nodes),
        edge_index=edge_index_llm,
        edge_attr=edge_attr_llm,
        y=torch.tensor(target_value, dtype=torch.float32)
    )
    return data


class BrainNetworkDataset(Dataset):
    def __init__(self, sentences_id, dataset_path, targets, llm_layer, network_density, from_sparse_data=False):
        super().__init__(None, transform=None, pre_transform=None)
        self.sentences_id = sentences_id
        self.dataset_path = dataset_path
        self.targets = targets
        self.llm_layer = llm_layer
        self.network_density = network_density
        self.from_sparse_data = from_sparse_data

    def len(self):
        return len(self.sentences_id)

    def get(self, idx):
        network_id = self.sentences_id[idx]
        data = wrap_data(self.dataset_path, network_id, self.llm_layer, self.targets[idx], self.network_density, self.from_sparse_data)
        return data


def get_brain_network_dataloader(
    dataset_filename,
    network_density=1.0,
    from_sparse_data=False,
    llm_model_name="gpt2",
    ckpt_step=-1,
    llm_layer=0,
    dataset_path="data/graph_probing",
    batch_size=32,
    eval_batch_size=32,
    num_workers=4,
    prefetch_factor=2,
    test_set_ratio=0.2,
    in_memory=True,
    shuffle=True,
    seed=42,
    target=["perplexities"],
    dataset_name="openwebtext",
    normalize_targets=True,
    **kwargs
):

    train_data_split, test_data_split, targets, path = prepare_data(
        dataset_filename,
        ckpt_step,
        llm_model_name,
        dataset_path,
        test_set_ratio,
        seed,
        target,
        dataset_name,
        normalize_targets,
    )

    if in_memory:
        data_list = []
        for network_id in tqdm(range(len(targets)), desc="Loading LLM brain data"):
            data = wrap_data(path, network_id, llm_layer, targets[network_id], network_density, from_sparse_data)
            data_list.append(data)

        train_dataset = [data_list[i] for i in train_data_split]
        test_dataset = [data_list[i] for i in test_data_split]
    else:
        train_dataset = BrainNetworkDataset(
            train_data_split,
            path,
            targets[train_data_split],
            llm_layer,
            network_density,
            from_sparse_data
        )
        test_dataset = BrainNetworkDataset(
            test_data_split,
            path,
            targets[test_data_split],
            llm_layer,
            network_density,
            from_sparse_data
        )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        **kwargs
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        **kwargs
    )

    return train_data_loader, test_data_loader


class BrainNetworkLinearDataset(TorchDataset):
    def __init__(self, sentences_id, dataset_path, targets, llm_layer, feature_name, feature_density=1.0):
        super().__init__()
        self.sentences_id = sentences_id
        self.dataset_path = dataset_path
        self.targets = targets
        self.llm_layer = llm_layer
        self.feature_name = feature_name
        self.feature_density = feature_density

        self.loaded_data = []
        for idx in tqdm(range(len(self.sentences_id)), desc="Loading linear data"):
            self.loaded_data.append(self._load_data(idx))

    def __len__(self):
        return len(self.sentences_id)

    def _load_data(self, idx):
        network_id = self.sentences_id[idx]
        feature_path = os.path.join(self.dataset_path, str(network_id), f"layer_{self.llm_layer}_{self.feature_name}.npy")
        feature = torch.from_numpy(np.load(feature_path)).float()
        if self.feature_name == "corr":
            triu_indices = torch.triu_indices(feature.shape[0], feature.shape[1], offset=1)
            feature = feature[triu_indices[0], triu_indices[1]]
        if self.feature_density < 1.0:
            threshold = torch.quantile(torch.abs(feature), 1 - self.feature_density)
            feature[torch.abs(feature) < threshold] = 0
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target

    def __getitem__(self, idx):
        return self.loaded_data[idx]


def get_brain_network_linear_dataloader(
    dataset_filename,
    feature_name,
    feature_density=1.0,
    llm_model_name="gpt2",
    ckpt_step=-1,
    llm_layer=0,
    dataset_path="data/graph_probing",
    batch_size=32,
    eval_batch_size=32,
    num_workers=4,
    prefetch_factor=2,
    test_set_ratio=0.2,
    shuffle=True,
    seed=42,
    target=["perplexities"],
    dataset_name="openwebtext",
    normalize_targets=True,
    **kwargs
):
    train_data_split, test_data_split, targets, path = prepare_data(
        dataset_filename,
        ckpt_step,
        llm_model_name,
        dataset_path,
        test_set_ratio,
        seed,
        target,
        dataset_name,
        normalize_targets,
    )

    train_dataset = BrainNetworkLinearDataset(
        train_data_split,
        path,
        targets[train_data_split],
        llm_layer,
        feature_name,
        feature_density=feature_density
    )
    test_dataset = BrainNetworkLinearDataset(
        test_data_split,
        path,
        targets[test_data_split],
        llm_layer,
        feature_name,
        feature_density=feature_density
    )

    train_data_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        **kwargs
    )
    test_data_loader = TorchDataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        **kwargs
    )

    return train_data_loader, test_data_loader
