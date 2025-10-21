import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse


def prepare_data(dataset_filename, test_set_ratio, seed):
    data = pd.read_csv(dataset_filename)
    indices = list(range(len(data["question"])))

    test_size = int(len(indices) * test_set_ratio)
    train_size = len(indices) - test_size
    train_data_split, test_data_split = random_split(indices, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    return train_data_split, test_data_split


class TruthfulQADataset(Dataset):
    def __init__(self, llm_model_name, ckpt_step, llm_layer, network_density, indices, from_sparse_data=False, in_memory=True, dataset_name="truthfulqa"):
        self.llm_model_name = llm_model_name
        self.ckpt_step = ckpt_step
        self.llm_layer = llm_layer
        if self.llm_layer == -1:
            self.dense_filename = "layer_average_corr.npy"
        else:
            self.dense_filename = f"layer_{self.llm_layer}_corr.npy"
        self.network_density = network_density
        self.network_indices = indices
        self.from_sparse_data = from_sparse_data
        self.in_memory = in_memory
        self.dataset_name = dataset_name

        if self.ckpt_step == -1:
            model_dir = self.llm_model_name
        else:
            model_dir = f"{self.llm_model_name}_step{self.ckpt_step}"
        self.data_dir = os.path.join("data/hallucination", self.dataset_name, model_dir)

        if self.in_memory:
            self.loaded_data = []
            for idx in tqdm(range(len(self.network_indices))):
                self.loaded_data.append(self._load_data(idx))

    def __len__(self):
        return len(self.network_indices)

    def _load_data(self, idx):
        question_idx = self.network_indices[idx]
        data_path = os.path.join(self.data_dir, str(question_idx))

        if not self.from_sparse_data:
            adj = np.load(os.path.join(data_path, self.dense_filename))
            percentile_threshold = self.network_density * 100
            threshold = np.percentile(np.abs(adj), 100 - percentile_threshold)
            adj[np.abs(adj) < threshold] = 0
            np.fill_diagonal(adj, 0)
            adj = torch.from_numpy(adj).float()
            edge_index, edge_attr = dense_to_sparse(adj)
            num_nodes = adj.shape[0]
        else:
            edge_index = torch.from_numpy(np.load(os.path.join(data_path, f"layer_{self.llm_layer}_sparse_{self.network_density}_edge_index.npy"))).long()
            edge_attr = torch.from_numpy(np.load(os.path.join(data_path, f"layer_{self.llm_layer}_sparse_{self.network_density}_edge_attr.npy"))).float()
            num_nodes = edge_index.max().item() + 1

        y = torch.from_numpy(np.load(os.path.join(data_path, "label.npy"))).long()
        x = torch.arange(num_nodes)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def __getitem__(self, idx):
        if self.in_memory:
            return self.loaded_data[idx]
        else:
            return self._load_data(idx)


def get_truthfulqa_dataloader(dataset_name, llm_model_name, ckpt_step, llm_layer, network_density, from_sparse_data, batch_size, eval_batch_size, num_workers, prefetch_factor, test_set_ratio, in_memory, seed):

    dataset_filename = f"data/hallucination/{dataset_name}.csv"
    train_data_split, test_data_split = prepare_data(dataset_filename, test_set_ratio, seed)

    train_dataset = TruthfulQADataset(llm_model_name, ckpt_step, llm_layer, network_density, train_data_split, from_sparse_data, in_memory, dataset_name=dataset_name)
    test_dataset = TruthfulQADataset(llm_model_name, ckpt_step, llm_layer, network_density, test_data_split, from_sparse_data, in_memory, dataset_name=dataset_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor)

    return train_loader, test_loader


class TruthfulQALinearDataset(TorchDataset):
    def __init__(self, llm_model_name, ckpt_step, llm_layer, feature_name, indices, feature_density=1.0, dataset_name="truthfulqa"):
        self.llm_model_name = llm_model_name
        self.ckpt_step = ckpt_step
        self.llm_layer = llm_layer
        self.feature_name = feature_name
        if self.feature_name in {"word2vec_average", "word2vec_token_count"}:
            self.dense_filename = f"{self.feature_name}.npy"
        elif self.llm_layer == -1:
            self.dense_filename = f"layer_average_{self.feature_name}.npy"
        else:
            self.dense_filename = f"layer_{self.llm_layer}_{self.feature_name}.npy"
        self.network_indices = indices
        self.feature_density = feature_density
        self.dataset_name = dataset_name

        if self.ckpt_step == -1:
            model_dir = self.llm_model_name
        else:
            model_dir = f"{self.llm_model_name}_step{self.ckpt_step}"
        self.data_dir = os.path.join("data/hallucination", self.dataset_name, model_dir)

        self.loaded_data = []
        for idx in tqdm(range(len(self.network_indices))):
            self.loaded_data.append(self._load_data(idx))

    def __len__(self):
        return len(self.network_indices)

    def _load_data(self, idx):
        question_idx = self.network_indices[idx]
        data_path = os.path.join(self.data_dir, str(question_idx))
        feature = np.load(os.path.join(data_path, self.dense_filename)).astype(np.float32)
        feature = torch.from_numpy(feature)
        if self.feature_name == "corr":
            triu_indices = torch.triu_indices(feature.shape[0], feature.shape[1], offset=1)
            feature = feature[triu_indices[0], triu_indices[1]]
        
        # Apply feature sparsification
        if self.feature_density < 1.0:
            threshold = torch.quantile(torch.abs(feature), 1 - self.feature_density)
            feature[torch.abs(feature) < threshold] = 0
            
        y = torch.from_numpy(np.load(os.path.join(data_path, "label.npy"))).long()

        return (feature, y)

    def __getitem__(self, idx):
        return self.loaded_data[idx]


def get_truthfulqa_linear_dataloader(feature_name, dataset_name, llm_model_name, ckpt_step, llm_layer, batch_size, eval_batch_size, num_workers, prefetch_factor, test_set_ratio=0.2, shuffle=True, seed=42, feature_density=1.0):

    dataset_filename = f"data/hallucination/{dataset_name}.csv"
    train_data_split, test_data_split = prepare_data(dataset_filename, test_set_ratio, seed)

    train_dataset = TruthfulQALinearDataset(llm_model_name, ckpt_step, llm_layer, feature_name, train_data_split, feature_density, dataset_name=dataset_name)
    test_dataset = TruthfulQALinearDataset(llm_model_name, ckpt_step, llm_layer, feature_name, test_data_split, feature_density, dataset_name=dataset_name)

    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, prefetch_factor=prefetch_factor)
    test_loader = TorchDataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor)

    return train_loader, test_loader
