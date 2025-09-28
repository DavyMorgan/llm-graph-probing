import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse


def wrap_data(path_1, path_2, network_id, llm_layer_1, llm_layer_2, network_density):
    percentile_threshold = network_density * 100

    network_1 = np.load(os.path.join(path_1, f"{network_id}/layer_{llm_layer_1}_corr.npy")).astype(np.float32)
    threshold_1 = np.percentile(np.abs(network_1), 100 - percentile_threshold)
    network_1[np.abs(network_1) < threshold_1] = 0
    np.fill_diagonal(network_1, 1.0)
    network_1 = torch.from_numpy(network_1)
    network_2 = np.load(os.path.join(path_2, f"{network_id}/layer_{llm_layer_2}_corr.npy")).astype(np.float32)
    threshold_2 = np.percentile(np.abs(network_2), 100 - percentile_threshold)
    network_2[np.abs(network_2) < threshold_2] = 0
    np.fill_diagonal(network_2, 1.0)
    network_2 = torch.from_numpy(network_2)

    edge_index_llm_1, edge_attr_llm_1 = dense_to_sparse(network_1)
    edge_index_llm_2, edge_attr_llm_2 = dense_to_sparse(network_2)
    data = BrainNetworkPairData(
        x_llm_1=torch.arange(network_1.shape[0]),
        edge_index_llm_1=edge_index_llm_1,
        edge_attr_llm_1=edge_attr_llm_1,
        x_llm_2=torch.arange(network_2.shape[0]),
        edge_index_llm_2=edge_index_llm_2,
        edge_attr_llm_2=edge_attr_llm_2,
    )
    return data


class BrainNetworkPairDataset(Dataset):
    def __init__(self, sentences_id, dataset_path_1, dataset_path_2, llm_layer_1, llm_layer_2, network_density):
        super().__init__(None, transform=None, pre_transform=None)
        self.sentences_id = sentences_id
        self.dataset_path_1 = dataset_path_1
        self.dataset_path_2 = dataset_path_2
        self.llm_layer_1 = llm_layer_1
        self.llm_layer_2 = llm_layer_2
        self.network_density = network_density

    def len(self):
        return len(self.sentences_id)

    def get(self, idx):
        network_id = self.sentences_id[idx]
        data = wrap_data(self.dataset_path_1, self.dataset_path_2, network_id, self.llm_layer_1, self.llm_layer_2, self.network_density)
        return data


class BrainNetworkPairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_llm_1':
            return self.x_llm_1.size(0)
        if key == 'edge_index_llm_2':
            return self.x_llm_2.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def get_brain_network_pair_dataloader(
    dataset_filename,
    network_density=1.0,
    llm_model_name_1="gpt2",
    ckpt_step_1=-1,
    llm_layer_1=7,
    llm_model_name_2="pythia-160m",
    ckpt_step_2=-1,
    llm_layer_2=7,
    dataset_path="data/graph_matching",
    batch_size=32,
    eval_batch_size=32,
    num_workers=0,
    prefetch_factor=2,
    test_set_ratio=0.2,
    in_memory=True,
    shuffle=True,
    **kwargs
):
    data = pd.read_csv(dataset_filename)
    sentences = data["sentences"].to_list()
    num_sentences = len(sentences)

    if ckpt_step_1 == -1:
        path_1 = os.path.join(dataset_path, llm_model_name_1)
    else:
        path_1 = os.path.join(dataset_path, f"{llm_model_name_1}_step{ckpt_step_1}")
    if ckpt_step_2 == -1:
        path_2 = os.path.join(dataset_path, llm_model_name_2)
    else:
        path_2 = os.path.join(dataset_path, f"{llm_model_name_2}_step{ckpt_step_2}")

    generator = torch.Generator().manual_seed(42)
    test_set_size = int(num_sentences * test_set_ratio)
    train_data_split, test_data_split = torch.utils.data.random_split(
        list(range(num_sentences)), [num_sentences - test_set_size, test_set_size], generator=generator)

    if in_memory:
        data_list = []
        for network_id in tqdm(range(num_sentences), desc="Loading LLM brain data"):
            data = wrap_data(path_1, path_2, network_id, llm_layer_1, llm_layer_2, network_density)
            data_list.append(data)

        train_dataset = [data_list[i] for i in train_data_split]
        test_dataset = [data_list[i] for i in test_data_split]
    else:
        train_dataset = BrainNetworkPairDataset(
            train_data_split,
            path_1,
            path_2,
            llm_layer_1,
            llm_layer_2,
            network_density,
        )
        test_dataset = BrainNetworkPairDataset(
            test_data_split,
            path_1,
            path_2,
            llm_layer_1,
            llm_layer_2,
            network_density,
        )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=['x_llm_1', 'x_llm_2'],
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        **kwargs
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        follow_batch=['x_llm_1', 'x_llm_2'],
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        **kwargs
    )

    return train_data_loader, test_data_loader
