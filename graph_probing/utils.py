from absl import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F
from scipy import stats


hf_model_name_map = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
}


llm_model_num_nodes_map = {
    "gpt2": 768,
    "gpt2-medium": 1024,
    "gpt2-large": 1280,
    "pythia-160m": 768,
    "pythia-410m": 1024,
    "pythia-1.4b": 2048,
    "pythia-2.8b": 2560,
    "pythia-6.9b": 4096,
    "pythia-12b": 5120,
    "qwen2.5-0.5b": 896,
    "qwen2.5-3b": 2048,
    "qwen2.5-7b": 3584,
    "qwen2.5-14b": 5120,
}


def test_fn(model, test_data_loader, device, return_raw_data=False):
    model.eval()
    with torch.no_grad():
        total_mse = 0.0
        total_mae = 0.0
        num_graphs = 0
        all_pred = []
        all_y = []
        for data in tqdm(test_data_loader, desc="Testing", leave=False):
            pred = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.batch.to(device))
            total_mse += F.mse_loss(pred, data.y.to(device)[:,None], reduction="sum").item()
            total_mae += F.l1_loss(pred, data.y.to(device)[:,None], reduction="sum").item()
            num_graphs += data.num_graphs
            all_pred.append(pred.cpu().detach().numpy())
            all_y.append(data.y.cpu().detach().numpy())
        test_mse = total_mse / num_graphs
        test_mae = total_mae / num_graphs
        all_pred = np.concatenate(all_pred).flatten()
        all_y = np.concatenate(all_y)
        test_r2 = r2_score(all_y, all_pred)
        test_pearsonr = stats.pearsonr(all_y, all_pred).statistic
        test_spearmanr = stats.spearmanr(all_y, all_pred).statistic

    if not return_raw_data:
        return test_mse, test_mae, test_r2, test_pearsonr, test_spearmanr
    else:
        return test_mse, test_mae, test_r2, test_pearsonr, test_spearmanr, all_y, all_pred


def eval_model(model, test_data_loader, device):
    test_mse, test_mae, test_r2, test_pearsonr, test_spearmanr, all_y, all_pred = test_fn(model, test_data_loader, device, return_raw_data=True)
    torch.cuda.empty_cache()
    logging.info(f"Test MSE: {test_mse:.4f}")
    logging.info(f"Test MAE: {test_mae:.4f}")
    logging.info(f"Test R2: {test_r2:.4f}")
    logging.info(f"Test Pearsonr: {test_pearsonr:.4f}")
    logging.info(f"Test Spearmanr: {test_spearmanr:.4f}")

    return all_y, all_pred


