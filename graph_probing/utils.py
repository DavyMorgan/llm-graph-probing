from absl import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F
from scipy import stats


def test_fn(model, test_data_loader, device, num_layers):
    model.eval()
    with torch.no_grad():
        total_mse = 0.0
        total_mae = 0.0
        num_graphs = 0
        all_pred = []
        all_y = []
        for data in tqdm(test_data_loader, desc="Testing", leave=False):
            if num_layers > 0:
                pred = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.batch.to(device))
                target = data.y.to(device).squeeze(-1)
                num_graphs += data.num_graphs
            else:
                activation, target = data
                pred = model(activation.to(device))
                target = target.to(device).squeeze(-1)
                num_graphs += activation.shape[0]

            total_mse += F.mse_loss(pred, target, reduction="sum").item()
            total_mae += F.l1_loss(pred, target, reduction="sum").item()
            all_pred.append(pred.cpu().detach().numpy())
            all_y.append(target.cpu().detach().numpy())

        all_pred = np.concatenate(all_pred).flatten()
        all_y = np.concatenate(all_y).flatten()
        mse = total_mse / num_graphs
        mae = total_mae / num_graphs
        r2 = r2_score(all_y, all_pred)
        pearsonr = stats.pearsonr(all_y, all_pred).statistic
        spearmanr = stats.spearmanr(all_y, all_pred).statistic

    return mse, mae, r2, pearsonr, spearmanr, all_y, all_pred


def eval_model(model, test_data_loader, device, num_layers):
    mse, mae, r2, pearsonr, spearmanr, all_y, all_pred = test_fn(model, test_data_loader, device, num_layers)
    torch.cuda.empty_cache()
    logging.info(f"Test MSE: {mse:.4f}")
    logging.info(f"Test MAE: {mae:.4f}")
    logging.info(f"Test R2: {r2:.4f}")
    logging.info(f"Test Pearsonr: {pearsonr:.4f}")
    logging.info(f"Test Spearmanr: {spearmanr:.4f}")

    return all_y, all_pred


