from absl import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F
from scipy import stats
import math


def haversine_distance(pred, target):
    R = 6371  # Earth radius in km
    
    # Convert degrees to radians
    lat1, lon1 = np.radians(pred[:, 0]), np.radians(pred[:, 1])
    lat2, lon2 = np.radians(target[:, 0]), np.radians(target[:, 1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    return distance.mean()


def haversine_r2(actual_coords, predicted_coords):
    # Step 1: Calculate the mean coordinates
    mean_coords = np.mean(actual_coords, axis=0)

    # Step 2: Calculate SS_tot and SS_res
    ss_tot = np.sum(haversine_distance(actual_coords, mean_coords[None, :])**2)
    ss_res = np.sum(haversine_distance(actual_coords, predicted_coords)**2)

    # Step 3: Calculate R^2
    r2 = 1 - (ss_res / ss_tot)

    return r2


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


def test_fn_space(model, test_data_loader, device, num_layers):
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
                target = data.y.to(device)
                num_graphs += data.num_graphs
            else:
                activation, target = data
                pred = model(activation.to(device))
                target = target.to(device)
                num_graphs += activation.shape[0]

            total_mse += F.mse_loss(pred, target, reduction="sum").item()
            total_mae += F.l1_loss(pred, target, reduction="sum").item()
            all_pred.append(pred.cpu().detach().numpy())
            all_y.append(target.cpu().detach().numpy())

        all_pred = np.concatenate(all_pred)
        all_y = np.concatenate(all_y)
        hav_dist = haversine_distance(all_pred, all_y)
        hav_mse = np.mean(hav_dist**2)
        hav_mae = np.mean(np.abs(hav_dist))
        hav_r2 = haversine_r2(all_pred, all_y)

    return hav_mse, hav_mae, hav_r2, all_y, all_pred


def eval_model(model, test_data_loader, device, num_layers):
    mse, mae, r2, pearsonr, spearmanr, all_y, all_pred = test_fn(model, test_data_loader, device, num_layers)
    torch.cuda.empty_cache()
    logging.info(f"Test MSE: {mse:.4f}")
    logging.info(f"Test MAE: {mae:.4f}")
    logging.info(f"Test R2: {r2:.4f}")
    logging.info(f"Test Pearsonr: {pearsonr:.4f}")
    logging.info(f"Test Spearmanr: {spearmanr:.4f}")

    return all_y, all_pred


def eval_model_space(model, test_data_loader, device, num_layers):
    hav_mse, hav_mae, hav_r2, all_y, all_pred = test_fn_space(model, test_data_loader, device, num_layers)
    torch.cuda.empty_cache()
    logging.info(f"Test Haversine MSE: {hav_mse:.4f}")
    logging.info(f"Test Haversine MAE: {hav_mae:.4f}")
    logging.info(f"Test Haversine R2: {hav_r2:.4f}")

    return all_y, all_pred


