from absl import app, flags, logging
import os
from setproctitle import setproctitle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)
from torch.utils.tensorboard import SummaryWriter

from graph_probing.dataset import get_brain_network_dataloader
from graph_probing.model import GCNRegressor, GCNRegressorLinear
from graph_probing.utils import llm_model_num_nodes_map, test_fn

flags.DEFINE_string("dataset_filename", "data/graph_probing/openwebtext-10k-gpt2.pkl", "The dataset filename.")
flags.DEFINE_float("network_density", 1.0, "The density of the network.")
flags.DEFINE_boolean("from_sparse_data", False, "Whether to use sparse data.")
flags.DEFINE_string("llm_model_name", "gpt2", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 0, "The layer of the LLM model.")
flags.DEFINE_integer("batch_size", 16, "The batch size.")
flags.DEFINE_integer("eval_batch_size", 16, "The evaluation batch size.")
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_integer("prefatch_factor", 4, "Prefetch factor.")
flags.DEFINE_boolean("linear_probing", False, "Whether to use linear probing.")
flags.DEFINE_integer("num_channels", 32, "The number of channels in GNN probes.")
flags.DEFINE_integer("num_layers", 1, "The number of GNN layers.")
flags.DEFINE_float("dropout", 0.0, "The dropout rate.")
flags.DEFINE_float("lr", 0.001, "The learning rate.")
flags.DEFINE_float("weight_decay", 1e-5, "The weight decay.")
flags.DEFINE_integer("num_epochs", 100, "The number of epochs.")
flags.DEFINE_float("test_set_ratio", 0.2, "The ratio of the test set.")
flags.DEFINE_boolean("in_memory", True, "In-memory dataset.")
flags.DEFINE_integer("early_stop_patience", 20, "The patience for early stopping.")
flags.DEFINE_integer("gpu_id", 7, "The GPU ID.")
flags.DEFINE_boolean("resume", False, "Whether to resume training from the best model.")
FLAGS = flags.FLAGS


def train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_name, device):
    test_mse, test_mae, test_r2, test_pearsonr, test_spearmanr = test_fn(model, test_data_loader, device)
    torch.cuda.empty_cache()
    logging.info(f"Initial Test MSE: {test_mse:.4f}")
    logging.info(f"Initial Test MAE: {test_mae:.4f}")
    logging.info(f"Initial Test R2: {test_r2:.4f}")
    logging.info(f"Initial Test Pearsonr: {test_pearsonr:.4f}")
    logging.info(f"Initial Test Spearmanr: {test_spearmanr:.4f}")
    writer.add_scalar("test/mse", test_mse, 0)
    writer.add_scalar("test/mae", test_mae, 0)
    writer.add_scalar("test/r2", test_r2, 0)
    writer.add_scalar("test/pearsonr", test_pearsonr, 0)
    writer.add_scalar("test/spearmanr", test_spearmanr, 0)

    model_save_path = os.path.join(
        f"saves/{save_model_name}/layer_{FLAGS.llm_layer}", 
        f"best_model_density-{FLAGS.network_density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}.pth"
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    best_test_mse = test_mse
    best_test_mae = test_mae
    best_test_r2 = test_r2
    best_test_pearsonr = test_pearsonr
    best_test_spearmanr = test_spearmanr
    best_epoch = 0
    epochs_no_improve = 0
    num_epochs = FLAGS.num_epochs
    for epoch in tqdm(range(num_epochs), position=0, desc="Training"):
        model.train()
        total_loss = 0.0
        num_graphs = 0
        for data in tqdm(train_data_loader, position=1, desc=f"Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            pred = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.batch.to(device))
            loss = F.mse_loss(pred, data.y.to(device)[:,None], reduction="mean")
            loss.backward()
            optimizer.step()
            num_graphs += data.num_graphs
            total_loss += loss.item()*data.num_graphs
        avg_loss = total_loss / num_graphs
        logging.info(f"Epoch {epoch + 1}, MSE Loss: {avg_loss:.4f}")
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        torch.cuda.empty_cache()

        test_mse, test_mae, test_r2, test_pearsonr, test_spearmanr = test_fn(model, test_data_loader, device)
        torch.cuda.empty_cache()
        logging.info(f"Test MSE: {test_mse:.4f}")
        logging.info(f"Test MAE: {test_mae:.4f}")
        logging.info(f"Test R2: {test_r2:.4f}")
        logging.info(f"Test Pearsonr: {test_pearsonr:.4f}")
        logging.info(f"Test Spearmanr: {test_spearmanr:.4f}")
        writer.add_scalar("test/mse", test_mse, epoch + 1)
        writer.add_scalar("test/mae", test_mae, epoch + 1)
        writer.add_scalar("test/r2", test_r2, epoch + 1)
        writer.add_scalar("test/pearsonr", test_pearsonr, epoch + 1)
        writer.add_scalar("test/spearmanr", test_spearmanr, epoch + 1)
        scheduler.step(test_mse)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_test_mae = test_mae
            best_test_r2 = test_r2
            best_test_pearsonr = test_pearsonr
            best_test_spearmanr = test_spearmanr
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= FLAGS.early_stop_patience:
                break

    logging.info(f"Best Epoch: {best_epoch}")
    logging.info(f"Best Test MSE: {best_test_mse:.4f}")
    logging.info(f"Best Test MAE: {best_test_mae:.4f}")
    logging.info(f"Best Test R2: {best_test_r2:.4f}")
    logging.info(f"Best Test Pearsonr: {best_test_pearsonr:.4f}")
    logging.info(f"Best Test Spearmanr: {best_test_spearmanr:.4f}")

    writer.add_text(
        "best_record",
        f"Best Epoch: {best_epoch}, "
        f"Best Test MSE: {best_test_mse:.4f}, "
        f"Best Test MAE: {best_test_mae:.4f}, "
        f"Best Test R2: {best_test_r2:.4f}, "
        f"Best Test Pearsonr: {best_test_pearsonr:.4f}, "
        f"Best Test Spearmanr: {best_test_spearmanr:.4f}",
        0
    )


def main(_):

    device = torch.device(f"cuda:{FLAGS.gpu_id}")

    if FLAGS.ckpt_step == -1:
        save_model_name = f"{FLAGS.llm_model_name}"
    else:
        save_model_name = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}"

    train_data_loader, test_data_loader = get_brain_network_dataloader(
        FLAGS.dataset_filename,
        network_density=FLAGS.network_density,
        from_sparse_data=FLAGS.from_sparse_data,
        llm_model_name=FLAGS.llm_model_name,
        ckpt_step=FLAGS.ckpt_step,
        llm_layer=FLAGS.llm_layer,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        num_workers=FLAGS.num_workers,
        prefatch_factor=FLAGS.prefatch_factor,
        test_set_ratio=FLAGS.test_set_ratio,
        in_memory=FLAGS.in_memory,
    )

    if not FLAGS.linear_probing:
        model = GCNRegressor(
            num_nodes=llm_model_num_nodes_map[FLAGS.llm_model_name],
            hidden_channels=FLAGS.num_channels,
            out_channels=FLAGS.num_channels,
            num_layers=FLAGS.num_layers,
            dropout=FLAGS.dropout,
        ).to(device)
    else:
        model = GCNRegressorLinear(
            num_nodes=llm_model_num_nodes_map[FLAGS.llm_model_name],
            hidden_channels=FLAGS.num_channels,
            out_channels=FLAGS.num_channels,
            num_layers=FLAGS.num_layers,
            dropout=FLAGS.dropout,
        ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    writer = SummaryWriter(log_dir=f"runs/{save_model_name}/layer_{FLAGS.llm_layer}")
    writer.add_hparams(
        {
            "linear_probing": FLAGS.linear_probing,
            "hidden_channels": FLAGS.num_channels, "out_channels": FLAGS.num_channels,
            "num_layers": FLAGS.num_layers, "dropout": FLAGS.dropout,
            "batch_size": FLAGS.batch_size, "lr": FLAGS.lr, "weight_decay": FLAGS.weight_decay
        },
        {"hparam/placeholder": 0}
    )

    if FLAGS.resume:
        model_save_path = os.path.join(
            f"saves/{save_model_name}/layer_{FLAGS.llm_layer}",
            f"best_model_density-{FLAGS.network_density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}.pth"
        )
        model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_name, device)


if __name__ == "__main__":
    setproctitle("llm graph probing")
    app.run(main)
