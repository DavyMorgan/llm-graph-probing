from absl import app, flags, logging
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)
from torch.utils.tensorboard import SummaryWriter

from graph_probing.dataset import get_brain_network_dataloader, get_brain_network_linear_dataloader
from graph_probing.utils import test_fn, test_fn_space

from utils.constants import hf_model_name_map
from utils.model_utils import get_num_nodes
from utils.probing_model import GCNProbe as GCNRegressor, MLPProbe as MLPRegressor

flags.DEFINE_string("dataset", "openwebtext", "The name of the dataset.")
flags.DEFINE_string("probe_input", "activation", "The input type for probing: activation, activation_avg, or corr.")
flags.DEFINE_float("density", 1.0, "The density of the input.")
flags.DEFINE_boolean("from_sparse_data", False, "Whether to use sparse data.")
flags.DEFINE_string("llm_model_name", "gpt2", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 0, "The layer of the LLM model.")
flags.DEFINE_integer("batch_size", 16, "The batch size.")
flags.DEFINE_integer("eval_batch_size", 16, "The evaluation batch size.")
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_integer("prefetch_factor", 4, "Prefetch factor.")
flags.DEFINE_boolean("nonlinear_activation", True, "Whether to use nonlinear activation.")
flags.DEFINE_integer("num_channels", 32, "The number of channels in GNN probes.")
flags.DEFINE_integer("num_layers", 1, "The number of GNN layers.")
flags.DEFINE_float("dropout", 0.0, "The dropout rate.")
flags.DEFINE_float("lr", 0.001, "The learning rate.")
flags.DEFINE_float("weight_decay", 1e-5, "The weight decay.")
flags.DEFINE_integer("num_epochs", 100, "The number of epochs.")
flags.DEFINE_float("test_set_ratio", 0.2, "The ratio of the test set.")
flags.DEFINE_boolean("in_memory", True, "In-memory dataset.")
flags.DEFINE_integer("early_stop_patience", 20, "The patience for early stopping.")
flags.DEFINE_integer("gpu_id", 0, "The GPU ID.")
flags.DEFINE_boolean("resume", False, "Whether to resume training from the best model.")
flags.DEFINE_integer("seed", 42, "The random seed.")
FLAGS = flags.FLAGS


def train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_name, device):
    model_save_path = os.path.join(
        f"saves/graph_probing/{save_model_name}/layer_{FLAGS.llm_layer}",
        f"best_model_density-{FLAGS.density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    best_metrics = {
        "mse": float("inf"),
        "mae": float("inf"),
        "r2": float("-inf"),
        "pearsonr": float("-inf"),
        "spearmanr": float("-inf"),
    }
    if FLAGS.dataset == "world_place":
        mse, mae, haversine, _, _ = test_fn_space(model, test_data_loader, device, num_layers=FLAGS.num_layers)
        torch.cuda.empty_cache()
        for metric, value in zip(["mse", "mae", "haversine"], [mse, mae, haversine]):
            logging.info(f"Initial Test {metric.capitalize()}: {value:.4f}")
            writer.add_scalar(f"test/{metric}", value, 0)
            best_metrics[metric] = value
    else:
        mse, mae, r2, pearsonr, spearmanr, _, _ = test_fn(model, test_data_loader, device, num_layers=FLAGS.num_layers)
        torch.cuda.empty_cache()
        for metric, value in zip(["mse", "mae", "r2", "pearsonr", "spearmanr"], [mse, mae, r2, pearsonr, spearmanr]):
            logging.info(f"Initial Test {metric.capitalize()}: {value:.4f}")
            writer.add_scalar(f"test/{metric}", value, 0)
            best_metrics[metric] = value
    best_metrics["epoch"] = 0
    epochs_no_improve = 0

    def get_pred_target_graph(data):
        pred = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.batch.to(device))
        target = data.y.to(device).squeeze(-1)
        batch_size = data.num_graphs
        return pred, target, batch_size

    def get_pred_target_linear(data):
        feature, target = data
        feature, target = feature.to(device), target.to(device).squeeze(-1)
        pred = model(feature)
        return pred, target, feature.shape[0]

    if FLAGS.num_layers > 0:
        get_pred_target = get_pred_target_graph
    else:
        get_pred_target = get_pred_target_linear

    num_epochs = FLAGS.num_epochs
    for epoch in tqdm(range(num_epochs), position=0, desc="Training"):
        model.train()
        total_loss = 0.0
        num_graphs = 0
        for data in tqdm(train_data_loader, position=1, desc=f"Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            pred, target, batch_size = get_pred_target(data)
            loss = F.mse_loss(pred, target, reduction="mean")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
            num_graphs += batch_size
        avg_loss = total_loss / num_graphs
        logging.info(f"Epoch {epoch + 1}, MSE Loss: {avg_loss:.4f}")
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        torch.cuda.empty_cache()

        if FLAGS.dataset == "world_place":
            mse, mae, haversine, _, _ = test_fn_space(model, test_data_loader, device, num_layers=FLAGS.num_layers)
            torch.cuda.empty_cache()
            for metric, value in zip(["mse", "mae", "haversine"], [mse, mae, haversine]):
                logging.info(f"Test {metric.capitalize()}: {value:.4f}")
                writer.add_scalar(f"test/{metric}", value, epoch + 1)
            scheduler.step(mse)

            if mse < best_metrics["mse"]:
                for metric, value in zip(["mse", "mae", "haversine"], [mse, mae, haversine]):
                    best_metrics[metric] = value
                best_metrics["epoch"] = epoch + 1
                torch.save(model.state_dict(), model_save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= FLAGS.early_stop_patience:
                    break
        else:
            mse, mae, r2, pearsonr, spearmanr, _, _ = test_fn(model, test_data_loader, device, num_layers=FLAGS.num_layers)
            torch.cuda.empty_cache()
            for metric, value in zip(["mse", "mae", "r2", "pearsonr", "spearmanr"], [mse, mae, r2, pearsonr, spearmanr]):
                logging.info(f"Test {metric.capitalize()}: {value:.4f}")
                writer.add_scalar(f"test/{metric}", value, epoch + 1)
            scheduler.step(mse)

            if mse < best_metrics["mse"]:
                for metric, value in zip(["mse", "mae", "r2", "pearsonr", "spearmanr"], [mse, mae, r2, pearsonr, spearmanr]):
                    best_metrics[metric] = value
                best_metrics["epoch"] = epoch + 1
                torch.save(model.state_dict(), model_save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= FLAGS.early_stop_patience:
                    break

    for key, value in best_metrics.items():
        logging.info(f"Best Test {key.capitalize()}: {value:.4f}")
    writer.add_text("best_record", " ".join([f"Best Test {key.capitalize()}: {value:.4f}" for key, value in best_metrics.items()]), 0)


def main(_):

    if FLAGS.gpu_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{FLAGS.gpu_id}")

    if FLAGS.ckpt_step == -1:
        save_model_name = f"{FLAGS.llm_model_name}/{FLAGS.dataset}"
    else:
        save_model_name = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}/{FLAGS.dataset}"

    hf_model_name = hf_model_name_map[FLAGS.llm_model_name]
    if FLAGS.dataset == "art":
        dataset_filename = "st_data/art.csv"
        target = "decade"
        num_output = 1
        normalize_targets = True
    elif FLAGS.dataset == "world_place":
        dataset_filename = "st_data/world_place.csv"
        target = ["latitude", "longitude"]
        num_output = 2
        normalize_targets = False
    else:
        revision = "main" if FLAGS.ckpt_step == -1 else f"step{FLAGS.ckpt_step}"
        if hf_model_name.startswith("EleutherAI") and revision != "main":
            dataset_filename = f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}-{revision}.csv"
        else:
            dataset_filename = f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}.csv"
        target = "perplexities"
        num_output = 1
        normalize_targets = True


    if FLAGS.num_layers > 0:
        train_data_loader, test_data_loader = get_brain_network_dataloader(
            dataset_filename,
            network_density=FLAGS.density,
            from_sparse_data=FLAGS.from_sparse_data,
            llm_model_name=FLAGS.llm_model_name,
            ckpt_step=FLAGS.ckpt_step,
            llm_layer=FLAGS.llm_layer,
            batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            num_workers=FLAGS.num_workers,
            prefetch_factor=FLAGS.prefetch_factor,
            test_set_ratio=FLAGS.test_set_ratio,
            in_memory=FLAGS.in_memory,
            seed=FLAGS.seed,
            target=target,
            dataset_name=FLAGS.dataset,
            normalize_targets=normalize_targets,
        )
        model = GCNRegressor(
            num_nodes=get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer),
            hidden_channels=FLAGS.num_channels,
            num_layers=FLAGS.num_layers,
            nonlinear_activation=FLAGS.nonlinear_activation,
            dropout=FLAGS.dropout,
            num_output=num_output,
        ).to(device)
    else:
        train_data_loader, test_data_loader = get_brain_network_linear_dataloader(
            dataset_filename,
            FLAGS.probe_input,
            feature_density=FLAGS.density,
            llm_model_name=FLAGS.llm_model_name,
            ckpt_step=FLAGS.ckpt_step,
            llm_layer=FLAGS.llm_layer,
            batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            num_workers=FLAGS.num_workers,
            prefetch_factor=FLAGS.prefetch_factor,
            test_set_ratio=FLAGS.test_set_ratio,
            seed=FLAGS.seed,
            target=target,
            dataset_name=FLAGS.dataset,
            normalize_targets=normalize_targets,
        )
        model = MLPRegressor(
            num_nodes=get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer, FLAGS.probe_input),
            hidden_channels=FLAGS.num_channels,
            num_layers=-FLAGS.num_layers,
            num_output=num_output,
        ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-10)
    writer = SummaryWriter(log_dir=f"runs/{save_model_name}/layer_{FLAGS.llm_layer}")
    writer.add_hparams(
        {
            "nonlinear_activation": FLAGS.nonlinear_activation,
            "hidden_channels": FLAGS.num_channels, "out_channels": FLAGS.num_channels,
            "num_layers": FLAGS.num_layers, "dropout": FLAGS.dropout,
            "batch_size": FLAGS.batch_size, "lr": FLAGS.lr, "weight_decay": FLAGS.weight_decay
        },
        {"hparam/placeholder": 0}
    )

    if FLAGS.resume:
        model_save_path = os.path.join(
            f"saves/graph_probing/{save_model_name}/layer_{FLAGS.llm_layer}",
            f"best_model_density-{FLAGS.density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
        )
        model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_name, device)


if __name__ == "__main__":
    app.run(main)
