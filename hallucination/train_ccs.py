from absl import app, flags, logging
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from hallucination.dataset import get_truthfulqa_ccs_dataloader
from hallucination.utils import test_fn_ccs
from utils.probing_model import MLPProbe as MLPClassifier
from utils.model_utils import get_num_nodes

flags.DEFINE_enum("dataset_name", "truthfulqa", ["truthfulqa", "halueval", "medhallu", "helm"], "Name of the dataset.")
flags.DEFINE_float("density", 1.0, "The density of the network/features.")
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 12, "The layer of the LLM model.")
flags.DEFINE_integer("batch_size", 16, "The batch size.")
flags.DEFINE_integer("eval_batch_size", 16, "The evaluation batch size.")
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_integer("prefetch_factor", 4, "Prefetch factor.")
flags.DEFINE_integer("hidden_channels", 32, "The hidden channels.")
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
    accuracy, precision, recall, f1, cm = test_fn_ccs(model, test_data_loader, device)
    torch.cuda.empty_cache()
    logging.info(f"Initial Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    for metric, value in zip(["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]):
        writer.add_scalar(f"test/{metric}", value, 0)

    model_save_path = os.path.join(
        f"saves/{save_model_name}/layer_{FLAGS.llm_layer}",
        f"best_model_density-{FLAGS.density}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-ccs.pth"
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    best_metrics = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "cm": cm.copy(),
        "epoch": 0
    }
    epochs_no_improve = 0
    
    def get_loss_batch_size_ccs(data):
        x_yes, x_no, y = data
        x_yes, x_no, y = x_yes.to(device), x_no.to(device), y.to(device)
        out_yes = model.forward(x_yes)
        out_no = model.forward(x_no)
        informative_loss = (torch.min(out_yes, out_no)**2).mean()
        consistent_loss = ((out_yes - (1 - out_no))**2).mean()
        loss = informative_loss + consistent_loss
        batch_size = x_yes.shape[0]
        return loss, batch_size

    for epoch in tqdm(range(FLAGS.num_epochs), position=0, desc="Training"):
        model.train()
        total_loss = 0.0
        num_graphs = 0
        for data in tqdm(train_data_loader, position=1, desc=f"Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            batch_size = 0
            loss, batch_size = get_loss_batch_size_ccs(data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
            num_graphs += batch_size
        
        avg_loss = total_loss / num_graphs
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        torch.cuda.empty_cache()

        accuracy, precision, recall, f1, cm = test_fn_ccs(model, test_data_loader, device)
        torch.cuda.empty_cache()
        logging.info(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")
        for metric, value in zip(["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]):
            writer.add_scalar(f"test/{metric}", value, epoch + 1)
        scheduler.step(accuracy)

        if accuracy > best_metrics["accuracy"]:
            for metric, value in zip(["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]):
                best_metrics[metric] = value
            best_metrics["epoch"] = epoch + 1
            best_metrics["cm"] = cm.copy()
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= FLAGS.early_stop_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    logging.info(f"Best Epoch: {best_metrics['epoch']}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        logging.info(f"Best Test {metric.capitalize()}: {best_metrics[metric]:.4f}")
    logging.info(f"Best Confusion Matrix:\n{best_metrics['cm']}")
 

def main(_):
    device = torch.device(f"cuda:{FLAGS.gpu_id}")
    assert FLAGS.num_layers <= 0, "CCS probing only supports MLP classifier."

    if FLAGS.ckpt_step == -1:
        model_dir = FLAGS.llm_model_name
    else:
        model_dir = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}"
    save_model_name = f"hallucination/{FLAGS.dataset_name}/{model_dir}"

    train_loader, test_loader = get_truthfulqa_ccs_dataloader(
        FLAGS.dataset_name,
        FLAGS.llm_model_name,
        FLAGS.ckpt_step,
        FLAGS.llm_layer,
        FLAGS.batch_size,
        FLAGS.eval_batch_size,
        FLAGS.num_workers,
        FLAGS.prefetch_factor,
        FLAGS.test_set_ratio,
        seed=FLAGS.seed,
        feature_density=FLAGS.density,
    )
    model = MLPClassifier(
        num_nodes=get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer, "ccs"),
        hidden_channels=FLAGS.hidden_channels,
        num_layers=-FLAGS.num_layers,
        num_output=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr=1e-6)
    writer = SummaryWriter(log_dir=f"runs/{save_model_name}/layer_{FLAGS.llm_layer}")
    writer.add_hparams(
        {
            "hidden_channels": FLAGS.hidden_channels,
            "num_layers": FLAGS.num_layers, "dropout": FLAGS.dropout,
            "batch_size": FLAGS.batch_size, "lr": FLAGS.lr, "weight_decay": FLAGS.weight_decay
        },
        {"hparam/placeholder": 0}
    )

    if FLAGS.resume:
        model_save_path = os.path.join(
            f"saves/{save_model_name}/layer_{FLAGS.llm_layer}",
            f"best_model_density-{FLAGS.density}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-ccs.pth"
        )
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    train_model(model, train_loader, test_loader, optimizer, scheduler, writer, save_model_name, device)


if __name__ == "__main__":
    app.run(main)
