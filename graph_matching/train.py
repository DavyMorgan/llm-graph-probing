from absl import app, flags, logging
import os
from tqdm import tqdm

import torch
torch.set_default_dtype(torch.float32)
from torch.utils.tensorboard import SummaryWriter

from graph_matching.dataset import get_brain_network_pair_dataloader
from graph_matching.loss import contrastive_loss_cosine
from graph_matching.model import GraphMatchingModel
from graph_matching.utils import test_fn
from utils.model_utils import get_num_nodes

flags.DEFINE_string("dataset_filename", "data/graph_matching/openwebtext-10k.csv", "The dataset filename.")
flags.DEFINE_float("network_density", 1.0, "The density of the network.")
flags.DEFINE_string("llm_model_name_1", "gpt2", "The name of the first LLM model.")
flags.DEFINE_string("llm_model_name_2", "gpt2", "The name of the second LLM model.")
flags.DEFINE_integer("ckpt_step_1", -1, "The checkpoint step.")
flags.DEFINE_integer("ckpt_step_2", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer_1", 6, "The layer of the LLM model.")
flags.DEFINE_integer("llm_layer_2", 6, "The layer of the LLM model.")
flags.DEFINE_integer("batch_size", 16, "The batch size.")
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_integer("prefetch_factor", 2, "Prefetch factor.")
flags.DEFINE_integer("eval_batch_size", 16, "The evaluation batch size.")
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
FLAGS = flags.FLAGS


def train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_name, device):
    test_loss, test_gauc, test_auc, test_sim_matrix = test_fn(model, test_data_loader, device)
    torch.cuda.empty_cache()
    logging.info(f"Initial Test Contrastive Loss: {test_loss:.4f}")
    logging.info(f"Initial Test GAUROC: {test_gauc:.4f}")
    logging.info(f"Initial Test AUROC: {test_auc:.4f}")
    writer.add_scalar("loss/test", test_loss, 0)
    writer.add_scalar("gauroc/test", test_gauc, 0)
    writer.add_scalar("auroc/test", test_auc, 0)
    writer.add_image("sim_matrix/test", test_sim_matrix, 0, dataformats="HW")
    writer.add_histogram("sim_matrix_histogram/test", test_sim_matrix, 0)

    model_save_path = os.path.join(
        f"saves/{save_model_name}/layer_{FLAGS.llm_layer_1}_{FLAGS.llm_layer_2}", 
        f"best_model_density-{FLAGS.network_density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}.pth"
    )
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    best_test_gauc = test_gauc
    best_test_auc = test_auc
    best_test_loss = test_loss
    best_epoch = 0
    epochs_no_improve = 0
    num_epochs = FLAGS.num_epochs
    for epoch in tqdm(range(num_epochs), position=0, desc="Training"):
        model.train()
        total_loss = 0.0
        for data in tqdm(train_data_loader, position=1, desc=f"Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            _, _, sim_matrix = model(data.to(device))
            loss = contrastive_loss_cosine(sim_matrix)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_data_loader)
        logging.info(f"Epoch {epoch + 1}, Contrastive Loss: {avg_loss:.4f}")
        writer.add_scalar("loss/train", avg_loss, epoch + 1)
        torch.cuda.empty_cache()

        test_loss, test_gauc, test_auc, test_sim_matrix = test_fn(model, test_data_loader, device)
        torch.cuda.empty_cache()
        logging.info(f"Test Contrastive Loss: {test_loss:.4f}")
        logging.info(f"Test GAUROC: {test_gauc:.4f}")
        logging.info(f"Test AUROC: {test_auc:.4f}")
        writer.add_scalar("loss/test", test_loss, epoch + 1)
        writer.add_scalar("gauroc/test", test_gauc, epoch + 1)
        writer.add_scalar("auroc/test", test_auc, epoch + 1)
        writer.add_image("sim_matrix/test", test_sim_matrix, epoch + 1, dataformats="HW")
        writer.add_histogram("sim_matrix_histogram/test", test_sim_matrix, epoch + 1)
        scheduler.step(test_auc)

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_test_gauc = test_gauc
            best_test_loss = test_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= FLAGS.early_stop_patience:
                break

    logging.info(f"Best Epoch: {best_epoch}")
    logging.info(f"Best Test Contrastive Loss: {best_test_loss:.4f}")
    logging.info(f"Best Test GAUROC: {best_test_gauc:.4f}")
    logging.info(f"Best Test AUROC: {best_test_auc:.4f}")

    writer.add_text(
        "best_record",
        f"Best Epoch: {best_epoch}, "
        f"Best Test Contrastive Loss: {best_test_loss:.4f}, "
        f"Best Test GAUROC: {best_test_gauc:.4f}, "
        f"Best Test AUROC: {best_test_auc:.4f}",
        0
    )


def main(_):

    device = torch.device(f"cuda:{FLAGS.gpu_id}")

    train_data_loader, test_data_loader = get_brain_network_pair_dataloader(
        dataset_filename=FLAGS.dataset_filename,
        network_density=FLAGS.network_density,
        llm_model_name_1=FLAGS.llm_model_name_1,
        ckpt_step_1=FLAGS.ckpt_step_1,
        llm_layer_1=FLAGS.llm_layer_1,
        llm_model_name_2=FLAGS.llm_model_name_2,
        ckpt_step_2=FLAGS.ckpt_step_2,
        llm_layer_2=FLAGS.llm_layer_2,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        num_workers=FLAGS.num_workers,
        prefetch_factor=FLAGS.prefetch_factor,
        test_set_ratio=FLAGS.test_set_ratio,
        in_memory=FLAGS.in_memory,
    )

    model = GraphMatchingModel(
        num_nodes_llm_1=get_num_nodes(FLAGS.llm_model_name_1, FLAGS.llm_layer_1),
        num_nodes_llm_2=get_num_nodes(FLAGS.llm_model_name_2, FLAGS.llm_layer_2),
        hidden_channels=FLAGS.num_channels,
        out_channels=FLAGS.num_channels,
        num_layers=FLAGS.num_layers,
        dropout=FLAGS.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr=1e-6)
    if FLAGS.ckpt_step_1 == -1:
        save_model_name_1 = f"{FLAGS.llm_model_name_1}"
    else:
        save_model_name_1 = f"{FLAGS.llm_model_name_1}_step{FLAGS.ckpt_step_1}"
    if FLAGS.ckpt_step_2 == -1:
        save_model_name_2 = f"{FLAGS.llm_model_name_2}"
    else:
        save_model_name_2 = f"{FLAGS.llm_model_name_2}_step{FLAGS.ckpt_step_2}"
    save_model_name = f"{save_model_name_1}_{save_model_name_2}"
    writer = SummaryWriter(log_dir=f"runs/{save_model_name}/layer_{FLAGS.llm_layer_1}_{FLAGS.llm_layer_2}")
    writer.add_hparams(
        {
            "hidden_channels": FLAGS.num_channels, "out_channels": FLAGS.num_channels,
            "num_layers": FLAGS.num_layers, "dropout": FLAGS.dropout,
            "batch_size": FLAGS.batch_size, "lr": FLAGS.lr, "weight_decay": FLAGS.weight_decay
        },
        {"hparam/placeholder": 0}
    )

    train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_name, device)


if __name__ == "__main__":
    app.run(main)
