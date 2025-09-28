from absl import app, flags, logging
import os

import torch
torch.set_default_dtype(torch.float32)

from graph_matching.dataset import get_brain_network_pair_dataloader
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
flags.DEFINE_float("test_set_ratio", 0.2, "The ratio of the test set.")
flags.DEFINE_boolean("in_memory", True, "In-memory dataset.")
flags.DEFINE_integer("gpu_id", 0, "The GPU ID.")
FLAGS = flags.FLAGS


def eval_model(model, test_data_loader, device):
    test_loss, test_gauc, test_auc, _ = test_fn(model, test_data_loader, device)
    torch.cuda.empty_cache()
    logging.info(f"Test Contrastive Loss: {test_loss:.4f}")
    logging.info(f"Test GAUROC: {test_gauc:.4f}")
    logging.info(f"Test AUROC: {test_auc:.4f}")


def main(_):

    device = torch.device(f"cuda:{FLAGS.gpu_id}")

    _, test_data_loader = get_brain_network_pair_dataloader(
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
    if FLAGS.ckpt_step_1 == -1:
        save_model_name_1 = f"{FLAGS.llm_model_name_1}"
    else:
        save_model_name_1 = f"{FLAGS.llm_model_name_1}_step{FLAGS.ckpt_step_1}"
    if FLAGS.ckpt_step_2 == -1:
        save_model_name_2 = f"{FLAGS.llm_model_name_2}"
    else:
        save_model_name_2 = f"{FLAGS.llm_model_name_2}_step{FLAGS.ckpt_step_2}"
    save_model_name = f"{save_model_name_1}_{save_model_name_2}"
    model_save_path = os.path.join(
        f"saves/{save_model_name}/layer_{FLAGS.llm_layer_1}_{FLAGS.llm_layer_2}", 
        f"best_model_density-{FLAGS.network_density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}.pth"
    )
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    eval_model(model, test_data_loader, device)


if __name__ == "__main__":
    app.run(main)
