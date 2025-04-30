from absl import app, flags
import os
from setproctitle import setproctitle

import numpy as np
import torch
torch.set_default_dtype(torch.float32)

from graph_probing.dataset import get_brain_network_dataloader
from graph_probing.model import GCNRegressor, GCNRegressorLinear
from graph_probing.utils import llm_model_num_nodes_map, eval_model


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
flags.DEFINE_float("test_set_ratio", 0.2, "The size of the test set.")
flags.DEFINE_boolean("in_memory", True, "In-memory dataset.")
flags.DEFINE_integer("gpu_id", 7, "The GPU ID.")
FLAGS = flags.FLAGS


def main(_):

    device = torch.device(f"cuda:{FLAGS.gpu_id}")

    _, test_data_loader = get_brain_network_dataloader(
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
    if FLAGS.ckpt_step == -1:
        save_model_name = f"{FLAGS.llm_model_name}"
    else:
        save_model_name = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}"
    model_save_path = os.path.join(
        f"saves/{save_model_name}/layer_{FLAGS.llm_layer}", 
        f"best_model_density-{FLAGS.network_density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}.pth"
    )
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

    all_y, all_pred = eval_model(model, test_data_loader, device)
    results = np.vstack((all_y, all_pred))
    np.save(f"saves/{save_model_name}/layer_{FLAGS.llm_layer}/results_density-{FLAGS.network_density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}.npy", results)


if __name__ == "__main__":
    setproctitle("llm graph probing")
    app.run(main)
