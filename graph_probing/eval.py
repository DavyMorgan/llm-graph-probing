from absl import app, flags
import os
from setproctitle import setproctitle

import numpy as np
import torch
torch.set_default_dtype(torch.float32)

from graph_probing.dataset import get_brain_network_dataloader, get_brain_network_linear_dataloader
from graph_probing.utils import eval_model

from utils.constants import hf_model_name_map
from utils.model_utils import get_num_nodes
from utils.probing_model import GCNProbe as GCNRegressor, MLPProbe as MLPRegressor

flags.DEFINE_string("dataset", "openwebtext", "The name of the dataset.")
flags.DEFINE_string("target", "perplexities", "The target column for regression.")
flags.DEFINE_string("probe_input", "activation", "The input type for linear probing: activation, activation_avg or corr.")
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
flags.DEFINE_float("test_set_ratio", 0.2, "The size of the test set.")
flags.DEFINE_boolean("in_memory", False, "In-memory dataset.")
flags.DEFINE_integer("gpu_id", 0, "The GPU ID.")
flags.DEFINE_integer("seed", 42, "The random seed.")
FLAGS = flags.FLAGS


def main(_):

    if FLAGS.gpu_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{FLAGS.gpu_id}")

    hf_model_name = hf_model_name_map[FLAGS.llm_model_name]
    if FLAGS.dataset == "art":
        assert FLAGS.target == "decade"
        dataset_filename = "st_data/art.csv"
    else:
        revision = "main" if FLAGS.ckpt_step == -1 else f"step{FLAGS.ckpt_step}"
        if hf_model_name.startswith("EleutherAI") and revision != "main":
            dataset_filename = f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}-{revision}.csv"
        else:
            dataset_filename = f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}.csv"

    if FLAGS.num_layers > 0:
        _, test_data_loader = get_brain_network_dataloader(
            dataset_filename,
            density=FLAGS.density,
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
            target=FLAGS.target,
            dataset_name=FLAGS.dataset,
        )

        model = GCNRegressor(
            num_nodes=get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer, FLAGS.probe_input),
            hidden_channels=FLAGS.num_channels,
            num_layers=FLAGS.num_layers,
            nonlinear_activation=FLAGS.nonlinear_activation,
            dropout=FLAGS.dropout,
            num_output=1,
        ).to(device)
    else:
        _, test_data_loader = get_brain_network_linear_dataloader(
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
            target=FLAGS.target,
            dataset_name=FLAGS.dataset
        )
        model = MLPRegressor(
            num_nodes=get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer, FLAGS.probe_input),
            hidden_channels=FLAGS.num_channels,
            num_layers=-FLAGS.num_layers,
            num_output=1,
        ).to(device)

    if FLAGS.ckpt_step == -1:
        save_model_name = f"{FLAGS.llm_model_name}/{FLAGS.dataset}"
    else:
        save_model_name = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}/{FLAGS.dataset}"
    model_save_path = os.path.join(
        f"saves/graph_probing/{save_model_name}/layer_{FLAGS.llm_layer}",
        f"best_model_density-{FLAGS.density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
    )
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

    all_y, all_pred = eval_model(model, test_data_loader, device, num_layers=FLAGS.num_layers)
    results = np.vstack((all_y, all_pred))
    np.save(f"saves/graph_probing/{save_model_name}/layer_{FLAGS.llm_layer}/results_density-{FLAGS.density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.npy", results)


if __name__ == "__main__":
    setproctitle("llm graph probing")
    app.run(main)
