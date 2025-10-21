from absl import app, flags, logging
import os

import torch

from hallucination.dataset import get_truthfulqa_dataloader, get_truthfulqa_linear_dataloader
from hallucination.utils import test_fn
from utils.probing_model import GCNProbe as GCNClassifier, MLPProbe as MLPClassifier
from utils.model_utils import get_num_nodes

flags.DEFINE_enum("dataset_name", "truthfulqa", ["truthfulqa", "halubench", "medhallu"], "Name of the dataset.")
flags.DEFINE_float("density", 1.0, "The density of the network/features.")
flags.DEFINE_boolean("from_sparse_data", False, "Whether to use sparse data.")
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 12, "The layer of the LLM model.")
flags.DEFINE_string("probe_input", "activation", "The input type for linear probing: activation, corr, word2vec_average, or word2vec_token_count.")
flags.DEFINE_integer("batch_size", 16, "The batch size.")
flags.DEFINE_integer("eval_batch_size", 16, "The evaluation batch size.")
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_integer("prefetch_factor", 4, "Prefetch factor.")
flags.DEFINE_integer("hidden_channels", 32, "The hidden channels.")
flags.DEFINE_integer("num_layers", 1, "The number of GNN layers.")
flags.DEFINE_float("dropout", 0.0, "The dropout rate.")
flags.DEFINE_float("test_set_ratio", 0.2, "The ratio of the test set.")
flags.DEFINE_boolean("in_memory", True, "In-memory dataset.")
flags.DEFINE_integer("gpu_id", 0, "The GPU ID.")
flags.DEFINE_integer("seed", 42, "The random seed.")
flags.DEFINE_boolean("use_constant_baseline", False, "Whether to evaluate a constant prediction baseline.")
flags.DEFINE_integer("baseline_label", 0, "Label to predict for the constant baseline (0 or 1).")
FLAGS = flags.FLAGS


class ConstantBaseline(torch.nn.Module):
    def __init__(self, label):
        super().__init__()
        if label not in (0, 1):
            raise ValueError("baseline_label must be 0 or 1.")
        self.label = label

    def forward(self, x, *args):
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 2, device=x.device)
        logits[:, self.label] = 1.0
        return logits


def main(_):
    device = torch.device(f"cuda:{FLAGS.gpu_id}")

    if FLAGS.use_constant_baseline:
        assert FLAGS.num_layers == 0, "Constant baseline requires num_layers=0."

    if FLAGS.ckpt_step == -1:
        model_dir = FLAGS.llm_model_name
    else:
        model_dir = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}"
    save_model_name = f"hallucination/{FLAGS.dataset_name}/{model_dir}"

    if FLAGS.num_layers > 0:
        _, test_loader = get_truthfulqa_dataloader(
            FLAGS.dataset_name,
            FLAGS.llm_model_name,
            FLAGS.ckpt_step,
            FLAGS.llm_layer,
            FLAGS.density,
            FLAGS.from_sparse_data,
            FLAGS.batch_size,
            FLAGS.eval_batch_size,
            FLAGS.num_workers,
            FLAGS.prefetch_factor,
            FLAGS.test_set_ratio,
            FLAGS.in_memory,
            FLAGS.seed,
        )
        model = GCNClassifier(
            num_nodes=get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer),
            hidden_channels=FLAGS.hidden_channels,
            num_layers=FLAGS.num_layers,
            dropout=FLAGS.dropout,
            num_output=2,
        ).to(device)
    else:
        _, test_loader = get_truthfulqa_linear_dataloader(
            FLAGS.probe_input,
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
        if FLAGS.use_constant_baseline:
            model = ConstantBaseline(FLAGS.baseline_label).to(device)
        else:
            model = MLPClassifier(
                num_nodes=get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer, FLAGS.probe_input),
                hidden_channels=FLAGS.hidden_channels,
                num_layers=-FLAGS.num_layers,
                num_output=2,
            ).to(device)

    if not FLAGS.use_constant_baseline:
        model_save_path = os.path.join(
            f"saves/{save_model_name}/layer_{FLAGS.llm_layer}",
            f"best_model_density-{FLAGS.density}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
        )
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    accuracy, precision, recall, f1, cm = test_fn(model, test_loader, device, num_layers=FLAGS.num_layers)
    torch.cuda.empty_cache()
    logging.info(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    app.run(main)
