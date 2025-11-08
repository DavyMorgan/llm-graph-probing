from absl import app, flags, logging
from tqdm import tqdm

import numpy as np
from sklearn.decomposition import PCA
import sklearn.metrics as sk
import torch

from hallucination.dataset import get_truthfulqa_linear_dataloader
from utils.model_utils import get_num_nodes

flags.DEFINE_enum("dataset_name", "truthfulqa", ["truthfulqa", "halueval", "medhallu", "helm"], "Name of the dataset.")
flags.DEFINE_float("density", 1.0, "The density of the network/features.")
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 12, "The layer of the LLM model.")
flags.DEFINE_float("test_set_ratio", 0.2, "The ratio of the test set.")
flags.DEFINE_boolean("in_memory", True, "In-memory dataset.")
flags.DEFINE_integer("gpu_id", 0, "The GPU ID.")
flags.DEFINE_boolean("resume", False, "Whether to resume training from the best model.")
flags.DEFINE_integer("seed", 42, "The random seed.")
FLAGS = flags.FLAGS


def get_measures(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    return auroc


def svd_embed_score(feature, label, begin_k, k_span, mean=1, svd=1, weight=0):
    device = torch.device(f"cuda:{FLAGS.gpu_id}")
    embed_generated = feature
    best_k = -1
    best_auroc_over_k = 0
    best_sign_over_k = None
    best_scores_over_k = None
    best_projection_over_k = None
    best_mean_over_k = None
    for k in tqdm(range(begin_k, k_span)):
        scores = None
        mean_recorded = None
        if mean:
            mean_recorded = embed_generated[:, :].mean(0)
            centered = embed_generated[:, :] - mean_recorded
        else:
            centered = embed_generated[:, :]

        if not svd:
            pca_model = PCA(n_components=k, whiten=False).fit(centered)
            projection = pca_model.components_.T
            mean_recorded = pca_model.mean_
            if weight:
                projection = pca_model.singular_values_ * projection
        else:
            _, sin_value, V_p = torch.linalg.svd(torch.from_numpy(centered).to(device))
            projection = V_p[:k, :].T.cpu().data.numpy()
            if weight:
                projection = sin_value[:k] * projection


        scores = np.matmul(centered, projection).mean(-1, keepdims=True)
        #scores = np.mean(np.matmul(centered, projection), dim=-1, keepdims=True)
        assert scores.shape[1] == 1
        scores = np.sqrt(np.square(scores).sum(axis=1))
        #scores = np.sqrt(np.sum(np.square(scores), axis=1))

        # not sure about whether true and false data the direction will point to,
        # so we test both. similar practices are in the representation engineering paper
        # https://arxiv.org/abs/2310.01405
        measures1 = get_measures(scores[label == 1], scores[label == 0])
        measures2 = get_measures(-scores[label == 1], -scores[label == 0])

        if measures1 > measures2:
            measures = measures1
            sign_layer = 1
        else:
            measures = measures2
            sign_layer = -1

        print('k: ', k, 'auroc: ', measures, 'mean: ', mean, 'svd: ', svd)

        if measures > best_auroc_over_k:
            best_auroc_over_k = measures
            best_k = k
            best_sign_over_k = sign_layer
            best_scores_over_k = scores
            best_projection_over_k = projection
            best_mean_over_k = mean_recorded


    return {'k': best_k,
            'best_auroc':best_auroc_over_k,
            'best_sign':best_sign_over_k,
            'best_scores':best_scores_over_k,
            'best_projection':best_projection_over_k,
            'best_mean':best_mean_over_k}


def train_model(train_data, test_data):
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    best_results = svd_embed_score(train_features, train_labels, begin_k=1, k_span=11, mean=0, svd=0, weight=0)

    scores = np.matmul(test_features, best_results['best_projection']).mean(-1, keepdims=True)
    assert scores.shape[1] == 1
    scores = np.sqrt(np.square(scores).sum(axis=1))
    scores *= best_results['best_sign']

    accuracy = sk.accuracy_score(test_labels.numpy(), (scores.numpy() > np.percentile(scores.numpy(), 100 - test_labels.sum()/len(test_labels))).astype(np.int32))
    logging.info(f'Test Accuracy: {accuracy:.4f}')


def main(_):
    train_data, test_data = get_truthfulqa_linear_dataloader(
        "activation",
        FLAGS.dataset_name,
        FLAGS.llm_model_name,
        FLAGS.ckpt_step,
        FLAGS.llm_layer,
        None,
        None,
        None,
        None,
        FLAGS.test_set_ratio,
        seed=FLAGS.seed,
        feature_density=FLAGS.density,
        return_all_data=True
    )

    train_model(train_data, test_data)


if __name__ == "__main__":
    app.run(main)
