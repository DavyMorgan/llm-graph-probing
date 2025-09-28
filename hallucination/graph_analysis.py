from absl import app, flags
from collections import defaultdict
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("layer", 12, "The layer to analyze.")
flags.DEFINE_enum("feature", "corr", ["corr", "activation"], "The feature to analyze.")
FLAGS = flags.FLAGS


def load_neural_topology_data(model_name, ckpt_step, layer, feature_name):
    """Load neural topology data for all question IDs and their true/false labels."""
    
    # Construct model directory path
    if ckpt_step == -1:
        model_dir = f"data/hallucination/{model_name}"
    else:
        model_dir = f"data/hallucination/{model_name}_step{ckpt_step}"
    
    # Load the original dataset to get question_id mappings
    dataset_file = "data/hallucination/truthfulqa-validation.csv"
    df = pd.read_csv(dataset_file)
    
    topology_data = defaultdict(lambda: {'true': [], 'false': []})
    
    # Iterate through each row (each row index corresponds to a directory)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading topology data"):
        question_id = row['question_id']
        label = row['label']
        
        # The directory name is the row index
        data_dir = os.path.join(model_dir, str(idx))
        feature_file = os.path.join(data_dir, f"layer_{layer}_{feature_name}.npy")

        if os.path.exists(feature_file):
            feature = np.load(feature_file)

            if feature_name == "corr":
                # Get upper triangular part and flatten
                triu_indices = np.triu_indices(feature.shape[0], k=1)
                feature = feature[triu_indices]

            if label == 1:  # True answer
                topology_data[question_id]['true'].append(feature)
            else:  # False answer
                topology_data[question_id]['false'].append(feature)
    
    return topology_data


def calculate_correlation_metrics(topology_data):
    """Calculate correlation metrics for each question_id."""
    
    question_metrics = {}
    
    for question_id, data in tqdm(topology_data.items(), desc="Processing questions"):
        true_topologies = data['true']
        false_topologies = data['false']

        if len(true_topologies) < 2 or len(false_topologies) < 2:
            continue
            
        # Concatenate all topologies for this question
        all_topologies = true_topologies + false_topologies
        
        # Create correlation matrix between all topology vectors
        topology_matrix = np.array(all_topologies)
        corr_matrix = np.corrcoef(topology_matrix)
        
        # Calculate indices for true-true, false-false, and true-false pairs
        num_true = len(true_topologies)
        num_false = len(false_topologies)
        
        # True-True correlations (upper triangular of true block)
        if num_true > 1:
            true_true_corrs = []
            for i in range(num_true):
                for j in range(i + 1, num_true):
                    true_true_corrs.append(corr_matrix[i, j])
            avg_true_true = np.mean(true_true_corrs) if true_true_corrs else 0
        else:
            avg_true_true = 0
            
        # False-False correlations (upper triangular of false block)
        if num_false > 1:
            false_false_corrs = []
            for i in range(num_true, num_true + num_false):
                for j in range(i + 1, num_true + num_false):
                    false_false_corrs.append(corr_matrix[i, j])
            avg_false_false = np.mean(false_false_corrs) if false_false_corrs else 0
        else:
            avg_false_false = 0
            
        # True-False correlations (cross-block)
        true_false_corrs = []
        for i in range(num_true):
            for j in range(num_true, num_true + num_false):
                true_false_corrs.append(corr_matrix[i, j])
        avg_true_false = np.mean(true_false_corrs) if true_false_corrs else 0
        
        # Calculate the metric: avg_true_true + avg_false_false - 2 * avg_true_false
        intra_vs_inter_metric = avg_true_true + avg_false_false - 2 * avg_true_false
        
        question_metrics[question_id] = {
            'avg_true_true': avg_true_true,
            'avg_false_false': avg_false_false,
            'avg_true_false': avg_true_false,
            'intra_vs_inter_metric': intra_vs_inter_metric,
            'num_true': num_true,
            'num_false': num_false
        }
    
    return question_metrics


def print_statistics(question_metrics):
    """Print comprehensive statistics of the metrics."""
    
    if not question_metrics:
        print("No data available for analysis.")
        return
        
    # Extract all metric values
    intra_vs_inter_values = [metrics['intra_vs_inter_metric'] for metrics in question_metrics.values()]
    true_true_values = [metrics['avg_true_true'] for metrics in question_metrics.values()]
    false_false_values = [metrics['avg_false_false'] for metrics in question_metrics.values()]
    true_false_values = [metrics['avg_true_false'] for metrics in question_metrics.values()]
    
    # Print individual question results
    print(f"\n--- Per-Question Results ---")
    for question_id in sorted(question_metrics.keys()):
        metrics = question_metrics[question_id]
        print(f"Q{question_id:3d}: {metrics['intra_vs_inter_metric']:8.6f} "
              f"(T-T: {metrics['avg_true_true']:6.4f}, "
              f"F-F: {metrics['avg_false_false']:6.4f}, "
              f"T-F: {metrics['avg_true_false']:6.4f}, "
              f"#T: {metrics['num_true']}, #F: {metrics['num_false']})")

    print(f"\n=== Neural Topology Correlation Analysis ===")
    print(f"Model: {FLAGS.llm_model_name}")
    if FLAGS.ckpt_step != -1:
        print(f"Checkpoint Step: {FLAGS.ckpt_step}")
    print(f"Layer: {FLAGS.layer}")
    print(f"Number of questions analyzed: {len(question_metrics)}")
    
    print(f"\n--- Intra vs Inter-group Metric (true-true + false-false - 2*true-false) ---")
    print(f"Mean: {np.mean(intra_vs_inter_values):.6f}")
    print(f"Std:  {np.std(intra_vs_inter_values):.6f}")
    print(f"Median: {np.median(intra_vs_inter_values):.6f}")
    print(f"Min: {np.min(intra_vs_inter_values):.6f}")
    print(f"Max: {np.max(intra_vs_inter_values):.6f}")
    print(f">0 ratio: {np.mean(np.array(intra_vs_inter_values) > 0):.4f}")
    
    print(f"\n--- True-True Correlations ---")
    print(f"Mean: {np.mean(true_true_values):.6f}")
    print(f"Std:  {np.std(true_true_values):.6f}")
    print(f"Median: {np.median(true_true_values):.6f}")
    print(f"Min: {np.min(true_true_values):.6f}")
    print(f"Max: {np.max(true_true_values):.6f}")

    print(f"\n--- False-False Correlations ---")
    print(f"Mean: {np.mean(false_false_values):.6f}")
    print(f"Std:  {np.std(false_false_values):.6f}")
    print(f"Median: {np.median(false_false_values):.6f}")
    print(f"Min: {np.min(false_false_values):.6f}")
    print(f"Max: {np.max(false_false_values):.6f}")
    
    print(f"\n--- True-False Correlations ---")
    print(f"Mean: {np.mean(true_false_values):.6f}")
    print(f"Std:  {np.std(true_false_values):.6f}")
    print(f"Median: {np.median(true_false_values):.6f}")
    print(f"Min: {np.min(true_false_values):.6f}")
    print(f"Max: {np.max(true_false_values):.6f}")

    return intra_vs_inter_values
    

def main(_):
    model_name = FLAGS.llm_model_name
    if FLAGS.ckpt_step == -1:
        model_dir = f"data/hallucination/{model_name}"
    else:
        model_dir = f"data/hallucination/{model_name}_step{FLAGS.ckpt_step}"
    
    print(f"Loading neural topology data from: {model_dir}")
    print(f"Analyzing layer: {FLAGS.layer}")
    
    # Load the neural topology data
    topology_data = load_neural_topology_data(FLAGS.llm_model_name, FLAGS.ckpt_step, FLAGS.layer, FLAGS.feature)

    # Calculate correlation metrics
    question_metrics = calculate_correlation_metrics(topology_data)
    
    # Print statistics
    diff = print_statistics(question_metrics)
    np.save(f"{FLAGS.llm_model_name}_layer_{FLAGS.layer}_{FLAGS.feature}_intra_vs_inter.npy", np.array(diff))


if __name__ == "__main__":
    app.run(main)