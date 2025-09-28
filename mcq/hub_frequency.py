from absl import app, flags
import os
import pickle
from tqdm import tqdm

import numpy as np

flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b-instruct", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 12, "The layer to analyze.")
flags.DEFINE_string("dataset", "mcq", "The name of the dataset, mcq or mcq_pro.")
flags.DEFINE_float("threshold", 0.1, "Threshold for sparsifying the correlation matrix.")
flags.DEFINE_integer("num_samples", None, "Number of samples to use. If None, use all available samples.")
flags.DEFINE_integer("top_k_hubs", 50, "Number of top hub nodes to analyze.")
FLAGS = flags.FLAGS


def get_hub_nodes(corr_matrix, top_k, verbose=False):
    """Identify hub nodes with highest degree from correlation matrix."""
    # Calculate degree as sum of absolute correlations (excluding diagonal)
    abs_corr = np.abs(corr_matrix)
    np.fill_diagonal(abs_corr, 0)  # Exclude self-connections
    degrees = np.sum(abs_corr, axis=1)
    
    # Get top-k hub nodes
    hub_indices = np.argsort(degrees)[-top_k:][::-1]  # Sort in descending order
    hub_degrees = degrees[hub_indices]
    
    if verbose:
        print(f"Top {top_k} hub nodes: {hub_indices}")
        print(f"Hub degrees: {hub_degrees}")
    
    return hub_indices, hub_degrees


def compute_hub_frequency(data_dir, layer, hub_nodes, num_samples=None):
    """Calculate frequency of hub nodes appearing as hubs in individual graphs."""
    # Get all question directories
    all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
    all_dirs.sort(key=int)
    
    if num_samples is not None:
        all_dirs = all_dirs[:num_samples]
    
    print(f"Computing hub frequencies from {len(all_dirs)} individual graphs...")
    
    hub_frequency = {hub: 0 for hub in hub_nodes}
    valid_graphs = 0
    top_k = len(hub_nodes)
    
    for question_dir in tqdm(all_dirs):
        corr_path = os.path.join(data_dir, question_dir, f"layer_{layer}_corr.npy")
        if os.path.exists(corr_path):
            corr_matrix = np.load(corr_path)
            if not np.any(np.isnan(corr_matrix)) and not np.any(np.isinf(corr_matrix)):
                # Get top-k hub nodes for this individual graph (verbose=False)
                individual_hubs, _ = get_hub_nodes(corr_matrix, top_k, verbose=False)
                
                # Update frequency count
                for hub in individual_hubs:
                    if hub in hub_frequency:
                        hub_frequency[hub] += 1
                
                valid_graphs += 1
    
    # Convert to frequency (proportion)
    hub_frequency_prop = {hub: count / valid_graphs for hub, count in hub_frequency.items()}
    
    print(f"Hub frequency analysis completed on {valid_graphs} graphs")
    
    # Print hub frequency results
    print("\n=== HUB FREQUENCY RESULTS ===")
    freq_values = list(hub_frequency_prop.values())
    print(f"Hub frequency statistics:")
    print(f"  Mean frequency: {np.mean(freq_values):.4f}")
    print(f"  Std frequency: {np.std(freq_values):.4f}")
    print(f"  Min frequency: {np.min(freq_values):.4f}")
    print(f"  Max frequency: {np.max(freq_values):.4f}")
    
    print(f"\nTop {top_k} most frequent hubs:")
    sorted_hubs = sorted(hub_frequency_prop.items(), key=lambda x: x[1], reverse=True)
    for i, (hub, freq) in enumerate(sorted_hubs[:top_k]):
        count = hub_frequency[hub]
        print(f"  {i+1}. Node {hub}: {count}/{valid_graphs} = {freq:.4f}")
    
    return hub_frequency, hub_frequency_prop, valid_graphs


def compute_average_correlation_matrix_online(data_dir, layer, num_samples=None):
    """Compute average correlation matrix by accumulating and dividing by count."""
    # Get all question directories
    all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
    all_dirs.sort(key=int)
    
    if num_samples is not None:
        all_dirs = all_dirs[:num_samples]
    
    print(f"Computing average correlation matrix from {len(all_dirs)} questions...")
    
    sum_corr_matrix = None
    count = 0
    
    for question_dir in tqdm(all_dirs):
        corr_path = os.path.join(data_dir, question_dir, f"layer_{layer}_corr.npy")
        if os.path.exists(corr_path):
            corr_matrix = np.load(corr_path)
            if not np.any(np.isnan(corr_matrix)) and not np.any(np.isinf(corr_matrix)):
                if sum_corr_matrix is None:
                    sum_corr_matrix = corr_matrix.copy()
                else:
                    sum_corr_matrix += corr_matrix
                count += 1
            else:
                print(f"Warning: Skipping {question_dir} due to NaN or Inf values")
        else:
            print(f"Warning: Correlation matrix not found for question {question_dir}")
    
    if sum_corr_matrix is None or count == 0:
        return None, 0
    
    # Compute average by dividing by count
    avg_corr_matrix = sum_corr_matrix / count
    
    print(f"Successfully computed average from {count} correlation matrices")
    print(f"Average correlation matrix shape: {avg_corr_matrix.shape}")
    
    return avg_corr_matrix, count


def sparsify_correlation_matrix(corr_matrix, threshold):
    """Sparsify correlation matrix by thresholding."""
    print(f"Sparsifying correlation matrix with threshold {threshold}")
    
    # Take absolute values and apply threshold
    abs_corr = np.abs(corr_matrix)
    sparse_corr = np.where(abs_corr >= threshold, corr_matrix, 0)
    
    # Keep original signs but zero out weak correlations
    np.fill_diagonal(sparse_corr, 1.0)  # Ensure diagonal is 1
    
    num_edges = np.sum(sparse_corr != 0) - sparse_corr.shape[0]  # Subtract diagonal
    total_possible = sparse_corr.shape[0] * (sparse_corr.shape[0] - 1)
    sparsity = num_edges / total_possible
    
    print(f"Sparsified matrix has {num_edges} edges ({sparsity:.4f} density)")
    
    return sparse_corr


def save_results(model_name, ckpt_step, layer, threshold, count, hub_nodes, hub_degrees, hub_frequency, hub_frequency_prop, valid_graphs):
    # Determine save directory
    if ckpt_step == -1:
        save_model_name = model_name
    else:
        save_model_name = f"{model_name}_step{ckpt_step}"
    
    output_dir = f"saves/mcq/{save_model_name}/layer_{layer}/community_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comprehensive results
    results_file = os.path.join(output_dir, f"analysis_thresh_{threshold}.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump({
            'hub_nodes': hub_nodes.tolist(),
            'hub_degrees': hub_degrees.tolist(),
            'hub_frequency_count': hub_frequency,
            'hub_frequency_proportion': hub_frequency_prop,
            'threshold': threshold,
            'layer': layer,
            'num_samples_used': count,
            'valid_graphs_for_hubs': valid_graphs
        }, f)
    
    print(f"Analysis results saved to {results_file}")
    
    # Save as text file for easy inspection
    text_file = os.path.join(output_dir, f"analysis_thresh_{threshold}.txt")
    with open(text_file, 'w') as f:
        f.write(f"Network Analysis Results\n")
        f.write(f"Layer: {layer}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Number of samples used: {count}\n")
        f.write(f"Valid graphs for hub analysis: {valid_graphs}\n\n")
        
        f.write(f"\n=== HUB NODE ANALYSIS ===\n")
        f.write(f"Top {len(hub_nodes)} hub nodes from average correlation matrix:\n")
        for i, (hub, degree) in enumerate(zip(hub_nodes, hub_degrees)):
            freq_count = hub_frequency[hub]
            freq_prop = hub_frequency_prop[hub]
            f.write(f"Hub {i+1}: Node {hub} (degree={degree:.4f}, frequency={freq_count}/{valid_graphs}={freq_prop:.4f})\n")
        
        f.write(f"\nHub frequency statistics:\n")
        freq_values = list(hub_frequency_prop.values())
        f.write(f"Mean frequency: {np.mean(freq_values):.4f}\n")
        f.write(f"Std frequency: {np.std(freq_values):.4f}\n")
        f.write(f"Min frequency: {np.min(freq_values):.4f}\n")
        f.write(f"Max frequency: {np.max(freq_values):.4f}\n")
    
    print(f"Analysis summary saved to {text_file}")
    
    # Save hub frequency as separate numpy files for easy plotting
    hub_freq_file = os.path.join(output_dir, f"hub_frequency_thresh_{threshold}.npy")
    np.save(hub_freq_file, {
        'hub_nodes': hub_nodes,
        'hub_degrees': hub_degrees,
        'hub_frequency_count': np.array([hub_frequency[hub] for hub in hub_nodes]),
        'hub_frequency_proportion': np.array([hub_frequency_prop[hub] for hub in hub_nodes]),
        'valid_graphs': valid_graphs
    })
    print(f"Hub frequency data saved to {hub_freq_file}")


def main(_):
    # Determine data directory
    model_name = FLAGS.llm_model_name
    if FLAGS.ckpt_step == -1:
        data_dir = f"data/{FLAGS.dataset}/{model_name}"
    else:
        data_dir = f"data/{FLAGS.dataset}/{model_name}_step{FLAGS.ckpt_step}"
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Exiting.")
        return
    
    # Compute average correlation matrix by accumulating and dividing
    avg_corr_matrix, count = compute_average_correlation_matrix_online(
        data_dir, FLAGS.llm_layer, FLAGS.num_samples
    )
    
    if avg_corr_matrix is None:
        print("No valid correlation matrices found. Exiting.")
        return
    
    # Save average correlation matrix
    if FLAGS.ckpt_step == -1:
        save_model_name = f"{FLAGS.dataset}/{FLAGS.llm_model_name}"
    else:
        save_model_name = f"{FLAGS.dataset}/{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}"
    
    output_dir = os.path.join(f"saves/{save_model_name}/layer_{FLAGS.llm_layer}/community_analysis")
    os.makedirs(output_dir, exist_ok=True)
    avg_corr_file = os.path.join(output_dir, f"avg_correlation.npy")
    np.save(avg_corr_file, avg_corr_matrix)
    print(f"Average correlation matrix saved to {avg_corr_file}")
    
    # Get hub nodes from average correlation matrix (verbose=True)
    hub_nodes, hub_degrees = get_hub_nodes(avg_corr_matrix, FLAGS.top_k_hubs, verbose=True)
    
    # Calculate hub frequency in individual graphs
    hub_frequency, hub_frequency_prop, valid_graphs = compute_hub_frequency(
        data_dir, FLAGS.llm_layer, hub_nodes, FLAGS.num_samples
    )
    
    # Save all results
    save_results(
        FLAGS.llm_model_name, FLAGS.ckpt_step, FLAGS.llm_layer, FLAGS.threshold, count,
        hub_nodes, hub_degrees, hub_frequency, hub_frequency_prop, valid_graphs
    )
    

if __name__ == "__main__":
    app.run(main)