# LLM Graph Probing

Graph probing is a tool for learning the functional connectivity topology of neurons in large language models (LLMs) and relating it to language generation performance. The project was initially described in the paper [***"Learning Neural Topology of Language Models with Graph Probing"***](https://arxiv.org/abs/2506.01042).

![Loading Model Overview](assets/graph_probing.png "Model Overview")
---

## Installation

### Environment
* Tested OS: Linux
* Python: 3.10

### Dependencies
To install the package, run the following command:

```bash
pip install -r requirements.txt
```

You may install `torch-scatter` to accelerate GNN computation. Please refer to the [installation instructions](https://github.com/rusty1s/pytorch_scatter#installation) for your specific CUDA version. 

## Usage of Graph Probing
### Data Preparation
First, run the following command to generate textual data and corresponding perplexity scores for the LLMs you want to probe. The data will be saved in the `data/graph_probing` directory.

```bash
python -m graph_probing.construct_dataset --dataset <dataset_name> --llm_model_name <model_name> --ckpt_step <ckpt_step> --batch_size <batch_size>
```
`dataset`: The name of the dataset you want to use. In our experiments, we used `openwebtext` for all models.

`model_name`: The name of the LLM you want to probe. You can choose from the keys of `hf_model_name_map` in `utils/constants.py`.

`ckpt_step`: The checkpoint step of the LLM you want to probe. For pythia models, you can choose from `0`, `1`, `2`, `4`, `8`, `16`, `32`, `64`, `128`, `256`, `512`, and 143 evenly-spaced checkpoints from `1000` to `143000`. For other models, only `-1` is supported, which means the last checkpoint.

Then, run the following command to generate the neural topology. The data will be saved in the `data/graph_probing/<model_name>` directory.

```bash
python -m graph_probing.compute_llm_network --dataset <dataset_name> --llm_model_name <model_name> --ckpt_step <ckpt_step> --llm_layer <layer_id> --batch_size <batch_size> --network_density <network_density>
```
`llm_layer`: The layer ID of the LLM you want to probe.

`network_density`: The density of the neural graph you want to generate. You can choose any value that is greater than `0` and less than or equal to `1`. If you set it to be less than `1`, the graph will be sparsified. 

Other parameters are the same as the previous step.

### Training Probes
Run the following command to train the probes. The trained probes will be saved in the `saves/graph_probing/<model_name>` directory.

```bash
python -m graph_probing.train --dataset <dataset_name> --probe_input <probe_input> --density <density> --from_sparse_data --llm_model_name <model_name> --ckpt_step <ckpt_step> --llm_layer <layer_id> --batch_size <batch_size> --eval_batch_size <eval_batch_size> --nonlinear_activation --num_channels <num_channels> --num_layers <num_layers> --lr <learning_rate> --in_memory --gpu_id 0
```
`probe_input`: The input type for probing. You can choose from `activation` and `corr`. If you set it to be `activation`, the probe will take neuron activations as input (baselines). If you set it to be `corr`, the probe will take neuron correlation as input (ours).

`from_sparse_data`: If you are training probes on sparse graphs, set it to be `--from_sparse_data`. Otherwise, set it to be `--nofrom_sparse_data`.

`nonlinear_activation`: If you are using nonlinear activation, set it to be `--nonlinear_activation`. Otherwise, set it to be `--nononlinear_activation`.

`num_channels`: The number of channels in the GNN probes or the number of hidden dimensions for MLP probes. The default value is `32`.

`num_layers`: If it is greater than 0, it represents the number of layers in the GNN probes. If it is equal to 0, it means you are using simple linear probes. If it is less than 0, it means you are using MLP probes, and its absolute value represents the number of hidden layers in the MLP.

`learning_rate`: The learning rate for training the probes. For linear probes on correlation graphs, we recommend setting it to be `0.00001`. For other settings, we recommend setting it to be `0.001`.

`in_memory`: Load all graphs into memory before training. This will speed up the training process. If you have enough memory, set it to be `--in_memory`. Otherwise, set it to be `--noin_memory`.

Other parameters are the same as previous steps.

### Evaluating GNN Probes
GNN probes are evaluated automatically during and after training. You can also run the following command to evaluate the saved GNN probes. The evaluation results containing the predicted and groundtruth perplexity scores will be saved in the `saves/graph_probing/<model_name>` directory.

```bash
python -m graph_probing.eval --dataset <dataset_name> --probe_input <probe_input> --density <network_density> --from_sparse_data --llm_model_name <model_name> --ckpt_step <ckpt_step> --llm_layer <layer_id> --batch_size <batch_size> --eval_batch_size <eval_batch_size> --nonlinear_activation --num_channels <num_channels> --num_layers <num_layers> --in_memory --gpu_id 0
```
All parameters are the same as the previous training step.

## Causal Intervention on MMLU benchmark
We provide the code for causal intervention on MMLU benchmark in the `mcq` directory.

First, run the following command to prepare the MMLU dataset. The data will be saved in the `data/mcq` directory.

```bash
python -m mcq.construct_dataset
```

Then, run the following command to compare different intervention methods.

```bash
python -m mcq.intervene --llm_model_name <model_name> --llm_layer <layer_id> --intervention_frac <intervention_fraction>
```

To compute hub nodes frequency, run the following command:

```bash
python -m mcq.compute_llm_network --llm_model_name <model_name> --llm_layer <layer_id>
python -m mcq.hub_frequency --llm_model_name <model_name> --llm_layer <layer_id>
```


## Hallucination Detection on TruthfulQA dataset

The `hallucination` package constructs neural graphs for TruthfulQA answers and trains probes that classify truthful versus hallucinated generations.

### Data Preparation
Generate the TruthfulQA validation split with labels for true and false answers. The CSV is saved to `data/hallucination/truthfulqa-validation.csv`.

```bash
python -m hallucination.construct_dataset
```

### Compute Neural Topology
Extract hidden-state correlations (and optionally sparse graphs) for every answer. Results are written under `data/hallucination/<model_name>[_step<ckpt>]`.

```bash
python -m hallucination.compute_llm_network --dataset_filename data/hallucination/truthfulqa-validation.csv \
  --llm_model_name <model_name> --ckpt_step <ckpt_step> --llm_layer <layer_id> \
  --batch_size <batch_size> --gpu_id <gpu_id_list> --num_workers <num_workers> \
  --network_density <density> [--sparse] [--resume]
```
All parameters are the same as previous steps.

### Train Hallucination Probes
Train probes on the extracted representations. Models are saved to `saves/hallucination/<model_name>/layer_<layer_id>` and TensorBoard logs under `runs/`.

```bash
python -m hallucination.train --dataset_filename data/hallucination/truthfulqa-validation.csv \
  --llm_model_name <model_name> --ckpt_step <ckpt_step> --llm_layer <layer_id> \
  --probe_input <activation|corr> --density <density> [--from_sparse_data] \
  --batch_size <batch_size> --eval_batch_size <eval_batch_size> --num_layers <num_layers> \
  --hidden_channels <hidden_dim> --dropout <dropout> --lr <learning_rate> --gpu_id <gpu_id>
```
All parameters are the same as previous steps.

### Evaluate Saved Probes
Load the best checkpoint and report accuracy, precision, recall, F1, and the confusion matrix.

```bash
python -m hallucination.eval --dataset_filename data/hallucination/truthfulqa-validation.csv \
  --llm_model_name <model_name> --ckpt_step <ckpt_step> --llm_layer <layer_id> \
  --probe_input <activation|corr> --density <density> --num_layers <num_layers> --gpu_id <gpu_id>
```
Ensure the flag values match those used during training so the correct checkpoint is loaded.

### Optional: Graph Statistics
Summarize intra- versus inter-answer correlations for each question to analyze topology differences.

```bash
python -m hallucination.graph_analysis --llm_model_name <model_name> --ckpt_step <ckpt_step> --layer <layer_id> --feature <corr|activation>
```
The script prints aggregate statistics and stores per-question metrics for downstream inspection.

## Graph Matching
Graph matching extends the graph probing framework to learn the topological similarity between two LLMs.


![Loading Model Overview](assets/graph_matching.png "Model Overview")
---
### Data Preparation
Run the following commands to generate textual dataset, and corresponding neural graphs for two LLMs. The data will be saved in the `data/graph_matching` directory.

```bash
python -m graph_matching.construct_dataset --dataset <dataset_name> 
python -m graph_matching.compute_llm_network --dataset_filename <dataset_filename> --llm_model_name <model_name_1> --llm_layer <layer_id_1> --batch_size <batch_size> 
python -m graph_matching.compute_llm_network --dataset_filename <dataset_filename> --llm_model_name <model_name_2> --llm_layer <layer_id_2> --batch_size <batch_size>
```
All parameters are the same as previous steps.

### Training Graph Matching Model
Run the following command to train the graph matching model. The trained model will be saved in the `saves/<model_name_1>_<model_name_2>` directory.

```bash
python -m graph_matching.train --dataset_filename <dataset_filename> --llm_model_name_1 <model_name_1> --llm_model_name_2 <model_name_2> --llm_layer_1 <layer_id_1> --llm_layer_2 <layer_id_2> --batch_size <batch_size> --eval_batch_size <eval_batch_size> --num_channels <num_channels> --num_layers <num_layers>
```
All parameters are the same as previous steps.

### Evaluating Graph Matching Model
Evaluation will be performed automatically during and after training. You can also run the following command to evaluate the saved graph matching model. 

```bash
python -m graph_matching.eval --dataset_filename <dataset_filename> --llm_model_name_1 <model_name_1> --llm_model_name_2 <model_name_2> --llm_layer_1 <layer_id_1> --llm_layer_2 <layer_id_2> --batch_size <batch_size> --eval_batch_size <eval_batch_size> --num_channels <num_channels> --num_layers <num_layers>
```
All parameters are the same as previous steps.

## Citation
If you use this code in your project, please consider citing the following paper:
```bibtex
@article{zheng2025probing,
  title={Probing Neural Topology of Large Language Models}, 
  author={Zheng, Yu and Yuan, Yuan and Li, Yong and Santi, Paolo},
  journal={arXiv preprint arXiv:2506.01042},
  year={2025}
}
```

## License
Please see the [license](LICENSE) for further details.
