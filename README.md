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
python -m graph_probing.construct_dataset --dataset <dataset_name> --llm_model_name <model_name> --ckpt_step <ckpt_step> --batch_size <batch_size> --gpu_id 0 --gpu_id 1 --gpu_id 2 --gpu_id 3
```
`dataset_name`: The name of the dataset you want to use. You can choose from `openwebtext` and `pile`. In our experiments, we used `pile` for pythia models and `openwebtext` for other models.

`model_name`: The name of the LLM you want to probe. You can choose from `gpt2`, `gpt2-medium`, `gpt2-large`, `pythia-160m`, `pythia-410m`, `pythia-1.4b`, `pythia-2.8b`, `pythia-6.9b`, `pythia-12b`, `qwen2.5-0.5b`, `qwen2.5-3b`, `qwen2.5-7b`, and `qwen2.5-14b`.

`ckpt_step`: The checkpoint step of the LLM you want to probe. For pythia models, you can choose from `0`, `1`, `2`, `4`, `8`, `16`, `32`, `64`, `128`, `256`, `512`, and 143 evenly-spaced checkpoints from `1000` to `143000`. For other models, only `-1` is supported, which means the last checkpoint.

Then, run the following command to generate the neural graphs. The data will be saved in the `data/graph_probing/<model_name>` directory.

```bash
python -m graph_probing.compute_llm_network_sentence --dataset_filename <dataset_filename> --llm_model_name <model_name> --ckpt_step <ckpt_step> --llm_layer <layer_id> --batch_size <batch_size> --gpu_id 0 --gpu_id 1 --gpu_id 2 --gpu_id 3 --num_workers 20 --network_density <network_density>
```
`dataset_filename`: The filename of the dataset you want to use, which is generated in the previous step. For example, if you use `openwebtext` and `gpt2`, the filename will be `data/graph_probing/openwebtext-10k-gpt2.pkl`.

`llm_layer`: The layer ID of the LLM you want to probe.

`network_density`: The density of the neural graph you want to generate. You can choose any value that is greater than `0` and less than or equal to `1`. If you set if to be less than `1`, the graph will be sparsified. 

Other parameters are the same as the previous step.

### Training GNN Probes
Run the following command to train the GNN probes. The trained probes will be saved in the `saves/<model_name>` directory.

```bash
python -m graph_probing.train --dataset_filename <dataset_filename> --network_density <network_density> --from_sparse_data --llm_model_name <model_name> --ckpt_step <ckpt_step> --llm_layer <layer_id> --batch_size <batch_size> --eval_batch_size <eval_batch_size> --nolinear_probing --num_channels <num_channels> --num_layers <num_layers> --in_memory --gpu_id 0
```
`from_sparse_data`: If you are training probes on sparse graphs, set it to be `--from_sparse_data`. Otherwise, set it to be `--nofrom_sparse_data`.

`linear_probing`: If you are training linear probes, set it to be `--linear_probing`. Otherwise, set it to be `--nolinear_probing`.

`num_channels`: The number of channels in the GNN probes. The default value is `32`.

`num_layers`: The number of layers in the GNN probes. The default value is `1`.

`in_memory`: Load all graphs into memory before training. This will speed up the training process. If you have enough memory, set it to be `--in_memory`. Otherwise, set it to be `--noin_memory`.

Other parameters are the same as previous steps.

### Evaluating GNN Probes
GNN probes are evaluated automatically during and after training. You can also run the following command to evaluate the saved GNN probes. The evaluation results containing the predicted and groundtruth perplexity scores will be saved in the `saves/<model_name>` directory.

```bash
python -m graph_probing.eval --dataset_filename <dataset_filename> --network_density <network_density> --from_sparse_data --llm_model_name <model_name> --ckpt_step <ckpt_step> --llm_layer <layer_id> --batch_size <batch_size> --eval_batch_size <eval_batch_size> --nolinear_probing --num_channels <num_channels> --num_layers <num_layers> --in_memory --gpu_id 0
```
All parameters are the same as the previous training step.

## Graph Matching
Graph matching extends the graph probing framework to learn the topological similarity between two LLMs.


![Loading Model Overview](assets/graph_matching.png "Model Overview")
---
### Data Preparation
Run the following commands to generate textual dataset, and corresponding neural graphs for two LLMs. The data will be saved in the `data/graph_matching` directory.

```bash
python -m graph_matching.construct_dataset --dataset <dataset_name> 
python -m graph_matching.compute_llm_network_sentence --dataset_filename <dataset_filename> --llm_model_name <model_name_1> --llm_layer <layer_id_1> --batch_size <batch_size> --gpu_id 0 --gpu_id 1 --gpu_id 2 --gpu_id 3 --num_workers 20
python -m graph_matching.compute_llm_network_sentence --dataset_filename <dataset_filename> --llm_model_name <model_name_2> --llm_layer <layer_id_2> --batch_size <batch_size> --gpu_id 0 --gpu_id 1 --gpu_id 2 --gpu_id 3 --num_workers 20
```
All parameters are the same as previous steps.

### Training Graph Matching Model
Run the following command to train the graph matching model. The trained model will be saved in the `saves/<model_name_1>_<model_name_2>` directory.

```bash
python -m graph_matching.train --dataset_filename <dataset_filename> --llm_model_name_1 <model_name_1> --llm_model_name_2 <model_name_2> --llm_layer_1 <layer_id_1> --llm_layer_2 <layer_id_2> --batch_size <batch_size> --eval_batch_size <eval_batch_size> --num_channels <num_channels> --num_layers <num_layers> --in_memory --gpu_id 0
```
All parameters are the same as previous steps.

### Evaluating Graph Matching Model
Evaluation will be performed automatically during and after training. You can also run the following command to evaluate the saved graph matching model. 

```bash
python -m graph_matching.eval --dataset_filename <dataset_filename> --llm_model_name_1 <model_name_1> --llm_model_name_2 <model_name_2> --llm_layer_1 <layer_id_1> --llm_layer_2 <layer_id_2> --batch_size <batch_size> --eval_batch_size <eval_batch_size> --num_channels <num_channels> --num_layers <num_layers> --in_memory --gpu_id 0
```
All parameters are the same as previous steps.

## License
Please see the [license](LICENSE) for further details.
