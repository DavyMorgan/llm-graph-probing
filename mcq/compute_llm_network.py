from absl import app, flags
from multiprocessing import Process, Queue
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.constants import hf_model_name_map
from utils.model_utils import wrap_chat_template

flags.DEFINE_string("dataset_filename", "data/mcq/mmlu-test.csv", "The dataset filename.")
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b-instruct", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_multi_integer("llm_layer", [12], "The layer list.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [0, 1], "The GPU ID.")
flags.DEFINE_integer("num_workers", 30, "Number of processes for computing networks.")
flags.DEFINE_boolean("resume", False, "Resume from the last generation.")
flags.DEFINE_boolean("sparse", False, "Whether to generate sparse networks.")
flags.DEFINE_float("network_density", 1.0, "The density of the network.")
flags.DEFINE_boolean("random", False, "Whether to generate random data.")
FLAGS = flags.FLAGS


def run_llm(
    rank,
    num_producers,
    queue,
    dataset_filename,
    model_name,
    ckpt_step,
    gpu_id,
    batch_size,
    layer_list,
    resume,
    p_save_path,
    random
):
    padding_side = "left"
    if ckpt_step == -1:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=f"cuda:{gpu_id}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side, revision=f'step{ckpt_step}')
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=f'step{ckpt_step}', device_map=f"cuda:{gpu_id}")

    data = pd.read_csv(dataset_filename)
    original_input_texts = data["questions"].to_list()
    original_answers = data["answers"].to_list()
    num_questions = len(original_input_texts)
    
    if not resume:
        input_texts = original_input_texts[rank::num_producers]
        answers = original_answers[rank::num_producers]
        question_indices = list(range(rank, num_questions, num_producers))
    else:
        input_texts = []
        answers = []
        question_indices = []
        for question_idx in range(rank, num_questions, num_producers):
            if not os.path.exists(f"{p_save_path}/{question_idx}"):
                input_texts.append(original_input_texts[question_idx])
                answers.append(original_answers[question_idx])
                question_indices.append(question_idx)

    if len(input_texts) > 0:
        input_texts = wrap_chat_template(input_texts, tokenizer, model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        choice_tokens = [tokenizer.encode(c, add_special_tokens=False)[0] for c in ['A', 'B', 'C', 'D']]

        with torch.no_grad():
            for i in tqdm(range(0, len(input_texts), batch_size), position=rank, desc=f"Producer {rank}"):
                inputs = tokenizer(input_texts[i:i+batch_size], padding=True, truncation=False, return_tensors="pt")
                batch_input_ids = inputs["input_ids"]
                batch_attention_mask = inputs["attention_mask"]

                model_output = model(
                    input_ids=batch_input_ids.to(model.device),
                    attention_mask=batch_attention_mask.to(model.device),
                    output_hidden_states=True,
                )
                
                # Get LLM prediction
                last_token_logits = model_output.logits[:, -1, :]
                choice_logits = last_token_logits[:, choice_tokens].cpu().numpy()
                choice_logits = choice_logits/np.sum(choice_logits, axis=-1, keepdims=True)
                llm_predictions = np.argmax(choice_logits, axis=-1)
                
                # Get correctness label
                batch_answers = answers[i:i+batch_input_ids.shape[0]]
                is_correct = (llm_predictions == np.array(batch_answers)).astype(int)

                batch_hidden_states = torch.stack(model_output.hidden_states[1:]).cpu().numpy()
                batch_hidden_states = batch_hidden_states[layer_list]
                if random:
                    batch_hidden_states = np.random.rand(*batch_hidden_states.shape)
                
                actual_batch_size = batch_hidden_states.shape[1]
                batch_question_indices = question_indices[i:i+actual_batch_size]
                
                queue.put((batch_hidden_states, batch_attention_mask.numpy(), batch_question_indices, is_correct))


def run_corr(queue, layer_list, p_save_path, worker_idx, sparse=False, network_density=1.0):
    from torch_geometric.utils import dense_to_sparse
    with torch.no_grad():
        while True:
            batch = queue.get(block=True)
            if batch == "STOP":
                break
            hidden_states, attention_mask, question_indices, is_correct_labels = batch
            for i, question_idx in enumerate(question_indices):
                p_dir_name = f"{p_save_path}/{question_idx}"
                os.makedirs(p_dir_name, exist_ok=True)
                
                # Save correctness label
                np.save(f"{p_dir_name}/is_correct.npy", is_correct_labels[i])

                for j, layer_idx in enumerate(layer_list):
                    layer_hidden_states = hidden_states[j, i]
                    sentence_attention_mask = attention_mask[i]
                    sentence_hidden_states = layer_hidden_states[sentence_attention_mask == 1].T
                    corr = np.corrcoef(sentence_hidden_states)
                    
                    if not sparse:
                        np.save(f"{p_dir_name}/layer_{layer_idx}_corr.npy", corr)
                    else:
                        percentile_threshold = network_density * 100
                        threshold = np.percentile(np.abs(corr), 100 - percentile_threshold)
                        corr[np.abs(corr) < threshold] = 0
                        np.fill_diagonal(corr, 1.0)
                        corr = torch.from_numpy(corr)
                        edge_index, edge_attr = dense_to_sparse(corr)
                        edge_index = edge_index.numpy()
                        edge_attr = edge_attr.numpy()
                        np.save(f"{p_dir_name}/layer_{layer_idx}_sparse_{network_density}_edge_index.npy", edge_index)
                        np.save(f"{p_dir_name}/layer_{layer_idx}_sparse_{network_density}_edge_attr.npy", edge_attr)
                    
                    activation = sentence_hidden_states[:, -1]
                    np.save(f"{p_dir_name}/layer_{layer_idx}_activation.npy", activation)

    print(f"Worker {worker_idx} finished processing.")


def main(_):
    model_name = FLAGS.llm_model_name
    if FLAGS.ckpt_step == -1:
        dir_name = f"data/mcq/{model_name}"
    else:
        dir_name = f"data/mcq/{model_name}_step{FLAGS.ckpt_step}"
    os.makedirs(dir_name, exist_ok=True)

    layer_list = FLAGS.llm_layer

    queue = Queue()

    producers = []
    hf_model_name = hf_model_name_map.get(model_name, model_name)
    for i, gpu_id in enumerate(FLAGS.gpu_id):
        p = Process(
            target=run_llm,
            args=(
                i,
                len(FLAGS.gpu_id),
                queue,
                FLAGS.dataset_filename,
                hf_model_name,
                FLAGS.ckpt_step,
                gpu_id,
                FLAGS.batch_size,
                layer_list,
                FLAGS.resume,
                dir_name,
                FLAGS.random,
            )
        )
        p.start()
        producers.append(p)

    num_workers = FLAGS.num_workers
    consumers = []
    for worker_idx in range(num_workers):
        p = Process(
            target=run_corr,
            args=(queue, layer_list, dir_name, worker_idx, FLAGS.sparse, FLAGS.network_density))
        p.start()
        consumers.append(p)

    for producer in producers:
        producer.join()
    for _ in range(num_workers):
        queue.put("STOP")
    for consumer in consumers:
        consumer.join()


if __name__ == "__main__":
    app.run(main)
