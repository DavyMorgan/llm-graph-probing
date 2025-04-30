from absl import app, flags
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
from setproctitle import setproctitle
from tqdm import tqdm

from multiprocessing import Process, Queue
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from graph_matching.utils import hf_model_name_map

flags.DEFINE_string("dataset_filename", "data/graph_matching/pile-10k.pkl", "The dataset filename.")
flags.DEFINE_string("llm_model_name", "gpt2", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_multi_integer("llm_layer", [0], "Layer IDs for network construction.")
flags.DEFINE_integer("batch_size", 8, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [7], "The GPU ID.")
flags.DEFINE_integer("num_workers", 10, "Number of processes for computing networks.")
flags.DEFINE_boolean("resume", False, "Resume from the last generation.")
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
):
    padding_side = "right" if model_name.startswith("gpt2") else "left"
    if ckpt_step == -1:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=f"cuda:{gpu_id}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side, revision=f'step{ckpt_step}')
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=f'step{ckpt_step}', device_map=f"cuda:{gpu_id}")

    with open(dataset_filename, "rb") as f:
        data = pickle.load(f)
        original_input_texts = data["sentences"]
        num_sentences = len(original_input_texts)
        if not resume:
            input_texts = original_input_texts[rank::num_producers]
            sentence_indices = list(range(rank, num_sentences, num_producers))
        else:
            input_texts = []
            sentence_indices = []
            for sentence_idx in range(rank, num_sentences, num_producers):
                if not os.path.exists(f"{p_save_path}/{sentence_idx}"):
                    input_texts.append(original_input_texts[sentence_idx])
                    sentence_indices.append(sentence_idx)

    if len(input_texts) > 0:
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(input_texts, padding=True, truncation=False, return_tensors="pt")

        with torch.no_grad():
            for i in tqdm(range(0, len(inputs["input_ids"]), batch_size), position=rank, desc=f"Producer {rank}"):
                model_output = model(
                    input_ids=inputs["input_ids"][i:i+batch_size].to(model.device),
                    attention_mask=inputs["attention_mask"][i:i+batch_size].to(model.device),
                    output_hidden_states=True,
                )
                batch_hidden_states = torch.stack(model_output.hidden_states[1:]).cpu().numpy()  # layer activations (num_layers, B, L, D)
                batch_hidden_states = batch_hidden_states[layer_list]
                batch_attention_mask = inputs["attention_mask"][i:i+batch_size].numpy()  # (B, L)
                actual_batch_size = batch_hidden_states.shape[1]
                batch_sentence_indices = sentence_indices[i:i+actual_batch_size]
                queue.put((batch_hidden_states, batch_attention_mask, batch_sentence_indices))


def run_corr(queue, layer_list, p_save_path, worker_idx):
    while True:
        batch = queue.get(block=True)
        if batch == "STOP":
            break
        hidden_states, attention_mask, sentence_indices = batch
        for i, sentence_idx in enumerate(sentence_indices):
            for j, layer_idx in enumerate(layer_list):
                layer_hidden_states = hidden_states[j, i]
                sentence_attention_mask = attention_mask[i]
                sentence_hidden_states = layer_hidden_states[sentence_attention_mask == 1].T
                corr = np.corrcoef(sentence_hidden_states)
                p_dir_name = f"{p_save_path}/{sentence_idx}"
                os.makedirs(p_dir_name, exist_ok=True)
                np.save(f"{p_dir_name}/layer_{layer_idx}_corr.npy", corr)

    print(f"Worker {worker_idx} finished processing.")


def main(_):
    model_name = FLAGS.llm_model_name
    if FLAGS.ckpt_step == -1:
        dir_name = f"data/graph_matching/{model_name}"
    else:
        dir_name = f"data/graph_matching/{model_name}_step{FLAGS.ckpt_step}"
    os.makedirs(dir_name, exist_ok=True)

    layer_list = FLAGS.llm_layer
    queue = Queue()
    producers = []
    hf_model_name = hf_model_name_map[model_name]
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
            )
        )
        p.start()
        producers.append(p)

    num_workers = FLAGS.num_workers
    consumers = []
    for worker_idx in range(num_workers):
        p = Process(
            target=run_corr,
            args=(queue, layer_list, dir_name, worker_idx))
        p.start()
        consumers.append(p)

    for producer in producers:
        producer.join()
    for _ in range(num_workers):
        queue.put("STOP")
    for consumer in consumers:
        consumer.join()


if __name__ == "__main__":
    setproctitle("debug@zhengyu")
    app.run(main)
