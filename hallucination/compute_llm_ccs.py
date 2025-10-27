from absl import app, flags
from multiprocessing import Process, Queue
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from hallucination.utils import format_prompt_ccs
from utils.constants import hf_model_name_map
from utils.model_utils import load_tokenizer_and_model

flags.DEFINE_enum(
    "dataset_name",
    "truthfulqa",
    ["truthfulqa", "halueval", "medhallu", "helm"],
    "The dataset to process."
)
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_multi_integer("llm_layer", [12], "The layer list.")
flags.DEFINE_integer("batch_size", 16, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [0, 1], "The GPU ID.")
flags.DEFINE_integer("num_workers", 40, "Number of processes for computing networks.")
flags.DEFINE_boolean("resume", False, "Resume from the last generation.")
flags.DEFINE_boolean("random", False, "Whether to generate random data.")
FLAGS = flags.FLAGS

_WORD2VEC_MODEL = None


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
    random,
    suffix
):
    tokenizer, model = load_tokenizer_and_model(model_name, ckpt_step, gpu_id)

    df = pd.read_csv(dataset_filename)
    num_questions = len(df)
    original_input_questions = df["question"].tolist()[rank::num_producers]
    original_input_answers = df["answer"].tolist()[rank::num_producers]
        
    if not resume:
        input_texts = format_prompt_ccs(original_input_questions, original_input_answers, model_name, tokenizer, suffix)
        question_indices = list(range(rank, num_questions, num_producers))
    else:
        input_texts = []
        question_indices = []
        for i, question_idx in enumerate(range(rank, num_questions, num_producers)):
            if not os.path.exists(f"{p_save_path}/{question_idx}"):
                input_texts.extend(format_prompt_ccs([original_input_questions[i]], [original_input_answers[i]], model_name, tokenizer, suffix))
                question_indices.append(question_idx)

    if len(input_texts) > 0:
        tokenizer.pad_token = tokenizer.eos_token

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
                
                batch_hidden_states_all_layers = torch.stack(model_output.hidden_states[1:]).cpu().float().numpy() # (num_layers, batch_size, seq_length, hidden_size)
                batch_hidden_states = batch_hidden_states_all_layers[layer_list]
                if random:
                    batch_hidden_states = np.random.rand(*batch_hidden_states.shape)
                
                actual_batch_size = batch_hidden_states.shape[1]
                batch_question_indices = question_indices[i:i+actual_batch_size]

                queue.put((batch_hidden_states, batch_attention_mask.numpy(), batch_question_indices))


def run_save(queue, layer_list, p_save_path, worker_idx, suffix):
    with torch.no_grad():
        while True:
            batch = queue.get(block=True)
            if batch == "STOP":
                break
            hidden_states, attention_mask, question_indices = batch
            for i, question_idx in enumerate(question_indices):
                sentence_attention_mask = attention_mask[i]

                p_dir_name = f"{p_save_path}/{question_idx}"
                os.makedirs(p_dir_name, exist_ok=True)
                
                for j, layer_idx in enumerate(layer_list):
                    layer_hidden_states = hidden_states[j, i]
                    sentence_hidden_states = layer_hidden_states[sentence_attention_mask == 1].T
                    activation = sentence_hidden_states[:, -1]
                    np.save(f"{p_dir_name}/layer_{layer_idx}_activation_ccs_{suffix}.npy", activation)

    print(f"Worker {worker_idx} finished processing.")


def main(_):
    dataset_filename = os.path.join("data/hallucination", f"{FLAGS.dataset_name}.csv")
    dataset_dir = os.path.join("data/hallucination", FLAGS.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    model_name = FLAGS.llm_model_name
    if FLAGS.ckpt_step == -1:
        model_dir = model_name
    else:
        model_dir = f"{model_name}_step{FLAGS.ckpt_step}"
    dir_name = os.path.join(dataset_dir, model_dir)
    os.makedirs(dir_name, exist_ok=True)

    layer_list = FLAGS.llm_layer

    for suffix in ["yes", "no"]:
        print(f"Processing suffix: {suffix}")
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
                    dataset_filename,
                    hf_model_name,
                    FLAGS.ckpt_step,
                    gpu_id,
                    FLAGS.batch_size,
                    layer_list,
                    FLAGS.resume,
                    dir_name,
                    FLAGS.random,
                    suffix
                )
            )
            p.start()
            producers.append(p)

        num_workers = FLAGS.num_workers
        consumers = []
        for worker_idx in range(num_workers):
            p = Process(
                target=run_save,
                args=(queue, layer_list, dir_name, worker_idx, suffix)
            )
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
