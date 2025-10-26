from absl import app, flags
from multiprocessing import Process, Queue
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

from evaluate import load
perplexity_revised = load("graph_probing/perplexity_revised.py", module_type="metric")
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from hallucination.utils import format_prompt
from utils.constants import hf_model_name_map

flags.DEFINE_enum(
    "dataset_name",
    "truthfulqa",
    ["truthfulqa", "halueval", "medhallu", "helm"],
    "The dataset to process."
)
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("batch_size", 16, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [0, 1], "The GPU ID.")
flags.DEFINE_integer("num_workers", 4, "Number of processes for saving results.")
flags.DEFINE_boolean("resume", False, "Resume from the last generation.")
FLAGS = flags.FLAGS


def run_ppl(
    rank,
    num_producers,
    queue,
    dataset_filename,
    model_name,
    revision,
    gpu_id,
    batch_size,
    resume,
    p_save_path,
):
    df = pd.read_csv(dataset_filename)
    num_questions = len(df)
    original_input_questions = df["question"].tolist()[rank::num_producers]
    original_input_answers = df["answer"].tolist()[rank::num_producers]

    # Load tokenizer for formatting prompts
    if model_name.startswith("EleutherAI"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not resume:
        input_texts = format_prompt(original_input_questions, original_input_answers, model_name, tokenizer)
        question_indices = list(range(rank, num_questions, num_producers))
    else:
        input_texts = []
        question_indices = []
        for i, question_idx in enumerate(range(rank, num_questions, num_producers)):
            if not os.path.exists(f"{p_save_path}/{question_idx}/perplexity.npy"):
                input_texts.extend(format_prompt([original_input_questions[i]], [original_input_answers[i]], model_name, tokenizer))
                question_indices.append(question_idx)

    if len(input_texts) > 0:
        results = perplexity_revised.compute(
            model_id=model_name,
            revision=revision,
            add_start_token=False,
            batch_size=batch_size,
            predictions=input_texts,
            device=f"cuda:{gpu_id}",
            position=rank,
        )
        perplexities = results["perplexities"]

        # Put results in queue for consumer processes to save
        for question_idx, ppl in zip(question_indices, perplexities):
            queue.put((question_idx, ppl))


def run_save(queue, p_save_path, worker_idx):
    """Consumer process to save perplexity results."""
    while True:
        batch = queue.get(block=True)
        if batch == "STOP":
            break
        question_idx, perplexity = batch

        p_dir_name = f"{p_save_path}/{question_idx}"
        os.makedirs(p_dir_name, exist_ok=True)

        # Save perplexity
        np.save(f"{p_dir_name}/perplexity.npy", perplexity)

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

    hf_model_name = hf_model_name_map.get(model_name, model_name)
    revision = "main" if FLAGS.ckpt_step == -1 else f"step{FLAGS.ckpt_step}"

    queue = Queue()

    producers = []
    for i, gpu_id in enumerate(FLAGS.gpu_id):
        p = Process(
            target=run_ppl,
            args=(
                i,
                len(FLAGS.gpu_id),
                queue,
                dataset_filename,
                hf_model_name,
                revision,
                gpu_id,
                FLAGS.batch_size,
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
            target=run_save,
            args=(queue, dir_name, worker_idx))
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
