from absl import app, flags
import itertools
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

from datasets import load_dataset
from evaluate import load
perplexity_revised = load("graph_probing/perplexity_revised.py", module_type="metric")
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from utils.constants import hf_model_name_map

flags.DEFINE_string("dataset", "openwebtext", "The name of the dataset.")
flags.DEFINE_string("llm_model_name", "gpt2", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [0, 1], "The GPU ID.")
FLAGS = flags.FLAGS


def split_examples(examples):
    text = examples["text"]
    sentences = text.split("\n\n")

    segments = []
    current_segment = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        # If current_segment is not empty and adding the sentence would exceed the limit,
        # finish the current segment and start a new one.
        if current_segment and (current_word_count + sentence_word_count > 800):
            segments.append("\n\n".join(current_segment))
            current_segment = [sentence]
            current_word_count = sentence_word_count
        else:
            current_segment.append(sentence)
            current_word_count += sentence_word_count

    # Append any remaining sentences as the last segment.
    if current_segment:
        segments.append("\n\n".join(current_segment))

    return {"sentences": segments}


def run_ppl(rank, queue, model_name, revision, gpu_id, batch_size, all_sentences, p_sentence_indices):
    sentences = [all_sentences[i] for i in p_sentence_indices[rank]]
    if model_name.startswith("EleutherAI"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_sentences = tokenizer(sentences, padding=False, truncation=False)
    filtered_sentences = []
    for idx, sentence in enumerate(tqdm(sentences)):
        if 256 <= len(tokenized_sentences["input_ids"][idx]) < 1024:
            filtered_sentences.append(sentence)
    results = perplexity_revised.compute(
        model_id=model_name,
        revision=revision,
        add_start_token=False,
        batch_size=batch_size,
        predictions=filtered_sentences,
        device=f"cuda:{gpu_id[rank]}",
        position=rank,
    )
    perplexities = results["perplexities"]
    queue.put((rank, filtered_sentences, perplexities))


def main(_):
    if FLAGS.dataset == "openwebtext":
        dataset = load_dataset("sam2ai/openwebtext-10k", split="train", streaming=False)
    else:
        raise ValueError(f"Unknown dataset: {FLAGS.dataset}")

    dataset = dataset.filter(lambda x: x["text"] != "")
    dataset = dataset.map(split_examples, remove_columns=dataset.column_names)
    all_sentences = list(itertools.chain(*dataset["sentences"]))
    p_sentence_indices = np.array_split(np.arange(len(all_sentences)), len(FLAGS.gpu_id))

    hf_model_name = hf_model_name_map[FLAGS.llm_model_name]
    revision = "main" if FLAGS.ckpt_step == -1 else f"step{FLAGS.ckpt_step}"

    with mp.Manager() as manager:
        queue = manager.Queue()
        mp.spawn(
            run_ppl,
            args=(queue, hf_model_name, revision, FLAGS.gpu_id, FLAGS.batch_size, all_sentences, p_sentence_indices),
            nprocs=len(FLAGS.gpu_id),
            join=True,
        )
        mp_results = [queue.get() for _ in range(len(FLAGS.gpu_id))]
        print(f"Received {len(mp_results)} results.")

    mp_results.sort(key=lambda x: x[0])
    filtered_sentences = []
    perplexities = []
    for _, sentences, ppls in mp_results:
        filtered_sentences.extend(sentences)
        perplexities.extend(ppls)

    min_ppl = np.percentile(perplexities, 1)
    max_ppl = np.percentile(perplexities, 99)
    print(f"min perplexity: {min_ppl}")
    print(f"max perplexity: {max_ppl}")

    saved_sentences = []
    saved_perplexities = []
    for sentence, ppl in tqdm(zip(filtered_sentences, perplexities)):
        if min_ppl <= ppl <= max_ppl:
            saved_sentences.append(sentence)
            saved_perplexities.append(ppl)

    saved_dataset = {
        "sentences": saved_sentences,
        "perplexities": saved_perplexities
    }

    if hf_model_name.startswith("EleutherAI") and revision != "main":
        save_path = f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}-{revision}.csv"
    else:
        save_path = f"data/graph_probing/{FLAGS.dataset}-10k-{FLAGS.llm_model_name}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(saved_dataset)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    app.run(main)
