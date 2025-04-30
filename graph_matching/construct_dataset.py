from absl import app, flags
import itertools
import os
import pickle
from tqdm import tqdm

import numpy as np
import torch.multiprocessing as mp

from datasets import load_dataset
from transformers import AutoTokenizer

flags.DEFINE_enum("dataset", "openwebtext", ["openwebtext", "pile"], "The name of the dataset.")
flags.DEFINE_integer("num_workers", 10, "Number of processes for computing sentences.")
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


def run_ppl(rank, queue, all_sentences, p_sentence_indices):
    sentences = [all_sentences[i] for i in p_sentence_indices[rank]]
    tokenizer_1 = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer_3 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer_4 = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    tokenizer_5 = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")

    tokenized_sentences_1 = tokenizer_1(sentences, padding=False, truncation=False)
    tokenized_sentences_2 = tokenizer_2(sentences, padding=False, truncation=False)
    tokenized_sentences_3 = tokenizer_3(sentences, padding=False, truncation=False)
    tokenized_sentences_4 = tokenizer_4(sentences, padding=False, truncation=False)
    tokenized_sentences_5 = tokenizer_5(sentences, padding=False, truncation=False)
    filtered_sentences = []
    for idx, sentence in enumerate(tqdm(sentences)):
        num_tokens_1 = len(tokenized_sentences_1["input_ids"][idx])
        num_tokens_2 = len(tokenized_sentences_2["input_ids"][idx])
        num_tokens_3 = len(tokenized_sentences_3["input_ids"][idx])
        num_tokens_4 = len(tokenized_sentences_4["input_ids"][idx])
        num_tokens_5 = len(tokenized_sentences_5["input_ids"][idx])
        min_num_tokens = min(num_tokens_1, num_tokens_2, num_tokens_3, num_tokens_4, num_tokens_5)
        max_num_tokens = max(num_tokens_1, num_tokens_2, num_tokens_3, num_tokens_4, num_tokens_5)
        if 256 <= min_num_tokens < 1024 and 256 <= max_num_tokens < 1024:
            filtered_sentences.append(sentence)
    queue.put((rank, filtered_sentences))


def main(_):
    if FLAGS.dataset == "openwebtext":
        dataset = load_dataset("sam2ai/openwebtext-10k", split="train", streaming=False)
    elif FLAGS.dataset == "pile":
        dataset = load_dataset("NeelNanda/pile-10k", split="train", streaming=False)
    else:
        raise ValueError(f"Dataset {FLAGS.dataset} not supported.")

    save_path = f"data/graph_matching/{FLAGS.dataset}-10k.pkl"
    dataset = dataset.filter(lambda x: x["text"] != "")
    dataset = dataset.map(split_examples, remove_columns=dataset.column_names)
    all_sentences = list(itertools.chain(*dataset["sentences"]))
    p_sentence_indices = np.array_split(np.arange(len(all_sentences)), FLAGS.num_workers)

    with mp.Manager() as manager:
        queue = manager.Queue()
        mp.spawn(
            run_ppl,
            args=(queue, all_sentences, p_sentence_indices),
            nprocs=FLAGS.num_workers,
            join=True,
        )
        mp_results = [queue.get() for _ in range(FLAGS.num_workers)]
        print(f"Received {len(mp_results)} results.")

    mp_results.sort(key=lambda x: x[0])
    filtered_sentences = []
    for _, sentences in mp_results:
        filtered_sentences.extend(sentences)

    saved_dataset = {
        "sentences": filtered_sentences,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(saved_dataset, f)


if __name__ == "__main__":
    app.run(main)
