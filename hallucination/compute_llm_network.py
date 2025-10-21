from absl import app, flags
from multiprocessing import Process, Queue
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import gensim.downloader as gensim_downloader
from gensim.utils import tokenize

from utils.constants import hf_model_name_map, BASE_MODELS, QWEN_CHAT_MODELS
from utils.model_utils import load_tokenizer_and_model

flags.DEFINE_enum(
    "dataset_name",
    "truthfulqa",
    ["truthfulqa", "halueval"],
    "The dataset to process."
)
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_multi_integer("llm_layer", [12], "The layer list.")
flags.DEFINE_integer("batch_size", 16, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [0, 1], "The GPU ID.")
flags.DEFINE_integer("num_workers", 40, "Number of processes for computing networks.")
flags.DEFINE_boolean("resume", False, "Resume from the last generation.")
flags.DEFINE_boolean("sparse", False, "Whether to generate sparse networks.")
flags.DEFINE_float("network_density", 1.0, "The density of the network.")
flags.DEFINE_boolean("random", False, "Whether to generate random data.")
FLAGS = flags.FLAGS

_WORD2VEC_MODEL = None


def get_word2vec_model():
    global _WORD2VEC_MODEL
    if _WORD2VEC_MODEL is None:
        _WORD2VEC_MODEL = gensim_downloader.load("word2vec-google-news-300")
    return _WORD2VEC_MODEL


def compute_word2vec_embedding(text, model):
    # Tokenize text while keeping case and removing punctuation accents
    tokens = list(tokenize(text, lowercase=False, deacc=True))
    tokens_in_vocab = [token for token in tokens if token in model.key_to_index]
    if not tokens_in_vocab:
        embedding = np.zeros(model.vector_size, dtype=np.float32)
    else:
        embedding = model.get_mean_vector(tokens_in_vocab).astype(np.float32)
    return embedding


def format_prompt(questions, answers, model_name, tokenizer):
    prompts = []
    for question, answer in zip(questions, answers):
        if model_name in QWEN_CHAT_MODELS:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(answer)}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, 
                return_tensors=None, return_dict=False, 
                tokenize=False, continue_final_message=False, enable_thinking=False)
        elif model_name in BASE_MODELS:
            prompt = f"Question: {question} Answer: {str(answer)}"
        else:
            raise NotImplementedError(f"Model {model_name} is not supported.")
        prompts.append(prompt)
    return prompts


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
    tokenizer, model = load_tokenizer_and_model(model_name, ckpt_step, gpu_id)

    df = pd.read_csv(dataset_filename)
    num_questions = len(df)
    original_input_questions = df["question"].tolist()[rank::num_producers]
    original_input_answers = df["answer"].tolist()[rank::num_producers]
    original_labels = df["label"].tolist()[rank::num_producers]
        
    if not resume:
        input_texts = format_prompt(original_input_questions, original_input_answers, model_name, tokenizer)
        labels = original_labels
        question_indices = list(range(rank, num_questions, num_producers))
    else:
        input_texts = []
        labels = []
        question_indices = []
        for i, question_idx in enumerate(range(rank, num_questions, num_producers)):
            if not os.path.exists(f"{p_save_path}/{question_idx}"):
                input_texts.extend(format_prompt([original_input_questions[i]], [original_input_answers[i]], model_name, tokenizer))
                labels.append(original_labels[i])
                question_indices.append(question_idx)

    if len(input_texts) > 0:
        tokenizer.pad_token = tokenizer.eos_token
        word2vec_model = get_word2vec_model()
        sample_word2vec_embeddings = np.stack([compute_word2vec_embedding(text, word2vec_model) for text in input_texts], axis=0)
        answer_tokens = [list(tokenizer.tokenize(answer)) for answer in original_input_answers]
        sample_word_token_counts = [len(tokens) for tokens in answer_tokens]

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
                
                # Get correctness label
                batch_labels = labels[i:i+batch_input_ids.shape[0]]

                batch_hidden_states_all_layers = torch.stack(model_output.hidden_states[1:]).cpu().float().numpy() # (num_layers, batch_size, seq_length, hidden_size)
                batch_hidden_states_layer_average = batch_hidden_states_all_layers.mean(axis=-1) # (num_layers, batch_size, seq_length)
                batch_hidden_states = batch_hidden_states_all_layers[layer_list]
                if random:
                    batch_hidden_states = np.random.rand(*batch_hidden_states.shape)
                
                actual_batch_size = batch_hidden_states.shape[1]
                batch_question_indices = question_indices[i:i+actual_batch_size]

                batch_word2vec_embeddings = sample_word2vec_embeddings[i:i+actual_batch_size]
                batch_word_token_counts = sample_word_token_counts[i:i+actual_batch_size]
                queue.put((batch_hidden_states_layer_average, batch_hidden_states, batch_attention_mask.numpy(), batch_question_indices, batch_labels, batch_word2vec_embeddings, batch_word_token_counts))


def run_corr(queue, layer_list, p_save_path, worker_idx, sparse=False, network_density=1.0):
    from torch_geometric.utils import dense_to_sparse
    with torch.no_grad():
        while True:
            batch = queue.get(block=True)
            if batch == "STOP":
                break
            hidden_states_layer_average, hidden_states, attention_mask, question_indices, labels, word2vec_embeddings, word_token_counts = batch
            for i, question_idx in enumerate(question_indices):
                sentence_attention_mask = attention_mask[i]

                p_dir_name = f"{p_save_path}/{question_idx}"
                os.makedirs(p_dir_name, exist_ok=True)
                
                # Save correctness label
                np.save(f"{p_dir_name}/label.npy", labels[i])

                layer_average_hidden_states = hidden_states_layer_average[:, i, sentence_attention_mask == 1]
                layer_average_corr = np.corrcoef(layer_average_hidden_states)
                np.save(f"{p_dir_name}/layer_average_corr.npy", layer_average_corr)

                layer_average_activation = layer_average_hidden_states[:, -1]
                np.save(f"{p_dir_name}/layer_average_activation.npy", layer_average_activation)

                layer_average_degree = np.abs(layer_average_corr).sum(axis=1)
                np.save(f"{p_dir_name}/layer_average_degree.npy", layer_average_degree)

                np.save(f"{p_dir_name}/word2vec_average.npy", word2vec_embeddings[i])
                np.save(f"{p_dir_name}/word2vec_token_count.npy", word_token_counts[i])

                for j, layer_idx in enumerate(layer_list):
                    layer_hidden_states = hidden_states[j, i]
                    sentence_hidden_states = layer_hidden_states[sentence_attention_mask == 1].T
                    corr = np.corrcoef(sentence_hidden_states)
                    
                    degree = np.abs(corr).sum(axis=1)
                    np.save(f"{p_dir_name}/layer_{layer_idx}_degree.npy", degree)

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
