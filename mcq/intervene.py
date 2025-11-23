from absl import app, flags
from functools import partial
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
torch.set_grad_enabled(False)
import torch.multiprocessing as mp

from transformer_lens import HookedTransformer

from utils.constants import hf_model_name_map
from utils.model_utils import get_num_nodes, wrap_chat_template

flags.DEFINE_string("dataset_filename", "data/mcq/mmlu-test.csv", "The dataset filename.")
flags.DEFINE_integer("num_questions", 20000, "Number of questions.")
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b-instruct", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 12, "The layer of the LLM model.")
flags.DEFINE_float("intervention_frac", None, "Intervention fraction.")
flags.DEFINE_integer("intervention_num", None, "Number of nodes to intervene on.")
flags.DEFINE_boolean("largest", True, "Whether to ablate the largest nodes.")
flags.DEFINE_multi_integer("gpu_id", [0, 1], "The GPU ID.")
flags.DEFINE_integer("num_processes", None, "Number of processes. If None, equals number of GPUs.")
flags.DEFINE_boolean("skip_original", False, "Whether to skip original inference.")
flags.DEFINE_boolean("skip_random", False, "Whether to skip random ablation.")
flags.DEFINE_boolean("skip_degree", False, "Whether to skip degree ablation.")
flags.DEFINE_boolean("skip_activation", False, "Whether to skip activation ablation.")
flags.DEFINE_boolean("skip_weighted", False, "Whether to skip weighted activation ablation.")
FLAGS = flags.FLAGS


def ablation_hook(
    value,
    hook,
    index_to_ablate
):
    value[:, :, index_to_ablate] = 0.
    return value


def random_ablation_hook(
    value,
    hook,
    num_nodes,
    num_nodes_to_ablate
):
    index_to_ablate = torch.randperm(num_nodes)[:num_nodes_to_ablate]
    value[:, :, index_to_ablate] += torch.randn_like(value[:, :, index_to_ablate])
    return value


def degree_ablation_hook(
    value,
    hook,
    num_nodes_to_ablate,
    largest
):
    time_series = value[0].transpose(-2, -1) # D * T
    corr = torch.corrcoef(time_series)
    degree = torch.abs(corr).sum(dim=-1)
    _, index_to_ablate_top = torch.topk(degree, num_nodes_to_ablate, largest=largest)
    value[:, :, index_to_ablate_top] = 0.0
    return value


def activation_ablation_hook(
    value,
    hook,
    num_nodes_to_ablate,
    largest
):
    time_series = value[0].transpose(-2, -1) # D * T
    activation = torch.abs(time_series[:, -1])
    _, index_to_ablate_top = torch.topk(activation, num_nodes_to_ablate, largest=largest)
    value[:, :, index_to_ablate_top] = 0.0
    return value


def w_activation_ablation_hook(
    value,
    hook,
    num_nodes_to_ablate,
    largest,
    w_q_norm
):
    time_series = value[0].transpose(-2, -1) # D * T
    activation = torch.norm(time_series, p=2, dim=-1)
    score = activation * w_q_norm
    _, index_to_ablate_top = torch.topk(score, num_nodes_to_ablate, largest=largest)
    value[:, :, index_to_ablate_top] = 0.0
    return value

def run_intervention(
    rank,
    queues,
    model_name,
    ckpt_step,
    layer,
    inference_hook,
    r_inference_hook,
    random_inference_hook,
    gpu_ids,
    all_questions,
    all_answers,
    p_question_indices,
    num_nodes_to_ablate,
    largest,
    skip_original,
    skip_random,
    skip_degree,
    skip_activation,
    skip_weighted
):
    original_queue, intervened_queue, r_intervened_queue, random_intervened_queue, w_intervened_queue = queues
    questions = [all_questions[i] for i in p_question_indices[rank]]
    answers = [all_answers[i] for i in p_question_indices[rank]]

    gpu_id = gpu_ids[rank % len(gpu_ids)]
    if ckpt_step == -1:
        model = HookedTransformer.from_pretrained(model_name, device=f"cuda:{gpu_id}")
    else:
        model = HookedTransformer.from_pretrained(model_name, revision=f'step{ckpt_step}', device=f"cuda:{gpu_id}")
    
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"

    w_q_norm = model.W_V[layer + 1].abs().sum(dim=(0, 2))
    w_activation_inference_hook = partial(
        w_activation_ablation_hook,
        num_nodes_to_ablate=num_nodes_to_ablate,
        largest=largest,
        w_q_norm=w_q_norm
    )
    
    choice_tokens = [model.tokenizer.encode(c, add_special_tokens=False)[0] for c in ['A', 'B', 'C', 'D']]

    original_correct = []
    intervened_correct = []
    r_intervened_correct = []
    random_intervened_correct = []
    w_intervened_correct = []

    for i in tqdm(range(len(questions)), position=rank, desc=f"Worker {rank}"):
        question = questions[i]
        answer = answers[i]
        
        text = wrap_chat_template([question], model.tokenizer, model_name)[0]
        tokens = model.to_tokens(text).to(torch.device(f"cuda:{gpu_id}"))

        # Original
        if not skip_original:
            logits = model(tokens, return_type="logits")
            last_token_logits = logits[0, -1, :]
            choice_logits = last_token_logits[choice_tokens]
            prediction = torch.argmax(choice_logits).item()
            original_correct.append(int(prediction == answer))
        else:
            original_correct.append(0)

        # Intervention
        if not skip_degree:
            intervened_logits = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=[(
                    f"blocks.{layer}.hook_resid_post",
                    inference_hook,
                )]
            )
            last_token_logits = intervened_logits[0, -1, :]
            choice_logits = last_token_logits[choice_tokens]
            prediction = torch.argmax(choice_logits).item()
            intervened_correct.append(int(prediction == answer))
        else:
            intervened_correct.append(0)


        # Reverse intervention
        if not skip_activation:
            r_intervened_logits = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=[(
                    f"blocks.{layer}.hook_resid_post",
                    r_inference_hook,
                )]
            )
            last_token_logits = r_intervened_logits[0, -1, :]
            choice_logits = last_token_logits[choice_tokens]
            prediction = torch.argmax(choice_logits).item()
            r_intervened_correct.append(int(prediction == answer))
        else:
            r_intervened_correct.append(0)


        # Random intervention
        if not skip_random:
            random_intervened_logits = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=[(
                    f"blocks.{layer}.hook_resid_post",
                    random_inference_hook,
                )]
            )
            last_token_logits = random_intervened_logits[0, -1, :]
            choice_logits = last_token_logits[choice_tokens]
            prediction = torch.argmax(choice_logits).item()
            random_intervened_correct.append(int(prediction == answer))
        else:
            random_intervened_correct.append(0)


        # Weighted activation intervention
        if not skip_weighted:
            w_intervened_logits = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=[(
                    f"blocks.{layer}.hook_resid_post",
                    w_activation_inference_hook,
                )]
            )
            last_token_logits = w_intervened_logits[0, -1, :]
            choice_logits = last_token_logits[choice_tokens]
            prediction = torch.argmax(choice_logits).item()
            w_intervened_correct.append(int(prediction == answer))
        else:
            w_intervened_correct.append(0)

    original_queue.put((rank, original_correct))
    intervened_queue.put((rank, intervened_correct))
    r_intervened_queue.put((rank, r_intervened_correct))
    random_intervened_queue.put((rank, random_intervened_correct))
    w_intervened_queue.put((rank, w_intervened_correct))


def run_mp_intervention(hf_model_name, inference_hook, r_inference_hook, random_inference_hook, all_questions, all_answers, p_question_indices, num_processes, num_nodes_to_ablate):
    with mp.Manager() as manager:
        original_queue = manager.Queue()
        intervened_queue = manager.Queue()
        r_intervened_queue = manager.Queue()
        random_intervened_queue = manager.Queue()
        w_intervened_queue = manager.Queue()
        queues = (original_queue, intervened_queue, r_intervened_queue, random_intervened_queue, w_intervened_queue)

        mp.spawn(
            run_intervention,
            args=(queues, hf_model_name, FLAGS.ckpt_step, FLAGS.llm_layer, inference_hook, r_inference_hook, random_inference_hook, FLAGS.gpu_id, all_questions, all_answers, p_question_indices, num_nodes_to_ablate, FLAGS.largest, FLAGS.skip_original, FLAGS.skip_random, FLAGS.skip_degree, FLAGS.skip_activation, FLAGS.skip_weighted),
            nprocs=num_processes,
            join=True,
        )
        
        original_mp_results = [original_queue.get() for _ in range(num_processes)]
        intervened_mp_results = [intervened_queue.get() for _ in range(num_processes)]
        r_intervened_mp_results = [r_intervened_queue.get() for _ in range(num_processes)]
        random_intervened_mp_results = [random_intervened_queue.get() for _ in range(num_processes)]
        w_intervened_mp_results = [w_intervened_queue.get() for _ in range(num_processes)]

    original_mp_results.sort(key=lambda x: x[0])
    original_correct = []
    for _, l in original_mp_results:
        original_correct.extend(l)
    original_correct_ratio = np.mean(original_correct)

    intervened_mp_results.sort(key=lambda x: x[0])
    intervened_correct = []
    for _, l in intervened_mp_results:
        intervened_correct.extend(l)
    intervened_correct_ratio = np.mean(intervened_correct)

    r_intervened_mp_results.sort(key=lambda x: x[0])
    r_intervened_correct = []
    for _, l in r_intervened_mp_results:
        r_intervened_correct.extend(l)
    r_intervened_correct_ratio = np.mean(r_intervened_correct)
    
    random_intervened_mp_results.sort(key=lambda x: x[0])
    random_intervened_correct = []
    for _, l in random_intervened_mp_results:
        random_intervened_correct.extend(l)
    random_intervened_correct_ratio = np.mean(random_intervened_correct)

    w_intervened_mp_results.sort(key=lambda x: x[0])
    w_intervened_correct = []
    for _, l in w_intervened_mp_results:
        w_intervened_correct.extend(l)
    w_intervened_correct_ratio = np.mean(w_intervened_correct)

    return original_correct_ratio, intervened_correct_ratio, r_intervened_correct_ratio, random_intervened_correct_ratio, w_intervened_correct_ratio, original_correct, intervened_correct, r_intervened_correct, random_intervened_correct, w_intervened_correct


def main(_):
    assert FLAGS.intervention_num or FLAGS.intervention_frac, "You must specify either intervention_num or intervention_frac."
    if FLAGS.intervention_num is not None and FLAGS.intervention_frac is not None:
        raise ValueError("Cannot specify both intervention_num and intervention_frac.")
    
    data = pd.read_csv(FLAGS.dataset_filename)
    if FLAGS.num_questions < len(data):
        data = data.sample(n=FLAGS.num_questions, random_state=42).reset_index(drop=True)
    all_questions = data["questions"]
    all_answers = data["answer_indices"]

    hf_model_name = hf_model_name_map[FLAGS.llm_model_name]
    num_nodes = get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer)
    if FLAGS.intervention_num is not None:
        num_nodes_to_ablate = FLAGS.intervention_num
    else:
        num_nodes_to_ablate = int(num_nodes * FLAGS.intervention_frac)

    num_processes = FLAGS.num_processes if FLAGS.num_processes is not None else len(FLAGS.gpu_id)
    p_question_indices = np.array_split(np.arange(len(all_questions)), num_processes)
    
    random_inference_hook = partial(random_ablation_hook, num_nodes=num_nodes, num_nodes_to_ablate=num_nodes_to_ablate)
    degree_inference_hook = partial(degree_ablation_hook, num_nodes_to_ablate=num_nodes_to_ablate, largest=FLAGS.largest)
    activation_inference_hook = partial(activation_ablation_hook, num_nodes_to_ablate=num_nodes_to_ablate, largest=FLAGS.largest)
    degree_abl_title = "Degree Abl."
    activation_abl_title = "Activation Abl."

    original_correct_ratio, degree_intervened_correct_ratio, activation_intervened_correct_ratio, random_intervened_correct_ratio, w_intervened_correct_ratio, original_correct, degree_intervened_correct, activation_intervened_correct, random_intervened_correct, w_intervened_correct = run_mp_intervention(hf_model_name, degree_inference_hook, activation_inference_hook, random_inference_hook, all_questions, all_answers, p_question_indices, num_processes, num_nodes_to_ablate)
    
    if not FLAGS.skip_original:
        print(f"Original correct ratio: {original_correct_ratio:.4f}")
    else:
        print(f"Original correct ratio: Skipped")
    print("="*20)
    
    if not FLAGS.skip_degree:
        print(f"{degree_abl_title} correct ratio: {degree_intervened_correct_ratio:.4f} ({(degree_intervened_correct_ratio - original_correct_ratio) / original_correct_ratio * 100:+.2f if not FLAGS.skip_original else 'N/A'}%)")
    else:
        print(f"{degree_abl_title} correct ratio: Skipped")

    if not FLAGS.skip_activation:
        print(f"{activation_abl_title} correct ratio: {activation_intervened_correct_ratio:.4f} ({(activation_intervened_correct_ratio - original_correct_ratio) / original_correct_ratio * 100:+.2f if not FLAGS.skip_original else 'N/A'}%)")
    else:
        print(f"{activation_abl_title} correct ratio: Skipped")

    if not FLAGS.skip_weighted:
        print(f"Weighted Activation Abl. correct ratio: {w_intervened_correct_ratio:.4f} ({(w_intervened_correct_ratio - original_correct_ratio) / original_correct_ratio * 100:+.2f if not FLAGS.skip_original else 'N/A'}%)")
    else:
        print(f"Weighted Activation Abl. correct ratio: Skipped")

    if not FLAGS.skip_random:
        print(f"Random Abl. correct ratio: {random_intervened_correct_ratio:.4f} ({(random_intervened_correct_ratio - original_correct_ratio) / original_correct_ratio * 100:+.2f if not FLAGS.skip_original else 'N/A'}%)")
    else:
        print(f"Random Abl. correct ratio: Skipped")

    if not FLAGS.skip_degree and not FLAGS.skip_random:
        rel_drop_diff = (random_intervened_correct_ratio - degree_intervened_correct_ratio) / original_correct_ratio * 100 if original_correct_ratio > 0 else 0
        targeted_change = degree_intervened_correct_ratio - original_correct_ratio
        random_change = random_intervened_correct_ratio - original_correct_ratio
        if random_change != 0:
            ratio = targeted_change / random_change
        else:
            ratio = np.nan
        
        print(f"Targeted-Random Drop (%): {rel_drop_diff:.2f}")
        print(f"Targeted/Random Ratio: {ratio:.2f}" if not np.isnan(ratio) else "nan")
    else:
        print("Targeted-Random Drop (%): Skipped")
        print("Targeted/Random Ratio: Skipped")


if __name__ == "__main__":
    app.run(main)
