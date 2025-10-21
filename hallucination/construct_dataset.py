from absl import app, flags
import os
from typing import List, Dict

from tqdm import tqdm
from datasets import load_dataset
import pandas as pd

flags.DEFINE_enum(
    "dataset_name",
    "truthfulqa",
    ["truthfulqa", "halueval", "medhallu"],
    "Which dataset to construct."
)
flags.DEFINE_string("output_dir", "data/hallucination", "Directory to save the constructed dataset.")
FLAGS = flags.FLAGS


def _build_truthfulqa() -> List[Dict]:
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    records = []
    for i, example in enumerate(tqdm(dataset, desc="TruthfulQA examples")):
        question = example["question"]

        for true_answer in example["correct_answers"]:
            records.append({"question_id": i, "question": question, "answer": true_answer, "label": 1})

        for false_answer in example["incorrect_answers"]:
            records.append({"question_id": i, "question": question, "answer": false_answer, "label": 0})

    return records

def _build_halueval() -> List[Dict]:
    dataset = load_dataset("pminervini/HaluEval", "qa", split="data")

    records = []
    for i, example in enumerate(tqdm(dataset, desc="HaluEval examples")):
        knowledge = example["knowledge"]
        question = example["question"]
        records.append({"question_id": i, "question": f"{knowledge} {question}", "answer": example["right_answer"], "label": 1})
        records.append({"question_id": i, "question": f"{knowledge} {question}", "answer": example["hallucinated_answer"], "label": 0})

    return records


def _build_medhallu() -> List[Dict]:
    dataset = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial", split="train")

    records = []
    for i, example in enumerate(tqdm(dataset, desc="MedHallu examples")):
        knowledge = example["Knowledge"]
        question = example["Question"]
        records.append({"question_id": i, "question": f"{knowledge} {question}", "answer": example["Ground Truth"], "label": 1})
        records.append({"question_id": i, "question": f"{knowledge} {question}", "answer": example["Hallucinated Answer"], "label": 0})

    return records


def main(_):
    if FLAGS.dataset_name == "truthfulqa":
        records = _build_truthfulqa()
    elif FLAGS.dataset_name == "halueval":
        records = _build_halueval()
    elif FLAGS.dataset_name == "medhallu":
        records = _build_medhallu()
    else:
        raise ValueError(f"Unsupported dataset: {FLAGS.dataset_name}")

    df = pd.DataFrame(records)
    output_path = os.path.join(FLAGS.output_dir, f"{FLAGS.dataset_name}.csv")
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    app.run(main)
