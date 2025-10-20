from absl import app, flags
import os
from typing import Dict

from tqdm import tqdm
from datasets import load_dataset
import pandas as pd

flags.DEFINE_enum(
    "dataset_name",
    "truthfulqa",
    ["truthfulqa", "halubench"],
    "Which dataset to construct."
)
flags.DEFINE_string("output_dir", "data/hallucination", "Directory to save the constructed dataset.")
flags.DEFINE_string("truthfulqa_split", "validation", "Split to use for the TruthfulQA dataset.")
flags.DEFINE_string("halubench_split", "test", "Split to use for the HaluBench dataset.")
FLAGS = flags.FLAGS


def _build_truthfulqa(split: str, output_dir: str) -> str:
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split=split)

    records = []
    for i, example in enumerate(tqdm(dataset, desc="TruthfulQA examples")):
        question = example["question"]

        for true_answer in example["correct_answers"]:
            records.append({"question_id": i, "question": question, "answer": true_answer, "label": 1})

        for false_answer in example["incorrect_answers"]:
            records.append({"question_id": i, "question": question, "answer": false_answer, "label": 0})

    df = pd.DataFrame(records)
    output_path = os.path.join(output_dir, f"truthfulqa-{split}.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _normalize_halubench_label(raw_label: str, mapping: Dict[str, int]) -> int:
    try:
        return mapping[raw_label.upper()]
    except KeyError as error:
        raise ValueError(f"Unexpected HaluBench label: {raw_label}") from error


def _build_halubench(split: str, output_dir: str) -> str:
    dataset = load_dataset("PatronusAI/HaluBench", split=split)
    dataset = dataset.filter(lambda example: example["source_ds"] != "DROP")
    label_mapping = {"PASS": 1, "FAIL": 0}

    records = []
    for i, example in enumerate(tqdm(dataset, desc="HaluBench examples")):
        label = _normalize_halubench_label(example["label"], label_mapping)
        records.append(
            {
                "question_id": i,
                "question": f"{example['passage']} {example['question']}",
                "answer": example["answer"],
                "label": label,
                "source_ds": example["source_ds"],
                "example_id": example["id"],
            }
        )

    df = pd.DataFrame(records)
    output_path = os.path.join(output_dir, f"halubench-{split}.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main(_):
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    if FLAGS.dataset_name == "truthfulqa":
        output_path = _build_truthfulqa(FLAGS.truthfulqa_split, FLAGS.output_dir)
    elif FLAGS.dataset_name == "halubench":
        output_path = _build_halubench(FLAGS.halubench_split, FLAGS.output_dir)
    else:
        raise ValueError(f"Unsupported dataset: {FLAGS.dataset_name}")

    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    app.run(main)
