from absl import app, flags
import os
from typing import List, Dict
import urllib.error
import urllib.request

from tqdm import tqdm
from datasets import load_dataset
import pandas as pd

flags.DEFINE_enum(
    "dataset_name",
    "truthfulqa",
    ["truthfulqa", "halueval", "medhallu", "helm"],
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


def _build_helm() -> List[Dict]:
    import json
    
    raw_urls = [
        "https://raw.githubusercontent.com/oneal2000/MIND/refs/heads/main/helm/data/falcon40b/data.json",
        "https://raw.githubusercontent.com/oneal2000/MIND/refs/heads/main/helm/data/gptj7b/data.json",
        "https://raw.githubusercontent.com/oneal2000/MIND/refs/heads/main/helm/data/llamabase7b/data.json",
        "https://raw.githubusercontent.com/oneal2000/MIND/refs/heads/main/helm/data/llamachat13b/data.json",
        "https://raw.githubusercontent.com/oneal2000/MIND/refs/heads/main/helm/data/llamachat7b/data.json",
        "https://raw.githubusercontent.com/oneal2000/MIND/refs/heads/main/helm/data/mpt7b/data.json",
        "https://raw.githubusercontent.com/oneal2000/MIND/refs/heads/main/helm/data/opt7b/data.json"
    ]
    save_paths = []
    for url in raw_urls:
        model_name = url.split("/")[-2]
        save_path = os.path.join(FLAGS.output_dir, f"helm_{model_name}.json")
        save_paths.append(save_path)
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"Downloaded HELM {model_name} dataset to {save_path}")
        except urllib.error.URLError as e:
            print(f"Error downloading file: {e}")
        except IOError as e:
            print(f"Error writing file to path {save_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to download HELM dataset from {url}: {e}")

    records = []
    for save_path in save_paths:
        with open(save_path, 'r') as f:
            data = json.load(f)

        for question_id, (_, entry) in enumerate(tqdm(data.items(), desc="HELM examples")):
            prompt = entry["prompt"]
            for sentence_data in entry["sentences"]:
                sentence = sentence_data["sentence"]
                label = sentence_data["label"]
                if "\nlabel" in sentence or "\nlable" in sentence or "\nLabel" in sentence or "\nLable" in sentence or "\nlebel" in sentence or "\nLebel" in sentence:
                    continue
                records.append({
                    "question_id": question_id,
                    "question": prompt,
                    "answer": sentence,
                    "label": label
                })
    
    return records


def main(_):
    if FLAGS.dataset_name == "truthfulqa":
        records = _build_truthfulqa()
    elif FLAGS.dataset_name == "halueval":
        records = _build_halueval()
    elif FLAGS.dataset_name == "medhallu":
        records = _build_medhallu()
    elif FLAGS.dataset_name == "helm":
        records = _build_helm()
    else:
        raise ValueError(f"Unsupported dataset: {FLAGS.dataset_name}")

    df = pd.DataFrame(records)
    df = df.drop_duplicates().reset_index(drop=True)
    output_path = os.path.join(FLAGS.output_dir, f"{FLAGS.dataset_name}.csv")
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    app.run(main)
