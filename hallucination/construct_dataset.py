from absl import app
import os
from tqdm import tqdm

from datasets import load_dataset
import pandas as pd


def main(_):
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    data = []
    for i, example in enumerate(tqdm(dataset)):
        question = example["question"]
        
        for true_answer in example["correct_answers"]:
            data.append({"question_id": i, "question": question, "answer": true_answer, "label": 1})

        for false_answer in example["incorrect_answers"]:
            data.append({"question_id": i, "question": question, "answer": false_answer, "label": 0})

    df = pd.DataFrame(data)
    output_path = os.path.join("data/hallucination", f"truthfulqa-validation.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} prompts to {output_path}")


if __name__ == "__main__":
    app.run(main)
