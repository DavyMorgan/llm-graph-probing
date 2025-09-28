from absl import app
import os
import pandas as pd
from tqdm import tqdm

from datasets import load_dataset


subjects = [
    'abstract_algebra', 
    'anatomy', 
    'astronomy', 
    'business_ethics', 
    'clinical_knowledge', 
    'college_biology', 
    'college_chemistry', 
    'college_computer_science', 
    'college_mathematics', 
    'college_medicine', 
    'college_physics', 
    'computer_security', 
    'conceptual_physics', 
    'econometrics', 
    'electrical_engineering', 
    'elementary_mathematics', 
    'formal_logic', 
    'global_facts', 
    'high_school_biology', 
    'high_school_chemistry', 
    'high_school_computer_science', 
    'high_school_european_history', 
    'high_school_geography', 
    'high_school_government_and_politics', 
    'high_school_macroeconomics', 
    'high_school_mathematics', 
    'high_school_microeconomics', 
    'high_school_physics', 
    'high_school_psychology', 
    'high_school_statistics', 
    'high_school_us_history', 
    'high_school_world_history', 
    'human_aging', 
    'human_sexuality', 
    'international_law', 
    'jurisprudence', 
    'logical_fallacies', 
    'machine_learning', 
    'management', 
    'marketing', 
    'medical_genetics', 
    'miscellaneous', 
    'moral_disputes', 
    'moral_scenarios', 
    'nutrition', 
    'philosophy', 
    'prehistory', 
    'professional_accounting', 
    #'professional_law', 
    'professional_medicine', 
    'professional_psychology', 
    'public_relations', 
    'security_studies', 
    'sociology', 
    'us_foreign_policy', 
    'virology', 
    'world_religions']

def format_question(question, choices):
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(ord('A') + i)}) {choice}\n"
    prompt += "Answer:"
    return prompt

def main(_):
    dataset = load_dataset("cais/mmlu", "all", split="test")

    questions = []
    answers = []
    for example in tqdm(dataset):
        if example["subject"] in subjects:
            question = example["question"]
            choices = example["choices"]
            answer = example["answer"]

            formatted_question = format_question(question, choices)
            questions.append(formatted_question)
            answers.append(answer)

    output_path = os.path.join("data/mcq", f"mmlu-test.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"questions": questions, "answers": answers, "answer_indices": answers})
    df.to_csv(output_path, index=False)

    print(f"Saved {len(questions)} questions to {output_path}")

if __name__ == "__main__":
    app.run(main)
