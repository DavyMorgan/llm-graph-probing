hf_model_name_map = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "qwen1.5-0.5b": "Qwen/Qwen1.5-0.5B",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
}

QWEN_BASE_MODELS = [
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
]
QWEN_CHAT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]
QWEN_MODELS = [*QWEN_BASE_MODELS, *QWEN_CHAT_MODELS]

OPENAI_BASE_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large"
]
OPENAI_MODELS = [*OPENAI_BASE_MODELS]

PYTHIA_BASE_MODELS = [
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-160m-seed1",
    "EleutherAI/pythia-160m-seed2",
    "EleutherAI/pythia-160m-seed3"
]
PYTHIA_MODELS = [*PYTHIA_BASE_MODELS]

BASE_MODELS = [
    *OPENAI_BASE_MODELS,
    *PYTHIA_BASE_MODELS,
    *QWEN_BASE_MODELS,
]
