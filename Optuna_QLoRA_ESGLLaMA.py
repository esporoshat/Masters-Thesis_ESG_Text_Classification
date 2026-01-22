import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
import torch
import csv
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import re
import optuna
import random

# ========== CONFIG ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Making sure huggingface datasets use the same seed
os.environ["PYTHONHASHSEED"] = str(SEED)


model_path = "saved_model"
TRAIN_PATH = "train_augmented_esg.csv"
VAL_PATH = "val_esg.csv"
TEST_PATH = "test_esg.csv"
TEXT_COL = "Text"
LABEL_COL = "ESG_Category"
LABELS = ["Environment", "Social", "Governance"]
LOG_FILE = "optuna_trials_log.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========== LOGGING FUNCTION ==========
def log_trial_result(trial_id, params, f1_score):
    log_exists = Path(LOG_FILE).exists()
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["trial_id", "learning_rate", "batch_size", "epochs", "grad_accum", "macro_f1"])
        writer.writerow([
            trial_id,
            params.get("learning_rate"),
            params.get("per_device_train_batch_size"),
            params.get("num_train_epochs"),
            params.get("gradient_accumulation_steps"),
            f1_score
        ])

# ========== MODEL INIT FUNCTION ==========
def model_init():
    tokenizer_local = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer_local.pad_token = tokenizer_local.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)
    return model

# ========== LOAD DATA ==========
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
train_dataset = load_dataset("csv", data_files={"train": TRAIN_PATH})["train"]
val_dataset = load_dataset("csv", data_files={"validation": VAL_PATH})["validation"]
test_dataset = load_dataset("csv", data_files={"test": TEST_PATH})["test"]

# ========== PREPROCESS ==========
def preprocess(example):
    prompt = f"""<s>[INST] <<SYS>>\nYou are an ESG classification assistant.\nClassify the text into exactly one category: Environment, Social, or Governance.\n<</SYS>>\n\nText: {example[TEXT_COL]}\nAnswer ONLY with a single label:[/INST]"""
    answer = f" {example[LABEL_COL]}"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels
    }

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

# ========== DATA COLLATOR ==========
class CustomDataCollatorForSeq2Seq:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        batch_keys = features[0].keys()
        batch = {}
        for key in batch_keys:
            max_length = max(len(feature[key]) for feature in features)
            padded = []
            for feature in features:
                seq = feature[key]
                pad_length = max_length - len(seq)
                if key == "labels":
                    padded.append(seq + [self.label_pad_token_id] * pad_length)
                else:
                    padded.append(seq + [self.tokenizer.pad_token_id] * pad_length)
            batch[key] = torch.tensor(padded, dtype=torch.long)
        return batch

data_collator = CustomDataCollatorForSeq2Seq(tokenizer)

# ========== PROMPT GENERATION ==========
def generate_prediction(example, model_instance):
    prompt = f"""<s>[INST] <<SYS>>\nYou are an ESG classification assistant.\nClassify the text into exactly one category: Environment, Social, or Governance.\n<</SYS>>\n\nText: {example[TEXT_COL]}\nAnswer ONLY with a single label:[/INST]"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_ids = model_instance.generate(input_ids, max_new_tokens=10, do_sample=False)
    output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    match = re.search(r'\b(Environment|Social|Governance)\b', output_text, re.IGNORECASE)
    return match.group(1).capitalize() if match else "None"

# ========== METRICS COMPUTATION ==========
def compute_metrics(eval_preds):
    preds, true_labels = [], []
    current_model = trainer.model
    for example in val_dataset:
        pred = generate_prediction(example, current_model)
        preds.append(pred)
        true_labels.append(example[LABEL_COL])
    macro_f1 = f1_score(true_labels, preds, average="macro")
    trial_id = trainer.state.trial_number if hasattr(trainer.state, "trial_number") else "manual"
    if hasattr(trainer.args, "learning_rate"):
        log_trial_result(trial_id, {
            "learning_rate": trainer.args.learning_rate,
            "per_device_train_batch_size": trainer.args.per_device_train_batch_size,
            "num_train_epochs": trainer.args.num_train_epochs,
            "gradient_accumulation_steps": trainer.args.gradient_accumulation_steps
        }, macro_f1)
    return {"eval_macro_f1": macro_f1}

# ========== HYPERPARAMETER SEARCH SPACE ==========
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1.5e-4, 3e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [1, 2]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [4, 8]),
    }

# ========== CUSTOM TRAINER ==========
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss.unsqueeze(0), outputs) if return_outputs else loss.unsqueeze(0)

    def training_step(self, *args, **kwargs):
        output = super().training_step(*args, **kwargs)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return output

# ========== TRAINING ARGS ==========
training_args = TrainingArguments(
    output_dir="./qlora_esg_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
    save_total_limit=2,
    fp16=True,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,
    seed = SEED
)

trainer = CustomTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ========== RUN HYPERPARAMETER SEARCH ==========
best_run = trainer.hyperparameter_search(
    hp_space=hp_space,
    direction="maximize",
    backend="optuna",
    n_trials=10,
    storage="sqlite:///optuna_esg_v2.db",
    study_name="qlora_esg_f1_v2",
    load_if_exists=True
)

print("Best hyperparameters found:")
print(best_run.hyperparameters)

for k, v in best_run.hyperparameters.items():
    setattr(trainer.args, k, v)

trainer.train()
#--------------------Evaluate------------------------
model = trainer.model
test_preds = []
test_labels = []
test_texts = []

for example in test_dataset:
    pred = generate_prediction(example, model)
    test_preds.append(pred)
    test_labels.append(example[LABEL_COL])
    test_texts.append(example[TEXT_COL])

# Save predictions to CSV
results_df = pd.DataFrame({
    "text": test_texts,
    "true_label": test_labels,
    "predicted_label": test_preds
})
results_df.to_csv("test_preds_optuna.csv", index=False)

print("Test predictions saved to test_preds_optuna.csv")

