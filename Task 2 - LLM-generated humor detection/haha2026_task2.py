"""
HAHA 2026 — Task 2: LLM-Generated Humor Detection
===================================================
Files:
  task2_trial.tsv    — 12 labeled examples  → training data
  task2_dev_gold.tsv — 350 labeled examples → used for training in test phase (all labeled data)
  task2_dev.tsv      — 350 unlabeled        → submit predictions to Codabench (dev phase)
  task2_test.tsv     — 550 unlabeled        → submit predictions to Codabench (eval phase)

Label: 'tag' column, values: 'human' | 'machine'
Metric: F1 of the 'machine' class (= the competition's 'automatic' class)
"""


# !pip install transformers datasets scikit-learn accelerate torch scipy sentencepiece



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")



#   Spanish models (best to worst for this task):
#   "dccuchile/bert-base-spanish-wwm-cased"   ← BETO
#   "xlm-roberta-base"                         
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

DATA_DIR    = Path(".")   # folder that contains the .tsv files
MAX_LEN     = 256
BATCH_SIZE  = 16          # reduce to 8 if you hit GPU OOM
EPOCHS      = 8
LR          = 2e-5
SEED        = 42

LABEL2ID = {"human": 0, "machine": 1}
ID2LABEL = {0: "human",  1: "machine"}

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

trial    = pd.read_csv(DATA_DIR / "task2_trial.tsv",    sep="\t")   # 12 rows, has 'tag'
dev_gold = pd.read_csv(DATA_DIR / "task2_dev_gold.tsv", sep="\t")   # 350 rows, has 'tag'
dev      = pd.read_csv(DATA_DIR / "task2_dev.tsv",      sep="\t")   # 350 rows, no 'tag'
test     = pd.read_csv(DATA_DIR / "task2_test.tsv",     sep="\t")   # 550 rows, no 'tag'

# Training set = trial + dev_gold (362 labeled examples)
train = pd.concat([trial, dev_gold]).reset_index(drop=True)

print(f"Train: {len(train)} | Dev-gold (local eval): {len(dev_gold)} | Dev (submit): {len(dev)} | Test (submit): {len(test)}")
print("Train label distribution:", train["tag"].value_counts().to_dict())

# ─────────────────────────────────────────────────────────────────────────────
# Build input text
# ─────────────────────────────────────────────────────────────────────────────
# Concatenating headline + joke gives the model the crucial context:
# does the joke match the headline in a "machine-like" way?

SEP = " [SEP] "

def make_text(df):
    return (df["headline"].fillna("") + SEP + df["joke"].fillna("")).tolist()

train_texts  = make_text(train)
train_labels = train["tag"].map(LABEL2ID).tolist()

gold_texts   = make_text(dev_gold)
gold_labels  = dev_gold["tag"].map(LABEL2ID).tolist()

dev_texts    = make_text(dev)
test_texts   = make_text(test)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenize
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_texts(texts, labels=None):
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    d = dict(enc)
    if labels is not None:
        d["labels"] = labels
    return Dataset.from_dict(d)

train_ds = tokenize_texts(train_texts, train_labels)
gold_ds  = tokenize_texts(gold_texts,  gold_labels)
dev_ds   = tokenize_texts(dev_texts)
test_ds  = tokenize_texts(test_texts)

train_ds.set_format("torch")
gold_ds.set_format("torch")
dev_ds.set_format("torch")
test_ds.set_format("torch")

print("Tokenization done.")

# ─────────────────────────────────────────────────────────────────────────────
#  Model with class-weighted loss
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

# Upweight 'machine' class slightly to push its F1 higher
# (classes are balanced 50/50, but we still slightly favor the target class)

CLASS_WEIGHTS = torch.tensor([1.0, 1.3]).to(DEVICE)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_machine": f1_score(labels, preds, pos_label=1, average="binary"),
        "f1_macro":   f1_score(labels, preds, average="macro"),
        "accuracy":   float((preds == labels).mean()),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./haha_task2_ckpt",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=32,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_machine",
    greater_is_better=True,
    fp16=(DEVICE == "cuda"),
    seed=SEED,
    logging_steps=10,
    report_to="none",
    save_total_limit=2,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=gold_ds,        # use dev_gold for monitoring during training
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("\nStarting training...")
trainer.train()

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate locally on dev_gold
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOCAL EVALUATION on dev_gold (350 labeled examples)")
print("="*60)

raw = trainer.predict(gold_ds)
preds_gold = np.argmax(raw.predictions, axis=-1)

print(classification_report(gold_labels, preds_gold, target_names=["human", "machine"]))
print(f">>> F1 'machine' (competition metric): {f1_score(gold_labels, preds_gold, pos_label=1):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Generate submission files
# ─────────────────────────────────────────────────────────────────────────────

def save_submission(ids, preds_int, filename):
    labels = [ID2LABEL[p] for p in preds_int]
    df = pd.DataFrame({"id": ids, "tag": labels})
    df.to_csv(filename, sep="\t", index=False)
    print(f"Saved {filename} — distribution: {pd.Series(labels).value_counts().to_dict()}")
    return df

# Dev submission (scored in the dev phase on Codabench)
raw_dev  = trainer.predict(dev_ds)
preds_dev = np.argmax(raw_dev.predictions, axis=-1)
save_submission(dev["id"], preds_dev, "task2_dev_submission.tsv")   # rename to task2.tsv before uploading

# Test submission (scored in the eval phase on Codabench)
raw_test  = trainer.predict(test_ds)
preds_test = np.argmax(raw_test.predictions, axis=-1)
save_submission(test["id"], preds_test, "task2_test_submission.tsv")  # rename to task2.tsv before uploading

print("\nDone! Upload task2_dev_submission.tsv or task2_test_submission.tsv to Codabench.")
print("Zip the file first if the platform requires it:")
print("  zip task2_submission.zip task2_dev_submission.tsv")


