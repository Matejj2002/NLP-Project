"""
HAHA 2026 — Task 2: Multi-model ensemble
=========================================
Trains BETO and xlm-roberta-large (different architectures → more diverse errors)
then soft-votes their logits. More effective than same-model/different-seed ensemble.
"""

# !pip install transformers datasets scikit-learn accelerate torch

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

# ─────────────────────────────────────────────────────────────────────────────
# Config — each entry is (model_name, batch_size, seed)
# ─────────────────────────────────────────────────────────────────────────────
MODELS = [
    ("dccuchile/bert-base-spanish-wwm-cased", 16, 42),   # BETO
    ("xlm-roberta-large",                      8, 42),   # XLM-R large (smaller batch for memory)
]

DATA_DIR = Path(".")
MAX_LEN  = 256
EPOCHS   = 8
LR       = 2e-5

LABEL2ID = {"human": 0, "machine": 1}
ID2LABEL = {0: "human",  1: "machine"}
CLASS_WEIGHTS = torch.tensor([1.0, 1.3]).to(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

trial    = pd.read_csv(DATA_DIR / "task2_trial.tsv",    sep="\t")
dev_gold = pd.read_csv(DATA_DIR / "task2_dev_gold.tsv", sep="\t")
dev      = pd.read_csv(DATA_DIR / "task2_dev.tsv",      sep="\t")
test     = pd.read_csv(DATA_DIR / "task2_test.tsv",     sep="\t")

train = pd.concat([trial, dev_gold]).reset_index(drop=True)

print(f"Train: {len(train)} | Dev-gold: {len(dev_gold)} | Dev: {len(dev)} | Test: {len(test)}")
print("Train label distribution:", train["tag"].value_counts().to_dict())

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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_datasets(model_name):
    print(f"  Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(texts, labels=None):
        enc = tok(texts, truncation=True, padding="max_length", max_length=MAX_LEN)
        d = dict(enc)
        if labels is not None:
            d["labels"] = labels
        ds = Dataset.from_dict(d)
        ds.set_format("torch")
        return ds

    return (
        tokenize(train_texts, train_labels),
        tokenize(gold_texts,  gold_labels),
        tokenize(dev_texts),
        tokenize(test_texts),
    )

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_machine": f1_score(labels, preds, pos_label=1, average="binary"),
        "f1_macro":   f1_score(labels, preds, average="macro"),
        "accuracy":   float((preds == labels).mean()),
    }

def save_submission(ids, preds_int, filename):
    labels = [ID2LABEL[p] for p in preds_int]
    df = pd.DataFrame({"id": ids, "tag": labels})
    df.to_csv(filename, sep="\t", index=False)
    print(f"Saved {filename} — {pd.Series(labels).value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# Ensemble loop — one run per model
# ─────────────────────────────────────────────────────────────────────────────
all_gold_logits = []
all_dev_logits  = []
all_test_logits = []

for i, (model_name, batch_size, seed) in enumerate(MODELS):
    print(f"\n{'='*60}")
    print(f"  MODEL {i+1}/{len(MODELS)}: {model_name}  (seed={seed})")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds, gold_ds, dev_ds, test_ds = get_datasets(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=f"./haha_task2_ckpt_{model_name.replace('/', '_')}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=batch_size,
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
        seed=seed,
        logging_steps=10,
        report_to="none",
        save_total_limit=1,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=gold_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    all_gold_logits.append(trainer.predict(gold_ds).predictions)
    all_dev_logits.append(trainer.predict(dev_ds).predictions)
    all_test_logits.append(trainer.predict(test_ds).predictions)

    model_f1 = f1_score(gold_labels, np.argmax(all_gold_logits[-1], axis=-1), pos_label=1)
    print(f"{model_name} — F1 machine (dev_gold): {model_f1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Soft vote (average logits across models)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("ENSEMBLE RESULTS (soft vote: BETO + xlm-roberta-large)")
print(f"{'='*60}")

avg_gold  = np.mean(all_gold_logits, axis=0)
avg_dev   = np.mean(all_dev_logits,  axis=0)
avg_test  = np.mean(all_test_logits, axis=0)

preds_gold = np.argmax(avg_gold,  axis=-1)
preds_dev  = np.argmax(avg_dev,   axis=-1)
preds_test = np.argmax(avg_test,  axis=-1)

print(classification_report(gold_labels, preds_gold, target_names=["human", "machine"]))
print(f">>> Ensemble F1 'machine': {f1_score(gold_labels, preds_gold, pos_label=1):.4f}")

save_submission(dev["id"],  preds_dev,  "task2_dev_submission_ensemble.tsv")
save_submission(test["id"], preds_test, "task2_test_submission_ensemble.tsv")

print("\nDone! Upload task2_dev_submission_ensemble.tsv or task2_test_submission_ensemble.tsv to Codabench.")
