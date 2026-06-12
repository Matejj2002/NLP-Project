"""
HAHA 2026 — Task 2: Exploratory Data Analysis
Saves 4 separate figures: fig1a, fig1b, fig2a, fig2b
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

DATA_DIR = Path(".")

trial    = pd.read_csv(DATA_DIR / "task2_trial.tsv",    sep="\t")
dev_gold = pd.read_csv(DATA_DIR / "task2_dev_gold.tsv", sep="\t")
train    = pd.concat([trial, dev_gold]).reset_index(drop=True)

train["text"]         = train["headline"].fillna("") + " " + train["joke"].fillna("")
train["text_len"]     = train["text"].str.len()
train["headline_len"] = train["headline"].fillna("").str.len()
train["joke_len"]     = train["joke"].fillna("").str.len()
train["avg_word_len"] = train["text"].apply(
    lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0
)

COLORS = {"human": "#4472C4", "machine": "#ED7D31"}
LABELS = ["human", "machine"]

# ── Figure 1a: Class Balance ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts = train["tag"].value_counts().reindex(LABELS)
bars = ax.bar(LABELS, counts.values, color=[COLORS[l] for l in LABELS], width=0.5)
ax.set_title("Class Balance", fontsize=12, fontweight="bold")
ax.set_xlabel("Tag")
ax.set_ylabel("Count")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("task2_fig1a_class_balance.png", dpi=150, bbox_inches="tight")
print("Saved task2_fig1a_class_balance.png")
plt.close()

# ── Figure 1b: Headline length vs Joke length ────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
for label in LABELS:
    subset = train[train["tag"] == label]
    ax.scatter(subset["headline_len"], subset["joke_len"],
               alpha=0.4, label=label, color=COLORS[label], s=20)
ax.set_title("Headline Length vs Joke Length", fontsize=12, fontweight="bold")
ax.set_xlabel("Headline length (chars)")
ax.set_ylabel("Joke length (chars)")
ax.legend(title="Tag")
plt.tight_layout()
plt.savefig("task2_fig1b_lengths_scatter.png", dpi=150, bbox_inches="tight")
print("Saved task2_fig1b_lengths_scatter.png")
plt.close()

# ── Figure 2a: Total Text Length Distribution ─────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
for label in LABELS:
    subset = train[train["tag"] == label]["text_len"]
    ax.hist(subset, bins=20, alpha=0.5, color=COLORS[label], label=label)
    kde = gaussian_kde(subset)
    x   = np.linspace(subset.min(), subset.max(), 200)
    ax.plot(x, kde(x) * len(subset) * (subset.max() - subset.min()) / 20,
            color=COLORS[label], linewidth=2)
ax.set_title("Text Length Distribution by Class", fontsize=11, fontweight="bold")
ax.set_xlabel("Number of characters (Headline + Joke)")
ax.set_ylabel("Frequency")
ax.legend(title="tag")
plt.tight_layout()
plt.savefig("task2_fig2a_text_length.png", dpi=150, bbox_inches="tight")
print("Saved task2_fig2a_text_length.png")
plt.close()

# ── Figure 2b: Average Word Length ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
for label in LABELS:
    subset = train[train["tag"] == label]["avg_word_len"]
    kde = gaussian_kde(subset)
    x   = np.linspace(subset.min(), subset.max(), 200)
    ax.fill_between(x, kde(x), alpha=0.4, color=COLORS[label], label=label)
    ax.plot(x, kde(x), color=COLORS[label], linewidth=2)
ax.set_title("Average Word Length", fontsize=12, fontweight="bold")
ax.set_xlabel("avg_word_length")
ax.set_ylabel("Density")
ax.legend(title="tag")
plt.tight_layout()
plt.savefig("task2_fig2b_word_length.png", dpi=150, bbox_inches="tight")
print("Saved task2_fig2b_word_length.png")
plt.close()
