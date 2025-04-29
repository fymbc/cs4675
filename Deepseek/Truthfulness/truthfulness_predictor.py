#Imports
import os
import time
import requests
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tqdm import tqdm
import csv

# Config
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-bf14f88d01d44b568bb8140b9687b6eb")
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
SAMPLE_LIMIT = 1000
FEVER_SPLIT = "labelled_dev"

# Build prompt
def build_prompt(claim: str) -> str:
    return (
        f"Given the following statement:\n\n\"{claim}\"\n\n"
        "Determine if it is TRUE or FALSE. "
        "ONLY OUTPUT 1 (TRUE) OR 0 (FALSE). DO NOT WRITE ANYTHING ELSE."
    )

# Call DeepSeek API
def call_deepseek(prompt: str) -> int:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5,
        "temperature": 0.0,
    }
    resp = requests.post(DEEPSEEK_ENDPOINT, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    if text == "1":
        return 1
    if text == "0":
        return 0
    return -1

# Group and save results
def group_and_save(samples, y_true, y_pred):
    groups = {"correct": [], "false_positive": [], "false_negative": []}
    for claim, true_lab, pred in zip(samples, y_true, y_pred):
        if pred == -1:
            continue
        if pred == true_lab:
            groups["correct"].append((claim, true_lab, pred))
        elif pred == 1 and true_lab == 0:
            groups["false_positive"].append((claim, true_lab, pred))
        else:
            groups["false_negative"].append((claim, true_lab, pred))

    for grp, items in groups.items():
        fname = f"{grp}_results.csv"
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["claim", "label", "prediction"])
            for c, l, p in items:
                writer.writerow([c, l, p])
        print(f"  • Saved {len(items)} examples