#Imports
import os
import time
import csv
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from anthropic import Anthropic

# Config
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
SAMPLE_LIMIT = 1000
FEVER_SPLIT = "labelled_dev"

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Build prompt
def build_prompt(claim: str) -> str:
    return (
        f'Given the following statement:\n\n"{claim}"\n\n'
        "Determine if it is TRUE or FALSE. ONLY OUTPUT 1 (TRUE) OR 0 (FALSE)."
    )

# Call Anthropic API
def call_anthropic(prompt: str) -> int:
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=5,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
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
            writer.writerows(items)
        print(f"  • Saved {len(items)} examples to {fname}")

# Main function
def main():
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Please set your ANTHROPIC_API_KEY environment variable.")

    print(f"Loading FEVER split '{FEVER_SPLIT}'…")
    ds = load_dataset("fever", "v1.0", split=FEVER_SPLIT, trust_remote_code=True)
    df = pd.DataFrame(ds)
    df = df[df["label"].isin(["SUPPORTS", "REFUTES"])]
    df["true_label"] = df["label"].map({"SUPPORTS": 1, "REFUTES": 0})

    print(f"  • {len(df)} claims after filtering.")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if SAMPLE_LIMIT:
        df = df.iloc[:SAMPLE_LIMIT]
    print(f"  • Evaluating {len(df)} statements…")

    y_true, y_pred = [], []
    for claim in tqdm(df["claim"], desc="Anthropic calls"):
        prompt = build_prompt(claim)
        try:
            pred = call_anthropic(prompt)
        except Exception as e:
            print("API error:", e)
            pred = -1
        y_pred.append(pred)
        y_true.append(df.loc[len(y_true), "true_label"])
        print(f"[{len(y_true)}] Claim: \"{claim}\" → Prediction: {pred}, True Label: {df.loc[len(y_true)-1, 'true_label']}")
        time.sleep(0.1)

    valid_idx = [i for i, p in enumerate(y_pred) if p != -1]
    y_true_val = [y_true[i] for i in valid_idx]
    y_pred_val = [y_pred[i] for i in valid_idx]
    claims_val = [df["claim"].iloc[i] for i in valid_idx]

    print("\nComputing metrics…")
    acc   = accuracy_score(y_true_val, y_pred_val)
    prec1 = precision_score(y_true_val, y_pred_val, pos_label=1, zero_division=0)
    rec1  = recall_score(y_true_val, y_pred_val, pos_label=1, zero_div_
