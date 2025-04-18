#!/usr/bin/env python3
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

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
# 1) Your DeepSeek credentials & model
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-bf14f88d01d44b568bb8140b9687b6eb")
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# 2) How many statements to process (set None for the entire split)
SAMPLE_LIMIT = 1000

# 3) Which FEVER split to use
FEVER_SPLIT = "labelled_dev"  # or "train" / "test" if available

# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------
def build_prompt(claim: str) -> str:
    return (
        f"Given the following statement:\n\n\"{claim}\"\n\n"
        "Determine if it is TRUE or FALSE. "
        "ONLY OUTPUT 1 (TRUE) OR 0 (FALSE). DO NOT WRITE ANYTHING ELSE."
    )

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
    # extract the first choice
    text = data["choices"][0]["message"]["content"].strip()
    if text == "1":
        return 1
    if text == "0":
        return 0
    # unexpected output
    return -1

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
        print(f"  • Saved {len(items)} examples to {fname}")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    if DEEPSEEK_API_KEY.startswith("<"):
        raise RuntimeError("Please set your DEEPSEEK_API_KEY environment variable.")

    print(f"Loading FEVER split '{FEVER_SPLIT}'…")
    ds = load_dataset(
    "fever",                # dataset name
    "v1.0",                 # pick the config (v1.0 or v2.0)
    split=FEVER_SPLIT,      # e.g. "validation"
    trust_remote_code=True  # so it won’t prompt you again
    )

    # Filter to only SUPPORTS / REFUTES
    df = pd.DataFrame(ds)
    df = df[df["label"].isin(["SUPPORTS", "REFUTES"])]
    df["true_label"] = df["label"].map({
        "SUPPORTS": 1,
        "REFUTES":  0
    })

    print(f"  • {len(df)} claims after filtering to true/false.")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if SAMPLE_LIMIT:
        df = df.iloc[:SAMPLE_LIMIT]
    print(f"  • Evaluating {len(df)} statements…")

    y_true, y_pred = [], []
    for claim in tqdm(df["claim"], desc="DeepSeek calls"):
        prompt = build_prompt(claim)
        try:
            pred = call_deepseek(prompt)
        except Exception as e:
            print("API error:", e)
            pred = -1
        y_pred.append(pred)
        y_true.append(df.loc[len(y_true), "true_label"])
        time.sleep(0.1)  # small delay to ease rate limits

    # Filter out failures
    valid_idx = [i for i, p in enumerate(y_pred) if p != -1]
    y_true_val = [y_true[i] for i in valid_idx]
    y_pred_val = [y_pred[i] for i in valid_idx]
    claims_val = [df["claim"].iloc[i] for i in valid_idx]

    print("\nComputing metrics…")
    acc      = accuracy_score(y_true_val, y_pred_val)
    prec1    = precision_score(y_true_val, y_pred_val, pos_label=1, zero_division=0)
    rec1     = recall_score(y_true_val, y_pred_val, pos_label=1, zero_division=0)
    f11      = f1_score(y_true_val, y_pred_val, pos_label=1, zero_division=0)
    prec0    = precision_score(y_true_val, y_pred_val, pos_label=0, zero_division=0)
    rec0     = recall_score(y_true_val, y_pred_val, pos_label=0, zero_division=0)
    f10      = f1_score(y_true_val, y_pred_val, pos_label=0, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true_val, y_pred_val, labels=[0,1]).ravel()
    print(f"""
--- Results (n={len(y_true_val)}) ---
Accuracy:               {acc:.4f}
Precision (TRUE=1):     {prec1:.4f}
Recall (TRUE=1):        {rec1:.4f}
F1 (TRUE=1):            {f11:.4f}

Precision (FALSE=0):    {prec0:.4f}
Recall (FALSE=0):       {rec0:.4f}
F1 (FALSE=0):           {f10:.4f}

Confusion Matrix:
               Pred=F  Pred=T
Actual=F ({tn+fp}):     {tn:<5} {fp:<5}
Actual=T ({fn+tp}):     {fn:<5} {tp:<5}
""")

    # Save aggregate metrics
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric","value"])
        for k,v in [
            ("samples_evaluated", len(y_true_val)),
            ("accuracy", f"{acc:.4f}"),
            ("prec_true", f"{prec1:.4f}"),
            ("recall_true", f"{rec1:.4f}"),
            ("f1_true", f"{f11:.4f}"),
            ("prec_false", f"{prec0:.4f}"),
            ("recall_false", f"{rec0:.4f}"),
            ("f1_false", f"{f10:.4f}"),
            ("TN", tn), ("FP", fp), ("FN", fn), ("TP", tp),
        ]:
            writer.writerow([k, v])
    print("Saved aggregate metrics to results.csv")

    # Save grouped examples
    group_and_save(claims_val, y_true_val, y_pred_val)

if __name__ == "__main__":
    main()
