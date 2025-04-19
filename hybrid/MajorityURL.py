import math
import pandas as pd
import csv
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub
from kagglehub import KaggleDatasetAdapter
import json

# --- LLM API Client Imports ---
import anthropic
import openai
import google.generativeai as genai
import requests

# --- API Keys and Model Settings ---
ANTHROPIC_API_KEY = ""
DEEPSEEK_API_KEY = ""
GEMINI_API_KEY = ""
OPENAI_API_KEY = ""

ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
GEMINI_MODEL = "gemini-1.5-pro-latest"
OPENAI_MODEL = "gpt-4o"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)


# --- Prompt Template ---
def build_single_prompt(url):
    return (
        f"Given this URL: {url}, determine if it is a phishing website or not. "
        "ONLY OUTPUT 1 (PHISHING) OR 0 (NOT PHISHING). DO NOT WRITE ANYTHING ELSE."
    )


# --- Model Call Functions ---
def call_anthropic(prompt):
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0].text if hasattr(content[0], "text") else str(content[0])
        elif hasattr(content, "text"):
            content = content.text
        return str(content).strip()
    except Exception as e:
        print(f"Anthropic API Error: {e}")
        return "-1"

def map_anthropic_prediction(pred):
    try:
        return 0 if int(pred) == 1 else 1
    except ValueError:
        return -1


def call_deepseek(prompt):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.1
    }

    try:
        response = requests.post(DEEPSEEK_API_ENDPOINT, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        choice = response_data['choices'][0]
        content = choice['message']['content'].strip()
        return content if content in ("0", "1") else "-1"
    except Exception as e:
        print(f"DeepSeek API Error: {e}")
        return "-1"

def map_deepseek_prediction(pred):
    try:
        return 0 if int(pred) == 1 else 1
    except ValueError:
        return -1


def call_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        if response and response.text:
            stripped = response.text.strip()
            return stripped if stripped in ("0", "1") else "-1"
        return "-1"
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "-1"

def map_gemini_prediction(pred):
    try:
        return 0 if int(pred) == 1 else 1
    except ValueError:
        return -1


def call_openai_model(prompt):
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a phishing detection assistant. Output only 0 or 1."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        return content if content in ("0", "1") else "-1"
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "-1"

def map_openai_prediction(pred):
    try:
        return 0 if int(pred) == 1 else 1
    except ValueError:
        return -1


# --- Load Dataset ---
def load_dataset():
    file_path = "new_data_urls.csv"
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "harisudhan411/phishing-and-legitimate-urls",
            file_path
        )
    except Exception:
        df = pd.read_csv(file_path)
    if "url" not in df.columns or "status" not in df.columns:
        raise ValueError("Dataset must have 'url' and 'status' columns")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    samples = [
        {"url": r["url"], "label": int(r["status"])}
        for _, r in df.iterrows()
        if isinstance(r["url"], str) and r["status"] in {0, 1}
    ][:1]
    return samples


# --- Majority Vote Logic ---
def majority_vote(predictions):
    counts = {0: 0, 1: 0}
    for p in predictions:
        if p in (0, 1):
            counts[p] += 1
    if counts[0] > counts[1]:
        return 0
    elif counts[1] > counts[0]:
        return 1
    else:
        return 0  # Default on tie


# --- Main Script ---
def main():
    print("Loading dataset...")
    samples = load_dataset()
    print(f"Loaded {len(samples)} samples.")

    results = []
    for item in tqdm(samples, desc="Processing URLs"):
        url = item["url"]
        label = item["label"]
        prompt = build_single_prompt(url)

        # Query all models
        raw_gpt = call_openai_model(prompt)
        pred_gpt = map_openai_prediction(raw_gpt)

        raw_gemini = call_gemini(prompt)
        pred_gemini = map_gemini_prediction(raw_gemini)

        raw_anthropic = call_anthropic(prompt)
        pred_anthropic = map_anthropic_prediction(raw_anthropic)

        raw_deepseek = call_deepseek(prompt)
        pred_deepseek = map_deepseek_prediction(raw_deepseek)

        preds = [pred_gpt, pred_gemini, pred_anthropic, pred_deepseek]
        valid_preds = [p for p in preds if p in (0, 1)]
        maj = majority_vote(valid_preds) if len(valid_preds) >= 3 else -1

        results.append({
            "url": url,
            "label": label,
            "gpt": pred_gpt,
            "gemini": pred_gemini,
            "anthropic": pred_anthropic,
            "deepseek": pred_deepseek,
            "majority_vote": maj
        })

    pd.DataFrame(results).to_csv("ensemble_results.csv", index=False)
    print("Saved all predictions to ensemble_results.csv")

    y_true = [r["label"] for r in results if r["majority_vote"] in (0, 1)]
    y_pred = [r["majority_vote"] for r in results if r["majority_vote"] in (0, 1)]

    if y_true and y_pred:
        acc = accuracy_score(y_true, y_pred)
        prec1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f11 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        prec0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        rec0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        f10 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        metrics = {
            "Accuracy": acc,
            "Precision (Legit=1)": prec1,
            "Recall (Legit=1)": rec1,
            "F1 (Legit=1)": f11,
            "Precision (Phish=0)": prec0,
            "Recall (Phish=0)": rec0,
            "F1 (Phish=0)": f10,
            "FPR": fpr,
            "FNR": fnr
        }
        pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).to_csv("ensemble_metrics.csv", index=False)
        print("Saved ensemble metrics to ensemble_metrics.csv")
    else:
        print("No valid majority vote predictions to compute metrics.")


if __name__ == "__main__":
    main()
