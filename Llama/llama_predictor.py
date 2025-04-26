import math
import pandas as pd
import csv
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub
from kagglehub import KaggleDatasetAdapter
import requests
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------
# Configuration and Constants for Together AI LLaMA
# ------------------------------------------------------------
TOGETHER_API_KEY = ""
LLAMA_MODEL = "meta-llama/Llama-3-8b-chat-hf"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
MAX_TOKENS = 10

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def build_single_prompt(item):
    return (
        f"Given this URL: {item['url']}, determine if it is a phishing website or not. "
        "ONLY OUTPUT 1 (PHISHING) OR 0 (NOT PHISHING). DO NOT WRITE ANYTHING ELSE."
    )

def call_llama_api(prompt_content):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_content}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }
    try:
        start_time = time.time()
        response = requests.post(TOGETHER_URL, headers=headers, json=payload)
        latency = time.time() - start_time
        result = response.json()
        return result['choices'][0]['message']['content'].strip(), latency
    except Exception as e:
        print("LLaMA API call failed:", e)
        return None, None

def fake_llama_response(prompt_content):
    # Simulate fake response and fake latency
    time.sleep(random.uniform(0.5, 2.0))
    return random.choice(["0", "1"]), random.uniform(0.5, 2.0)

def plot_latency_bars(latencies, query_ids):
    plt.figure(figsize=(10, 6))
    plt.bar(query_ids, latencies, color='skyblue')
    plt.xlabel('Query ID')
    plt.ylabel('Latency (seconds)')
    plt.title('Latency per Query')
    plt.savefig("latency_per_query.png", dpi=150)
    plt.show()

def plot_component_bars(prep_times, llama_times, query_ids):
    x = range(len(query_ids))
    plt.figure(figsize=(10, 6))
    plt.bar(x, prep_times, label='Preparation Time', color='lightgreen')
    plt.bar(x, llama_times, bottom=prep_times, label='LLaMA Response Time', color='lightcoral')
    plt.xlabel('Query ID')
    plt.ylabel('Time (seconds)')
    plt.title('Component-wise Latency')
    plt.xticks(x, query_ids)
    plt.legend()
    plt.savefig("component_latency.png", dpi=150)
    plt.show()

# ------------------------------------------------------------
# Main Script
# ------------------------------------------------------------
def main():
    # 5 manual samples
    samples = [
        {"url": "http://safe-example.com", "label": 0},
        {"url": "http://phish-login.com", "label": 1},
        {"url": "https://trustedsite.org", "label": 0},
        {"url": "http://bank-verification-required.com", "label": 1},
        {"url": "http://paypal-update-info.net", "label": 1},
    ]

    latencies = []
    prep_times = []
    llama_times = []
    true_labels = []
    predictions = []
    query_ids = [f"Q{i+1}" for i in range(len(samples))]

    print("Processing 5 Queries...")

    for item in samples:
        prep_start = time.time()
        prompt = build_single_prompt(item)
        prep_end = time.time()

        # Replace real call with fake if needed
        response, llama_latency = fake_llama_response(prompt)

        total_latency = (prep_end - prep_start) + llama_latency
        latencies.append(total_latency)
        prep_times.append(prep_end - prep_start)
        llama_times.append(llama_latency)

        if response in {"0", "1"}:
            predictions.append(int(response))
        else:
            predictions.append(-1)
        true_labels.append(item["label"])

    print("Done.")

    print("Plotting latencies...")
    plot_latency_bars(latencies, query_ids)
    plot_component_bars(prep_times, llama_times, query_ids)

    print("Analyzing predictions...")
    y_pred = [0 if p == 1 else 1 for p in predictions]
    acc = accuracy_score(true_labels, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Save results
    results_df = pd.DataFrame({
        "QueryID": query_ids,
        "TrueLabel": true_labels,
        "Prediction": predictions,
        "PrepTime": prep_times,
        "LlamaLatency": llama_times,
        "TotalLatency": latencies
    })
    results_df.to_csv("query_latency_results.csv", index=False)
    print("Saved all query results to query_latency_results.csv")

if __name__ == "__main__":
    main()
