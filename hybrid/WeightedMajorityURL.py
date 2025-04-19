import math
import pandas as pd
import csv
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub
from kagglehub import KaggleDatasetAdapter
import json

# --- LLM API Client Imports (fill in as needed) ---
import anthropic
import openai
import google.generativeai as genai
import requests
# llamastuff

# --- API Keys and Model Settings (fill in your own keys) ---
ANTHROPIC_API_KEY = ""
DEEPSEEK_API_KEY = ""
GEMINI_API_KEY = ""
OPENAI_API_KEY = ""
LLAMA_MODEL_PATH = ""
#llamastuff

# --- Model Names/Endpoints (anything you need to add for model access basically) ---
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

GEMINI_MODEL = "gemini-1.5-pro-latest"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

OPENAI_MODEL = "gpt-4o"

#llamastuff

#weights config + decison threshold
MODEL_WEIGHTS = {
    'anthropic': 0.25,
    'deepseek': 0.25,
    'gemini': 0.25,
    'openai': 0.25
}
DECISION_THRESHOLD = 0.0


# --- Prompt Template ---
def build_single_prompt(url):
    return (
        f"Given this URL: {url}, determine if it is a phishing website or not. "
        "ONLY OUTPUT 1 (PHISHING) OR 0 (NOT PHISHING). DO NOT WRITE ANYTHING ELSE."
    )

# --- Model Call Functions (implement your own API logic as needed) ---
def call_anthropic(prompt):
    """Call Anthropic API with error handling and response normalization"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract content from different response formats
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0].text if hasattr(content[0], "text") else str(content[0])
        elif hasattr(content, "text"):
            content = content.text
        
        # Normalize response to 0/1 string
        return str(content).strip()
    except Exception as e:
        print(f"Anthropic API Error: {e}")
        return "-1"  # Flag for error state

def map_anthropic_prediction(pred):
    """Map Anthropic's prediction to -1 (not phishing) or 1 (phishing)"""
    try:
        return 1 if int(pred) == 1 else -1
    except ValueError:
        return 0  # Use 0 for invalid predictions to exclude from sum

def call_deepseek(prompt):
    """Call DeepSeek API with comprehensive error handling and response normalization"""
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
        if 'choices' not in response_data or not response_data['choices']:
            print("Empty choices in DeepSeek response")
            return "-1"
            
        choice = response_data['choices'][0]
        if 'message' not in choice or 'content' not in choice['message']:
            print("Malformed DeepSeek response structure")
            return "-1"
            
        content = choice['message']['content'].strip()
        return content if content in ("0", "1") else "-1"

    except requests.exceptions.Timeout:
        print("DeepSeek API timeout")
        return "-1"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("DeepSeek rate limit hit - implement backoff")
        else:
            print(f"DeepSeek HTTP error: {e}")
        return "-1"
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"DeepSeek response parsing error: {e}")
        return "-1"

def map_deepseek_prediction(pred):
    """Map DeepSeek's prediction to -1/1"""
    try:
        return 1 if int(pred) == 1 else -1
    except ValueError:
        return 0

def call_gemini(prompt, max_output_tokens=10):
    """Call Gemini API with output token limit and error handling"""
    try:
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_output_tokens}
        )
        if response and response.text:
            stripped = response.text.strip()
            return stripped if stripped in ("0", "1") else "-1"
        return "-1"
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "-1"

# Add mapping function for Gemini predictions
def map_gemini_prediction(pred):
    """Map Gemini's prediction to -1/1"""
    try:
        return 1 if int(pred) == 1 else -1
    except ValueError:
        return 0

def call_openai_model(prompt):
    """Call OpenAI API using provided prompt"""
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
        return 1 if int(pred) == 1 else -1
    except ValueError:
        return 0


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
    ][:1000]
    return samples

# weighted decs
def weighted_decision(predictions):
    """
    Calculate weighted sum of predictions
    predictions: list of tuples (model_name, mapped_prediction)
    Returns: 1 (phishing) if sum >= threshold, 0 (not phishing) otherwise
    """
    total = 0.0
    valid_models = 0
    
    for model_name, pred in predictions:
        if pred not in (-1, 1):
            continue  # Skip invalid predictions
            
        total += MODEL_WEIGHTS[model_name] * pred
        valid_models += 1
        
    if valid_models == 0:
        return -1  # No valid predictions
    
    return 1 if total >= DECISION_THRESHOLD else 0

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

        # Get and map predictions
        raw_anthropic = call_anthropic(prompt)
        raw_deepseek = call_deepseek(prompt)
        raw_gemini = call_gemini(prompt)
        raw_openai = call_openai_model(prompt)

        preds = [
            ('anthropic', map_anthropic_prediction(raw_anthropic)),
            ('deepseek', map_deepseek_prediction(raw_deepseek)),
            ('gemini', map_gemini_prediction(raw_gemini)),
            ('openai', map_openai_prediction(raw_openai))
        ]

        final_decision = weighted_decision(preds)

        results.append({
            "url": url,
            "label": label,
            "anthropic_raw": raw_anthropic,
            "deepseek_raw": raw_deepseek,
            "gemini_raw": raw_gemini,
            "openai_raw": raw_openai,
            "anthropic_mapped": preds[0][1],
            "deepseek_mapped": preds[1][1],
            "gemini_mapped": preds[2][1],
            "openai_mapped": preds[3][1],
            "weighted_score": final_decision
        })

    # Save results and metrics
    pd.DataFrame(results).to_csv("weighted_ensemble_results.csv", index=False)
    print("Saved predictions to weighted_ensemble_results.csv")

    # Calculate metrics (updated for new scoring)
    y_true = [r["label"] for r in results if r["weighted_score"] != -1]
    y_pred = [r["weighted_score"] for r in results if r["weighted_score"] != -1]
    
    if y_true and y_pred:
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (Phish)": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            "Recall (Phish)": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            "F1 (Phish)": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
            "FPR": confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()[1] / (len(y_true) - sum(y_true))
        }
        pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).to_csv("weighted_metrics.csv", index=False)
        print("Saved metrics to weighted_metrics.csv")

if __name__ == "__main__":
    main()