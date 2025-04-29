#Imports
import openai
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import csv
import time

OPENAI_API_KEY = "" # Input API Key (Redacted from codebase)
client = openai.OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = "gpt-4o-mini"  # Model used during this project


# This function makes the call to the OpenAI API
def call_openai_api(prompt_content):
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a phishing detection assistant. Your sole output should be the digit 0 or 1."
                },
                {"role": "user", "content": prompt_content}
            ],
            max_tokens=10,      # Only 10 tokens needed (Asking for just 0 or 1 as a labeling)
            temperature=0.1     # Low temperature gave better results.
        )
        
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content.strip()
            return content
        else:
            print("Unexpected response structure from OpenAI:", response)
            return None

    except Exception as e:
        print(f"OpenAI API request failed: {e}")
        return None

# This function's purpose is to split the results collected from GPT into 3 areas:
# correct predictions and incorrect predictions (split into false positives and false negative).
def group_results(samples, predictions):
    grouped = {"correct": [], "false_positive": [], "false_negative": []}

    for sample, api_pred in zip(samples, predictions):

        mapped_pred = api_pred
        gt = int(sample["label"])
        result = {
            "url": sample["url"],
            "label": gt,
            "api_prediction": api_pred,
            "mapped_prediction": mapped_pred
        }

        if mapped_pred == gt:
            # A correct prediction
            grouped["correct"].append(result)
        else:
            # A false positive (Labeled 0, predicted 1)
            if mapped_pred == 1 and gt == 0:
                grouped["false_positive"].append(result)
            # A false negative (Labeled 1, predicted 0)
            elif mapped_pred == 0 and gt == 1:
                grouped["false_negative"].append(result)
    return grouped

# Saves the results into the csv files
def write_grouped_results_to_csv(grouped_results):
    for group, items in grouped_results.items():
        filename = f"{group}_results.csv"
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["group", "url", "label", "api_prediction", "mapped_prediction"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(item)
        print(f"Group '{group}' results saved to '{filename}'.")


def main():
    # output_100.txt is the file which contained the 100 url + html examples which we used to test the model's accuracy.
    print("Getting prompts from output_100.txt now.")
    
    try:
        with open("output_100.txt", "r") as file:
            lines = file.readlines()
        prompts = []
        ground_truth_labels = []
        for line in lines:
            parts = line.strip().rsplit(",", 1)
            if len(parts) == 2:
                prompt = parts[0].strip()
                ground_truth = parts[1].strip()
                
                if ground_truth in ["0", "1"]:
                    prompts.append(prompt)
                    ground_truth_labels.append(ground_truth)
                else:
                    print(f"Skipping invalid ground truth for prompt: {prompt}")
            else:
                print(f"Skipping malformed line: {line.strip()}")
        # Confirmation for script user     
        print(f"Successfully loaded {len(prompts)} prompts from the file.")
    # Check to make sure file is really there.
    except FileNotFoundError:
        print("The file, 'output_100.txt', was not found. Add it into the directory before you run this script again.")
        return

    if not prompts:
        print("No valid prompts.")
        return

    # This is a list which stores the API predicitons
    raw_api_predictions = []
    
    # Each sample will now go through the OPEN API calls.
    print("Calling OpenAI API for each prompt now.")
    for prompt in tqdm(prompts, desc="Processing prompts"):
        api_response_content = call_openai_api(prompt)
        prediction = -1 # For reference (failed API call)
        if api_response_content is not None:
            if api_response_content == "1":
                prediction = 1
            elif api_response_content == "0":
                prediction = 0
            else:
                print(f"Unexpected response content for prompt '{prompt}': '{api_response_content}'")
        raw_api_predictions.append(prediction)

        # Due to limited compute/rate limits, sleep was added in.
        time.sleep(1)

    valid_indices = [i for i, pred in enumerate(raw_api_predictions) if pred != -1]
    
    if len(valid_indices) == 0:
        print("No valid predictions received from the API!")
        return

    # Find the samples that are valid
    valid_samples = [{"url": prompts[i], "label": ground_truth_labels[i]} for i in valid_indices]
    y_pred_api = [raw_api_predictions[i] for i in valid_indices]
    y_true_mapped = [int(ground_truth_labels[i]) for i in valid_indices]  # Compared to groud truth to make sure.
    y_pred_mapped = [int(prediction) for prediction in y_pred_api]

    # Additional check of length
    if len(y_true_mapped) != len(y_pred_mapped):
        print(f"Length mismatch: y_true_mapped has {len(y_true_mapped)} samples, but y_pred_mapped has {len(y_pred_mapped)} samples.")
        return


    print("\n--- Evaluation Metrics (1, Phishing) ---")
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1 = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n--- Evaluation Metrics (0, Not Phishing) ---")
    prec0 = precision_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    rec0 = recall_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    f10 = f1_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    print(f"Precision: {prec0:.4f}")
    print(f"Recall: {rec0:.4f}")
    print(f"F1 Score: {f10:.4f}")

    try:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print("\nConfusion Matrix (Rows: Actual, Columns: Predicted)")
        print("                     Predicted Not Phishing (0)  Predicted Phishing (1)")
        print(f"Actual Not Phishing (0): {tn:<20}  {fp:<20}")
        print(f"Actual Phishing (1):     {fn:<20}  {tp:<20}")
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"False Positive Rate (FPR): {false_positive_rate:.4f}")
        print(f"False Negative Rate (FNR): {false_negative_rate:.4f}")
    except ValueError as e:
        print(f"\nCould not compute confusion matrix. Error: {e}")
        tn = fp = fn = tp = 0

    grouped_results = group_results(valid_samples, y_pred_api)

    write_grouped_results_to_csv(grouped_results)

if __name__ == "__main__":
    main()
