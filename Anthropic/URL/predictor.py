import math
import pandas as pd
import csv
import time  # for potential rate limiting/retries
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub
from kagglehub import KaggleDatasetAdapter
import anthropic  # Import the Anthropic client library
import matplotlib.pyplot as plt  # For plotting the table and confusion matrix

# ------------------------------------------------------------
# Configuration and Constants for Anthropic
# ------------------------------------------------------------
ANTHROPIC_API_KEY = ""  # actual key ommitted

# Anthropic API settings
# Adjust the model name and token count as needed.
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 10  # Limit tokens to get only a short answer (0/1)

# Create a client instance (reusing the client for all API calls)
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def build_single_prompt(item):
    """
    Build the prompt in Anthropic's conversation style.
    The prompt instructs the model to determine whether a URL is a phishing website.
    """
    prompt_content = (
        f"Given this URL: {item['url']}, determine if it is a phishing website or not. "
        "ONLY OUTPUT 1 (PHISHING) OR 0 (NOT PHISHING). DO NOT WRITE ANYTHING ELSE."
    )
    return prompt_content

def call_anthropic_api(prompt_content):
    """
    Call the Anthropic API using the official Anthropic client.
    Returns the stripped text extracted from the API's reply or None if an error occurs.
    """
    try:
        response_message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,  # Ensure the reply remains short
            messages=[
                {"role": "user", "content": prompt_content}
            ]
        )
        content = response_message.content

        # If content is a list, extract the text attribute from the first element.
        if isinstance(content, list):
            if len(content) > 0:
                first_item = content[0]
                if hasattr(first_item, "text"):
                    content = first_item.text
                else:
                    content = str(first_item)
            else:
                content = ""
        # If content is directly a TextBlock (or other object) with a text attribute, extract it.
        elif hasattr(content, "text"):
            content = content.text

        # Ensure we have a string; if not, convert it.
        if not isinstance(content, str):
            content = str(content)
        return content.strip()
    except Exception as e:
        print("API call failed:", e)
        return None

def group_results(samples, predictions):
    """
    Group the results into three categories: correct, false positive, and false negative.
    The function maps API predictions to the Kaggle label semantics:
      - The Kaggle dataset labels: 0 indicates phishing, 1 indicates legitimate.
      - The API is expected to output:
            "1" meaning phishing, which maps to Kaggle label 0.
            "0" meaning not phishing, which maps to Kaggle label 1.
    """
    grouped = {"correct": [], "false_positive": [], "false_negative": []}
    for sample, api_pred in zip(samples, predictions):
        if api_pred == 1:
            mapped_pred = 0  # API output 1 indicates phishing → matches Kaggle label 0
        elif api_pred == 0:
            mapped_pred = 1  # API output 0 indicates not phishing → matches Kaggle label 1
        else:
            print(f"Skipping grouping for URL {sample['url']} due to invalid prediction {api_pred}")
            continue

        result = {
            "url": sample["url"],
            "label": sample["label"],
            "api_prediction": api_pred,
            "mapped_prediction": mapped_pred
        }
        if mapped_pred == sample["label"]:
            grouped["correct"].append(result)
        else:
            if mapped_pred == 1 and sample["label"] == 0:
                grouped["false_positive"].append(result)
            elif mapped_pred == 0 and sample["label"] == 1:
                grouped["false_negative"].append(result)
            else:
                print(f"Unexpected grouping condition: Label={sample['label']}, MappedPred={mapped_pred}")
    return grouped

def write_grouped_results_to_csv(grouped_results):
    """
    Write the grouped results into separate CSV files (correct, false positive, false negative).
    Each CSV file includes the following columns:
    group, url, label, api_prediction, mapped_prediction.
    """
    for group, items in grouped_results.items():
        filename = f"{group}_results.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["group", "url", "label", "api_prediction", "mapped_prediction"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                row = {
                    "group": group,
                    "url": item["url"],
                    "label": item["label"],
                    "api_prediction": item["api_prediction"],
                    "mapped_prediction": item["mapped_prediction"]
                }
                writer.writerow(row)
        print(f"Group '{group}' results saved to '{filename}'.")

# ------------------------------------------------------------
# Main Script
# ------------------------------------------------------------
def main():
    # 1. Load the dataset from KaggleHub.
    print("Loading dataset from KaggleHub...")
    df = None
    file_path = "new_data_urls.csv"  # The file within the dataset

    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,  # Load as a Pandas DataFrame
            "harisudhan411/phishing-and-legitimate-urls",  # Dataset handle (owner/dataset-slug)
            file_path  # The path within the dataset
        )
        print("Successfully loaded dataset using kagglehub.load_dataset.")
    except Exception as e:
        print(f"Error loading dataset from KaggleHub: {e}")
        print("Ensure your Kaggle API credentials are configured and the dataset exists.")
        print(f"\nAttempting to load '{file_path}' from the current directory as a fallback...")
        try:
            df = pd.read_csv(file_path)
            print("Successfully loaded from local directory.")
        except Exception as e_local:
            print(f"Failed to load the dataset locally: {e_local}")
            return

    if df is None:
        print("DataFrame could not be loaded. Exiting.")
        return

    print("Total rows in dataset:", len(df))
    if "url" not in df.columns or "status" not in df.columns:
        print("Error: The dataset is missing required columns ('url' and 'status').")
        print("Available columns:", df.columns)
        return

    # Shuffle the DataFrame rows to randomize the order
    print("Shuffling the DataFrame rows...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("DataFrame successfully shuffled.")

    # 2. Prepare samples.
    samples = []
    for _, row in df.iterrows():
        if row["url"] and isinstance(row["url"], str) and not pd.isna(row["status"]):
            try:
                label = int(row["status"])
                if label in [0, 1]:  # Only process valid labels
                    sample = {
                        "url": row["url"],
                        "html": "",  # Dataset does not include HTML content.
                        "label": label  # 0 for phishing; 1 for legitimate.
                    }
                    samples.append(sample)
            except ValueError:
                continue

    sample_limit = 1000  # Adjust the number of samples for testing as needed
    samples = samples[:sample_limit]
    print(f"Prepared {len(samples)} valid samples for evaluation (limited to first {sample_limit}).")

    if not samples:
        print("No valid samples found to process.")
        return

    # 3. Process each sample with an Anthropic API call.
    raw_api_predictions = []  # To store raw API output (expected as "0" or "1")
    ground_truth_labels = []  # To store corresponding Kaggle labels

    print("Calling Anthropic API for each sample...")
    for item in tqdm(samples, desc="Processing samples"):
        prompt_content = build_single_prompt(item)
        api_response_content = call_anthropic_api(prompt_content)
        prediction = -1  # Default value if the API call fails or returns an unexpected response
        if api_response_content is not None:
            # We expect the API to return a single digit string: "1" or "0".
            if api_response_content == "1":
                prediction = 1  # Indicates phishing
            elif api_response_content == "0":
                prediction = 0  # Indicates not phishing
            else:
                print(f"Unexpected response for URL '{item['url']}': '{api_response_content}'")
        raw_api_predictions.append(prediction)
        ground_truth_labels.append(item["label"])
        # Optionally, uncomment the next line to add a delay and avoid rate limits.
        # time.sleep(0.1)

    # 4. Filter out samples where API predictions failed (prediction == -1).
    valid_indices = [i for i, pred in enumerate(raw_api_predictions) if pred != -1]

    if len(valid_indices) == 0:
        print("No valid predictions received from the API!")
        return

    valid_samples = [samples[i] for i in valid_indices]
    y_true_mapped = [ground_truth_labels[i] for i in valid_indices]  # Ground truth labels
    y_pred_api = [raw_api_predictions[i] for i in valid_indices]       # Raw API predictions

    # Map API predictions to match Kaggle label semantics:
    # API output: 1 (Phishing) → mapped to Kaggle label 0; 0 (Not phishing) → mapped to Kaggle label 1.
    y_pred_mapped = [0 if pred == 1 else 1 for pred in y_pred_api]

    print(f"\nProcessed {len(samples)} samples, received {len(valid_indices)} valid predictions.")

    # 5. Compute evaluation metrics.
    print("\n--- Evaluation Metrics (Positive Label = 1: Legitimate) ---")
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec_legit = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec_legit = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1_legit = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)

    print("\n--- Evaluation Metrics (Positive Label = 0: Phishing) ---")
    prec_phish = precision_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    rec_phish = recall_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    f1_phish = f1_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)

    try:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print("\nConfusion Matrix (Rows: Actual, Columns: Predicted)")
        print("                     Predicted Phishing (0)  Predicted Legitimate (1)")
        print(f"Actual Phishing (0)      {tn:<20}  {fp:<20}")
        print(f"Actual Legitimate (1)    {fn:<20}  {tp:<20}")

        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    except Exception as e:
        print(f"\nCould not compute confusion matrix: {e}")
        tn = fp = fn = tp = 0
        false_positive_rate = false_negative_rate = 0

    print("\n--- Overall & Specific Metrics ---")
    print(f"Accuracy:                     {acc:.4f}")
    print(f"Precision (Legitimate=1):     {prec_legit:.4f}")
    print(f"Recall (Legitimate=1):        {rec_legit:.4f}")
    print(f"F1 Score (Legitimate=1):      {f1_legit:.4f}")
    print(f"Precision (Phishing=0):       {prec_phish:.4f}")
    print(f"Recall (Phishing=0):          {rec_phish:.4f}")
    print(f"F1 Score (Phishing=0):        {f1_phish:.4f}")
    print(f"False Positive Rate (FPR):    {false_positive_rate:.4f}")
    print(f"False Negative Rate (FNR):    {false_negative_rate:.4f}")
    
    # ---------------------------------------------------------------------
    # 6. Save Overall & Specific Metrics to a CSV table.
    # ---------------------------------------------------------------------
    metrics_dict = {
        "Accuracy": acc,
        "Precision (Legit=1)": prec_legit,
        "Recall (Legit=1)": rec_legit,
        "F1 Score (Legit=1)": f1_legit,
        "Precision (Phishing=0)": prec_phish,
        "Recall (Phishing=0)": rec_phish,
        "F1 Score (Phishing=0)": f1_phish,
        "False Positive Rate (FPR)": false_positive_rate,
        "False Negative Rate (FNR)": false_negative_rate
    }
    # Convert the metrics to a DataFrame and save to CSV.
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    metrics_df.to_csv("metrics_results.csv", index=False)
    print("Saved metrics results to 'metrics_results.csv'.")
    
    # ---------------------------------------------------------------------
    # 7. Plot and Save the Confusion Matrix Image.
    # ---------------------------------------------------------------------
    # 7. Plot and Save the Confusion Matrix Image (with improved layout to avoid overlap).
    cm_data = [[tn, fp], [fn, tp]]

    fig2, ax2 = plt.subplots(figsize=(7, 6))  # Increase figure size
    im = ax2.imshow(cm_data, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax2)

    # Adjust tick labels with line breaks so text is less likely to overlap.
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Predicted\nPhishing (0)", "Predicted\nLegitimate (1)"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Actual\nPhishing (0)", "Actual\nLegitimate (1)"])

    # Add axis titles.
    ax2.set_ylabel("Actual Label")
    ax2.set_xlabel("Predicted Label")
    ax2.set_title("Confusion Matrix")

    # Annotate the cells with the count (TN, FP, FN, TP).
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, format(cm_data[i][j], "d"),
                    ha="center", va="center", color="red", fontsize=12)

    # Ensure there's enough room for all labels.
    plt.tight_layout()

    # Save the confusion matrix image to a file.
    fig2.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion matrix image to 'confusion_matrix.png'.")

    plt.show()


    # ---------------------------------------------------------------------
    # 8. Group results into correct, false positives, and false negatives.
    # ---------------------------------------------------------------------
    grouped_results = group_results(valid_samples, y_pred_api)

    # 9. Write grouped results into separate CSV files.
    write_grouped_results_to_csv(grouped_results)

if __name__ == "__main__":
    main()
    