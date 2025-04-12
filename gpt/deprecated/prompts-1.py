import os
import re
import zipfile
import sqlite3

# Define paths to the main zip file and the output file
outer_zip = "n96ncsr5g4-1.zip"
extraction_folder = "extracted_data_1"  # Folder where outer_zip's contents will be extracted
output_file = "prompts-1.txt"

# Inner folder name inside the extracted folder
inner_folder_name = "n96ncsr5g4-1"

# point to files inside the inner folder
index_db_file = os.path.join(extraction_folder, inner_folder_name, "index.sqlite")
dataset_folder = os.path.join(extraction_folder, inner_folder_name, "dataset")

def extract_zip(zip_path, target_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

def extract_html_from_zip(zip_path, target_folder):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

def read_index_db(db_file):

    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    try:
        cursor.execute('SELECT website, url, result FROM "index"')
    except sqlite3.OperationalError as e:
        print("Error querying the table:", e)
        connection.close()
        raise e

    index_data = cursor.fetchall()
    website_to_url_and_result = {row[0]: (row[1], row[2]) for row in index_data}
    connection.close()
    return website_to_url_and_result

def generate_prompts(dataset_folder, website_mapping, output_file):
    """
    Processes each dataset part and writes prompts incrementally to the output file.
    """
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Process dataset parts 1
        for part_number in range(1, 2):
            part_zip_filename = f"dataset_part_{part_number}.zip"
            part_zip_path = os.path.join(dataset_folder, part_zip_filename)
            part_extraction_folder = os.path.join(dataset_folder, f"dataset_part_{part_number}")
            
            if os.path.exists(part_zip_path):
                extract_html_from_zip(part_zip_path, part_extraction_folder)
                
                # Look for folder with hyphen (dataset-part-X)
                inner_folder = os.path.join(part_extraction_folder, f"dataset-part-{part_number}")
                html_folder = inner_folder if os.path.exists(inner_folder) else part_extraction_folder

                html_files = [f for f in os.listdir(html_folder) if f.endswith(".html")]
                print(f"Found {len(html_files)} HTML files in part {part_number}")
                
                for html_file in html_files:
                    file_path = os.path.join(html_folder, html_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            html_content = f.read().replace('\n', ' ')
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
                    
                    file_key = html_file.strip().lower()
                    url, _ = website_mapping.get(file_key, (None, None))
                    if url is not None:
                        prompt = (
                            f"Given this URL: {url} and its HTML code: {html_content}, "
                            "determine if it is a phishing website or not. ONLY OUTPUT 1 (PHISHING) OR 0 (NOT PHISHING). "
                            "DO NOT WRITE ANYTHING ELSE"
                        )
                        prompt = re.sub(r'\s+', ' ', prompt).strip()
                        f_out.write(prompt + "\n")
                    else:
                        print(f"No matching URL for file: {html_file}")
            else:
                print(f"Warning: {part_zip_path} does not exist.")
    print(f"Prompts have been incrementally saved to {output_file}")

def main():
    extract_zip(outer_zip, extraction_folder)
    print(f"Extracted {outer_zip} to {extraction_folder}")

    website_mapping = read_index_db(index_db_file)
    
    generate_prompts(dataset_folder, website_mapping, output_file)
    print(f"Prompts have been saved to {output_file}")

if __name__ == "__main__":
    main()
