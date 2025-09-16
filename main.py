import os
import json
import re
import importlib.util
import google.generativeai as genai

# ------------- CONFIGURATION -------------
GEMINI_API_KEY = "API Key"  # Replace with your API key
INPUT_DIR = "copied_docs"
OUTPUT_DIR = "gemini_predictions_v2"

# ------------- LOAD EXTRACTION LABELS -------------
def load_extraction_labels(file_path="extraction_labels.py"):
    spec = importlib.util.spec_from_file_location("extraction_labels", file_path)
    labels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labels)
    return labels.extraction_labels

# ------------- GEMINI SETUP -------------
def setup_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

# ------------- PROMPT BUILDER -------------
def build_prompt(text, doctype, keys):
    keys_str = ", ".join(keys)
    return f"""
You are an advanced structured data extraction model.

You will be given OCR text for a document of type: "{doctype}".
Your job is to extract values for ALL of these fields: {keys_str}.

Guidelines:
1. Always include every key exactly as provided in the list.
2. If a value is not found in the text, set it as an empty string "" (do NOT omit the key).
3. If a value is found, extract it exactly as it appears in the text.
4. Return a single valid JSON object with all keys present.
5. Do NOT include explanations, markdown, or extra text — only output valid JSON.

Example:
OCR TEXT:
"This is a Passport of Ali Khan. Date of Birth: 1995-07-21. Passport Number: PK1234567"

Expected JSON:
{{
    "name": "Ali Khan",
    "dob": "1995-07-21",
    "passport_number": "PK1234567",
    "nationality": ""
}}
OCR TEXT:
{text}
"""

# ------------- SAFE JSON PARSER -------------
def safe_parse_json(response_text, keys):
    # Try direct JSON parse first
    try:
        return json.loads(response_text)
    except:
        pass

    # Attempt to extract JSON substring using regex
    match = re.search(r'\{[\s\S]*\}', response_text)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except:
            pass

    # If all fails, return empty fields
    return {k: "" for k in keys}

# ------------- MAIN PROCESSING FUNCTION -------------
def process_documents():
    extraction_labels = load_extraction_labels()
    model = setup_gemini()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for doctype_folder in os.listdir(INPUT_DIR):
        folder_path = os.path.join(INPUT_DIR, doctype_folder)
        if not os.path.isdir(folder_path):
            continue

        output_folder = os.path.join(OUTPUT_DIR, doctype_folder)
        os.makedirs(output_folder, exist_ok=True)

        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                    data = json.load(f)

                doctype = data.get("doctype")
                text = data.get("text")
                keys = extraction_labels.get(doctype, [])

                if not keys:
                    print(f"⚠️ No extraction keys found for doctype {doctype}, skipping {file}")
                    continue

                prompt = build_prompt(text, doctype, keys)
                response = model.generate_content(prompt)

                response_json = safe_parse_json(response.text, keys)

                new_data = {
                    "id": data["id"],
                    "doctype": doctype,
                    "text": text,
                    "response": response_json
                }

                with open(os.path.join(output_folder, file), "w", encoding="utf-8") as out_f:
                    json.dump(new_data, out_f, indent=4)

    print(f"✅ Gemini predictions saved in '{OUTPUT_DIR}' folder!")

# ------------- RUN SCRIPT -------------
if __name__ == "__main__":
    process_documents()
