import json

QUESTIONS_FILE = "eval_questions.json"
METADATA_FILE = "data/metadata.json"
OUTPUT_FILE = "eval_answers.json"

# Load files
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    eval_questions = json.load(f)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Build lookup table
doc_lookup = {doc["doc_id"]: doc["text"] for doc in documents}

eval_answers = []

for item in eval_questions:
    expected_id = item["expected_doc_id"]
    doc_text = doc_lookup.get(expected_id, "")

    # Extract answer portion
    if "Answer:" in doc_text:
        answer = doc_text.split("Answer:", 1)[1].strip()
    else:
        answer = ""

    eval_answers.append({
        "query": item["query"],
        "reference_answer": answer,
        "generated_answer": ""
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(eval_answers, f, indent=2, ensure_ascii=False)

print(f"✅ eval_answers.json generated ({len(eval_answers)} items)")
