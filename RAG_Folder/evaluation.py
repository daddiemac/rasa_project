import json
import random

INPUT_FILE = "data/metadata.json"
OUTPUT_FILE = "eval_questions.json"
NUM_QUESTIONS = 1000

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

eval_questions = []

# Shuffle to avoid bias
random.shuffle(documents)

for doc in documents:
    text = doc.get("text", "")

    # Extract question line
    if text.startswith("Question:"):
        question = text.split("\n")[0].replace("Question:", "").strip()
    else:
        continue

    eval_questions.append({
        "query": question,
        "expected_doc_id": doc["doc_id"]
    })

    if len(eval_questions) >= NUM_QUESTIONS:
        break

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(eval_questions, f, indent=2, ensure_ascii=False)

print(f"✅ Generated {len(eval_questions)} eval questions")
print(f"📄 Saved to {OUTPUT_FILE}")
