import json
from pathlib import Path

# Paths
INPUT_FILE = Path(r"C:\Users\Nsikak Warrie\Downloads\bilingual_rasa_project\bilingual_rasa_project\rasa_project\RAG_Folder\data\dataset.json")
OUTPUT_FILE = Path(r"C:\Users\Nsikak Warrie\Downloads\bilingual_rasa_project\bilingual_rasa_project\rasa_project\RAG_Folder\data\metadata.json")

# Load dataset
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

for item in data:
    base_metadata = {
        "id": item.get("id"),
        "crop": item.get("crop", "unknown"),
        "category": item.get("category", "unknown"),
        "difficulty": item.get("difficulty", "unknown"),
        "source": "agric_dataset"
    }

    # English document
    if "question_en" in item and "answer_en" in item:
        documents.append({
            "doc_id": f'{item.get("crop","crop")}_{item.get("id","x")}_en',
            "text": f'Question: {item["question_en"]}\nAnswer: {item["answer_en"]}',
            "metadata": {
                **base_metadata,
                "language": "en"
            }
        })

    # Pidgin document
    if "question_pidgin" in item and "answer_pidgin" in item:
        documents.append({
            "doc_id": f'{item.get("crop","crop")}_{item.get("id","x")}_pidgin',
            "text": f'Question: {item["question_pidgin"]}\nAnswer: {item["answer_pidgin"]}',
            "metadata": {
                **base_metadata,
                "language": "pidgin"
            }
        })

# Save metadata.json
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print(f"✅ metadata.json generated successfully")
print(f"📦 Total documents: {len(documents)}")