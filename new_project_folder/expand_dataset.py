#!/usr/bin/env python3
"""
expand_dataset.py

Reads original dataset.json (1,253 Q&A pairs) and expands each intent
with paraphrased English, paraphrased Nigerian Pidgin, and authentic
Code-Switched farmer queries.

Outputs expanded_dataset.json containing multi-example intents annotated
with language tags ('en', 'pidgin', 'code_switched').
"""

import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "data" / "dataset.json"
OUTPUT_PATH = BASE_DIR / "data" / "expanded_dataset.json"

# Seed for reproducibility
random.seed(42)

# Common Nigerian Pidgin prefixes and code-switching patterns
PIDGIN_PREFIXES = [
    "Abeg ", "How far, ", "Oga, ", "Wetin be de way to ", "Tell me how to ",
    "How farmer fit ", "Weti person fit do to ", "Shey ", "Abi "
]

CODE_SWITCH_PATTERNS = [
    ("What is the best way to", "Wetin be the best way to"),
    ("How can I", "How farmer fit"),
    ("What are the symptoms of", "Wetin be the signs of"),
    ("How do I control", "How I fit control"),
    ("Why is", "Why e be say"),
    ("When should I", "Wetin time farmer suppose"),
    ("What type of soil", "Which kind soil"),
    ("How do farmers", "How farmers dey"),
    ("What causes", "Wetin dey cause"),
    ("How to prevent", "How to prevent make e no happen for"),
]

def generate_english_paraphrase(q_en: str, crop: str) -> str:
    """Generate syntactically varied English paraphrase."""
    q = q_en.strip()
    if q.startswith("What is ") or q.startswith("What are "):
        return q.replace("What is ", "Can you explain ").replace("What are ", "Could you list ")
    elif q.startswith("How does ") or q.startswith("How do "):
        return q.replace("How does ", "In what manner does ").replace("How do ", "What procedure is used to ")
    elif q.startswith("Why is ") or q.startswith("Why are "):
        return q.replace("Why is ", "What is the reason behind ").replace("Why are ", "For what reason are ")
    elif q.startswith("How can "):
        return q.replace("How can ", "What steps can be taken to ")
    else:
        return f"Please guide on: {q.lower()}"

def generate_pidgin_paraphrase(q_pid: str) -> str:
    """Generate orthographic and lexical variants for Nigerian Pidgin."""
    q = q_pid.strip()
    # Apply common orthographic variations (e.g., 'wetin' -> 'weti', 'dey' -> 'de')
    if "wetin" in q.lower():
        q_var = q.lower().replace("wetin", "weti")
    elif "dey" in q.lower():
        q_var = q.lower().replace("dey", "de")
    else:
        prefix = random.choice(PIDGIN_PREFIXES)
        q_var = f"{prefix}{q[0].lower()}{q[1:]}"
    return q_var.capitalize()

def generate_code_switched(q_en: str, q_pid: str, crop: str) -> str:
    """Create authentic code-switched query mixing English and Pidgin."""
    for en_pat, pid_pat in CODE_SWITCH_PATTERNS:
        if en_pat.lower() in q_en.lower():
            # Substitute English prefix with Pidgin prefix while retaining English technical terms
            cs = q_en.lower().replace(en_pat.lower(), pid_pat.lower())
            cs = cs.replace("maize", "maize crop").replace("cassava", "cassava farm")
            return cs.capitalize()
    
    # Fallback code-switched query
    return f"Abeg how farmer fit handle {q_en.lower().rstrip('?')}, as e dey affect {crop}?"

def expand_dataset():
    print(f"Reading original dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    expanded_records = []
    stats = {"en": 0, "pidgin": 0, "code_switched": 0}

    for item in data:
        item_id = item["id"]
        crop = item.get("crop", "crop")
        category = item.get("category", "general")
        difficulty = item.get("difficulty", "basic")
        q_en = item.get("question_en", "").strip()
        a_en = item.get("answer_en", "").strip()
        q_pid = item.get("question_pidgin", "").strip()
        a_pid = item.get("answer_pidgin", "").strip()

        intent = f"ask_{crop}_{item_id}"

        examples = []

        # 1. Original English
        if q_en:
            examples.append({
                "text": q_en,
                "language_tag": "en",
                "is_paraphrase": False
            })
            stats["en"] += 1

        # 2. Original Pidgin
        if q_pid:
            examples.append({
                "text": q_pid,
                "language_tag": "pidgin",
                "is_paraphrase": False
            })
            stats["pidgin"] += 1

        # 3. Paraphrased English
        if q_en:
            en_para = generate_english_paraphrase(q_en, crop)
            examples.append({
                "text": en_para,
                "language_tag": "en",
                "is_paraphrase": True
            })
            stats["en"] += 1

        # 4. Paraphrased Pidgin
        if q_pid:
            pid_para = generate_pidgin_paraphrase(q_pid)
            examples.append({
                "text": pid_para,
                "language_tag": "pidgin",
                "is_paraphrase": True
            })
            stats["pidgin"] += 1

        # 5. Authentic Code-Switched Query
        if q_en:
            cs_query = generate_code_switched(q_en, q_pid, crop)
            examples.append({
                "text": cs_query,
                "language_tag": "code_switched",
                "is_paraphrase": True
            })
            stats["code_switched"] += 1

        record = {
            "id": item_id,
            "intent": intent,
            "crop": crop,
            "category": category,
            "difficulty": difficulty,
            "answer_en": a_en,
            "answer_pidgin": a_pid,
            "examples": examples
        }
        expanded_records.append(record)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(expanded_records, f, indent=2, ensure_ascii=False)

    total_examples = sum(stats.values())
    print("Dataset expansion complete!")
    print(f"Total Intents: {len(expanded_records)}")
    print(f"Total Examples Generated: {total_examples}")
    print(f"Breakdown by Language: English={stats['en']}, Pidgin={stats['pidgin']}, Code-Switched={stats['code_switched']}")
    print(f"Average Examples per Intent: {total_examples / len(expanded_records):.2f}")

if __name__ == "__main__":
    expand_dataset()
