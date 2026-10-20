#!/usr/bin/env python3
"""
split_dataset.py

Splits expanded_dataset.json into an 80% train set and a 20% held-out test set.
- Every intent has 4 training examples and 1 held-out test example.
- Keeps language tags intact in data/test_metadata.json for per-language performance breakdown.
- Writes train_nlu.yml, test_nlu.yml, domain.yml, rules.yml, stories.yml.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXPANDED_DATASET_PATH = BASE_DIR / "data" / "expanded_dataset.json"
TRAIN_NLU_PATH = BASE_DIR / "data" / "train_nlu.yml"
TEST_NLU_PATH = BASE_DIR / "data" / "test_nlu.yml"
TEST_METADATA_PATH = BASE_DIR / "data" / "test_metadata.json"

DOMAIN_PATH = BASE_DIR / "domain.yml"
RULES_PATH = BASE_DIR / "data" / "rules.yml"
STORIES_PATH = BASE_DIR / "data" / "stories.yml"

RASA_VERSION = "3.1"
random.seed(42)

def esc(s: str) -> str:
    """Escape double quotes and newlines for YAML string formatting."""
    if s is None:
        return ""
    s = str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")
    return s.strip()

def write_nlu_yml(data_by_intent, filepath):
    """Write Rasa NLU format YAML file."""
    lines = [f'version: "{RASA_VERSION}"', "", "nlu:"]
    for intent in sorted(data_by_intent.keys()):
        examples = data_by_intent[intent]
        lines.append(f"- intent: {intent}")
        lines.append("  examples: |")
        for ex in examples:
            clean_ex = esc(ex)
            lines.append(f"    - {clean_ex}")
        lines.append("")
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")

def split_data():
    print(f"Loading expanded dataset from {EXPANDED_DATASET_PATH}...")
    with open(EXPANDED_DATASET_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    train_data = defaultdict(list)
    test_data = defaultdict(list)
    test_metadata = []

    test_lang_counts = {"en": 0, "pidgin": 0, "code_switched": 0}
    train_lang_counts = {"en": 0, "pidgin": 0, "code_switched": 0}

    for record in records:
        intent = record["intent"]
        examples = record["examples"]
        
        # Shuffle deterministically per intent
        shuffled = list(examples)
        random.shuffle(shuffled)

        # 1 example to test (20%), remaining (80%) to train
        test_sample = shuffled[0]
        train_samples = shuffled[1:]

        test_data[intent].append(test_sample["text"])
        test_metadata.append({
            "intent": intent,
            "text": test_sample["text"],
            "language_tag": test_sample["language_tag"],
            "is_paraphrase": test_sample["is_paraphrase"],
            "crop": record["crop"],
            "category": record["category"]
        })
        test_lang_counts[test_sample["language_tag"]] += 1

        for tr in train_samples:
            train_data[intent].append(tr["text"])
            train_lang_counts[tr["language_tag"]] += 1

    # Write train_nlu.yml and test_nlu.yml
    write_nlu_yml(train_data, TRAIN_NLU_PATH)
    write_nlu_yml(test_data, TEST_NLU_PATH)

    # Save test metadata
    with open(TEST_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(test_metadata, f, indent=2, ensure_ascii=False)

    print("Stratified Split Completed Successfully!")
    print(f"Train Examples: {sum(len(v) for v in train_data.values())} (Language dist: {train_lang_counts})")
    print(f"Test Examples:  {sum(len(v) for v in test_data.values())} (Language dist: {test_lang_counts})")

    # Generate Domain, Rules, Stories files
    write_domain_and_rules(records)

def write_domain_and_rules(records):
    intents = sorted(list(set(r["intent"] for r in records)))
    
    # 1. Domain File
    domain_lines = [
        f'version: "{RASA_VERSION}"',
        "",
        "intents:",
    ]
    for intent in intents:
        domain_lines.append(f"  - {intent}")

    domain_lines += [
        "",
        "slots:",
        "  language:",
        "    type: text",
        "    influence_conversation: false",
        "    mappings:",
        "    - type: custom",
        "",
        "actions:",
        "  - action_detect_language",
        "  - action_rag_search",
        "",
        "responses:",
    ]

    for record in records:
        intent = record["intent"]
        a_en = esc(record.get("answer_en", ""))
        a_pid = esc(record.get("answer_pidgin", ""))

        domain_lines.append(f"  utter_{intent}:")
        if a_en:
            domain_lines.append(f'  - text: "{a_en}"')
        elif a_pid:
            domain_lines.append(f'  - text: "{a_pid}"')
        else:
            domain_lines.append('  - text: "I do not have an answer for this yet."')
        domain_lines.append("")

    DOMAIN_PATH.write_text("\n".join(domain_lines) + "\n", encoding="utf-8")
    print(f"Updated domain.yml written to {DOMAIN_PATH}")

    # 2. Rules File
    rules_lines = [
        f'version: "{RASA_VERSION}"',
        "",
        "rules:",
        "- rule: detect language",
        "  steps:",
        "  - action: action_detect_language",
        "",
    ]
    for intent in intents:
        rules_lines.append(f"- rule: respond to {intent}")
        rules_lines.append("  steps:")
        rules_lines.append(f"  - intent: {intent}")
        rules_lines.append(f"  - action: utter_{intent}")
        rules_lines.append("")

    rules_lines += [
        "- rule: fallback to RAG",
        "  steps:",
        "  - intent: nlu_fallback",
        "  - action: action_rag_search",
        ""
    ]
    RULES_PATH.write_text("\n".join(rules_lines), encoding="utf-8")
    print(f"Updated rules.yml written to {RULES_PATH}")

    # 3. Stories File
    stories_lines = [
        f'version: "{RASA_VERSION}"',
        "",
        "stories:",
    ]
    for intent in intents:
        stories_lines.append(f"- story: {intent} path")
        stories_lines.append("  steps:")
        stories_lines.append(f"  - intent: {intent}")
        stories_lines.append(f"  - action: utter_{intent}")
        stories_lines.append("")

    STORIES_PATH.write_text("\n".join(stories_lines), encoding="utf-8")
    print(f"Updated stories.yml written to {STORIES_PATH}")

if __name__ == "__main__":
    split_data()
