#!/usr/bin/env python3
"""
generate_all_rasa_files.py

Reads dataset_fixed.json (list of objects) and writes:
 - nlu.yml
 - domain.yml
 - rules.yml
 - stories.yml

Dataset row format expected:
{
  "id": 1081,
  "crop": "maize",
  "category": "soil",
  "question_en": "...",
  "answer_en": "...",
  "question_pidgin": "...",
  "answer_pidgin": "..."
}
"""
import json
from collections import defaultdict
from pathlib import Path

# ---------- CONFIG ----------
DATASET_FILE = Path("dataset.json")   # adjust path if needed
NLU_FILE = Path("nlu.yml")
DOMAIN_FILE = Path("domain.yml")
RULES_FILE = Path("rules.yml")
STORIES_FILE = Path("stories.yml")
RASA_VERSION = "3.1"
# ----------------------------

def safe_examples_list(examples):
    """
    Return a deduplicated list preserving insertion order.
    Remove stray newlines and strip items.
    """
    seen = set()
    out = []
    for ex in examples:
        if not isinstance(ex, str):
            ex = str(ex)
        ex = ex.strip().replace("\n", " ").replace("\r", " ")
        if ex and ex not in seen:
            seen.add(ex)
            out.append(ex)
    return out

def write_nlu(data_by_intent, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"version: '{RASA_VERSION}'\n\nnlu:\n")
        for intent in sorted(data_by_intent.keys()):
            examples = safe_examples_list(data_by_intent[intent]["examples"])
            f.write(f"- intent: {intent}\n")
            f.write("  examples: |\n")
            for ex in examples:
                f.write(f"    - {ex}\n")
            f.write("\n")

def write_domain(intents_ordered, data_by_intent, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"version: '{RASA_VERSION}'\n\n")
        f.write("intents:\n")
        for intent in intents_ordered:
            f.write(f"  - {intent}\n")
        f.write("\nresponses:\n")
        for intent in intents_ordered:
            utt = f"utter_{intent}"
            f.write(f"  {utt}:\n")
            answers = safe_examples_list(data_by_intent[intent]["answers"])
            if not answers:
                answers = ["I'm sorry, I don't have an answer for that right now."]
            for ans in answers:
                quoted = json.dumps(ans, ensure_ascii=False)
                f.write(f"  - text: {quoted}\n")
            f.write("\n")

def write_rules(intents_ordered, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"version: '{RASA_VERSION}'\n\nrules:\n")
        for intent in intents_ordered:
            f.write(f"- rule: respond to {intent}\n")
            f.write("  steps:\n")
            f.write(f"  - intent: {intent}\n")
            f.write(f"  - action: utter_{intent}\n\n")

def write_stories(intents_ordered, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"version: '{RASA_VERSION}'\n\nstories:\n")
        for intent in intents_ordered:
            f.write(f"- story: {intent}_story\n")
            f.write("  steps:\n")
            f.write(f"  - intent: {intent}\n")
            f.write(f"  - action: utter_{intent}\n\n")

def main():
    with open(DATASET_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    data_by_intent = defaultdict(lambda: {"examples": [], "answers": []})
    intents_ordered = []

    for row in data:
        if "crop" not in row or "id" not in row:
            continue
        intent = f"ask_{row['crop']}_{row['id']}"
        if intent not in data_by_intent:
            intents_ordered.append(intent)
        q_en = row.get("question_en")
        q_pid = row.get("question_pidgin")
        if q_en:
            data_by_intent[intent]["examples"].append(q_en)
        if q_pid and q_pid != q_en:
            data_by_intent[intent]["examples"].append(q_pid)
        a_en = row.get("answer_en")
        a_pid = row.get("answer_pidgin")
        if a_en:
            data_by_intent[intent]["answers"].append(a_en)
        if a_pid and a_pid != a_en:
            data_by_intent[intent]["answers"].append(a_pid)

    write_nlu(data_by_intent, NLU_FILE)
    write_domain(intents_ordered, data_by_intent, DOMAIN_FILE)
    write_rules(intents_ordered, RULES_FILE)
    write_stories(intents_ordered, STORIES_FILE)

if __name__ == "__main__":
    main()
