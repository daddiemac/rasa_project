import json
from pathlib import Path

# ---------- CONFIG ----------
DATASET_FILE = Path(r"C:\Users\Nsikak Warrie\Downloads\bilingual_rasa_project\bilingual_rasa_project\rasa_project\new_project_folder\data\dataset.json")
NLU_FILE = Path("nlu.yml")
DOMAIN_FILE = Path("domain.yml")
RULES_FILE = Path("rules.yml")
STORIES_FILE = Path("stories.yml")
RASA_VERSION = "3.1"

def esc(s: str) -> str:
    """Escape double quotes and backslashes, collapse newlines."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\r", " ").replace("\n", " ")
    return s.strip()

# ---------- LOAD DATA ----------
text = DATASET_FILE.read_text(encoding="utf-8")
data = json.loads(text)

# ---------- NLU ----------
nlu_lines = [f'version: "{RASA_VERSION}"', "", "nlu:"]

for row in data:
    # safe extraction
    crop = row.get("crop")
    rid = row.get("id")
    if not crop or rid is None:
        continue

    intent = f"ask_{crop}_{rid}"
    q_en = esc(row.get("question_en", ""))
    q_pid = esc(row.get("question_pidgin", ""))

    nlu_lines.append(f"- intent: {intent}")
    nlu_lines.append("  examples: |")
    if q_en:
        nlu_lines.append(f"    - {q_en}")
    if q_pid and q_pid != q_en:
        nlu_lines.append(f"    - {q_pid}")
    nlu_lines.append("")

NLU_FILE.write_text("\n".join(nlu_lines) + "\n", encoding="utf-8")

# ---------- DOMAIN ----------
domain_lines = [
    f'version: "{RASA_VERSION}"',
    "",
    "intents:",
]

intents_seen = set()
responses_seen = set()

for row in data:
    crop = row.get("crop")
    rid = row.get("id")
    if not crop or rid is None:
        continue
    intent = f"ask_{crop}_{rid}"
    if intent not in intents_seen:
        domain_lines.append(f"  - {intent}")
        intents_seen.add(intent)

domain_lines += [
    "",
    "slots:",
    "  language:",
    "    type: text",
    "    influence_conversation: false",
    "",
    "actions:",
    "  - action_detect_language",
    "  - action_rag_search",
    "",
    "responses:",
]

for row in data:
    crop = row.get("crop")
    rid = row.get("id")
    if not crop or rid is None:
        continue

    intent = f"ask_{crop}_{rid}"
    response = f"utter_{intent}"
    if response in responses_seen:
        continue

    a_en = esc(row.get("answer_en", ""))
    a_pid = esc(row.get("answer_pidgin", ""))

    domain_lines.append(f"  {response}:")
    # English conditional response (only if we have text)
    if a_en:
        domain_lines.append("  - condition:")
        domain_lines.append("    - slot_was_set:")
        domain_lines.append("      - language: english")
        domain_lines.append(f'    text: "{a_en}"')
    # Pidgin conditional response
    if a_pid:
        domain_lines.append("  - condition:")
        domain_lines.append("    - slot_was_set:")
        domain_lines.append("      - language: pidgin")
        domain_lines.append(f'    text: "{a_pid}"')

    # if both empty, give a tiny default (keeps Rasa happy)
    if not a_en and not a_pid:
        domain_lines.append('  - text: "I do not have an answer for this yet."')

    domain_lines.append("")
    responses_seen.add(response)

DOMAIN_FILE.write_text("\n".join(domain_lines) + "\n", encoding="utf-8")

# ---------- RULES ----------
rules_lines = [
    f'version: "{RASA_VERSION}"',
    "",
    "rules:",
    "- rule: detect language",
    "  steps:",
    "  - action: action_detect_language",
    "",
]

for row in data:
    crop = row.get("crop")
    rid = row.get("id")
    if not crop or rid is None:
        continue
    intent = f"ask_{crop}_{rid}"
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

RULES_FILE.write_text("\n".join(rules_lines), encoding="utf-8")

# ---------- STORIES ----------
stories_lines = [
    f'version: "{RASA_VERSION}"',
    "",
    "stories:",
]

for row in data:
    crop = row.get("crop")
    rid = row.get("id")
    if not crop or rid is None:
        continue
    intent = f"ask_{crop}_{rid}"
    stories_lines.append(f"- story: {intent} path")
    stories_lines.append("  steps:")
    stories_lines.append(f"  - intent: {intent}")
    stories_lines.append(f"  - action: utter_{intent}")
    stories_lines.append("")

STORIES_FILE.write_text("\n".join(stories_lines), encoding="utf-8")

print("✅ nlu.yml, domain.yml, rules.yml, stories.yml generated successfully.")
