import json
import yaml
import requests
import uuid

RASA_URL = "http://localhost:5005/webhooks/rest/webhook"
NLU_FILE = "data/nlu.yml"
DOMAIN_FILE = "data/rules.yml"
OUTPUT_FILE = "rasa_responses.json"


def detect_language(text):
    pidgin_keywords = ["abi", "abeg", "howfar", "wetin", "wey", "na",
            "una", "dey", "fit", "sabi", "shey", "e dey", "i no"]
    for w in pidgin_keywords:
        if w in text.lower():
            return "pidgin"
    return "en"


def load_intent_to_utter_map(domain_path):
    with open(domain_path, "r", encoding="utf-8") as f:
        domain = yaml.safe_load(f)

    mapping = {}
    for rule in domain.get("rules", []):
        steps = rule.get("steps", [])
        intent = None
        utter = None
        for s in steps:
            if "intent" in s:
                intent = s["intent"]
            if "action" in s and s["action"].startswith("utter_"):
                utter = s["action"]
        if intent and utter:
            mapping[intent] = utter

    return mapping


def send_message(text):
    payload = {
        "sender": str(uuid.uuid4()),
        "message": text
    }
    r = requests.post(RASA_URL, json=payload)
    r.raise_for_status()
    return r.json()


def generate():
    with open(NLU_FILE, "r", encoding="utf-8") as f:
        nlu = yaml.safe_load(f)

    intent_to_utter = load_intent_to_utter_map(DOMAIN_FILE)
    results = []

    for item in nlu.get("nlu", []):
        intent = item.get("intent")
        examples = item.get("examples", "")

        utter_action = intent_to_utter.get(intent)
        if not utter_action:
            continue

        for line in examples.split("\n"):
            text = line.replace("-", "").strip()
            if not text:
                continue

            language = detect_language(text)
            bot_messages = send_message(text)

            for msg in bot_messages:
                if "text" in msg:
                    results.append({
                        "utter_action": utter_action,
                        "language": language,
                        "generated_text": msg["text"]
                    })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(results)} BLEU-ready responses")


if __name__ == "__main__":
    generate()
