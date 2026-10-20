from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import json
import os

DATA_DIR = os.path.join(os.getcwd(), "data")


# --------------------------------------------------
# LANGUAGE DETECTION ACTION
# --------------------------------------------------
class ActionDetectLanguage(Action):

    def name(self) -> Text:
        return "action_detect_language"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        message = tracker.latest_message.get("text", "").lower()

        pidgin_markers = [
            "abi", "abeg", "howfar", "wetin", "wey", "na",
            "una", "dey", "fit", "sabi", "shey", "e dey", "i no"
        ]

        language = "pidgin" if any(m in message for m in pidgin_markers) else "english"

        return [SlotSet("language", language)]


# --------------------------------------------------
# RAG FALLBACK ACTION
# --------------------------------------------------
class ActionRagSearch(Action):

    def name(self) -> Text:
        return "action_rag_search"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        meta_path = os.path.join(DATA_DIR, "metadata.json")

        if not os.path.exists(meta_path):
            dispatcher.utter_message(
                text="I no get this one for my training data yet."
            )
            return []

        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        query = tracker.latest_message.get("text", "").lower()
        keywords = [w for w in query.split() if len(w) > 3]

        matches = []
        for item in metadata:
            text = item.get("text", "").lower()
            if any(k in text for k in keywords):
                matches.append(item)
            if len(matches) >= 2:
                break

        if matches:
            dispatcher.utter_message(
                text="I no see am for my training data, but check this:"
            )
            for m in matches:
                dispatcher.utter_message(text=f"- {m.get('text')}")
        else:
            dispatcher.utter_message(
                text="Sorry, I no get answer for this yet. Try ask am another way."
            )

        return []
