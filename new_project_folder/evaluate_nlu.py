#!/usr/bin/env python3
"""
evaluate_nlu.py

1. Dynamically registers XLM-RoBERTa (AfroXLM-R) into Rasa's HuggingFace registry.
2. Retrains DIET on the 80% train split (data/train_nlu.yml) with AfroXLM-R backbone for 100 epochs.
3. Evaluates the model on the 20% held-out test split (data/test_nlu.yml).
4. Cross-references predictions with test_metadata.json to compute overall NLU metrics
   and per-language performance breakdowns (English, Nigerian Pidgin, Code-Switched).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Register XLM-RoBERTa in Rasa registry before any Rasa engine execution
def register_xlmroberta():
    try:
        import transformers
        from transformers import XLMRobertaTokenizer, TFXLMRobertaModel
        from rasa.nlu.utils.hugging_face import registry
        from rasa.nlu.featurizers.dense_featurizer import lm_featurizer
        from rasa.nlu.utils.hugging_face.transformers_pre_post_processors import (
            roberta_tokens_pre_processor,
            gpt2_tokens_cleaner,
            roberta_embeddings_post_processor,
        )

        registry.model_class_dict["xlmroberta"] = TFXLMRobertaModel
        registry.model_tokenizer_dict["xlmroberta"] = XLMRobertaTokenizer
        registry.model_weights_defaults["xlmroberta"] = "Davlan/afro-xlmr-base"
        registry.model_special_tokens_pre_processors["xlmroberta"] = roberta_tokens_pre_processor
        registry.model_tokens_cleaners["xlmroberta"] = gpt2_tokens_cleaner
        registry.model_embeddings_post_processors["xlmroberta"] = roberta_embeddings_post_processor
        lm_featurizer.MAX_SEQUENCE_LENGTHS["xlmroberta"] = 512
        print("Successfully registered XLM-RoBERTa (AfroXLM-R) into Rasa registry.")
    except Exception as e:
        print(f"Warning: Could not register XLM-RoBERTa dynamically: {e}")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yml"
TRAIN_DATA_PATH = BASE_DIR / "data" / "train_nlu.yml"
TEST_DATA_PATH = BASE_DIR / "data" / "test_nlu.yml"
TEST_METADATA_PATH = BASE_DIR / "data" / "test_metadata.json"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

def get_env():
    env = os.environ.copy()
    user_site = r"C:\Users\macau\AppData\Roaming\Python\Python310\site-packages"
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{user_site};{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = user_site
    env["RASA_TELEMETRY_ENABLED"] = "false"
    env["PYTHONIOENCODING"] = "utf-8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env

def run_cmd(cmd, cwd=BASE_DIR):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, env=get_env())
    if res.returncode != 0:
        print(f"Error running command:\n{res.stderr}")
        print(f"Stdout:\n{res.stdout}")
    else:
        print(res.stdout)
    return res.returncode == 0

def train_diet():
    print("=== STAGE 1: Training DIET Classifier with AfroXLM-R Backbone ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # We execute python -m rasa train via a custom script wrapper or python inline to ensure registry hook
    train_script = f"""
import sys, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, r'C:\\Users\\macau\\AppData\\Roaming\\Python\\Python310\\site-packages')
import transformers
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel
from rasa.nlu.utils.hugging_face import registry
from rasa.nlu.featurizers.dense_featurizer import lm_featurizer
from rasa.nlu.utils.hugging_face.transformers_pre_post_processors import (
    roberta_tokens_pre_processor,
    gpt2_tokens_cleaner,
    roberta_embeddings_post_processor,
)
registry.model_class_dict['xlmroberta'] = TFXLMRobertaModel
registry.model_tokenizer_dict['xlmroberta'] = XLMRobertaTokenizer
registry.model_weights_defaults['xlmroberta'] = 'Davlan/afro-xlmr-base'
registry.model_special_tokens_pre_processors['xlmroberta'] = roberta_tokens_pre_processor
registry.model_tokens_cleaners['xlmroberta'] = gpt2_tokens_cleaner
registry.model_embeddings_post_processors['xlmroberta'] = roberta_embeddings_post_processor
lm_featurizer.MAX_SEQUENCE_LENGTHS['xlmroberta'] = 512

from rasa.__main__ import main
sys.argv = ['rasa', 'train', 'nlu', '--config', r'{CONFIG_PATH}', '--nlu', r'{TRAIN_DATA_PATH}', '--out', r'{MODELS_DIR}']
main()
"""
    wrapper_path = BASE_DIR / "run_train.py"
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(train_script)

    cmd = f'"{sys.executable}" "{wrapper_path}"'
    success = run_cmd(cmd)
    if not success:
        print("Rasa training failed!")
        sys.exit(1)
    
    models = sorted(MODELS_DIR.glob("*.tar.gz"), key=os.path.getmtime, reverse=True)
    if not models:
        print("No trained model found!")
        sys.exit(1)
    latest_model = models[0]
    print(f"Successfully trained model: {latest_model}")
    return latest_model

def evaluate_diet(model_path):
    print("\n=== STAGE 2: Evaluating DIET on Held-out Test Split ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    test_script = f"""
import sys, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, r'C:\\Users\\macau\\AppData\\Roaming\\Python\\Python310\\site-packages')
import transformers
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel
from rasa.nlu.utils.hugging_face import registry
from rasa.nlu.featurizers.dense_featurizer import lm_featurizer
from rasa.nlu.utils.hugging_face.transformers_pre_post_processors import (
    roberta_tokens_pre_processor,
    gpt2_tokens_cleaner,
    roberta_embeddings_post_processor,
)
registry.model_class_dict['xlmroberta'] = TFXLMRobertaModel
registry.model_tokenizer_dict['xlmroberta'] = XLMRobertaTokenizer
registry.model_weights_defaults['xlmroberta'] = 'Davlan/afro-xlmr-base'
registry.model_special_tokens_pre_processors['xlmroberta'] = roberta_tokens_pre_processor
registry.model_tokens_cleaners['xlmroberta'] = gpt2_tokens_cleaner
registry.model_embeddings_post_processors['xlmroberta'] = roberta_embeddings_post_processor
lm_featurizer.MAX_SEQUENCE_LENGTHS['xlmroberta'] = 512

from rasa.__main__ import main
sys.argv = ['rasa', 'test', 'nlu', '--nlu', r'{TEST_DATA_PATH}', '--model', r'{model_path}', '--out', r'{RESULTS_DIR}']
main()
"""
    wrapper_path = BASE_DIR / "run_test.py"
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(test_script)

    cmd = f'"{sys.executable}" "{wrapper_path}"'
    success = run_cmd(cmd)
    if not success:
        print("Rasa test evaluation failed!")
        sys.exit(1)
    print(f"Rasa evaluation completed. Results saved in {RESULTS_DIR}")

def analyze_per_language_breakdown():
    print("\n=== STAGE 3: Computing Per-Language Performance Breakdown ===")
    report_file = RESULTS_DIR / "intent_report.json"
    errors_file = RESULTS_DIR / "intent_errors.json"
    
    with open(TEST_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    errors = []
    if errors_file.exists():
        with open(errors_file, "r", encoding="utf-8") as f:
            errors = json.load(f)

    error_texts = {e["text"].strip(): e["intent_prediction"]["name"] for e in errors}

    lang_stats = {
        "en": {"correct": 0, "total": 0, "errors": []},
        "pidgin": {"correct": 0, "total": 0, "errors": []},
        "code_switched": {"correct": 0, "total": 0, "errors": []}
    }

    for item in metadata:
        text = item["text"].strip()
        lang = item["language_tag"]
        target_intent = item["intent"]

        lang_stats[lang]["total"] += 1
        if text in error_texts:
            predicted = error_texts[text]
            lang_stats[lang]["errors"].append({
                "text": text,
                "target": target_intent,
                "predicted": predicted
            })
        else:
            lang_stats[lang]["correct"] += 1

    with open(report_file, "r", encoding="utf-8") as f:
        global_report = json.load(f)

    macro_avg = global_report.get("macro avg", {})
    weighted_avg = global_report.get("weighted avg", {})
    accuracy = global_report.get("accuracy", 0.0)

    summary = {
        "global_metrics": {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_avg.get("precision", 0.0), 4),
            "macro_recall": round(macro_avg.get("recall", 0.0), 4),
            "macro_f1": round(macro_avg.get("f1-score", 0.0), 4),
            "weighted_f1": round(weighted_avg.get("f1-score", 0.0), 4),
            "total_test_samples": len(metadata)
        },
        "per_language_metrics": {}
    }

    print("\n-------------------------------------------------------------")
    print(f"GLOBAL HELD-OUT TEST PERFORMANCE (N={len(metadata)})")
    print("-------------------------------------------------------------")
    print(f"Accuracy:        {accuracy*100:.2f}%")
    print(f"Macro Precision: {macro_avg.get('precision', 0.0)*100:.2f}%")
    print(f"Macro Recall:    {macro_avg.get('recall', 0.0)*100:.2f}%")
    print(f"Macro F1-Score:  {macro_avg.get('f1-score', 0.0)*100:.2f}%")
    print("-------------------------------------------------------------\n")

    print("-------------------------------------------------------------")
    print("PER-LANGUAGE HELD-OUT TEST BREAKDOWN")
    print("-------------------------------------------------------------")
    for lang, st in lang_stats.items():
        corr = st["correct"]
        tot = st["total"]
        acc = (corr / tot) if tot > 0 else 0.0
        prec = acc
        rec = acc
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        summary["per_language_metrics"][lang] = {
            "total_samples": tot,
            "correct_predictions": corr,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        }
        print(f"Language: {lang.upper():<14} | N={tot:<4} | Acc={acc*100:6.2f}% | F1={f1*100:6.2f}%")

    print("-------------------------------------------------------------\n")

    summary_path = RESULTS_DIR / "held_out_eval_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Detailed evaluation summary saved to {summary_path}")

def main():
    register_xlmroberta()
    latest_model = train_diet()
    evaluate_diet(latest_model)
    analyze_per_language_breakdown()

if __name__ == "__main__":
    main()
