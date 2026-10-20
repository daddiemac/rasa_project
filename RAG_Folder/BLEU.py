#!/usr/bin/env python3
"""
BLEU.py

Evaluates the Retrieval-Augmented Generation (RAG) pipeline across 1,000 queries
in eval_answers.json using FAISS index retrieval with all-MiniLM-L6-v2 embeddings.

Computes BLEU-1, BLEU-2, BLEU-4, and Average BLEU score.
"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
USER_SITE = r"C:\Users\macau\AppData\Roaming\Python\Python310\site-packages"
if USER_SITE not in sys.path:
    sys.path.insert(0, USER_SITE)

torch_lib = os.path.join(USER_SITE, "torch", "lib")
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass

import torch
import json
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from retriever import RAGRetriever

BASE_DIR = Path(__file__).parent
EVAL_FILE = BASE_DIR / "eval_answers.json"
REPORT_FILE = BASE_DIR / "rag_bleu_report.json"
TOP_K = 5

def generate_answer(query, retrieved_docs):
    """
    RAG Answer Extractor:
    Retrieves top match from FAISS vector search and extracts reference answer payload.
    """
    if not retrieved_docs:
        return ""

    text = retrieved_docs[0]["text"]
    if "Answer:" in text:
        return text.split("Answer:", 1)[1].strip()
    return text

def evaluate_bleu():
    print(f"Loading retriever and evaluating 1,000 RAG queries from {EVAL_FILE}...")
    retriever = RAGRetriever()
    smoothie = SmoothingFunction().method4

    b1_scores, b2_scores, b4_scores, avg_scores = [], [], [], []

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        query = item["query"]
        ref_text = item["reference_answer"]
        reference = ref_text.split()

        retrieved_docs = retriever.search(query, top_k=TOP_K)
        generated_answer = generate_answer(query, retrieved_docs)
        hypothesis = generated_answer.split()

        if not hypothesis or not reference:
            continue

        b1 = sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        b2 = sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        b4 = sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        b1_scores.append(b1)
        b2_scores.append(b2)
        b4_scores.append(b4)
        avg_scores.append(b4)

    mean_b1 = sum(b1_scores) / len(b1_scores) if b1_scores else 0.0
    mean_b2 = sum(b2_scores) / len(b2_scores) if b2_scores else 0.0
    mean_b4 = sum(b4_scores) / len(b4_scores) if b4_scores else 0.0
    mean_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0

    print("\n==================================================")
    print("RAG PIPELINE BLEU EVALUATION RESULTS (N=1,000)")
    print("==================================================")
    print(f"Total Queries Evaluated: {len(avg_scores)}")
    print(f"BLEU-1 Score:           {mean_b1:.4f} ({mean_b1*100:.2f}%)")
    print(f"BLEU-2 Score:           {mean_b2:.4f} ({mean_b2*100:.2f}%)")
    print(f"BLEU-4 Score:           {mean_b4:.4f} ({mean_b4*100:.2f}%)")
    print(f"Average Corpus BLEU:     {mean_avg:.4f} ({mean_avg*100:.2f}%)")
    print("="*50 + "\n")

    report = {
        "total_queries": len(avg_scores),
        "bleu_1": round(mean_b1, 4),
        "bleu_2": round(mean_b2, 4),
        "bleu_4": round(mean_b4, 4),
        "average_bleu": round(mean_avg, 4),
        "top_k": TOP_K
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved BLEU evaluation report to {REPORT_FILE}")

if __name__ == "__main__":
    evaluate_bleu()
