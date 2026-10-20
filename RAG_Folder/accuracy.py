import json
from retriever import RAGRetriever

TOP_K = 100
EVAL_FILE = "eval_questions.json"


def evaluate():
    retriever = RAGRetriever()

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    TP = FP = FN = TN = 0

    for item in eval_data:
        query = item["query"]
        expected_id = item["expected_doc_id"]

        results = retriever.search(query, top_k=TOP_K)
        retrieved_ids = [r["doc_id"] for r in results]

        if expected_id in retrieved_ids:
            TP += 1
        else:
            FN += 1

    total = len(eval_data)

    # Binary metrics
    recall = TP / (TP + FN) if (TP + FN) else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    accuracy = TP / total if total else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    confusion_matrix = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }

    print("\n📊 RAG RETRIEVER EVALUATION")
    print("=" * 35)
    print(f"Queries evaluated: {total}")
    print(f"Top-K: {TOP_K}\n")

    print("Confusion Matrix:")
    print(f"  TP: {TP}")
    print(f"  FP: {FP}")
    print(f"  FN: {FN}")
    print(f"  TN: {TN}\n")

    print(f"Recall:    {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")


if __name__ == "__main__":
    evaluate()
