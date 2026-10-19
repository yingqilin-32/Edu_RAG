"""Offline evaluation data and denominators; never queries or modifies the database."""

from collections import Counter
from hashlib import sha256
import json
from statistics import mean


METRICS = ("faithfulness", "factual_correctness", "context_precision", "context_recall", "answer_accuracy")
ANSWERED = {"answered", "exercises_found"}


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def export_samples(rows, references):
    """Only final generation contexts count as evidence; no candidate/gold leakage."""
    seen, samples = set(), []
    for row in rows:
        identity = (row["policy"], row["id"])
        if identity in seen:
            raise ValueError(f"Duplicate policy/case: {identity}")
        seen.add(identity)
        if row["answerable"] and not references.get(row["id"], "").strip():
            raise ValueError(f"Missing reference: {row['id']}")
        response = row["response"]
        sources = response.get("sources", [])
        if any(not s.get("content", "").strip() for s in sources):
            raise ValueError(f"Empty final source: {identity}")
        samples.append({"id": row["id"], "policy": row["policy"], "answerable": row["answerable"],
                        "partition": row["partition"], "status": response["status"],
                        "user_input": row["question"], "response": response.get("answer", ""),
                        "retrieved_contexts": [s["content"] for s in sources],
                        "retrieved_context_ids": [s["id"] for s in sources],
                        "reference": references.get(row["id"]),
                        "known_ambiguity": row["id"] == "complex-07"})
    return samples


def metric_inputs(sample, metric):
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    # Refusals and clarification templates are not substantive answers.
    if not sample["answerable"] or sample["status"] not in ANSWERED or not sample["response"].strip():
        return None
    columns = {"faithfulness": ("user_input", "response", "retrieved_contexts"),
               "factual_correctness": ("response", "reference"),
               "context_precision": ("user_input", "reference", "retrieved_contexts"),
               "context_recall": ("user_input", "reference", "retrieved_contexts"),
               "answer_accuracy": ("user_input", "response", "reference")}[metric]
    result = {key: sample[key] for key in columns}
    if "retrieved_contexts" in result and not result["retrieved_contexts"]:
        return None
    return result


def summarize(rows):
    policies = {}
    for policy in sorted({r["policy"] for r in rows}):
        group = [r for r in rows if r["policy"] == policy]
        positives = [r for r in group if r["answerable"]]
        negatives = [r for r in group if not r["answerable"]]
        metrics = {}
        for name in METRICS:
            values = [r["metrics"][name]["value"] for r in positives if r["metrics"][name]["status"] == "scored"]
            errors = sum(r["metrics"][name]["status"] == "error" for r in positives)
            metrics[name] = {"mean_conditional": mean(values) if values else None,
                             "scored": len(values), "positive_total": len(positives), "errors": errors,
                             "not_applicable": len(positives)-len(values)-errors}
            # This utility explicitly penalizes abstention on answerable questions;
            # it is not the standard Ragas conditional metric or a binary accuracy.
            if name == "answer_accuracy":
                metrics[name]["all_positive_utility_abstention_zero"] = (
                    sum(values)/len(positives) if positives and not errors else None)
        policies[policy] = {"positives": len(positives), "negatives": len(negatives),
                            "completed": sum(r["status"] in ANSWERED for r in positives),
                            "positive_statuses": dict(Counter(r["status"] for r in positives)),
                            "negative_status_flags": sum(r["status"] in ANSWERED for r in negatives),
                            "metrics": metrics}
    return policies
