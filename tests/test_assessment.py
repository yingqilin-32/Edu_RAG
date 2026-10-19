from copy import deepcopy
import json
from pathlib import Path

import pytest
import httpx

from edu_rag.assessment import METRICS, digest, export_samples, metric_inputs, summarize
from edu_rag.config import Settings
from scripts.evaluate_ragas import JudgeHTTP


def saved_row(status="answered", answerable=True):
    return {"id": "case-1", "policy": "adaptive", "partition": "test", "answerable": answerable,
            "question": "2加3？", "response": {"status": status, "answer": "5。[C1]",
                "sources": [{"id": "final", "content": "加法求和"}],
                "workflow": {"searches": [{"sources": [{"id": "unused", "content": "unselected"}]}]}}}


def test_export_preserves_only_actual_final_contexts_and_reference_is_separate():
    sample = export_samples([saved_row()], {"case-1": "5"})[0]
    assert sample["retrieved_contexts"] == ["加法求和"]
    assert sample["retrieved_context_ids"] == ["final"]
    inputs = metric_inputs(sample, "faithfulness")
    assert "reference" not in inputs
    assert "5" not in inputs["retrieved_contexts"]
    assert metric_inputs(sample, "context_recall")["reference"] == "5"


@pytest.mark.parametrize("status,answerable", [("refused", True), ("needs_clarification", True),
                                               ("budget_exceeded", True), ("answered", False)])
def test_abstentions_and_negatives_are_not_scored_as_faithful_answers(status, answerable):
    sample = export_samples([saved_row(status, answerable)], {"case-1": "5"})[0]
    assert all(metric_inputs(sample, name) is None for name in METRICS)


def test_invalid_export_is_rejected():
    with pytest.raises(ValueError, match="Missing reference"):
        export_samples([saved_row()], {})
    with pytest.raises(ValueError, match="Duplicate"):
        export_samples([saved_row(), saved_row()], {"case-1": "5"})
    row = saved_row()
    row["response"]["sources"][0]["content"] = ""
    with pytest.raises(ValueError, match="Empty final source"):
        export_samples([row], {"case-1": "5"})


def test_conditional_means_and_all_positive_utility_have_explicit_denominators():
    samples = export_samples([saved_row()], {"case-1": "5"})
    first = {**samples[0], "metrics": {name: {"status": "scored", "value": .75} for name in METRICS}}
    second = {**deepcopy(first), "id": "unanswered", "status": "refused",
              "metrics": {name: {"status": "not_applicable", "value": None} for name in METRICS}}
    report = summarize([first, second])["adaptive"]
    assert report["completed"] == 1
    accuracy = report["metrics"]["answer_accuracy"]
    assert accuracy["scored"] == 1 and accuracy["positive_total"] == 2
    assert accuracy["mean_conditional"] == .75
    assert accuracy["all_positive_utility_abstention_zero"] == .375
    first["metrics"]["answer_accuracy"] = {"status": "error", "value": None}
    assert summarize([first, second])["adaptive"]["metrics"]["answer_accuracy"]["all_positive_utility_abstention_zero"] is None


def test_all_frozen_v5_positives_have_references_and_export_is_repeatable():
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (root/"data/v5/runs/v2/results.jsonl").read_text(encoding="utf-8").splitlines()]
    references = json.loads((root/"data/evaluation/ragas/references.json").read_text(encoding="utf-8"))["references"]
    samples = export_samples(rows, references)
    assert len(samples) == 276 and sum(s["answerable"] for s in samples) == 96
    assert sum(metric_inputs(s, "faithfulness") is not None for s in samples) == 77
    assert digest(samples) == digest(export_samples(rows, references))
    assert sum(s["known_ambiguity"] for s in samples) == 3


def test_judge_cache_is_exact_and_never_stores_credentials(tmp_path):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"model": "fake", "choices": [{"message": {"content": "{}"}}],
                                        "usage": {"total_tokens": 9}})
    client = JudgeHTTP(tmp_path, Settings(_env_file=None), max_calls=1, transport=httpx.MockTransport(respond))
    try:
        kwargs = {"headers": {"Authorization": "Bearer PRIVATE_KEY"}, "json": {"messages": ["hello"]}}
        client.post("https://example.test/v1/chat/completions", **kwargs)
        client.post("https://example.test/v1/chat/completions", **kwargs)
        assert len(calls) == 1 and client.calls == 1 and client.tokens == 9
        assert [r["cache_hit"] for r in client.served] == [False, True]
        assert "PRIVATE_KEY" not in next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
        with pytest.raises(RuntimeError, match="budget"):
            client.post("https://example.test/v1/chat/completions", headers={}, json={"messages": ["different"]})
    finally:
        client.client.close()


def test_judge_balance_failure_stops_further_paid_requests(tmp_path):
    client = JudgeHTTP(tmp_path, Settings(_env_file=None), transport=httpx.MockTransport(lambda r: httpx.Response(402)))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.post("https://example.test", headers={}, json={})
        assert client.calls == 1 and client.halted
        with pytest.raises(RuntimeError, match="balance"):
            client.post("https://example.test", headers={}, json={"different": True})
        assert client.calls == 1 and client.served[0]["http_status"] == 402
    finally:
        client.client.close()
