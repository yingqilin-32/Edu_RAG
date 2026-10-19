"""Paired rerank/filter/gate ablation on frozen plans, pools and original responses."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import log2
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import httpx
from sqlalchemy import text

from edu_rag import rag
from edu_rag.config import PROJECT_ROOT, get_settings
from edu_rag.db import get_engine
from edu_rag.embedding import get_encoder
from edu_rag.gating import GatedWorkflow, POLICIES
from edu_rag.workflow import WorkflowRequest
from evaluate_v4 import corpus_fingerprint, write_json
from evaluate_four_way import inventory
from verify_v3 import SnapshotEngine, digest

DIRECTORY = PROJECT_ROOT / "data/v4/rerank_ablation/v2"
ARMS = (*POLICIES, "no_pre_generation_gate")
SUCCESS = {"answered", "exercises_found"}


def suite():
    data = json.loads((PROJECT_ROOT/"data/v4/eval_corrected.json").read_text(encoding="utf-8"))
    labels = [
        (["rectangle-perimeter"], None), (["rectangle-area"], None), (["square-area"], None),
        (["unit-fraction"], None), (["money-units", "decimal-money"], None), (["length-units"], None),
        (["addition-subtraction"], 3), (["reading-clocks"], 1),
        (["rectangle-area", "rectangle-perimeter"], 3), (["square-area"], 3),
        (["division-with-remainder"], 3), (["decimal-money", "money-units"], None),
        (["time-duration"], 3), (["addition-subtraction"], 3),
        (["rectangle-area"], 3), (["square-area"], 3), (["money-units"], 1),
        (["division-with-remainder"], 3), ([], 3), ([], 1)]
    for case in data["cases"]:
        case["partition"] = "reused_regression"
        if case["answerable"]:
            topics, grade = labels[int(case["id"].split("-")[1])-1]
            case["gold"] = {"topics": topics, "grade": grade, "kind": "exercise" if case["category"] == "exercise_search" else "knowledge"}
            if case["id"] == "pos-15": case["gold"].update(difficulty=2, question_type="calculation")
            if case["id"] in ("pos-16", "pos-17", "pos-18"): case["gold"]["question_type"] = "calculation"
            if case["id"] == "pos-17": case["gold"]["semester"] = "second"
    fresh = [
        ("三年级数学：61人乘船，每条船最多坐9人，至少要几条船？", ["division-with-remainder"], "7条"),
        ("三年级数学：47支笔每6支装一盒，能装满几盒，还余几支？", ["division-with-remainder"], "7盒，余5支"),
        ("三年级数学：90个杯子，每箱最多装8个，全部运走至少需要几箱？", ["division-with-remainder"], "12箱"),
        ("三年级数学：图书馆原有586本书，新添127本，再借出218本，还剩几本？", ["addition-subtraction"], "495本"),
        ("三年级数学：8个小组，每组9名同学，一共有多少名同学？", ["multiplication-facts"], "72人"),
        ("三年级数学：每袋7颗糖，6袋再加5颗，共有多少颗糖？", ["multiplication-facts", "addition-subtraction"], "47颗"),
        ("三年级数学：活动上午9时45分开始，35分钟后结束，结束时刻是几点几分？", ["time-duration"], "上午10时20分"),
        ("三年级数学：下午2时25分开始，下午3时10分结束，经过了多少分钟？", ["time-duration"], "45分钟"),
        ("三年级数学：书包15元，铅笔2元5角，合计多少钱？", ["decimal-money"], "17元5角"),
        ("三年级数学：2米长的绳子剪去85厘米，还剩多少厘米？", ["length-units"], "115厘米"),
        ("三年级数学：长方形纸板的四条边依次为9、4、9、4厘米，沿边围一圈需要多少厘米绳子？", ["rectangle-perimeter"], "26厘米"),
        ("三年级数学：长方形纸长12厘米、宽5厘米，把纸面全部涂色，涂色面积是多少？", ["rectangle-area"], "60平方厘米"),
    ]
    for i, (question, topics, answer) in enumerate(fresh, 1):
        data["cases"].append({"id": f"fresh-{i:02}", "question": question, "answerable": True,
                              "category": "problem_solving", "partition": "new_authored_variants",
                              "expected_answer": answer, "gold": {"topics": topics, "grade": 3, "kind": "knowledge"}})
    data.update(set="paired-rerank-and-gate-v1", note="60 negative + 20 positive reused regression cases; 12 new authored positive variants. Development comparison, not an independent representative holdout.")
    return data


class MemoClient:
    """Cache exact request bodies, never headers/keys; original provider response is auditable."""
    def __init__(self, directory, settings):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=httpx.Timeout(settings.deepseek_timeout, connect=10), follow_redirects=False)
        self.served = []

    def post(self, url, *, headers, json):
        key = digest({"url": url, "body": json})
        path, cached = self.directory/(key+".json"), False
        if path.exists():
            item, cached = __import__("json").loads(path.read_text(encoding="utf-8")), True
        else:
            start = perf_counter()
            response = self.client.post(url, headers=headers, json=json)
            response.raise_for_status()
            item = {"request_body": json, "response": response.json(), "elapsed_ms": (perf_counter()-start)*1000,
                    "recorded_at": datetime.now(timezone.utc).isoformat()}
            if item["response"].get("choices", [{}])[0].get("finish_reason") == "stop":
                write_json(path, item)
        self.served.append({"key": key, "cache_hit": cached, "provider_elapsed_ms": item["elapsed_ms"],
                            "usage": item["response"].get("usage", {})})
        return httpx.Response(200, json=item["response"], request=httpx.Request("POST", url))


class PairedWorkflow(GatedWorkflow):
    def __init__(self, *, frozen_plan, frozen_result, bypass_gate=False, **kwargs):
        self.frozen_plan, self.frozen_result, self.bypass_gate = frozen_plan, frozen_result, bypass_gate
        super().__init__(**kwargs)

    def _understand(self, state):
        plan = deepcopy(self.frozen_plan)
        decision = "retrieve" if plan["status"] == "ready" else "clarify" if plan["status"] == "needs_clarification" else "refuse"
        return {"plan": plan, "decision": decision, "reason": plan.get("message", ""),
                "clarification": plan.get("message", "") if decision == "clarify" else ""}

    def _retrieve(self, state):
        result, plan = deepcopy(self.frozen_result), deepcopy(state["plan"])
        if result.get("query_plan", {}).get("coverage"):
            plan["coverage"] = result["query_plan"]["coverage"]
        return {"result": result, "plan": plan, "retrieval_count": 1,
                "searches": [{"attempt": 0, "mode": "hybrid", "effective_scope": plan["effective_scope"],
                              "filters": plan["filters"], "semantic_query": plan["semantic_query"], "bm25_query": plan["bm25_query"]}]}

    def _evaluate(self, state):
        if not self.bypass_gate:
            return super()._evaluate(state)
        if state.get("fault"):
            return {"decision": "refuse", "selected": []}
        return {"decision": "answer" if state["selected"] else "refuse", "reason": "Ablation: pre-generation evidence gate bypassed",
                "assessment": {"gate_enabled": False, "selected_sources": len(state["selected"])}}


def relevance(source, case):
    gold = case.get("gold", {})
    if gold.get("grade") and source["grade"] != gold["grade"]: return 0
    if gold.get("topics") and source.get("knowledge_point_code") not in gold["topics"]: return 0
    if any(source.get(k) != gold[k] for k in ("semester", "difficulty", "question_type") if k in gold): return 0
    if gold.get("kind") == "exercise": return 2 if source.get("exercise_id") else 0
    return 2 if source.get("document_id") else 1


def rank_metrics(row, bank):
    if not row["answerable"] or row["category"] == "knowledge_summary": return None
    ids = row["selected_ids"]
    scores = [relevance(bank[i], row) for i in ids]
    ideal = sorted((relevance(s, row) for s in bank.values()), reverse=True)[:row["budget"]]
    def dcg(values): return sum((2**v-1)/log2(i+2) for i, v in enumerate(values))
    return {"ndcg": dcg(scores)/dcg(ideal) if dcg(ideal) else 0,
            "precision": sum(s > 0 for s in scores)/row["budget"],
            "all_required_topics_present": set(row["gold"].get("topics", [])) <= {bank[i].get("knowledge_point_code") for i in ids}}


def summarize(rows, bank):
    report = {}
    for arm in ARMS:
        items = [r for r in rows if r["arm"] == arm]
        pos, neg = [r for r in items if r["answerable"]], [r for r in items if not r["answerable"]]
        ranked = [rank_metrics(r, bank) for r in items if rank_metrics(r, bank) is not None]
        report[arm] = {
            "count": len(items), "positives": len(pos), "positive_completed": sum(r["response"]["status"] in SUCCESS for r in pos),
            "positive_completion_rate": mean(r["response"]["status"] in SUCCESS for r in pos) if pos else None,
            "negatives": len(neg), "false_answers": sum(r["response"]["status"] in SUCCESS for r in neg),
            "missing_input_clarifications": sum(r["category"] == "missing_input" and r["response"]["status"] == "needs_clarification" for r in neg),
            "unsupported_generation_attempts": sum(any(c["stage"] == "generate" for c in r["response"]["workflow"]["calls"]) for r in neg),
            "service_errors": [r["id"] for r in items if r["response"]["workflow"].get("error")],
            "status_counts": dict(Counter(r["response"]["status"] for r in items)),
            "ndcg_at_context_budget": mean(r["ndcg"] for r in ranked) if ranked else None,
            "precision_at_context_budget": mean(r["precision"] for r in ranked) if ranked else None,
            "topic_coverage_rate": mean(r["all_required_topics_present"] for r in ranked) if ranked else None,
            "nominal_model_calls": sum(len(r["response"]["workflow"]["calls"]) for r in items),
            "nominal_total_tokens": sum(r["response"]["usage"]["total_tokens"] for r in items),
            "median_provider_time_ms": median(sum(c["provider_elapsed_ms"] for c in r["requests"]) for r in items) if items else None,
            "positive_failures": [r["id"] for r in pos if r["response"]["status"] not in SUCCESS],
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    data = suite()
    if not (DIRECTORY/"suite.json").exists(): write_json(DIRECTORY/"suite.json", data)
    assert json.loads((DIRECTORY/"suite.json").read_text(encoding="utf-8")) == data
    code_paths = ["src/edu_rag/"+n+".py" for n in ("gating", "workflow", "query", "rag", "retrieval", "llm", "scope", "knowledge", "embedding")]+["scripts/evaluate_rerank_gate.py"]
    signature = {"suite_sha256": digest(data), "code": {p: sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in code_paths},
                 "model": get_settings().deepseek_model, "max_retries": 0, "arms": ARMS,
                 "design": "Frozen first-pass query plans/candidate pools; same context count budget, generation and post-verification. Exact prompt cache shared across arms. Gold labels never sent to model.",
                 "adoption_rule": "Prefer retrieval_order unless a rerank arm completes more positive cases with no added negative false answers or service errors. Ranking metrics are secondary; no statistically powered equivalence claim."}
    settings, client = get_settings(), MemoClient(DIRECTORY/"provider_cache", get_settings())
    frozen_path = DIRECTORY/"frozen_inputs.json"
    if not frozen_path.exists():
        old = {r["id"]: r["response"] for r in map(json.loads, (PROJECT_ROOT/"data/v4/runs/v2/results.jsonl").read_text(encoding="utf-8").splitlines())}
        engine, frozen = get_engine(), {}
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
            with conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))
                before = corpus_fingerprint(conn)
                assert before == json.loads((PROJECT_ROOT/"data/v4/runs/v2/signature.json").read_text(encoding="utf-8"))["corpus_sha256"]
                enc = get_encoder()
                enc.encode(["warmup"])
                for case in data["cases"]:
                    request = WorkflowRequest(question=case["question"], query_mode="deepseek", retrieval_mode="hybrid", max_retries=0)
                    if case["id"] in old:
                        previous = old[case["id"]]
                        plan = deepcopy(previous["query_plan"])
                        if previous["workflow"]["searches"]:
                            first = previous["workflow"]["searches"][0]
                            plan.update(semantic_query=first["semantic_query"], bm25_query=first["bm25_query"])
                    else:
                        plan = rag.construct_plan(request, engine=SnapshotEngine(conn), settings=settings, client=client)
                    service = GatedWorkflow(engine=SnapshotEngine(conn), encoder=enc, settings=settings, client=client, trace_file=None)
                    state = {"request": request.model_dump(), "plan": plan, "retrieval_count": 0, "retry_count": 0, "searches": []}
                    result = service._retrieve(state)["result"] if plan["status"] == "ready" else None
                    frozen[case["id"]] = {"plan": plan, "result": result}
                    print("frozen "+case["id"], flush=True)
                write_json(DIRECTORY/"inventory.json", inventory(conn))
                frozen_manifest = {"corpus_sha256": before, "cases": frozen}
        with engine.connect() as conn: assert corpus_fingerprint(conn) == before
        write_json(frozen_path, frozen_manifest)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    signature.update(corpus_sha256=frozen["corpus_sha256"], frozen_inputs_sha256=digest(frozen))
    # JSON roundtrip normalizes tuples and makes resumes exact.
    signature = json.loads(json.dumps(signature))
    if (DIRECTORY/"signature.json").exists():
        assert json.loads((DIRECTORY/"signature.json").read_text(encoding="utf-8")) == signature, "Frozen experiment changed; use a new version directory."
    else:
        write_json(DIRECTORY/"signature.json", signature)
        for relative in code_paths:
            target = DIRECTORY/"code"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((PROJECT_ROOT/relative).read_bytes())
    bank = {s["id"]: s for item in frozen["cases"].values() if item["result"] for s in item["result"]["sources"]}
    # Add every corpus chunk to the independent ideal-ranking denominator.
    for item in json.loads((DIRECTORY/"inventory.json").read_text(encoding="utf-8")):
        if item["id"] not in bank: bank[item["id"]] = {**item, "knowledge_point_code": item["topic_code"]}
    if args.prepare_only:
        print(json.dumps({"prepared": len(frozen["cases"]), "signature": signature}, ensure_ascii=False), flush=True)
        return
    ledger = DIRECTORY/"results.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()] if ledger.exists() else []
    done = {(r["id"], r["arm"]) for r in rows}
    if not args.summarize_only:
        for case in data["cases"]:
            inputs = frozen["cases"][case["id"]]
            for arm in ARMS:
                if (case["id"], arm) in done: continue
                start = len(client.served)
                service = PairedWorkflow(frozen_plan=inputs["plan"], frozen_result=inputs["result"],
                                         bypass_gate=arm == "no_pre_generation_gate",
                                         context_policy=arm if arm in POLICIES else "retrieval_order",
                                         settings=settings, client=client, trace_file=None)
                response = service.run(WorkflowRequest(question=case["question"], max_retries=0))
                searches = response["workflow"]["searches"]
                selection = searches[0].get("context_selection", {}) if searches else {}
                row = {**case, "arm": arm, "response": response, "requests": client.served[start:],
                       "selected_ids": selection.get("selected_ids", []), "budget": selection.get("budget", 5)}
                with ledger.open("a", encoding="utf-8") as handle: handle.write(json.dumps(row, ensure_ascii=False, default=str)+"\n")
                rows.append(row)
                print(json.dumps({"completed": len(rows), "total": len(data["cases"])*len(ARMS), "id": case["id"], "arm": arm,
                                  "status": response["status"], "error": response["workflow"]["error"]}, ensure_ascii=False), flush=True)
                write_json(DIRECTORY/"progress.json", {"completed": len(rows), "total": len(data["cases"])*len(ARMS)})
    cache = [json.loads(p.read_text(encoding="utf-8")) for p in (DIRECTORY/"provider_cache").glob("*.json")]
    report = {"signature": signature, "complete": len(rows) == len(data["cases"])*len(ARMS),
              "metrics": summarize(rows, bank), "unique_provider_calls": len(cache),
              "actual_provider_total_tokens": sum(c["response"].get("usage", {}).get("total_tokens", 0) for c in cache),
              "notes": [data["note"], "Completion is not independent answer correctness; audit outputs separately.",
                        "Latency is sum of original provider durations for each arm (cache hits replay original durations); excludes common planning/retrieval and is not HTTP end-to-end latency.",
                        "Negative controls retain curriculum preflight and post-answer verification; this isolates the pre-generation gate only.",
                        "nDCG averages 30 non-summary positives, with source-level gold relevance 2 for topic-matching knowledge cards or requested exercises, 1 for related examples. Summaries are excluded."]}
    write_json(DIRECTORY/"report.json", report)
    client.client.close()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
