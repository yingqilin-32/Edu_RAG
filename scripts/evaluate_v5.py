"""Frozen complex-query comparison with common planning, corpus and cost limits."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from statistics import mean, median

from sqlalchemy import text

from edu_rag import rag
from edu_rag.adaptive import AdaptiveRequest, AdaptiveWorkflow, ADAPTIVE_VERSION
from edu_rag.config import PROJECT_ROOT, get_settings
from edu_rag.db import get_engine
from edu_rag.embedding import get_encoder
from edu_rag.gating import GatedWorkflow
from edu_rag.workflow import WorkflowRequest
from evaluate_rerank_gate import MemoClient
from evaluate_v4 import corpus_fingerprint, write_json
from verify_v3 import SnapshotEngine, digest

DIRECTORY = PROJECT_ROOT/"data/v5/runs/v2"
POLICIES = ("single_pass", "always_retry", "adaptive")
SUCCESS = {"answered", "exercises_found"}
PRICING = {"verified_at": "2026-10-19", "model": "deepseek-flash",
           "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
           "off_peak_rmb_per_million": {"input_hit": .02, "input_miss": 1, "output": 4},
           "peak_rmb_per_million": {"input_hit": .04, "input_miss": 2, "output": 8},
           "note": "Estimated range using published rates, not a provider invoice. Missing cache counters are conservatively treated as cache misses."}


def build_suite():
    old = json.loads((PROJECT_ROOT/"data/v4/rerank_ablation/v2/suite.json").read_text(encoding="utf-8"))
    cases = [{**c, "partition": "reused_negative" if not c["answerable"] else "simple_control"}
             for c in old["cases"] if not c["answerable"] or c["id"] in {f"pos-{i:02}" for i in range(1, 9)}]
    complex_cases = [
        ("三年级数学：长方形长12分米、宽80厘米，先统一成分米，再求周长和面积。", "40分米；96平方分米", ["length-units", "rectangle-perimeter", "rectangle-area"]),
        ("三年级数学：正方形周长32厘米，先求边长，再求面积，并说明两种单位的区别。", "边长8厘米；面积64平方厘米", ["square-area"]),
        ("三年级数学：86人乘船，每船最多8人，每条船租金9元，至少租几条船，共需多少元？", "11条船；99元", ["division-with-remainder", "two-digit-multiplication"]),
        ("三年级数学：有2米绳子，围一个长30厘米、宽20厘米的长方形，先求周长，再求剩余绳长。", "周长100厘米；剩100厘米", ["length-units", "rectangle-perimeter"]),
        ("三年级数学：买4元5角的本子和3元8角的笔，付10元，共花多少元，找回多少元？", "8.3元；找1.7元", ["decimal-money"]),
        ("三年级数学：活动9时35分开始持续50分钟，7组同学每组8人参加，求结束时刻和总人数。", "10:25；56人", ["time-duration", "multiplication-facts"]),
        ("三年级数学：原有586本书，添127本后借出218本，剩下的每9本装一包，剩几本书，能装几包？", "495本；55包", ["addition-subtraction", "division-check"]),
        ("三年级数学：12盒铅笔每盒27支，每8支装一个袋子，先算总支数，再算全部装完至少要几袋。", "324支；41袋", ["two-digit-multiplication", "division-with-remainder"]),
        ("三年级数学：3千克加500克的糖果，每500克装一袋，先统一质量单位，再求能装几袋。", "3500克；7袋", ["mass-units", "division-check"]),
        ("三年级数学：活动共2小时15分钟，前两项分别用35分钟和40分钟，先换算总分钟数，再求剩余时间。", "135分钟；剩60分钟", ["time-duration", "addition-subtraction"]),
        ("三年级数学：同一个整体已用六分之一和六分之三，求合计用了几分之几，还剩几分之几，不必约分。", "用4/6；剩2/6", ["same-denominator-fractions"]),
        ("三年级数学：长方形周长34厘米，宽7厘米，先求长，再求面积，并写出使用的公式。", "长10厘米；面积70平方厘米", ["rectangle-perimeter", "rectangle-area"]),
        ("三年级数学：正方形面积49平方厘米，先求边长和周长，再与长7厘米宽3厘米的长方形比较周长，相差多少？", "边长7；周长28厘米与20厘米；差8厘米", ["square-area", "rectangle-perimeter"]),
        ("三年级数学：1米20厘米加80厘米的绳子，围长60厘米宽40厘米的长方形，先统一长度，再判断绳子是否刚好够。", "200厘米；周长200厘米；刚好够", ["length-units", "rectangle-perimeter"]),
        ("三年级数学：分类表记录三类图书分别23本、18本、19本，求总数；每9本装一箱，全部装完至少要几箱？", "60本；7箱", ["data-classification", "addition-subtraction", "division-with-remainder"]),
        ("三年级数学：每袋7颗糖，6袋再加5颗，共几颗？再按每盒8颗装盒，能装满几盒，余几颗？", "47颗；5盒余7颗", ["multiplication-facts", "division-with-remainder"]),
        ("三年级数学：96元买每本8元的书，能买几本？把这些书平均放到3个架子上，每架几本？", "12本；每架4本", ["division-check"]),
        ("三年级数学：7组同学每组8人，每人准备2瓶水，先算人数，再算水的总瓶数。", "56人；112瓶", ["multiplication-facts", "two-digit-multiplication"]),
        ("三年级数学：长8厘米宽6厘米的长方形，与边长7厘米的正方形，分别求面积，再比较哪个大，大多少？", "48与49平方厘米；正方形大1平方厘米", ["rectangle-area", "square-area"]),
        ("三年级数学：用周长24厘米的正方形示例，分别说明求边长、面积的步骤，并解释周长与面积的单位为什么不同。", "边长6厘米；面积36平方厘米；边线长度与内部大小", ["square-area"]),
        ("三年级数学：一根2米的绳子先剪35厘米，再剪65厘米，剩下部分围正方形，先求剩余长度，再求正方形边长。", "剩100厘米；边长25厘米", ["length-units", "square-area"]),
        ("三年级数学：53人每船最多8人，比较6条船与7条船能否坐下，解释余数为什么不能直接丢掉。", "6条坐48人还剩5人；至少7条", ["division-with-remainder"]),
        ("三年级数学：一张长方形纸长9厘米宽4厘米，另一张正方形纸边长6厘米，分别比较周长和面积。", "周长26与24厘米；面积均36平方厘米", ["rectangle-perimeter", "rectangle-area", "square-area"]),
        ("三年级数学：用同一个整体解释六分之一与三分之一谁大，再计算同分母分数五分之一加五分之二。", "1/3大于1/6；1/5+2/5=3/5", ["unit-fraction", "same-denominator-fractions"]),
    ]
    for i, (question, expected, topics) in enumerate(complex_cases, 1):
        cases.append({"id": f"complex-{i:02}", "question": question, "answerable": True,
                      "partition": "new_authored_complex", "category": "complex", "expected_answer": expected,
                      "gold_topics": topics})
    return {"set": "v5-complex-branches-cost-v1", "cases": cases,
            "note": "60 reused negatives, 8 reused simple positives, 24 new authored complex positives; small development sample, not a representative holdout.",
            "adoption_rule": "Report measured completion, safety and cost independently; no requirement to claim a gain. Prefer single-pass default retries if adaptive does not improve complex completion and adds cost. Preserve adaptive as an opt-in bounded recovery mode."}


class FrozenWorkflow(AdaptiveWorkflow):
    def __init__(self, *, frozen, **kwargs):
        self.frozen = frozen
        super().__init__(**kwargs)

    def _understand(self, state):
        plan = deepcopy(self.frozen["plan"])
        if plan.get("origin") in ("deepseek", "rules_fallback"):
            # Common planning is replayed in every arm for nominal per-request budgets.
            usage = plan.get("usage", {})
            self.meter.records.append({"stage": "understand", "status": "ok", "usage": usage,
                                       "replayed_common_plan": True, "elapsed_ms": 0,
                                       "input_and_output_estimate": usage.get("total_tokens", 0), "output_cap": 1000})
            state["calls"].append({"stage": "understand", "status": "fallback" if plan["origin"] == "rules_fallback" else "ok", "attempt": 0, "usage": usage, "model": plan.get("model")})
        decision = "retrieve" if plan["status"] == "ready" else "clarify" if plan["status"] == "needs_clarification" else "refuse"
        return {"plan": plan, "decision": decision, "reason": plan.get("message", ""),
                "clarification": plan.get("message", "") if decision == "clarify" else ""}

    def _retrieve(self, state):
        result, plan = deepcopy(self.frozen["result"]), deepcopy(state["plan"])
        if result.get("query_plan", {}).get("coverage"): plan["coverage"] = result["query_plan"]["coverage"]
        return {"result": result, "plan": plan, "retrieval_count": 1,
                "searches": [{"attempt": 0, "mode": result["retrieval"]["mode"],
                              "semantic_query": plan["semantic_query"], "bm25_query": plan["bm25_query"],
                              "effective_scope": plan["effective_scope"], "filters": plan["filters"],
                              "source_references": [source_reference(s) for s in result["sources"]],
                              "retrieval": result["retrieval"]}]}


def source_reference(source):
    from edu_rag.workflow import source_reference as reference
    return reference(source)


def price(records, rates):
    total = 0
    for r in records:
        usage = r["usage"]
        prompt, completion = usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0
        hits = usage.get("prompt_cache_hit_tokens", 0) or 0
        total += (min(prompt, hits)*rates["input_hit"]+max(0, prompt-hits)*rates["input_miss"]+completion*rates["output"])/1_000_000
    return total


def metrics(rows):
    output = {}
    for policy in POLICIES:
        items = [r for r in rows if r["policy"] == policy]
        pos = [r for r in items if r["answerable"]]
        neg = [r for r in items if not r["answerable"]]
        budgets = [r["response"]["workflow"]["budget"] for r in items]
        groups = {}
        for partition in ("simple_control", "new_authored_complex", "reused_negative"):
            selected = [r for r in items if r["partition"] == partition]
            groups[partition] = {"count": len(selected), "completed": sum(r["response"]["status"] in SUCCESS for r in selected),
                                 "model_calls": sum(r["response"]["workflow"]["budget"]["model_calls"] for r in selected),
                                 "total_tokens": sum(r["response"]["usage"]["total_tokens"] for r in selected)}
        output[policy] = {"cases": len(items), "positive_completed": sum(r["response"]["status"] in SUCCESS for r in pos),
                          "positives": len(pos), "false_answer_statuses": sum(r["response"]["status"] in SUCCESS for r in neg),
                          "negatives": len(neg), "partitions": groups,
                          "service_errors": [r["id"] for r in items if r["response"]["workflow"]["error"] not in (None, "BudgetExceeded")],
                          "budget_stops": [r["id"] for r in items if r["response"]["status"] == "budget_exceeded"],
                          "nominal_model_calls": sum(b["model_calls"] for b in budgets),
                          "max_observed_model_calls": max((b["model_calls"] for b in budgets), default=0),
                          "nominal_total_tokens": sum(r["response"]["usage"]["total_tokens"] for r in items),
                          "median_tokens": median(r["response"]["usage"]["total_tokens"] for r in items) if items else None,
                          "retry_distribution": dict(Counter(r["response"]["workflow"]["retry_count"] for r in items)),
                          "stop_reasons": dict(Counter(r["response"]["workflow"]["stop_reason"] for r in items)),
                          "bounds_respected": all(b["model_calls"] <= b["max_model_calls"] for b in budgets) and all(r["response"]["workflow"]["retry_count"] <= 2 and len(r["response"]["workflow"]["branch_history"]) <= 6 for r in items),
                          "estimated_rmb_off_peak": sum(price(b["records"], PRICING["off_peak_rmb_per_million"]) for b in budgets),
                          "estimated_rmb_peak": sum(price(b["records"], PRICING["peak_rmb_per_million"]) for b in budgets),
                          "positive_failures": [r["id"] for r in pos if r["response"]["status"] not in SUCCESS]}
    if len(rows) == 276:
        index = {(r["id"], r["policy"]): r for r in rows}
        for policy in POLICIES[1:]:
            pairs = [(r, index[(r["id"], policy)]) for r in rows if r["policy"] == "single_pass" and r["answerable"]]
            output[policy]["gained"] = [a["id"] for a,b in pairs if a["response"]["status"] not in SUCCESS and b["response"]["status"] in SUCCESS]
            output[policy]["lost"] = [a["id"] for a,b in pairs if a["response"]["status"] in SUCCESS and b["response"]["status"] not in SUCCESS]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    suite = build_suite()
    if (DIRECTORY/"suite.json").exists(): assert json.loads((DIRECTORY/"suite.json").read_text(encoding="utf-8")) == suite
    else: write_json(DIRECTORY/"suite.json", suite)
    settings, engine = get_settings(), get_engine()
    client = MemoClient(DIRECTORY/"provider_cache", settings)
    paths = ["src/edu_rag/"+n+".py" for n in ("adaptive", "budget", "gating", "workflow", "query", "rag", "retrieval", "llm", "scope", "knowledge", "embedding")]+["scripts/evaluate_v5.py"]
    signature = {"suite_sha256": digest(suite), "version": ADAPTIVE_VERSION,
                 "code": {p: sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in paths},
                 "model": settings.deepseek_model, "policies": POLICIES,
                 "request": {"max_retries": 2, "max_model_calls": 10, "max_total_tokens": 30000, "max_branches": 3, "max_elapsed_seconds": 90}}
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            before = corpus_fingerprint(conn)
            encoder = get_encoder()
            encoder.encode(["warmup"])
            frozen_path = DIRECTORY/"frozen_inputs.json"
            if not frozen_path.exists():
                old = json.loads((PROJECT_ROOT/"data/v4/rerank_ablation/v2/frozen_inputs.json").read_text(encoding="utf-8"))
                assert before == old["corpus_sha256"]
                frozen = {}
                for case in suite["cases"]:
                    if case["id"] in old["cases"]:
                        frozen[case["id"]] = old["cases"][case["id"]]
                    else:
                        request = AdaptiveRequest(question=case["question"])
                        plan = rag.construct_plan(request, engine=SnapshotEngine(conn), settings=settings, client=client)
                        base = WorkflowRequest(**{k:v for k,v in request.model_dump().items() if k in WorkflowRequest.model_fields})
                        service = GatedWorkflow(engine=SnapshotEngine(conn), encoder=encoder, trace_file=None)
                        result = service._retrieve({"request": base.model_dump(), "plan": plan, "retry_count": 0, "retrieval_count": 0, "searches": []})["result"] if plan["status"] == "ready" else None
                        frozen[case["id"]] = {"plan": plan, "result": result}
                    print("frozen "+case["id"], flush=True)
                write_json(frozen_path, {"corpus_sha256": before, "cases": frozen})
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            assert before == frozen["corpus_sha256"]
            signature.update(corpus_sha256=before, frozen_inputs_sha256=digest(frozen))
            signature = json.loads(json.dumps(signature))
            sig_path = DIRECTORY/"signature.json"
            if sig_path.exists(): assert json.loads(sig_path.read_text(encoding="utf-8")) == signature, "Signature changed; preserve this run and use a new version."
            else:
                write_json(sig_path, signature)
                write_json(DIRECTORY/"pricing.json", PRICING)
                for p in paths:
                    target = DIRECTORY/"code"/p
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((PROJECT_ROOT/p).read_bytes())
            if args.prepare_only:
                print(json.dumps({"prepared": len(suite["cases"]), "corpus": before}), flush=True)
                return
            ledger = DIRECTORY/"results.jsonl"
            rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()] if ledger.exists() else []
            done = {(r["id"], r["policy"]) for r in rows}
            if not args.summarize_only:
                for case in suite["cases"]:
                    for policy in POLICIES:
                        if (case["id"], policy) in done: continue
                        start = len(client.served)
                        service = FrozenWorkflow(frozen=frozen["cases"][case["id"]], retry_policy=policy,
                                                 engine=SnapshotEngine(conn), encoder=encoder, client=client, settings=settings, trace_file=None)
                        response = service.run(AdaptiveRequest(question=case["question"], **signature["request"]))
                        row = {**case, "policy": policy, "response": response, "provider_requests": client.served[start:]}
                        with ledger.open("a", encoding="utf-8") as handle: handle.write(json.dumps(row, ensure_ascii=False, default=str)+"\n")
                        rows.append(row)
                        print(json.dumps({"completed": len(rows), "total": 276, "id": case["id"], "policy": policy,
                                          "status": response["status"], "calls": response["workflow"]["budget"]["model_calls"],
                                          "stop": response["workflow"]["stop_reason"], "error": response["workflow"]["error"]}), flush=True)
                        write_json(DIRECTORY/"progress.json", {"completed": len(rows), "total": 276})
    with engine.connect() as conn: after = corpus_fingerprint(conn)
    cache = [json.loads(p.read_text(encoding="utf-8")) for p in (DIRECTORY/"provider_cache").glob("*.json")]
    report = {"signature": signature, "complete": len(rows) == 276, "metrics": metrics(rows),
              "corpus_unchanged": before == after, "corpus_after": after,
              "unique_provider_calls": len(cache), "actual_experiment_tokens": sum(c["response"].get("usage", {}).get("total_tokens", 0) for c in cache),
              "notes": [suite["note"], "Same frozen plan/first retrieval, ranking/filter/gate/generation/verification prompts and count budgets. Only retry policy and branch retrieval differ.",
                        "Common planning calls/usage replayed for nominal request budget and cost; raw provider cache counts actual distinct API calls, including new-case planning.",
                        "Completion is not independent answer correctness. Audit core facts and any negative status flags separately.",
                        "Budget call limit is hard; total-token admission uses a conservative estimate, not an exact provider tokenizer. Local DB/embedding work is not forcibly interrupted.",
                        "Always-retry control uses the same new nodes and focused single-query rewrites; it is not a bit-identical rerun of historical V4."]}
    write_json(DIRECTORY/"report.json", report)
    print(json.dumps({"report": str(DIRECTORY/"report.json"), "metrics": report["metrics"], "corpus_unchanged": report["corpus_unchanged"]}, ensure_ascii=False, indent=2), flush=True)
    client.client.close()


if __name__ == "__main__":
    main()
