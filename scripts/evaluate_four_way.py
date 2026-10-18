"""Fixed-corpus four-way ablation; plans are cached once and shared by both rankers."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import log2
from statistics import median
from time import perf_counter
from unittest.mock import patch

from edu_rag import __version__, rag
from edu_rag.config import PROJECT_ROOT, get_settings
from edu_rag.db import get_engine
from edu_rag.embedding import get_encoder
from edu_rag.query import DIFFICULTY_LEVELS, PLAN_VERSION, PLANNER_PROMPT
from sqlalchemy import text
from verify_v3 import SnapshotEngine, digest, manifest

EXPERIMENTS = {"raw_bm25": ("off", "bm25"), "raw_hybrid": ("off", "hybrid"),
               "query_bm25": ("planned", "bm25"), "query_hybrid": ("planned", "hybrid")}
DIRECTORY = PROJECT_ROOT / "data/v3/four_way"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")


def inventory(conn):
    return [dict(row) for row in conn.execute(text(
        "SELECT m.id,m.course_id,m.knowledge_point_id,m.document_id,m.exercise_id,c.code AS course_code,"
        "c.grade,c.semester,c.textbook_edition,c.subject,c.curriculum,k.code AS topic_code,k.title AS topic_title,"
        "k.parent_id,e.code AS exercise_code,e.question_type,e.difficulty "
        "FROM public.edu_chunks m JOIN public.edu_courses c ON c.id=m.course_id "
        "LEFT JOIN public.edu_knowledge_points k ON k.id=m.knowledge_point_id "
        "LEFT JOIN public.edu_exercises e ON e.id=m.exercise_id ORDER BY m.id"
    )).mappings()]


def request_for(case, rows, *, query_mode="off", mode="hybrid"):
    scope = case.get("scope", {})
    if "course" in case:
        scope = {"course_id": next(row["course_id"] for row in rows if row["course_code"] == case["course"])}
    return rag.SearchRequest(question=case["question"], **scope, query_mode=query_mode, retrieval_mode=mode,
                             top_k=10 if case["group"] == "summary" else 5,
                             summary_max_sources=10, candidate_k=20, rrf_k=60)


def gold_ids(case, rows):
    """Only explicit evaluation labels and stored metadata, independent of query planning."""
    if case["group"] == "negative":
        return set()
    selected = []
    for row in rows:
        if case.get("course") and row["course_code"] != case["course"]:
            continue
        if any(str(row[key]) != str(value) for key, value in case.get("scope", {}).items()):
            continue
        if case.get("topics") and row["topic_code"] not in case["topics"]:
            continue
        if case["kind"] == "knowledge" and row["document_id"] is None:
            continue
        if case["kind"] == "exercise" and row["exercise_id"] is None:
            continue
        if case.get("exercise_code") and row["exercise_code"] != case["exercise_code"]:
            continue
        if case.get("question_type") and row["question_type"] != case["question_type"]:
            continue
        if case.get("difficulty") and row["difficulty"] not in case["difficulty"]:
            continue
        selected.append(str(row["id"]))
    return set(selected)


def score(case, result, gold, by_id, planning_ms=0):
    sources = result["sources"]
    selected = sources[:5] if case["group"] == "retrieval" else sources[:10]
    labels = [str(source["id"]) in gold for source in selected]
    rank = next((i for i, relevant in enumerate(labels, 1) if relevant), None)
    stats = {"sources_returned": len(selected), "rank": rank,
             "search_ms": result["retrieval"]["timing_ms"]["total"],
             "estimated_request_ms": result["retrieval"]["timing_ms"]["total"] + planning_ms}
    if case["group"] == "retrieval":
        ideal = sum(1/log2(i+2) for i in range(min(5, len(gold))))
        stats.update(hit_at_5=int(rank is not None), mrr_at_5=1/rank if rank else 0,
                     precision_at_5=sum(labels)/5,
                     returned_precision_at_5=sum(labels)/len(labels) if labels else 0,
                     recall_at_5=sum(labels)/len(gold),
                     ndcg_at_5=sum(int(label)/log2(i+2) for i,label in enumerate(labels))/ideal)
    elif case["group"] == "summary":
        topics = {str(by_id[identity]["knowledge_point_id"]) for identity in gold}
        covered = {str(source["knowledge_point_id"]) for source in selected if str(source["id"]) in gold}
        stats.update(topic_coverage=len(covered)/len(topics), topics_covered=len(covered), total_topics=len(topics),
                     context_precision=sum(labels)/len(labels) if labels else 0, source_budget=10)
    else:
        stats.update(safe_empty=int(not sources), expected=case["expected"],
                     actual_status=(result.get("query_plan") or {}).get("status", "no_plan"))
        stats["correct_action"] = int(not sources and (case["expected"] != "needs_clarification" or stats["actual_status"] == "needs_clarification"))
    return stats


def summarize(results):
    summary = {}
    for group in ("retrieval", "knowledge_qa", "exercise_search", "problem_solving", "summary", "negative"):
        rows = [row for row in results if row["group"] == group or row["group"] == "retrieval" and row["intent"] == group]
        summary[group] = {}
        for name in EXPERIMENTS:
            values = [row["experiments"][name]["metrics"] for row in rows]
            keys = (("topic_coverage", "context_precision") if group == "summary" else
                    ("safe_empty", "correct_action") if group == "negative" else
                    ("hit_at_5", "mrr_at_5", "precision_at_5", "returned_precision_at_5", "recall_at_5", "ndcg_at_5"))
            summary[group][name] = {"queries": len(values), **{key: sum(v[key] for v in values)/len(values) for key in keys},
                                    "median_search_ms": median(v["search_ms"] for v in values),
                                    "median_estimated_request_ms": median(v["estimated_request_ms"] for v in values)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-mode", choices=("deepseek", "rules"), default="deepseek")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--with-answer-check", action="store_true")
    args = parser.parse_args()
    suite_path = DIRECTORY / "eval.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    settings, engine = get_settings(), get_engine()
    cache_path = DIRECTORY / f"plans_{args.plan_mode}.json"
    results = []
    report = {"version": __version__, "verified_at_utc": datetime.now(timezone.utc).isoformat(), "suite": suite["set"],
              "note": suite["note"], "parser": args.plan_mode, "results": results, "paid_api_attempts_this_run": 0}
    output = DIRECTORY / f"report_{args.plan_mode}.json"
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            assert conn.scalar(text("SHOW transaction_read_only")) == "on"
            rows = inventory(conn)
            before = digest({"base": manifest(conn), "tags": rows})
            by_id = {str(row["id"]): row for row in rows}
            gold = {case["id"]: gold_ids(case, rows) for case in suite["cases"]}
            assert all(gold[case["id"]] for case in suite["cases"] if case["group"] != "negative"), "A positive gold label is missing; fix labels before any API call"
            print(f"Preflight: {len(rows)} chunks, {len(suite['cases'])} cases, all positive gold labels exist", flush=True)
            snapshot_engine = SnapshotEngine(conn)
            if args.preflight_only:
                with patch("edu_rag.rag.get_engine", return_value=snapshot_engine), patch("edu_rag.rag.generate", side_effect=AssertionError("Exercise search generated an answer")):
                    case = next(case for case in suite["cases"] if case["id"] == "ex05")
                    result = rag.ask(request_for(case, rows, query_mode="rules", mode="bm25"))
                    assert result["status"] == "exercises_found" and len(result["exercises"]) == len(gold[case["id"]])
                    assert all(s["difficulty"] == 2 and s["question_type"] == "calculation" for s in result["sources"])
                    missing = next(case for case in suite["cases"] if case["id"] == "neg02")
                    assert rag.ask(request_for(missing, rows, query_mode="rules", mode="bm25"))["sources"] == []
                print("Live tag filtering, exact stored exercise presentation, and medium-level empty result passed; zero paid calls", flush=True)
                return
            signature = {"eval_sha256": sha256(suite_path.read_bytes()).hexdigest(), "corpus_sha256": before,
                         "planner_prompt_sha256": sha256(PLANNER_PROMPT.encode()).hexdigest(), "plan_version": PLAN_VERSION,
                         "code_sha256": sha256(b"".join((PROJECT_ROOT / f"src/edu_rag/{name}.py").read_bytes() for name in ("query","rag","retrieval","scope","knowledge"))).hexdigest(),
                         "model": settings.deepseek_model if args.plan_mode == "deepseek" else None}
            cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"signature": signature, "plans": {}}
            assert cache["signature"] == signature, "Frozen-plan cache differs from code, corpus or labels. Preserve it and create a separately versioned experiment."
            for i, case in enumerate(suite["cases"], 1):
                if case["id"] in cache["plans"]:
                    if cache["plans"][case["id"]]["plan"]["origin"] != "rules_fallback":
                        continue
                    cache.setdefault("failed_attempts", []).append({"id": case["id"], **cache["plans"][case["id"]]})
                request = request_for(case, rows, query_mode=args.plan_mode)
                started = perf_counter()
                # One call at most per question. Both rankers replay this same validated plan.
                plan = rag.construct_plan(request, engine=snapshot_engine)
                entry = {"plan": plan, "planning_ms": (perf_counter()-started)*1000, "created_at_utc": datetime.now(timezone.utc).isoformat()}
                cache["plans"][case["id"]] = entry
                write_json(cache_path, cache)
                if args.plan_mode == "deepseek" and plan["origin"] in ("deepseek", "rules_fallback"):
                    report["paid_api_attempts_this_run"] += 1
                if plan["origin"] == "rules_fallback":
                    raise RuntimeError("DeepSeek failed schema or service validation; partial cache preserved. Do not mix fallback plans into the DeepSeek ablation.")
                if i % 5 == 0:
                    print(f"Frozen plans {i}/{len(suite['cases'])}", flush=True)
            print("All plans frozen; loading and warming the local dense encoder", flush=True)
            encoder = get_encoder()
            encoder.encode(["四组消融评测预热"])
            mode_names = list(EXPERIMENTS)
            solve_result = None
            for i, case in enumerate(suite["cases"]):
                frozen = cache["plans"][case["id"]]
                row = {"id": case["id"], "question": case["question"], "group": case["group"], "intent": case["intent"],
                       "gold_chunk_ids": sorted(gold[case["id"]]), "plan": frozen["plan"],
                       "planning_ms": frozen["planning_ms"], "experiments": {}}
                order = mode_names[i % 4:] + mode_names[:i % 4]
                for name in order:
                    planned, mode = EXPERIMENTS[name]
                    request = request_for(case, rows, query_mode=args.plan_mode if planned == "planned" else "off", mode=mode)
                    result = rag.search(request, engine=snapshot_engine, encoder=encoder,
                                        prepared_plan=frozen["plan"] if planned == "planned" else None)
                    fields = ("id", "citation", "document_id", "exercise_id", "knowledge_point_id", "knowledge_point_code", "exercise_code",
                              "question_type", "difficulty", "source_title", "similarity", "dense_rank", "bm25_rank", "bm25_score", "rrf_score")
                    row["experiments"][name] = {"metrics": score(case, result, gold[case["id"]], by_id,
                                                frozen["planning_ms"] if planned == "planned" else 0),
                                                "sources": [{key: source.get(key) for key in fields} for source in result["sources"]],
                                                "retrieval": result["retrieval"], "coverage": (result.get("query_plan") or {}).get("coverage")}
                    if name == "query_hybrid" and case["id"] == "solve02":
                        solve_result = result
                results.append(row)
                if (i+1) % 5 == 0:
                    print(f"Compared {i+1}/{len(suite['cases'])} cases × 4 experiments", flush=True)
                write_json(output, report)
            if args.with_answer_check:
                assert solve_result is not None
                report["paid_api_attempts_this_run"] += 1
                answer = rag.generate(solve_result["question"], solve_result["sources"], query_plan=solve_result["query_plan"])
                report["answer_smoke_check"] = answer
                write_json(output, report)
                assert answer["status"] == "answered" and "22" in answer["answer"] and "24" in answer["answer"]
            assert digest({"base": manifest(conn), "tags": inventory(conn)}) == before
            write_json(DIRECTORY / "corpus_inventory.json", rows)
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        after = digest({"base": manifest(conn), "tags": inventory(conn)})
    assert before == after, "Corpus drift invalidates the comparison"
    report.update(signature=signature, corpus={"chunks": len(rows), "courses": len({r["course_id"] for r in rows}),
                  "before_sha256": before, "after_sha256": after, "unchanged": True},
                  configuration={"top_k":5, "summary_budget_all_modes":10, "candidate_k":20, "rrf_k":60,
                                 "difficulty_levels":DIFFICULTY_LEVELS, "embedding_model":encoder.model_id,
                                 "same_plan_replayed_for_both_rankers":True, "warmup_excluded":True, "mode_order_rotated":True,
                                 "scope_common_to_all_modes":"The caller-selected course or explicit scope; query-derived metadata is added only in planned modes.",
                                 "timing_note":"Warmed CPU search time includes database network calls. Planned request estimates add the same measured planner latency to both rankers. Single execution, not a load benchmark."},
                  summary=summarize(results),
                  planner={"correct_final_intents":sum(row["plan"]["intent"] == row["intent"] for row in results),
                           "queries":len(results), "guarded_intents":sum(bool(row["plan"].get("intent_guard_applied")) for row in results),
                           "deepseek_plans":sum(row["plan"]["origin"] == "deepseek" for row in results),
                           "total_tokens":sum(row["plan"].get("usage",{}).get("total_tokens",0) for row in results)})
    changes = {}
    for left,right in (("raw_bm25","query_bm25"),("raw_hybrid","query_hybrid"),("query_bm25","query_hybrid")):
        changes[left+"_to_"+right] = {kind:[row["id"] for row in results if row["group"] == "retrieval" and predicate(
            row["experiments"][left]["metrics"]["ndcg_at_5"], row["experiments"][right]["metrics"]["ndcg_at_5"])]
            for kind,predicate in (("improved",lambda a,b:b>a+1e-12),("regressed",lambda a,b:b<a-1e-12),("unchanged",lambda a,b:abs(b-a)<=1e-12))}
    report["paired_changes_ndcg_at_5"] = changes
    write_json(output, report)
    print(json.dumps({"summary":report["summary"],"paired_changes":changes,"planner":report["planner"],"corpus_unchanged":True},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
