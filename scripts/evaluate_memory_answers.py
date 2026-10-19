"""Ragas answer matching against independent dialog references; no DB or generation."""

import asyncio
import importlib.metadata
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from edu_rag.assessment import digest
from edu_rag.config import PROJECT_ROOT, get_settings
from evaluate_ragas import JudgeHTTP, build_judge, read, write

ROOT = PROJECT_ROOT/"data/v6/runs/v2"
OUT = ROOT/"answer_evaluation"


async def run():
    from ragas.metrics.collections import AnswerAccuracy
    if importlib.metadata.version("ragas") != "0.4.3":
        raise RuntimeError("Use the pinned Ragas evaluation environment")
    suite = read(ROOT/"suite.json")
    dialogs = {d["id"]: d for d in suite["dialogs"]}
    rows = [json.loads(l) for l in (ROOT/"results.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 3*sum(len(d["turns"]) for d in suite["dialogs"]):
        raise RuntimeError("Wait for the paired Memory run to complete")
    settings = get_settings()
    manifest = {"inputs_sha256": digest(rows), "suite_sha256": digest(suite), "model": settings.deepseek_model,
                "ragas": "0.4.3", "metric": "AnswerAccuracy", "max_retries": 1,
                "reference_use": "Only in evaluator; user_input contains full canonical user dialog, not the system's rewritten question.",
                "code_sha256": digest(Path(__file__).read_text(encoding="utf-8"))}
    if (OUT/"manifest.json").exists() and read(OUT/"manifest.json") != manifest:
        raise ValueError("Evaluation inputs changed; preserve the old output before a new run")
    write(OUT/"manifest.json", manifest)
    client = JudgeHTTP(OUT/"provider_cache", settings, max_calls=120, max_tokens=250000)
    metric = AnswerAccuracy(llm=build_judge(settings, client), max_retries=1)
    scores = []
    try:
        for row in rows:
            if "answerable" not in row["turn"]:
                continue
            answer, turn = row["response"], row["turn"]
            item = {"id": row["dialog_id"]+f"-{row['turn_index']}", "mode": row["mode"],
                    "turn": turn, "response": answer["answer"], "status": answer["status"],
                    "resolved_question": answer["resolved_question"], "assessment": row["assessment"]}
            if not turn["answerable"]:
                result = {"status": "not_applicable_negative", "value": None}
            elif turn.get("exercise"):
                result = {"status": "exercise_metadata_check", "value": float(row["assessment"]["routing_and_conditions_pass"])}
            elif answer["status"] != "answered":
                result = {"status": "not_answered", "value": None}
            else:
                history = dialogs[row["dialog_id"]]["turns"][:row["turn_index"]]
                inputs = {"user_input": "根据下列用户对话条件回答最后一轮问题：\n"+"\n".join(f"{i+1}. {t['question']}" for i, t in enumerate(history)),
                          "response": answer["answer"], "reference": turn["reference"]}
                key = digest({"manifest": {k: v for k, v in manifest.items() if k not in ("inputs_sha256",)}, "inputs": inputs})
                path = OUT/"metric_cache"/(key+".json")
                if path.exists():
                    result = read(path)
                else:
                    before = len(client.served)
                    try:
                        value = float((await metric.ascore(**inputs)).value)
                        if not math.isfinite(value):
                            raise ValueError("Undefined judge score")
                        result = {"status": "scored", "value": value,
                                  "provider_keys": [c["key"] for c in client.served[before:]]}
                        write(path, result)
                    except Exception as exc:
                        result = {"status": "error", "value": None, "error_type": type(exc).__name__}
            scores.append({**item, "answer_accuracy": result})
            write(OUT/"scores.json", scores)
            print(item["id"], row["mode"], result["status"], result["value"], flush=True)
        write(OUT/"requests.json", client.served)
        write(OUT/"invocation.json", {"new_http_attempts": client.calls, "new_tokens": client.tokens})
    finally:
        client.client.close()


if __name__ == "__main__":
    asyncio.run(run())
