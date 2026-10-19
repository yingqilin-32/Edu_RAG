"""Offline quality/cost report for Memory ablation. Requires saved Ragas scores."""

from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from edu_rag.assessment import digest
from edu_rag.config import PROJECT_ROOT

ROOT = PROJECT_ROOT/"data/v6/runs/v2"
MODES = ("none", "history", "structured")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    source, original = read(ROOT/"suite.json"), read(ROOT/"report.json")
    rows = [json.loads(l) for l in (ROOT/"results.jsonl").read_text(encoding="utf-8").splitlines()]
    scores = read(ROOT/"answer_evaluation/scores.json")
    signature = read(ROOT/"signature.json")
    code = {p.relative_to(ROOT/"code").as_posix(): p.read_text(encoding="utf-8") for p in (ROOT/"code").rglob("*.py")}
    assert digest(code) == signature["code_sha256"]
    assert all((PROJECT_ROOT/name).read_text(encoding="utf-8") == value for name, value in code.items())
    assert digest(source) == signature["suite_sha256"]
    assert len(rows) == 3*sum(len(d["turns"]) for d in source["dialogs"])
    assert len(scores) == sum("answerable" in r["turn"] for r in rows)
    assert digest(rows) == read(ROOT/"answer_evaluation/manifest.json")["inputs_sha256"]
    summary, scored_lookup = {}, {(s["mode"], s["id"]): s for s in scores}
    prices = read(PROJECT_ROOT/"data/v5/runs/v2/pricing.json")
    for mode in MODES:
        items = [s for s in scores if s["mode"] == mode]
        positive = [s for s in items if s["turn"]["answerable"]]
        mathematical = [s for s in positive if not s["turn"].get("exercise")]
        valid = [s for s in mathematical if s["answer_accuracy"]["status"] == "scored"]
        errors = [s for s in positive if s["answer_accuracy"]["status"] == "error"]
        strict = [s for s in positive if s["assessment"]["routing_and_conditions_pass"] and s["answer_accuracy"]["value"] == 1]
        requests = [r for r in rows if r["mode"] == mode and "answerable" in r["turn"]]
        usage = [c["usage"] for r in requests for c in r["response"]["workflow"]["budget"]["records"]]
        hit = sum(u.get("prompt_cache_hit_tokens", 0) for u in usage)
        miss = sum(u.get("prompt_tokens", 0)-u.get("prompt_cache_hit_tokens", 0) for u in usage)
        output = sum(u.get("completion_tokens", 0) for u in usage)
        costs = {label: (hit*price["input_hit"]+miss*price["input_miss"]+output*price["output"])/1000000
                 for label, price in (("off_peak", prices["off_peak_rmb_per_million"]), ("peak", prices["peak_rmb_per_million"]))}
        summary[mode] = {**original["modes"][mode], "strict_followup_success": len(strict),
                         "strict_success_ids": [s["id"] for s in strict], "answer_evaluation_errors": len(errors),
                         "ragas_math_scored": len(valid), "ragas_math_total": len(mathematical),
                         "ragas_answer_accuracy_conditional": sum(s["answer_accuracy"]["value"] for s in valid)/len(valid) if valid else None,
                         "ragas_answer_utility_abstention_zero": sum(s["answer_accuracy"]["value"] for s in valid)/len(mathematical) if mathematical and not errors else None,
                         "followup_cost_estimate_rmb": costs,
                         "strict_failures": [{"id": s["id"], "status": s["status"], "assessment": s["assessment"], "score": s["answer_accuracy"]}
                                             for s in positive if s not in strict]}
    common = set.intersection(*(set(summary[m]["strict_success_ids"]) for m in MODES))
    paired = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode and r["dialog_id"]+f"-{r['turn_index']}" in common]
        paired[mode] = {"followups": len(selected), "model_calls": sum(r["response"]["workflow"]["budget"]["model_calls"] for r in selected),
                        "tokens": sum(r["response"]["usage"]["total_tokens"] for r in selected)}
    first_turns_equal = all(len({tuple(c["key"] for c in r["provider_requests"]) for r in rows if r["dialog_id"] == d["id"] and r["turn_index"] == 1}) == 1 for d in source["dialogs"])
    result = {"modes": summary, "common_strict_success_ids": sorted(common), "common_success_costs": paired,
              "first_turn_provider_requests_identical": first_turns_equal,
              "budget_calls_respected": all(r["response"]["workflow"]["budget"]["model_calls"] <= 10 for r in rows),
              "core_code_and_inputs_unchanged": True, "corpus_unchanged": original["corpus_unchanged"],
              "metric_note": "Strict follow-up success combines saved routing/keyword/scope checks and full Ragas answer match, or deterministic exercise tag checks; not independently human annotated accuracy.",
              "negative_sample_note": "Only three negative follow-ups per arm; do not claim a <=5% production false-answer guarantee.",
              "latency_note": "API latency proxy replays original provider duration for cached prompts; real end-to-end smoke tests are separate.",
              "compression_note": "Structured mode retains four recent records; lower cost can reflect forgetting and safe abstention, not equal-quality compression."}
    (ROOT/"analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    n = summary["none"]["positive_followups"]
    lines = ["# V6 Memory 三组对照", "", "2026-10-19。固定14段原创对话，每组36轮，共108轮；每组14道可回答追问、3道不可回答追问。其余19轮用于建立上下文。", "",
             "三组复用V5检索、LLM重排、证据门控、生成与核验，最多2次重检、3条分支、10次模型调用、30,000 tokens；追问还原也计入总预算。", "",
             "| 指标 | 无记忆 | 完整历史 | 结构化记忆（最近4条） |", "| --- | ---: | ---: | ---: |"]
    def row(label, values): lines.append("| "+label+" | "+" | ".join(str(v) for v in values)+" |")
    row("可回答追问已作答", [f"{summary[m]['positive_answered']}/{n}" for m in MODES])
    row("严格追问成功", [f"{summary[m]['strict_followup_success']}/{n}" for m in MODES])
    row("Ragas数学答案匹配分（已答）", [f"{summary[m]['ragas_answer_accuracy_conditional']:.3f} (n={summary[m]['ragas_math_scored']})" if summary[m]['ragas_answer_accuracy_conditional'] is not None else "N/A" for m in MODES])
    row("数学全题效用（未答按0）", [f"{summary[m]['ragas_answer_utility_abstention_zero']:.3f}" if summary[m]['ragas_answer_utility_abstention_zero'] is not None else "N/A" for m in MODES])
    row("负例状态误放行", [f"{summary[m]['negative_status_flags']}/3" for m in MODES])
    row("17个追问的模型调用", [summary[m]["model_calls_followups"] for m in MODES])
    row("17个追问的tokens", [f"{summary[m]['tokens_followups']:,}" for m in MODES])
    row("36轮全部tokens", [f"{summary[m]['tokens_all_turns']:,}" for m in MODES])
    row("平均记忆上下文字符数", [f"{summary[m]['avg_context_characters_followups']:.1f}" for m in MODES])
    row("模型耗时代理中位数（秒）", [f"{summary[m]['median_api_latency_proxy_ms']/1000:.2f}" for m in MODES])
    row("17个追问的费用估计", [f"¥{summary[m]['followup_cost_estimate_rmb']['off_peak']:.4f}—{summary[m]['followup_cost_estimate_rmb']['peak']:.4f}" for m in MODES])
    lines.extend(["", "严格追问成功要求：状态正确、已列关键条件及来源范围检查通过，并且Ragas答案匹配=1.0；找题以真实标签和返回数量校验替代Ragas。关键词检查是粗粒度代理，不能证明所有语义条件保真。Ragas数字是匹配分，非数学正确率；生成与裁判均为DeepSeek，同模型家族偏差仍在。", "",
                  "## 成本怎么比较", "", "无记忆组在缺题干时可能直接澄清，调用较少。恢复原本不能回答的追问会增加检索、生成和核验调用；不能把总成本增加直接解释为效率降低。共同严格成功子集另保存在 analysis.json。", "",
                  "完整历史最多12轮；结构化记忆是最近4条原题/已通过工作流核验的结果及范围的确定性记录，尚未实现LLM叙述摘要。短对话中键名等结构开销可能使其更长；较早原题被丢弃时，安全澄清会同时降低成本和完成率，不能称为同质量压缩收益。", "",
                  "long-history专门测试超出结构化窗口后回指最初的题；参数修改、补全题干、明确换题、年级切换、未知图片与教材出处分别保存逐题结果。历史回答即使通过模型核验也可能有错；Memory不能作为教材来源或已证明的事实。", "",
                  "## 可复核性与限制", "",
                  f"首轮三组提供商请求完全相同：{first_turns_equal}。所有轮次模型调用≤10：{result['budget_calls_respected']}。检索使用只读一致快照，前后语料指纹一致：{original['corpus_unchanged']}。独立事务的后验检查另保存为 database_audit.json。", "",
                  "Ragas裁判收到独立保存的完整用户对话和参考答案，不使用系统还原后的问题冒充金标准；参考答案从未传入被测应用。相同提供商请求共享原始响应缓存，属于配对开发实验，非108次独立试验。原始响应、调用量、还原问题、会话轨迹、输入与代码签名均已保存。", "",
                  "每组只有3道负例，不能由0/3推断实际错误作答率≤5%。本次题集与此前单轮Ragas题集不同，不能直接比较两份报告的绝对匹配分。所有费用按记录usage和2026-10-19公开价格估算，非账单。耗时代理仅为提供商原始时长之和，不是端到端延迟；真实HTTP冒烟记录另存 live_integration.json。", "",
                  "复现：主环境运行 scripts/evaluate_memory.py；独立Ragas环境运行 scripts/evaluate_memory_answers.py；最后主环境运行 scripts/report_memory.py。配置、运行及本机Memory限制见 [V6说明](../../../../docs/V6-memory.md)。", ""])
    (ROOT/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({m: {k: summary[m][k] for k in ("positive_answered", "strict_followup_success", "tokens_followups", "model_calls_followups")} for m in MODES}, ensure_ascii=False))


if __name__ == "__main__":
    main()
