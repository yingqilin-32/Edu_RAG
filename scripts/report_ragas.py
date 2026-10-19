"""Generate a human-readable Ragas report from saved scores; zero network calls."""

from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from edu_rag.assessment import digest
from edu_rag.config import PROJECT_ROOT

DIRECTORY = PROJECT_ROOT / "data/evaluation/ragas/runs/v1"
LABELS = {"faithfulness": "证据支持度", "factual_correctness": "事实一致性 F1",
          "context_precision": "最终上下文相关性", "context_recall": "最终上下文覆盖率",
          "answer_accuracy": "参考答案匹配度"}
POLICIES = ("single_pass", "always_retry", "adaptive")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pct(value):
    return "N/A" if value is None else f"{value*100:.2f}%"


def main():
    report = read(DIRECTORY/"report.json")
    rows = [json.loads(l) for l in (DIRECTORY/"scores.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = read(DIRECTORY/"manifest.json")
    current_code = digest({p: (PROJECT_ROOT/p).read_text(encoding="utf-8") for p in
                          ("scripts/evaluate_ragas.py", "src/edu_rag/assessment.py")})
    references = read(PROJECT_ROOT/"data/evaluation/ragas/references.json")
    assert current_code == manifest["identity"]["evaluation_code_sha256"], "Evaluation code changed"
    assert digest(references) == manifest["references_sha256"], "References changed"
    assert digest(read(DIRECTORY/"samples.json")) == manifest["samples_sha256"], "Export changed"
    assert len(rows) == manifest["sample_rows"], "Partial evaluation"
    usage = report["provider"]["usage"]
    pricing = read(PROJECT_ROOT/"data/v5/runs/v2/pricing.json")
    hit = usage["prompt_cache_hit_tokens"]
    miss = max(usage["prompt_tokens"]-hit, 0)
    estimated_cost = {period: (hit*rates["input_hit"]+miss*rates["input_miss"]+
                               usage["completion_tokens"]*rates["output"])/1000000
                      for period, rates in (("off_peak", pricing["off_peak_rmb_per_million"]),
                                             ("peak", pricing["peak_rmb_per_million"]))}
    actual_files = [read(p) for p in (DIRECTORY/"provider_cache").glob("*.json")]
    # All Ragas calls in this run, including any partial metrics, remain auditable.
    all_usage = {k: sum(f["response"].get("usage", {}).get(k, 0) for f in actual_files)
                 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    audit = {"input_manifest_verified": True, "evaluation_code_verified": True,
             "reference_file_verified": True, "rows": len(rows),
             "distinct_completed_answer_texts": len({r["response"] for r in rows if r["status"] == "answered"}),
             "metric_statuses": dict(Counter(v["status"] for r in rows for v in r["metrics"].values())),
             "all_saved_provider_responses": len(actual_files), "all_saved_usage": all_usage,
             "estimated_evaluation_cost_rmb": estimated_cost, "pricing_asof": pricing["verified_at"],
             "cost_note": "Distinct saved judge responses including pilot reused by full run; not RAG generation cost or provider invoice.",
             "known_ambiguity": [r for r in rows if r["id"] == "complex-07"],
             "recovered_case": [r for r in rows if r["id"] == "complex-17"]}
    audit["assistant_review_of_low_answer_accuracy"] = {
        "reviewer": "Codex assistant, not independent human annotation",
        "case_ids": ["pos-03", "pos-06", "pos-07"],
        "observation": "Each scored 0.75: the two Ragas judge prompts returned 2 and 4. Reading the answers confirms correct core square perimeter, unit conversions and borrowing rules. References add details not explicitly requested in every question. The rating-only judge output gives no reason, so verbosity or judge disagreement is a possible explanation, not a proven cause. Keep original references and scores unchanged.",
        "ambiguous_case_caveat": "complex-07 scores AnswerAccuracy=1.0 despite the previously flagged ambiguous final remaining-book count. This false acceptance limits use as a binary mathematical verifier."}
    first_attempt = DIRECTORY/"first_attempt/report.json"
    if first_attempt.exists():
        initial = read(first_attempt)
        audit["first_attempt"] = {"metric_errors": initial["errors"], "invocation": initial["this_invocation"],
                                   "resolution": "Exact-input success already obtained by another policy; replayed in clean environment with no new HTTP calls. First scores, report and request ledger preserved."}
    (DIRECTORY/"audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    policies = report["policies"]
    lines = ["# V5 Ragas 评估结果", "", "2026-10-19。Ragas 0.4.3，裁判 DeepSeek `deepseek-flash`。评价保存的系统输出，没有重新生成答案或访问数据库。", "",
             "92题 × 三种策略 = 276条记录；每组32道正例、60道负例。标准分数仅针对有实质回答的正例，是0—1分数的百分比展示，不是二元正确率。", "",
             "| 指标 | 单轮 | 固定单路重试 | 按需分支 |", "| --- | ---: | ---: | ---: |"]
    lines.append("| 已回答正例 | "+" | ".join(f"{policies[p]['completed']}/{policies[p]['positives']}" for p in POLICIES)+" |")
    for name, label in LABELS.items():
        lines.append("| "+label+" | "+" | ".join(pct(policies[p]["metrics"][name]["mean_conditional"])+
                     f" (n={policies[p]['metrics'][name]['scored']})" for p in POLICIES)+" |")
    lines.append("| 全正例效用（未回答按0） | "+" | ".join(pct(policies[p]["metrics"]["answer_accuracy"]["all_positive_utility_abstention_zero"]) for p in POLICIES)+" |")
    lines.extend(["", "“全正例效用”是参考答案匹配分总和÷32，单独惩罚应答题上的拒答；不是标准 Ragas 指标。所有原始分数、分母、错误和N/A见 report.json。", "",
                  "## 配对和分片", "", "三组共同已回答的 "+str(len(report["common_completed_case_ids"]))+" 题：", "",
                  "| 指标 | 单轮 | 固定单路重试 | 按需分支 |", "| --- | ---: | ---: | ---: |"])
    for name, label in LABELS.items():
        lines.append("| "+label+" | "+" | ".join(pct(report["common_completed"][p]["metrics"][name]["mean_conditional"]) for p in POLICIES)+" |")
    lines.extend(["", "| 分片 | 单轮：已答/总数；匹配分 | 固定重试 | 按需分支 |", "| --- | --- | --- | --- |"])
    for partition in ("simple_control", "new_authored_complex"):
        groups = report["partitions"][partition]
        lines.append("| "+partition+" | "+" | ".join(f"{groups[p]['completed']}/{groups[p]['positives']}；"+
                     pct(groups[p]["metrics"]["answer_accuracy"]["mean_conditional"]) for p in POLICIES)+" |")
    lines.extend(["", "排除已知歧义题 complex-07 后的全正例效用（分母31）："+
                  "、".join(p+" "+pct(report["excluding_known_ambiguous_case"][p]["metrics"]["answer_accuracy"]["all_positive_utility_abstention_zero"]) for p in POLICIES)+"。", "",
                  "## 如何解释", "",
                  "- 五项标准分数测量不同问题，不能把相关性或参考匹配高分称作完全无幻觉。教材卡包含方法，题干包含具体数字；标准 Faithfulness 不在核验阶段附加题干，因此会对正确代入给低分。complex-17 的匹配分1.0、证据支持度0就是已复核的例子，原始裁判理由保留在 provider_cache。",
                  "- ContextPrecision 衡量最终保留片段的排序质量，并不衡量初始候选召回或漏题；ContextRecall 是基于参考陈述的LLM归因，同样受数学推导和裁判宽严影响。",
                  "- complex-07 的库存495本、装55包、未装包0本需要区分。自动裁判可能接受正文包含正确中间结果的答案，仍须保留既有人工式复核警示，不能用自动分数抹掉末句歧义。",
                  "- 自动答案匹配扣分仅出现在 pos-03、pos-06、pos-07，各为0.75：两个判分提示分别给2和4。逐题阅读后，正方形周长、单位换算和退位的核心内容均正确；参考更详尽，可能影响评分。但裁判只返回分数，没有理由，不能确定扣分原因，也不能把97.12%直接称为数学正确率。原参考与分数均未修改。",
                  "- 三组每组60道负例的状态误放行仍为0；本次没有对负例正文另做语义裁判，不把该数值当作真实生产错误率。",
                  "- 参考答案由助手编写，生成与裁判使用同一模型家族，部分历史结果共享缓存；这不是独立人工评价或统计独立的77次实验。下一步应用未见过的真实问题和独立人工标注复核。", "",
                  "## 调用成本与复现", "",
                  f"累计保存 {len(actual_files)} 个不同裁判响应（含复用的试运行响应），共 {all_usage['total_tokens']:,} tokens。全量评估错误数：{report['errors']}。生产问答原来的调用成本未计入这里。", "",
                  "首次执行与缓存复核的调用量分开保留；瞬时失败的HTTP尝试可能没有usage，已知token总量仅含实际返回usage的响应。首次单轮 pos-08 的 Faithfulness 请求失败，同输入在另一组已成功；以精确输入缓存完成复核，原始失败状态和请求账本保留在 first_attempt/。", "",
                  f"按 {pricing['verified_at']} 的 [DeepSeek官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) 与实际缓存命中计数估算，评估费用约 ¥{estimated_cost['off_peak']:.4f}—¥{estimated_cost['peak']:.4f}，不是账单金额。", "",
                  "题集、参考、代码哈希、标准导出、逐项裁判请求、原始响应及调用量均保留在同目录；pip锁文件为 requirements-eval-lock.txt。完整步骤及指标依据见 [接入说明](../../../../../docs/ragas-evaluation.md)。", ""])
    (DIRECTORY/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "errors": report["errors"], "judge_responses": len(actual_files),
                      "tokens": all_usage["total_tokens"], "estimated_cost_rmb": estimated_cost}, ensure_ascii=False))


if __name__ == "__main__":
    main()
