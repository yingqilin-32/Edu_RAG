# Ragas 离线评估

按用户提供的 [All-in-RAG 常用评估工具](https://github.com/yingqilin-32/all-in-rag/blob/main/docs/chapter6/19_common_tools.md)，本项目接入 **Ragas 0.4.3**。它评价已保存的 RAG 系统输出，不涉及训练、微调或模型权重评分。LlamaIndex Evaluation 更适合其原有框架；Phoenix 可用于后续轨迹展示。本次直接复用项目已有 LangGraph 轨迹和实验数据。

## 输入与判分约定

使用 `data/v5/runs/v2/results.jsonl` 的92题、三种策略，共276条历史结果。每组32道可回答题（8道简单、24道复杂），60道不可回答题；全部仍是开发/回归题，不是新独立测试集。

字段映射：`question → user_input`，最终 `answer → response`，最终 `sources[].content → retrieved_contexts`，同序 `sources[].id → retrieved_context_ids`。仅使用实际提供给生成环节的最终片段，不混入初次候选池、未选片段或参考答案。`reference` 另外保存于 [参考答案](../data/evaluation/ragas/references.json)，由助手根据原知识卡及 V5 原有答案补充计算过程，在首次判分前固定，尚未经独立人工标注。

标准指标来自 `ragas.metrics.collections`，保留其原始提示词和计算公式：

| 指标 | 观察什么 | 本次设置 |
| --- | --- | --- |
| Faithfulness | 回答陈述能否由检索正文支持 | 标准分解陈述和NLI判定 |
| FactualCorrectness | 回答与参考事实的重合程度 | F1，atomicity=low，coverage=low |
| ContextPrecision | 有用的最终片段是否排在前面 | 使用参考答案判定相关性 |
| ContextRecall | 参考答案陈述能否由最终正文覆盖 | 使用参考答案；不是文档ID召回率 |
| AnswerAccuracy | 回答和参考答案的匹配程度 | 两个顺序互换的判分提示，每个最多1次尝试 |

这五项均不依赖向量接口，因此不需要 OpenAI 密钥或另一个 embedding 模型。DeepSeek JSON 适配器实现 Ragas 的 `InstructorBaseRagasLLM`，按其 Pydantic schema 返回结构化结果；用现有 `.env` 的模型、API地址和密钥，关闭思考、JSON温度0、输出最多4096tokens。没有隐式 SDK 重试。两个判分提示使用同一个模型，**不等于两个独立评委**。

有两个必要的分母约定：

- 无实质答案、拒答、澄清、预算停止，以及不可回答题，都不计算这五项标准回答指标，记录 `not_applicable`。逐项报告有效样本数、正例总数、错误数，不能把“拒答很忠实”当成高质量答案。
- 另列 `all_positive_utility_abstention_zero`：已判分 AnswerAccuracy 的分数总和除以全部32道正例，未作答按0计。这是本项目的覆盖与质量综合效用，**不是标准 Ragas 分数，也不是二元正确率**。报告同时给出三组都回答的共同子集，避免仅比较不同难度的已答题均分。

60道负例仅重读原有状态，记录状态误放行数，未新增语义评审。因此不能用本次 Ragas 的结果重新证明实际错误作答率或生产≤5%保证。原来的不恰当澄清措辞等限制仍在。

## 数学题的指标边界

用户题干含有具体数字，知识卡通常只含方法。标准 Faithfulness 的核验阶段只收到检索正文，可能把正确代入也判作缺少支持。试运行中 `complex-17` 的“96÷8=12本、12÷3=4本”被 AnswerAccuracy 判为1.0，但 Faithfulness 为0，原始裁判理由是知识卡未出现96、8、12、3、4等具体条件。这是该指标在本任务的适用性限制，不能直接解释为模型100%幻觉；同样不能通过向正文塞入参考答案来提高分数。

FactualCorrectness 也会受到参考答案详略及额外正确例子的影响；相关性高并不等于数值运算正确。`complex-07` 原题中的“剩几本书”有库存与装包余数歧义，保留原题、原输出、原参考，并提供剔除这一题的敏感性统计。Ragas 自动判分不替代数学核对与独立人工复核。

## 安装和复现

Ragas 0.4.3 的实际依赖与后端 LangChain 1.x 不兼容，因此使用**完全独立的 Python 3.11 venv**；不要在此环境安装本项目的完整生产依赖，也不要在后端 Conda 环境直接升级 LangChain。

```powershell
# 在 Edu_RAG_V0 目录执行
$ragPython = 'D:\Anaconda\envs\edu-rag\python.exe'
& $ragPython -m venv .runtime/ragas-eval
$evalPython = '.\.runtime\ragas-eval\Scripts\python.exe'
& $evalPython -m pip install -r requirements-eval-lock.txt
& $evalPython -m pip check

# 导出数据，不调用API，不访问数据库：
& $ragPython scripts/evaluate_ragas.py
# 实际判分；相同输入自动复用缓存：
& $evalPython scripts/evaluate_ragas.py --run
```

一轮最多400次新HTTP调用、150万tokens，token上限在响应后检查，最后一次可能超出；可用 `--max-calls`、`--max-tokens` 收紧。401/402/403后停止后续新HTTP请求。其他失败记录为错误，不伪装成0分。中断后可重新运行，复用精确请求缓存；改变模型、题集、参考或评估代码要用新的 `--output` 目录。

`--case-id` 可重复指定试运行样例，务必使用单独目录，不能把试运行当作全量报告。标准输入、参考答案、判分参数及评估代码保存哈希；缓存仅保存请求正文和原始响应，不保存HTTP头或密钥。Ragas遥测默认关闭，不需要外部评估平台账号。

输出位于 `data/evaluation/ragas/runs/v1/`：`manifest.json` 固定输入和判分设置，`samples.json` 是标准字段导出，`scores.jsonl` 含逐题分数及原始判分请求索引，`provider_cache/` 保存原始判分提示和输出，`report.json` 含三组统计、分片、共同已答子集、排除歧义题的敏感性分析和token消耗。数据库完全不在本次脚本执行路径内，也不重新生成RAG答案。

直接阅读 [本次评估结果](../data/evaluation/ragas/runs/v1/report.md)。完整依赖锁定在 `requirements-eval-lock.txt`；`requirements-eval.txt` 说明关键兼容版本。第一次新接口判分在兼容的引导环境执行，之后在上述干净venv中重放缓存复核，避免继承后端依赖。复核不产生新增HTTP调用；原始失败及调用账本保留在 `first_attempt/`。

本次验证：172项项目自动测试通过；独立评估环境 `pip check` 无依赖冲突；完整276条结果复核没有新增模型HTTP请求，最终无未解决的评估错误。累计288个不同裁判响应、318,807个已知tokens，按当日定价估算约¥0.24—¥0.49。全量32题正例的效用从单轮75.78%升至两种重试策略的78.91%；共同已答25题的各项均分完全相同，收益来自多回答了1题，而非原有答案质量普遍提高。

理论依据：[Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)、[FactualCorrectness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/)、[ContextPrecision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)、[ContextRecall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)。
