# Edu RAG · V6

当前版本 **0.6.0** 新增本机会话 Memory。14段对话、108轮配对开发实验中，可回答追问完成数为：无记忆3/14、完整历史14/14、最近4条结构化记忆13/14。完整历史的36轮总tokens比无记忆多11.5%；精简模式因遗忘一条较早原题而澄清，不能把少答题带来的节省算成同质量压缩收益。方法、成本和限制见 [V6 Memory说明](docs/V6-memory.md)、[三组结果](data/v6/runs/v2/report.md)。

新增独立环境的 Ragas 离线评估：对 V5 已保存的三组结果检查证据支持、参考答案匹配及最终上下文相关性与覆盖率；逐题数据、适用边界与复现方法见 [Ragas 评估说明](docs/ragas-evaluation.md)。

V5（**0.5.0**）阶段：使用 LangGraph 管理证据不足后的分支检索、查询改写与有限重试；默认最多2次重检、每轮3个检索分支、10次模型调用。没有新证据时停止，调用预算在HTTP边界执行；总tokens采用保守预估准入并记录实际usage。

92题三组对照中，复杂题完成数为单轮17/24、固定单路重试18/24、按需分支18/24。相同完成数下按需策略少用19.2%调用、8.4% tokens；相对单轮多恢复1题，但调用和tokens也增加。三组均0/60负例状态误放行。助手复核另发现一条已完成答案结尾数量表述不清，因此完成率不能当准确率。162项自动测试与8项真实HTTP集成通过，见 [V5工作流、成本与实验报告](docs/V5.md)。

V4 使用 **LangGraph 编排查询理解 → 检索 → 重排与过滤 → 独立证据门控 → 回答／澄清／拒答**，沿用四类意图、年级和教材范围、题库标签，以及BGE-M3、BM25和RRF。历史0.4.1验收说明见 [V4工作流与验收口径](docs/V4.md)。

V4.1 固定查询与候选池完成 92 题 × 5 组对照：原顺序、仅重排、仅过滤、重排加过滤的正例完成数分别为29/32、30/32、32/32、32/32，保留门控时均为0/60负例状态误放行。重排改善排序，完成率主要收益来自过滤；默认保留重排加过滤并独立评估最终上下文。关闭生成前门控有2次状态误放行，其中一次正文实际为拒答，不能称为两次事实幻觉。143项测试通过，方法、成本、逐题结果与限制见 [重排及门控对照报告](docs/rerank-gate-ablation.md)。

V4第二轮合成回归验收：不可回答题错误作答0/60（样本内≤5%），可回答题18/20作答；129项自动测试通过，无服务异常，最多2次重检。首轮失败与正例标签审计完整保留，不能把回归结果推广为生产保证，详见 [完整验收报告](data/v4/runs/v2/report.json)。

V3.2曾固定114个向量片段和51个开发问题完成四组实验：36道检索问题中，DeepSeek查询构建使BM25的nDCG@5从0.7916升至0.9899，加入Dense后指标持平。这支持以BM25作为检索对照；V4默认混合检索并增加证据工作流，不能把V3.2的数字直接解释成V4的作答质量。历史数据、限制和复现方法见 [四类意图与对照实验](docs/four-intent-ablation.md)、[逐题完整报告](data/v3/four_way/report_deepseek.json)。V3.1说明和旧数据保留在 [查询构建说明](docs/query-construction.md)、[历史验收记录](data/v3/query_construction/report.json)。

检索入口在 `src/edu_rag/rag.py` 的 `search()`，词项、BM25 和 RRF 在 `src/edu_rag/retrieval.py`。工作台的“资料问答”可切换三种方式，点“仅检索”查看排名、分数和候选。选择本地解析或关闭查询构建时无需调用 DeepSeek；选择 DeepSeek 解析时会单独产生解析调用。无需新增依赖、数据库迁移或重新生成向量。详见 [V3 实现与使用](docs/V3.md)、[实测记录](docs/V3-verification.md) 和 [逐题对照数据](data/v3/retrieval_report.json)。

## 快速使用

已完成 V0 初始化的本机，在项目根目录运行：

```powershell
$ragPython = 'D:\Anaconda\envs\edu-rag\python.exe'
& $ragPython -m pip install -r requirements-v4-lock.txt
& $ragPython -m pip install --no-deps --no-build-isolation -e .
& $ragPython -m edu_rag.ingest demo
& $ragPython -m uvicorn edu_rag.main:app --host 127.0.0.1 --port 8000
```

打开 [学习工作台](http://127.0.0.1:8000)，可以直接浏览知识点、正文、练习题和答案解析。问答需要本机 `.env` 中的 `DEEPSEEK_API_KEY`；仅浏览知识库无需调用模型 API。

全新 V0 样例升级后的预期总数为 **16 个知识点、17 份资料、51 道题、69 个真实向量片段**。这些内容是原创功能样例，不是完整课程体系。样例原文见 [三年级数学知识库](data/v1/math_g3.md)，导入源为 [JSON 数据](data/v1/math_g3.json)。

页面支持年级、教材版本和学期筛选。暂无匹配资料时显示空结果，不会扩大到其他课程。当前已收录三年级全年样例，以及一年级上、下学期原创补充资料；筛选功能不会自动补齐尚未导入的教材。

一年级资料参考用户提供网页的主题目录，讲解和 30 道练习由本项目重新编写，未直接转载原文或图片。已导入的数据库合计 **3 门课程、31 个知识点、81 道题、114 个真实向量片段**。来源、范围和导入记录见 [一年级入库说明](docs/G1-import.md)；可阅读 [上学期知识库](data/g1/math_first.md) 和 [下学期知识库](data/g1/math_second.md)。

完整操作与导入格式见 [V2 说明](docs/V2.md)，[V1 说明](docs/V1.md) 保留历史范围。V2 复用 V0 数据库结构，无需重建表；下方保留 V0 的首次初始化流程。

## V0 首次初始化参考

V0 提供 Conda 环境、Supabase 数据库迁移、原创演示数据、数据库检查和最小 FastAPI 后端。以下内容保留其初始化操作。

数据库使用 Supabase PostgreSQL + pgvector。V0 阶段不调用语言模型、不生成 Embedding，不需要 Docker。部署到你自己的 Supabase 项目后，数据才会出现在云端；解压代码本身不会创建云端数据库。

## 1. 放到你的 Windows 文件夹

将压缩包中的项目文件解压到：

```text
D:\2026-2027telecom_sudparis\Edu_RAG
```

进入实际包含 `README.md`、`environment.yml`、`pyproject.toml` 和 `supabase` 文件夹的项目根目录。当前解压后多了一层 `Edu_RAG_V0`，下方命令按这个实际路径编写。

在支持 Conda 的 PowerShell / Anaconda Prompt 中运行：

```powershell
Set-Location 'D:\2026-2027telecom_sudparis\Edu_RAG\Edu_RAG_V0'
conda env create -f environment.yml
conda activate edu-rag
python --version
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

Python 应显示 3.11.x。`requirements-lock.txt` 固定本次验证过的依赖版本；项目以 editable 方式安装，之后修改源码无需重新安装。若你已有同名 Conda 环境，先查看 `conda env list`，不要重复创建。

如果 PowerShell 的脚本执行策略阻止环境激活，可直接使用环境内的 Python。后续命令中的 `python` 替换为 `& $ragPython`：

```powershell
$ragPython = 'D:\Anaconda\envs\edu-rag\python.exe'
& $ragPython --version
& $ragPython -m pytest -q
```

## 2. 准备 Supabase 项目

打开 [Supabase Dashboard](https://supabase.com/dashboard)，为本项目创建一个 **Free** 项目，名称可以是 `edu-rag`。地区选择离你近的可用区域。保存数据库密码，它与 Supabase 网站登录密码不同。

已有独立的 Edu RAG 项目可以直接使用。不要将下面的迁移运行到无关项目。

进入项目顶部 **Connect → Session pooler**，取出这几个参数：

| `.env` 配置 | 从哪里取 |
| --- | --- |
| `DB_HOST` | Session pooler 的 Host，完整复制，不按地区猜测 |
| `DB_PORT` | Session pooler 显示的端口，通常为 5432 |
| `DB_NAME` | Database，通常为 postgres |
| `DB_USER` | User，通常为 postgres.你的项目引用 |
| `DB_PASSWORD` | 创建项目时设置的数据库密码 |

Session pooler 可供 IPv4 网络连接。V0 使用它连接数据库，不需要在本机启动 PostgreSQL，也不需要 Supabase CLI。

```powershell
Copy-Item .env.example .env
notepad .env
```

只在本机填写实际值。密码保留在引号中；本项目分别读取各项参数，密码中的 `@` 等字符无需做 URL 编码。不要将 `.env` 上传到 GitHub，也不要在聊天中粘贴密码、service role key 或完整连接字符串。

## 3. 构建数据库

如果你希望先在云端看到表：打开该项目的 **SQL Editor → New query**，复制 `supabase/bootstrap_v0.sql` 的全部内容并运行一次。该文件包含建表和演示数据，运行成功即完成本节云端操作。不要在已初始化的项目上重复运行这个合并文件；后续使用下面的迁移命令。

合并文件也会记录迁移版本，因此之后本机执行 `migrate` 会正常跳过，`seed` 可以安全重复执行。

如果由本机直接完成初始化，配置 `.env` 后运行以下命令：

按顺序执行：

```powershell
python -m edu_rag.db migrate
python -m edu_rag.db seed
python -m edu_rag.db check
```

`migrate` 开启 pgvector、创建项目表、关联约束和精确向量查询函数；重复运行会跳过已应用版本。`seed` 导入原创演示数据，重复运行不会覆盖或重复插入已有演示记录。

全新数据库预期结果：

| 内容 | 数量 |
| --- | ---: |
| 课程 | 1 |
| 知识点 | 2 |
| 资料 | 1 |
| 练习题 | 3 |
| 题目与知识点的关联 | 3 |
| 检索片段 | 5 |
| 已生成向量的片段 | **0** |

向量数为 0 是 V0 的正常状态。真实 Embedding 将在 V1 中生成，当前没有用随机向量伪装检索效果。

Supabase 的 Table Editor 中选择 `public` schema，可看到 `edu_` 开头的表。完整字段说明见 [数据库设计](docs/database.md)。

## 4. 启动最小后端

```powershell
python -m uvicorn edu_rag.main:app --reload --host 127.0.0.1 --port 8000
```

- API 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 应用存活检查：[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- 云端数据库检查：[http://127.0.0.1:8000/health/db](http://127.0.0.1:8000/health/db)
- 演示课程列表：[http://127.0.0.1:8000/api/v0/courses](http://127.0.0.1:8000/api/v0/courses)

`/health` 成功只表示后端启动。`/health/db` 成功才表示环境已经连接到实际数据库并且项目表存在。当前是本机开发后端，尚未提供用户登录或聊天功能。

## 5. GitHub 关联

先在 [你的 GitHub 新建仓库页面](https://github.com/new) 创建空仓库，例如 `Edu_RAG`。私人代码建议先选择 Private；首次创建时不要额外添加 README。

新建空仓库后，在本机运行，最后两条的 URL 使用 GitHub 页面实际显示的地址：

```powershell
git init
git add .
git status
git commit -m "chore: initialize V0 Supabase database foundation"
git branch -M main
git remote add origin https://github.com/yingqilin-32/Edu_RAG.git
git push -u origin main
```

运行 `git status` 时确认 `.env` 不在提交列表中。仓库名称若不同，调整 remote URL。此项目包不假设远端仓库已经存在。

以后的更新：

```powershell
git add .
git commit -m "describe the change"
git push
```

## 测试与常见问题

```powershell
python -m pytest -q
```

Python 测试覆盖无数据库时的启动、数据库失败返回状态、凭据缺失和密码特殊字符处理。SQL 集成断言位于 `tests/database_contract.sql`，仅供空的、可丢弃的测试数据库使用，内部测试向量会回滚。不要把它当成正式入库脚本。

| 现象 | 检查内容 |
| --- | --- |
| 找不到 `conda` | 使用 Miniconda / Anaconda Prompt，或先完成 Conda 初始化 |
| 提示缺少 DB_HOST 等参数 | 确认 `.env` 在项目根目录，已填写而非只修改 `.env.example` |
| 连接超时、DNS 或认证失败 | 检查项目是否暂停、Session pooler 参数、密码和网络连通性 |
| 提示表不存在 | 先执行 migrate，再 seed / check |
| 提示 vector 不在 extensions schema | 已有扩展位于别处；先确认项目与扩展位置，再适配迁移，不自动移动共享扩展 |
| `.env` 已改但服务仍用旧配置 | 重启 Uvicorn，设置在进程内有缓存 |
| 端口 8000 被占用 | 启动命令改用其他端口，如 8001 |

## 后续迭代

V1 完成知识卡与向量问答，V2 完成清洗、切分及范围过滤，V3 完成 BM25/RRF 混合检索与同语料对照，V4完成重排和证据门控，V5完成按需分支、有限重试及调用预算。后续重点验证真实问题、查询范围规划、阈值泛化与答案质量。

参考项目：[Edu-RAG](https://github.com/easyroc/edu-rag)、[RAG Web UI 教程](https://github.com/rag-web-ui/rag-web-ui/blob/main/docs/tutorial/README.md)、[All-in-RAG](https://github.com/yingqilin-32/all-in-rag)。本项目骨架和演示数据为本次新建，未复制参考仓库代码。

官方说明：[pgvector](https://supabase.com/docs/guides/database/extensions/pgvector)、[连接数据库](https://supabase.com/docs/guides/database/connecting-to-postgres)、[RLS](https://supabase.com/docs/guides/database/postgres/row-level-security)。
