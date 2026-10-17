# Edu RAG · V0

一个从基础检索逐步演进的教育 RAG 项目。V0 提供 **Conda 环境、Supabase 数据库迁移、原创演示数据、数据库检查和最小 FastAPI 后端**。

当前数据库使用 Supabase PostgreSQL + pgvector。V0 不调用语言模型、不生成 Embedding，不需要 Docker。部署到你自己的 Supabase 项目后，数据才会出现在云端；解压代码本身不会创建云端数据库。

## 1. 放到你的 Windows 文件夹

将压缩包中的项目文件解压到：

```text
D:\2026-2027telecom_sudparis\Edu_RAG
```

确认该文件夹下直接有 `README.md`、`environment.yml`、`pyproject.toml` 和 `supabase` 文件夹。

在支持 Conda 的 PowerShell / Anaconda Prompt 中运行：

```powershell
Set-Location 'D:\2026-2027telecom_sudparis\Edu_RAG'
conda env create -f environment.yml
conda activate edu-rag
python --version
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

Python 应显示 3.11.x。`requirements-lock.txt` 固定本次验证过的依赖版本；项目以 editable 方式安装，之后修改源码无需重新安装。若你已有同名 Conda 环境，先查看 `conda env list`，不要重复创建。

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

V1：文本入库、BGE-M3 向量化、基础检索与带引用的回答。后续分别评测元数据过滤、BM25/RRF、重排、门控与 LangGraph。当前不提前安装这些模型或框架。

参考项目：[Edu-RAG](https://github.com/easyroc/edu-rag)、[RAG Web UI 教程](https://github.com/rag-web-ui/rag-web-ui/blob/main/docs/tutorial/README.md)、[All-in-RAG](https://github.com/yingqilin-32/all-in-rag)。本项目骨架和演示数据为本次新建，未复制参考仓库代码。

官方说明：[pgvector](https://supabase.com/docs/guides/database/extensions/pgvector)、[连接数据库](https://supabase.com/docs/guides/database/connecting-to-postgres)、[RLS](https://supabase.com/docs/guides/database/postgres/row-level-security)。

