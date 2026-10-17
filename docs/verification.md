# V0 验证记录

验证日期：2026-10-17。

## 项目包生成阶段的验证（保留原记录）

- 在当前 Linux 执行环境的 Python 3.12 上，4 项 Python 测试全部通过。
- 检查了固定依赖的 Python 版本声明，均兼容目标 Python 3.11。
- 使用 PGlite 0.5.10 的 PostgreSQL 18.6 与 pgvector 0.8.7 执行实际迁移和 SQL 集成断言。
- 初次建表成功，演示数据计数为 1 门课程、2 个知识点、1 份资料、3 道题、3 个题目知识点关联、5 个片段。
- seed 重复执行不会重复新增数据。
- 1024 维查询、模型隔离、课程和知识点过滤、余弦相似度通过验证。
- 不合法维度、无模型名向量、错误难度、双重片段来源和跨课程关联被拒绝。
- anon、authenticated 的 SELECT / INSERT / UPDATE / DELETE 共 8 次直接访问均被权限拒绝。
- 单独授予 anon SELECT 后，无 RLS 策略仍使其看到 0 行。
- service_role 在服务端权限范围内可访问演示题。
- 集成测试中的合成向量已回滚；演示数据保持 0 个已生成向量。

以上为项目包原有的生成环境验证记录；下面记录实际 Windows 本机与云端验收结果。

## Windows 本机与 Supabase 实际验收

- 项目根目录为 `D:\2026-2027telecom_sudparis\Edu_RAG\Edu_RAG_V0`。
- Conda 环境解释器为 `D:\Anaconda\envs\edu-rag\python.exe`，Python 3.11.17。
- 安装模型依赖后重新执行 `pip check`，结果为 `No broken requirements found`。
- 安装模型依赖后重新执行 Python 测试，4 项全部通过。仍有一项 Starlette TestClient / httpx 上游弃用提示，不影响测试结果。
- Supabase 项目 `Edu_RAG` 已完成 V0 schema 初始化、pgvector 扩展启用和原创演示数据导入。
- 本机只读 `db check` 返回 `status=ok`，pgvector 版本为 0.8.2，向量列为 1024 维。
- 实际数据计数：1 门课程、2 个知识点、1 份资料、3 道题、3 个题目知识点关联、5 个片段、0 个已生成向量的片段。
- 已通过实际 HTTP 请求验证 `/health`、`/health/db` 和 `/api/v0/courses` 返回 200。根路径尚无首页，返回 404 符合当前 API 路由定义。
- 已在云端查询确认 7 张 `edu_` 表全部开启 RLS；anon、authenticated 没有直接表访问权限和检索函数执行权限。
- Supabase 安全检查仅提示 7 项信息级的“RLS 已开启但无策略”。这是 V0 仅由后端访问数据库的预期配置，尚未开放前端直接访问。说明见 [Supabase RLS 检查文档](https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy)。
- GitHub 首次提交为 `b8fcef9`，本地 `main` 已跟踪 `origin/main`；`.env` 被 Git 忽略。
- 真实数据库参数与 DeepSeek 密钥仅保存在本机 `.env`；本记录不包含凭据。

## V1 模型环境预检

已安装的主要版本：

| 组件 | 版本 |
| --- | --- |
| FlagEmbedding | 1.4.2 |
| PyTorch | 2.14.1 |
| transformers | 5.20.1 |
| sentence-transformers | 6.1.0 |

- BGE-M3 模型配置、权重和 tokenizer 已存在于本机 Hugging Face 缓存。
- 在禁用 Hugging Face 网络访问的情况下，使用 CPU、FP32、batch size 1、max length 128 完成实际编码；输出形状为 `(1, 1024)`，所有数值有限，向量非零。
- 此次模型测试没有将向量写入 Supabase；云端演示片段的向量数保持为 0。
- DeepSeek 配置已能读取非空密钥、官方 API 地址和 `deepseek-flash` 模型名。尚未发起实际模型 API 请求，因此不声明密钥权限、余额和生成效果已验证。

## 下一阶段范围

V1 将实现资料入库、真实向量写入、问题检索与带引用的回答，并验证 DeepSeek API 的实际调用。当前 `requirements-lock.txt` 仍记录 V0 基础依赖；上表记录额外安装的模型环境，V1 接入时再纳入正式依赖声明和锁定文件。

