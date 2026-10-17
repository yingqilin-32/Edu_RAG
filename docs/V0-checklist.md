# V0 验收

- [x] 项目根目录为 `D:\2026-2027telecom_sudparis\Edu_RAG\Edu_RAG_V0`。
- [x] 已创建 `edu-rag` Conda 环境，环境内 Python 为 3.11.17；可通过解释器完整路径运行，无需依赖 PowerShell 激活脚本。
- [x] 依赖安装完成，`pip check` 无冲突，4 项 Python 测试通过。
- [x] 已创建独立的 Supabase 项目 `Edu_RAG`。
- [x] 本机 `.env` 填入实际数据库连接参数。
- [x] 云端已应用 V0 schema，`edu_schema_migrations` 中存在版本 `20261017000100`。
- [x] 原创演示数据已导入，各表行数符合预期；seed 幂等性另见生成阶段的验证记录。
- [x] 本机 `db check` 返回 ok，六个业务表有预期数据，embedded_chunks 为 0。
- [x] 已实测本机 `/health/db` 返回 200，`/api/v0/courses` 显示演示课程。
- [x] 已完成 GitHub 首次 push，本地 `main` 跟踪 `origin/main`，`.env` 被 Git 忽略。

验收日期：2026-10-17。具体证据和验证范围见 [verification.md](verification.md)。

## V1 模型环境准备

- [x] DeepSeek 密钥已写入本机 `.env`，配置读取使用 `SecretStr` 隐藏密钥。
- [x] FlagEmbedding 和 PyTorch 已安装，BGE-M3 权重已缓存。
- [x] CPU 离线编码验证通过，生成 `(1, 1024)` 的有限、非零向量。

DeepSeek API 实际请求、资料向量入库、检索和带引用的回答留到 V1 实现与验证。

