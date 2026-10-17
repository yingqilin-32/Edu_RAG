# V0 验收

- [ ] 项目文件位于 `D:\2026-2027telecom_sudparis\Edu_RAG`。
- [ ] 已激活 `edu-rag` Conda 环境，Python 为 3.11.x。
- [ ] 依赖安装完成，`python -m pytest -q` 通过。
- [ ] 已创建 / 选择独立的 Supabase Free 项目。
- [ ] 本机 `.env` 填入实际 Session pooler 参数。
- [ ] migrate 成功，并且再次执行时 applied 为空列表。
- [ ] seed 成功，并且重复执行后各表行数保持不变。
- [ ] db check 成功，六个业务表有预期数据，embedded_chunks 为 0。
- [ ] 本机 `/health/db` 返回 200，`/api/v0/courses` 显示演示课程。
- [ ] GitHub 空仓库已创建并完成首次 push，提交中不包含 `.env`。

这些方框是你实际运行后的验收记录，项目包中不预先标记为完成。

