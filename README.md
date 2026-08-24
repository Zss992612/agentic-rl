# Agentic RL on τ-bench

本工作区用于实现面向 7B 模型的 Retail 多轮工具 Agent 训练闭环，并在后续进行
Retail → Airline 零样本工具泛化实验。

## 当前阶段：Phase 0（官方环境验证）

当前不进行大规模训练。Phase 0 的退出条件是：

1. 固定并记录官方 benchmark 版本。
2. 跑通 Retail 数据、工具调用、状态重放和 Verifier。
3. 用真实 API 用户模拟器完成至少一条端到端多轮轨迹。
4. 明确 rollout 进程隔离、奖励语义和后续训练适配边界。

官方代码位于 `tau2-bench/`，当前版本为 `tau2==1.0.1`，本地提交为
`363133a`。官方仓库现在以 τ³-bench 名义继续维护；Retail/Airline 文本环境仍由
同一仓库提供。

## 已验证

- `tau2 check-data` 通过，数据目录可用。
- Retail 工具测试 15/15 通过。
- Gym 测试 26/29 通过；3 个失败均由缺少 `OPENAI_API_KEY` 导致默认 API 用户
  模拟器在首次响应时退出，不是工具或 Gym 依赖错误。
- Retail 任务 0 已完成真实工具执行、状态重放和官方 Verifier 正反例烟测。

运行离线 Verifier 烟测：

```bash
cd "/Users/projects/agentic rl"
UV_CACHE_DIR=/private/tmp/tau2-uv-cache \
  tau2-bench/.venv/bin/python phase0_retail_verifier_smoke.py
```

预期核心结果：

```text
reference:  env_reward=1, action_diagnostic=1
write_only: env_reward=1, action_diagnostic=0
reads_only: env_reward=0, action_diagnostic=0
```

这个结果说明 Retail 的主奖励按 `reward_basis` 聚合，核心是数据库终态；任务中的
`actions` 是一条参考解轨迹，并非通常意义上的逐步硬约束。后续 SFT 可以使用参考轨迹，
但 GRPO 奖励不能把“完全复现参考动作序列”误当作任务成功。

## 官方数据初步画像

- Retail：114 个任务，train/test/base = 74/40/114，16 个 Agent 工具。
- Airline：50 个任务，train/test/base = 30/20/50，14 个 Agent 工具。
- 两域工具名精确重合仅 3 个：`calculate`、`get_user_details`、
  `transfer_to_human_agents`。
- Retail 有 40 个任务包含非空 NL assertions；其默认裁判模型是
  `gpt-4.1-2025-04-14`。因此完整官方奖励除用户模拟 API 外，还可能包含裁判 API
  成本与方差，训练时必须把确定性 DB 奖励和 LLM 判分分开记录。

## 当前阻塞与下一步

当前唯一阻塞真实多轮在线烟测的是 API 凭据。不要把密钥写入代码或提交到 Git；在
`tau2-bench/.env` 中本地配置所选 provider 的密钥。凭据就绪后，先用单任务、
`max_concurrency=1`、固定 seed 运行 Retail，再扩到小批量并发。

旧离线参考项目中的算法、mask 和测试可以作为设计素材，但它的
`RealTau2Env.reset/step` 仍是 `NotImplementedError`，且其 reward/task schema 与当前
官方 v1.0.1 已有差异，不能直接作为在线适配层。

