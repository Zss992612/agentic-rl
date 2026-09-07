# Retail SFT 数据构建阶段总结

> 更新：目标模型适配、Assistant-only labels 和最终 Token 数据已于 2026-08-23 完成，详见 `docs/data/openai_chat_to_qwen_tokenized_dataset.md`。本文档保留为前一阶段的轨迹采集与筛选记录。

本文档总结项目从理解官方 τ²-bench Retail 环境，到构建出 256 条正式 SFT 轨迹为止的阶段性工作。重点记录实际执行结果、过程中遇到的问题、已经采用的解决方案，以及进入 SFT 训练前仍需完成的事项。

## 1. 本阶段目标

本阶段不是立即训练模型，而是先解决以下问题：

- 理解 Retail 任务、Agent Loop、工具环境和 Verifier 的工作方式。
- 固定 Train、Validation 和官方 Test 的任务边界，避免数据泄漏。
- 从真实多轮用户交互中采集候选轨迹。
- 不只依赖官方 reward，对轨迹进行过程和语义质量检查。
- 从候选数据中选择质量较高且具有一定多样性的轨迹。
- 转换为后续可适配具体 7B 模型的标准 SFT 消息格式。

本阶段最终产物为 256 条覆盖 64 个 Retail 训练任务的 SFT 数据。

## 2. 数据边界

官方 Retail 数据包含：

- Train：74 个任务。
- Test：40 个任务。

本项目将官方 74 个 Train 任务进一步固定划分为：

- 项目 Train：64 个任务，用于 SFT 数据采集和后续 GRPO rollout。
- 项目 Validation：10 个任务，只用于 checkpoint 选择和超参数比较。
- 官方 Test：40 个任务，只用于最终 Base、SFT 和 GRPO 对比。

Validation 和官方 Test 任务均未进入本次 SFT 数据。具体 Task ID 和分层依据记录在 `docs/data/sft_data_spec.md`。

## 3. 数据构建流程

整体数据流如下：

```text
τ²-bench 原始 results.json
        ↓
抽取并标准化 512 条候选轨迹
        ↓
第一阶段：硬性检查与任务内排序
        ↓
451 条 hard-pass 候选
        ↓
第二阶段：Teacher 语义检查
        ↓
435 条 accepted 候选
        ↓
质量、任务覆盖和多样性选择
        ↓
256 条最终候选轨迹
        ↓
标准 SFT messages + tools 格式
```

### 3.1 正式轨迹采样

采样对象为 64 个 Train 任务，每个任务采样 8 次：

| 档位 | Agent temperature | 每任务轨迹数 | 总轨迹数 |
|---|---:|---:|---:|
| low | 0.2 | 2 | 128 |
| medium | 0.6 | 4 | 256 |
| high | 0.9 | 2 | 128 |
| 合计 | — | 8 | 512 |

正式采样配置包括：

- Agent：`openai/deepseek-v4-flash`。
- User Simulator：`openai/deepseek-v4-flash`。
- User temperature：0.0，优先保持隐藏任务执行稳定。
- `parallel_tool_calls: false`，限制 Agent 单轮并行工具调用。
- `max_concurrency: 4`，最多并发运行 4 条 simulation。
- `auto_resume: true`，支持按 τ²-bench 已保存结果继续采样。

三档 temperature 只用于增加 Agent 行为和表达的候选多样性，不表示业务决策可以偏离 Retail Policy。

### 3.2 原始结果抽取

τ²-bench 的单条原始结果包含运行配置、完整任务、环境状态、评测详情和消息轨迹，体积较大，不适合直接送入筛选或训练。

抽取阶段保留：

- 轨迹、任务和采样标识。
- User、Assistant 和 Tool 消息。
- 工具名称、参数和工具返回。
- 官方 reward、termination reason 和评测信息。
- 轨迹长度、工具调用数量、错误数量等统计字段。

Agent 在工具调用时附带的非空 `content` 不会传给 User。正式抽取时将这类工具调用消息的 `content` 统一设为 `null`，保留工具调用本身，避免学生模型学习“文本和工具调用同时输出”的形式。

### 3.3 第一阶段：硬性检查与任务内排序

第一阶段使用确定性的本地规则处理 512 条候选，不调用外部 Teacher。

主要检查：

- simulation 是否正常结束。
- reward 和数据库检查是否通过。
- 消息及工具调用结构是否完整。
- 是否存在并行工具调用。
- 是否存在需要进一步检查的工具错误。
- 轨迹是否过长或工具调用过多。

实际结果：

- 候选总数：512。
- hard-pass：451。
- hard-rejected：61。
- 61 条拒绝轨迹均出现 `reward=0` 和 `db_mismatch`。
- 52 条 hard-pass 轨迹带有“工具错误需要检查”标记。
- 38 条带有“长轨迹”标记。
- 3 条带有“工具调用过多”标记。

任务内排序的目标不是直接决定最终数据，而是优先排列成功、较短、工具错误较少的候选，为后续语义检查和抽样审计提供顺序。

### 3.4 第二阶段：Teacher 语义检查

第二阶段将任务要求、Retail Policy、精简后的消息轨迹和客观统计发送给 Teacher，由 Teacher 判断：

- 用户模拟器是否保留了隐藏任务中的关键要求。
- Agent 的身份认证、确认和工具使用过程是否合理。
- 工具错误是否得到恢复。
- 最终回复是否与实际工具执行结果一致。

本地代码再根据 Teacher 返回的结构化字段生成 `accepted` 或 `rejected` decision。

实际结果：

- 输入：451 条 hard-pass 候选。
- accepted：435 条。
- rejected：16 条。
- Teacher 调用错误：0 条。

被拒绝轨迹中发现了提前转人工、任务未完成、付款方式未解决、缺少地址修改、未获得确认、最终总结不准确等问题。这证明 `reward=1` 不能单独作为 SFT 正样本标准。

### 3.5 最终多样性选择

从 435 条 accepted 候选中选择 256 条，不再强制每个任务固定保留 4 条，而是同时考虑：

- 每个任务至少保留 2 条，保证 64 个任务全部覆盖。
- Teacher 置信度和本地质量分。
- 工具错误、超长轨迹和工具调用过多等惩罚项。
- User/Assistant 文本 3-gram 差异。
- 工具名称、参数和调用顺序差异。
- low、medium、high 三档的总体分布。

实际结果：

- 最终轨迹：256 条。
- 任务覆盖：64/64。
- 每任务轨迹数：2～8 条。
- low：62 条。
- medium：123 条。
- high：71 条。
- 为任务覆盖选择：128 条。
- 为全局质量和多样性选择：128 条。
- 未选中的 accepted 候选：179 条，可作为后续扩充储备。

当前选择保留了 12 条带工具错误检查标记的轨迹和 2 条长轨迹。这些轨迹通过了语义检查，但仍属于训练前需要重点抽查的数据。

### 3.6 SFT 格式转换

最终每条 JSONL 数据包含：

- `trajectory_id`：轨迹唯一标识。
- `task_id`：Retail Task ID。
- `messages`：System、User、Assistant 和 Tool 多轮消息。
- `tools`：16 个 Retail 工具 Schema。
- `metadata`：采样档位、temperature、seed、来源和选择得分。

转换时执行了以下处理：

- 加入 τ²-bench 官方 Retail System Prompt 和 Policy，不自行编写业务规则。
- 加入 16 个工具的完整 Schema。
- 删除 τ²-bench 自动注入、并非 Teacher 生成的固定开场白。
- 删除末尾没有后续 Assistant target 的用户感谢、`###STOP###` 或 `###TRANSFER###`。
- 将 Assistant 工具调用转换为 `tool_calls.function` 结构。
- 保证 Assistant 工具调用与 Tool observation 通过 ID 一一配对。
- 不加入隐藏任务描述、Verifier 参考动作和数据库断言，避免答案泄漏。

结构检查结果：

- 256 条 JSONL，轨迹 ID 均唯一。
- 64 个训练任务全部覆盖。
- 每条均以 System 开始，随后为初始 User 消息，并以 Assistant 消息结束。
- 每条包含 16 个工具 Schema。
- 2064 次 Assistant 工具调用与 2064 条 Tool 返回全部配对。
- 不存在 Assistant 同时包含文本和工具调用的消息。
- 不存在残留的 `###STOP###`。

## 4. 过程中遇到的主要问题及解决方案

### 4.1 用户模拟器没有稳定执行隐藏任务

现象包括：

- 过晚改变需求，导致 Agent 已经完成不可逆操作。
- 在最终写工具执行前提前输出 `###STOP###`。
- 遗漏隐藏任务中的部分商品、地址修改或条件要求。

影响：合理 Agent 可能获得 `reward=0`，形成假阴性；错误轨迹也可能被误认为 Agent 能力不足。

已采用方案：

- 不只看 reward，增加用户意图保持和终止时机检查。
- SFT 中排除用户模拟器明显失真的轨迹。
- 保留原始官方指标，同时将用户模拟器问题作为独立诊断标签。

### 4.2 官方 Verifier 对过程错误覆盖不足

Verifier 主要关注最终数据库状态。并行工具调用、确认流程错误或回复不一致不一定被 reward 捕获，因此可能出现：

- Agent 过程合理，但受用户模拟器影响得到 0 分。
- Agent 过程违规，但最终数据库碰巧正确得到 1 分。

已采用方案：

- MVP 阶段不修改官方 reward，保证 benchmark 可比性。
- 在 SFT 数据侧增加硬性过程检查和 Teacher 语义检查。
- 为后续 GRPO 保留过程合规率、异常终止率和失败归因等诊断指标。

### 4.3 初次采样出现并行工具调用

早期轨迹中出现同一 Assistant turn 调用多个工具的情况，不符合 Retail Policy，也不适合作为当前单工具 Agent 范式的监督数据。

已采用方案：

- 在 Agent API 参数中设置 `parallel_tool_calls: false`。
- 先进行小规模重新采样验证。
- 验证通过后重新采集完整 512 条数据。
- 在第一阶段筛选中继续保留并行调用检查，防止只依赖 API 参数。

正式 512 条数据未发现并行工具调用。

### 4.4 工具调用同时附带内部文本

部分模型在发出工具调用时，同时返回类似思考或过渡语的 `content`。τ²-bench 此时把消息路由给环境执行工具，该内容不会传递给 User。

已采用方案：

- 不重写工具调用及其顺序。
- 抽取时只将工具调用消息的 `content` 标准化为 `null`。
- 最终数据中保证 Assistant 一轮只包含普通回复或工具调用之一。

### 4.5 只按 reward 筛选会保留错误数据

语义检查从 451 条 hard-pass 轨迹中进一步拒绝了 16 条，说明数据库成功不等于完整过程正确。

已采用方案：

- 将 `reward=1` 作为候选门槛，而不是最终录用条件。
- 使用“本地硬检查 + Teacher 语义检查 + 多样性选择”的三层流程。
- 保存 rejected reason 和 evidence，便于回溯筛选依据。

### 4.6 同一任务的不同轨迹前缀重复

由于同一个 Task 的用户目标固定，且 User temperature 为 0，同一任务常出现相似的初始请求、身份认证和查询过程。

当前判断：一定程度的重复是长链路任务的正常现象，但完全重复会降低有效数据量。

已采用方案：

- 最终选择时同时比较对话文本和完整工具轨迹。
- 不要求每个任务机械保留 4 条。
- 将其余 179 条 accepted 数据作为储备，而不是全部加入训练。

### 4.7 原始结果体积大且不适合直接训练

单个 τ²-bench `results.json` 同时包含环境、配置、任务定义、评测和轨迹信息，阅读和复用成本较高。

已采用方案：

- 将采集、抽取、硬筛选、语义筛选、最终选择和格式转换拆为独立阶段。
- 在 `src/agentic_rl/data` 中保存可复用逻辑。
- 在 `scripts/data` 中只保留短入口脚本。
- 每个阶段输出 JSONL 数据和对应 summary，保留可审计的中间产物。

## 5. 尚未解决或需要继续验证的问题

### 5.1 目标 7B 模型尚未确定

当前数据是模型无关的 messages + tools 中间格式。不同模型对工具调用格式和 Chat Template 的要求不同，因此尚未生成最终 token ID。

后续需要：

- 确定 Base 或 Instruct checkpoint。
- 使用目标 tokenizer 的 `apply_chat_template` 渲染消息和工具。
- 检查工具调用 JSON 在目标模板中的实际表现。

### 5.2 Assistant-only loss mask 尚未实现和验证

当前只记录了训练意图：System、User 和 Tool 作为上下文，Assistant 普通回复和工具调用参与 loss。

后续必须通过 token 级检查确认：

- Assistant 文本被正确监督。
- Assistant 工具调用被正确监督。
- System、User 和 Tool observation 的 label 全部为 `-100`。
- 多轮 Assistant 边界没有发生错位。

### 5.3 精确 token 长度尚未统计

当前每条数据包含约 7,056 字符的 System Prompt 和约 12,311 字符的工具 Schema。全部内容的字符数约为：

- 最短：23,754。
- 中位数：32,992。
- 最大：54,953。

根据英文和 JSON 的常见比例，最长轨迹可能接近或超过 16K token，但必须使用目标模型 tokenizer 才能确认。

后续需要决定：

- `max_seq_length`。
- 超长轨迹的截断、排除或压缩策略。
- 是否使用 length bucket、packing、Flash Attention 和 gradient checkpointing。

### 5.4 12 条工具错误轨迹仍需人工抽查

这 12 条轨迹通过了 Teacher 语义检查，可能属于“工具报错后正确恢复”，因此具有训练价值；也可能包含 Teacher 未识别的异常。

在正式训练前应逐条检查：

- 工具错误是否由合理探索造成。
- Agent 是否正确读取并恢复。
- 是否向用户声称了未完成的操作。
- 是否值得保留为错误恢复示例。

### 5.5 Teacher 判断仍存在模型误差

当前 Teacher 对 451 条轨迹全部成功返回结构化判断，但 Teacher 本身不是绝对真值。自由文本 reason 的命名也不完全统一。

后续应：

- 从 accepted 和 rejected 中分别随机抽样人工复核。
- 统计人工判断与 Teacher decision 的一致率。
- 固定 prompt version 和 judge model，保证实验可复现。

### 5.6 数据重复和长度权重仍需训练后验证

即使 User 和 Tool token 不参与 loss，它们仍参与前向计算；长轨迹也可能因为包含更多 Assistant token 而对训练产生更大权重。

后续需要结合 SFT 结果观察：

- Train loss 和 Validation 任务成功率是否同步改善。
- 模型是否只模仿固定认证和确认模板。
- 长任务、短任务和不同业务类型的提升是否均衡。
- 256 条数据是否足够；只有在 loss 和验证指标仍支持时才考虑加入 179 条储备轨迹。

### 5.7 SFT 和 GRPO 训练闭环尚未开始

当前完成的是数据构建，不代表已经完成 SFT。后续仍需实现：

- LoRA SFT 配置和训练。
- Base/SFT 的统一 Retail Validation 与 Test 评测。
- checkpoint 选择。
- veRL/vLLM 多轮在线 rollout 接入。
- GRPO 的 token mask、reward 回传和训练闭环。
- 用户模拟器噪声、Verifier 假阳性和假阴性对 GRPO 的影响诊断。

## 6. 当前主要产物

| 产物 | 作用 |
|---|---|
| `data/sft/retail_serial_v3/candidates.jsonl` | 512 条抽取后的候选轨迹 |
| `data/sft/retail_serial_v3/ranked_candidates.jsonl` | 第一阶段硬检查和任务内排序结果 |
| `data/sft/retail_serial_v3/semantic_screened_candidates.jsonl` | Teacher 语义检查结果 |
| `data/sft/retail_serial_v3/selected_candidates_256.jsonl` | 最终选择的 256 条候选 |
| `data/sft/retail_serial_v3/sft_train_256.jsonl` | 标准 SFT messages + tools 数据 |
| `data/sft/retail_serial_v3/manifest.json` | 数据来源、采样配置、System Prompt 和工具 Schema |
| `data/sft/retail_serial_v3/*_summary.json` | 各阶段统计与审计信息 |

相关实现：

- `src/agentic_rl/data/collection.py`：正式轨迹采集逻辑。
- `src/agentic_rl/data/extraction.py`：原始结果抽取和标准化。
- `src/agentic_rl/data/screening.py`：第一阶段硬检查与排序。
- `src/agentic_rl/data/semantic_screening.py`：第二阶段 Teacher 检查。
- `src/agentic_rl/data/selection.py`：质量、覆盖和多样性选择。
- `src/agentic_rl/data/sft_format.py`：最终 SFT 格式转换与结构验证。

## 7. 阶段性结论

本阶段完成的核心工作不是“从 reward=1 的轨迹中随便选 256 条”，而是建立了一条可回溯的数据构建链路：

```text
真实环境采样
→ 原始数据抽取
→ 结果硬检查
→ 过程语义检查
→ 质量与多样性选择
→ 训练格式转换
→ 结构一致性验证
```

当前 256 条数据已经具备进入“目标模型适配与 SFT 训练准备”阶段的基础，但在确定 7B 模型、验证 Chat Template、实现 assistant-only loss mask 和处理超长序列之前，不能直接宣称数据已经完全训练就绪。
