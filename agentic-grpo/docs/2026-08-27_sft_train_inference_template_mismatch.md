# Qwen3-4B SFT 训推 Chat Template 不一致复盘

日期：2026-08-27

## 1. 问题结论

本次普通版 `Qwen/Qwen3-4B` 在 SFT 后出现的工具调用能力异常，根因不是
256 条轨迹丢失、OpenAI 消息格式错误或 `labels.npy` 漏掉工具调用，而是：

> SFT 数据中的工具调用目标直接从 `<|im_start|>assistant\n` 后开始；在线推理使用
> 官方模板并关闭 thinking 时，却会先插入空的 `<think>\n\n</think>`，再让模型续写。

因此，模型训练时学习到的工具调用起始上下文，与推理时实际收到的起始上下文不一致。
这属于 token 协议层面的训练/推理不一致，而不是一般意义上的数据量不足。

## 2. 现象

使用官方 Qwen3-4B Chat Template 启动 vLLM 并加载 SFT step96 LoRA 后：

- 模型经常不发起本应执行的工具调用；
- 多轮对话中出现重复回复和循环；
- 出现类似“无法按商品名或购买日期查询订单，只能提供订单 ID”的错误拒绝；
- 初期评测平均 reward 一度只有约 `0.06`，与训练 loss 的正常下降明显矛盾；
- 同一批 SFT 数据在 Instruct 模型上表现正常，因此问题不像是轨迹整体失效。

改用不插入空 `<think>` 的模板后，前 10 个任务平均 reward 很快恢复到约 `0.7`；
最终 test 评测为 40 tasks × 4 trials，成功 `111/160`，Pass^1 / 平均 reward 为
`69.375%`。这一对照实验直接支持“官方推理模板前缀导致协议错位”的判断。

## 3. 官方模板的实际行为

官方普通版 Qwen3-4B 模板有两套相关逻辑。

### 3.1 渲染已有完整轨迹

模板会先找到最后一条真实 `user` 消息；工具返回虽然最终会被渲染成
`<|im_start|>user ... <tool_response>`，但源消息的角色仍是 `tool`，不属于真实用户请求。

对最后一条真实用户消息之后的 assistant 消息：

- 非最后一条 assistant 且没有 `reasoning_content`：直接渲染其文本或工具调用；
- 最后一条 assistant：模板会渲染一对空的 `<think>...</think>`，然后再渲染正文。

当前数据流水线对完整轨迹调用：

```python
tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=False,
)
```

因此一条长轨迹中的中间工具调用通常被训练为：

```text
<|im_start|>assistant
<tool_call>
{"name": "...", "arguments": {...}}
</tool_call><|im_end|>
```

### 3.2 在线生成下一条 assistant 消息

在线推理需要 `add_generation_prompt=True`。当同时设置
`enable_thinking=False` 时，官方模板会在生成位置预先写入：

```text
<|im_start|>assistant
<think>

</think>

```

然后模型才从这里继续生成。也就是说，推理时模型需要学习：

```text
空 think 标签 -> <tool_call>
```

但现有 SFT 的大多数中间工具调用监督目标是：

```text
assistant header -> <tool_call>
```

模型没有从本次 SFT 中充分学到前一种转移，工具调用能力因而明显退化。

## 4. 为什么不是 NPY 数据集做错了

数据流水线保留的是标准结构化消息：

- 用户请求：`role=user`；
- assistant 工具调用：`role=assistant` + `tool_calls`；
- 工具结果：`role=tool`；
- assistant 最终文本：`role=assistant` + `content`。

流水线随后使用目标 tokenizer 的 Chat Template 渲染整条轨迹，并构造
assistant-only labels。现有审计会检查：

- 源轨迹中的 assistant 数量与渲染后的 assistant span 数量一致；
- 每个结构化工具调用均渲染为 `<tool_call>`；
- 工具调用 payload 和 assistant 的 `<|im_end|>` 均进入监督目标；
- 非 assistant 上下文使用 `-100` mask；
- `input_ids.npy` 与 `labels.npy` 长度及 token ID 一致。

最终数据包含 256 条轨迹、2,638,896 个 token、415,318 个监督 token，工具调用也没有
因为“在 OpenAI 格式中放在 user 里”而被遗漏。真正的问题发生在同一个模板的
“完整轨迹渲染”与“下一轮在线生成”两个入口之间。

## 5. 当前采用的解决方案

当前选择在推理侧使用 no-think 模板：

```text
training/artifacts/chat_templates/qwen3_4b_no_think.jinja
```

该模板只移除 reasoning / 空 `<think>` 的插入逻辑，保留以下协议：

- system 与 tools 定义；
- user / assistant 角色边界；
- `<tool_call>` 的 JSON 格式；
- `<tool_response>` 的组织方式；
- `<|im_start|>` 与 `<|im_end|>`；
- `add_generation_prompt` 产生的 assistant header。

vLLM 启动时显式加载该模板，使生成位置变为：

```text
<|im_start|>assistant
```

这与现有 SFT 工具调用目标的起始上下文一致。vLLM 提示自定义模板与官方模板不同是
预期警告，不代表加载失败。

## 6. 备选方案与取舍

另一种方案是保留官方 non-thinking 推理模板，重新生成训练数据，让每一个可能生成的
assistant turn（包括工具调用）都在空 `<think>` 后开始，然后重新训练。该方案也能统一
协议，但需要重新制作数据和训练，且会把无实际内容的标签永久写进数据。

本阶段选择 no-think 推理模板，因为它改动更小、无需重训，并且最终评测已经验证恢复
有效。它不会移除 `<|im_end|>`，因此不会让模型丧失 assistant turn 的结束能力；整条任务
何时结束仍由 agent/environment 的停止条件与用户模拟器共同决定。

## 7. 防止再次发生

进入 GRPO 前应将 Chat Template 视为模型协议的一部分，而不是可随意替换的展示格式：

1. 固化 base model、tokenizer、Chat Template 与 LoRA 的版本和 SHA-256。
2. 训练数据元信息继续保存 `chat_template_sha256`。
3. vLLM/SGLang 启动时显式指定模板，不依赖运行时默认模板。
4. 增加训推契约测试：对同一条对话前缀，比较训练目标的下一段 token 与在线生成前缀。
5. 分别覆盖“直接回复”“首次工具调用”“工具返回后的再次调用”和“最终回复”。
6. 在正式批量评测或 rollout 前，先用固定样本检查渲染文本和首个生成 token。
7. 模板不一致应归为基础设施/协议错误，不能作为 reward 0 混入强化学习数据。

## 8. 相关文件与结果

- no-think 模板：`training/artifacts/chat_templates/qwen3_4b_no_think.jinja`
- 普通版 SFT 最终结果：`training/data/eval/results_sft_test.json`
- SFT 阶段交接：`training/docs/2026-08-27_qwen3_4b_sft_handoff.md`
- Instruct step96 评测：`training/docs/2026-08-25_sft_step96_test_evaluation.md`

