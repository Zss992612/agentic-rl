# OpenAI Chat 轨迹到 Qwen 可训练数据的转换总结

日期：2026-08-23

本文档记录 Retail SFT 数据从模型无关的 OpenAI Chat 兼容格式，转换为 Qwen3-4B-Instruct-2507 可直接读取的 Token 级训练数据的完整过程。

本文档承接：

- `docs/data/sft_data_spec.md`
- `docs/data/retail_sft_data_construction_summary.md`

前一阶段解决了轨迹采集、质量筛选和 OpenAI Chat 结构标准化；本阶段解决目标模型适配、Token 化、Assistant-only labels、全量长度审计和最终 NumPy 数据落盘。

## 1. 阶段结论

输入数据：

```text
data/sft/retail_serial_v3/sft_train_256.jsonl
```

目标模型：

```text
Qwen3-4B-Instruct-2507
```

最终数据：

```text
data/tokenized/retail_serial_v3_qwen3_4b_instruct_2507/
├── input_ids.npy
├── labels.npy
├── offsets.npy
├── metadata.jsonl
├── stats.json
└── manifest.json
```

最终结果：

| 指标 | 数值 |
|---|---:|
| 轨迹数 | 256 |
| 唯一 Task 数 | 64 |
| 总 Token 数 | 2,637,872 |
| Assistant 监督 Token 数 | 414,294 |
| Mask Token 数 | 2,223,578 |
| 加权监督比例 | 15.71% |
| 最短轨迹 | 6,145 Token |
| 中位轨迹 | 9,907 Token |
| 平均轨迹 | 10,304 Token |
| 最长轨迹 | 16,904 Token |
| 最终最大训练长度 | 17,408 Token |
| 截断轨迹 | 0 |
| 处理失败 | 0 |

整个过程未加载 Qwen 模型权重，不需要 PyTorch 或 GPU；只使用目标 tokenizer 完成本地预处理。

## 2. 为什么 OpenAI Chat 数据不能直接训练

输入 JSONL 已经具有规范的结构：

```json
{
  "trajectory_id": "...",
  "task_id": "0",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "chatcmpl-tool-...",
          "type": "function",
          "function": {
            "name": "get_order_details",
            "arguments": {"order_id": "#W2378156"}
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "chatcmpl-tool-...",
      "content": "{...}"
    }
  ],
  "tools": ["..."],
  "metadata": {"...": "..."}
}
```

但该格式只描述逻辑消息，不等于 Qwen 实际读取和生成的文本协议。正式训练前仍需确定：

- System Prompt 和 16 个工具定义如何进入上下文。
- Assistant 工具调用如何变成 Qwen 的 `<tool_call>` 格式。
- Tool observation 如何变成 Qwen 的 `<tool_response>` 格式。
- 哪些 Token 计算 loss。
- 多轮 Assistant target 的边界在哪里。
- 轨迹长度是否超过训练上下文。
- 最终数据如何高效存储和按轨迹读取。

因此本阶段没有直接对 JSONL 调用普通 `tokenizer.encode`，而是拆成六个可独立审计的检查点。

## 3. 环境和目标 Tokenizer

预处理虚拟环境：

```text
training/.venv
```

直接依赖：

```text
transformers==4.57.6
tokenizers==0.22.2
numpy==2.5.2
```

项目依赖组：

```toml
[dependency-groups]
preprocessing = [
    "numpy>=2,<3",
    "transformers>=4.51,<5",
]
```

Tokenizer-only 资源：

```text
artifacts/tokenizers/Qwen3-4B-Instruct-2507/
```

该目录约 16MB，只包含 tokenizer、词表、模型配置和 Chat Template，不包含 4B 权重。它与完整模型目录中的 tokenizer 文件 SHA-256 完全一致。

加载方式：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "artifacts/tokenizers/Qwen3-4B-Instruct-2507",
    use_fast=True,
    local_files_only=True,
)
```

Transformers 将其加载为 `Qwen2TokenizerFast` 是正常现象，这是 tokenizer 实现类名称，不表示模型版本错误。

## 4. 完整处理流程

```text
OpenAI Chat messages + tools JSONL
        ↓
Checkpoint 1：模型无关 Schema 检查
        ↓
Checkpoint 2：Qwen 官方 Chat Template 渲染
        ↓
Checkpoint 3：单条 encode → decode 往返审计
        ↓
Checkpoint 4：Assistant-only labels 和逐 Token 审计
        ↓
Checkpoint 5：256 条全量临时扫描
        ↓
Checkpoint 6：正式 NumPy 数据集落盘与哈希验证
```

每个检查点先确认上一层数据契约，再进入下一层，避免把 Schema、模板、Token 和 loss mask 的错误混在一起。

## 5. Checkpoint 1：模型无关 Schema 检查

核心代码：

```text
src/agentic_rl/preprocessing/schema.py
tests/preprocessing/test_schema.py
```

该阶段不加载 Qwen tokenizer，不渲染文本，也不生成 Token。

主要检查：

- 顶层必须包含 `trajectory_id`、`task_id`、`messages`、`tools` 和 `metadata`。
- 轨迹 ID 非空且唯一。
- 第一条消息必须为 System，第二条必须为 User。
- 最后一条必须为 Assistant 文本回复。
- Assistant 一轮只能包含普通文本或工具调用之一。
- 当前 Serial SFT 一轮最多调用一个工具。
- Assistant 工具调用后必须紧跟对应 Tool result。
- `tool_call_id` 必须一一匹配且不能重复。
- 工具名称必须存在于当前轨迹的工具定义中。
- 工具参数必须为严格可序列化 JSON。
- 每条轨迹包含完整的 16 个 Retail 工具定义。

结果：256 条全部通过。

这个检查点只回答“数据必要字段和逻辑关系是否正确”，不进行 Qwen 格式适配。

## 6. Checkpoint 2：Qwen Chat Template 渲染

核心代码：

```text
src/agentic_rl/preprocessing/rendering.py
scripts/preprocessing/inspect_rendered_trajectory.py
tests/preprocessing/test_rendering.py
```

使用目标 tokenizer 自带的官方模板：

```python
tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=False,
)
```

`add_generation_prompt=False` 的原因是训练轨迹已经包含最终 Assistant target，不需要在尾部额外添加一个空 Assistant 开头。

### 6.1 System 和工具定义

渲染后，每条轨迹从 Qwen System 消息开始，System Prompt 和 16 个工具 Schema 均位于对话开头。

### 6.2 Assistant 工具调用

原始 OpenAI Chat：

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "function": {
        "name": "get_order_details",
        "arguments": {"order_id": "#W2378156"}
      }
    }
  ]
}
```

Qwen 渲染结果：

```text
<|im_start|>assistant
<tool_call>
{"name": "get_order_details", "arguments": {"order_id": "#W2378156"}}
</tool_call><|im_end|>
```

工具名称和参数是 Agent 模型需要生成的内容，后续参与 loss。

### 6.3 Tool response

原始逻辑角色仍然是：

```json
{"role": "tool", "content": "..."}
```

Qwen 官方模板将其渲染为：

```text
<|im_start|>user
<tool_response>
工具执行结果
</tool_response><|im_end|>
```

外层 `user` 表示这是提供给 Agent 的外部输入，`<tool_response>` 表示其真实语义为环境返回，而不是普通用户发言。

原始 `tool_call_id` 是 API 和环境用于关联调用与结果的传输层 ID，不是模型需要生成的语义内容，因此不会进入最终 Qwen 文本。

### 6.4 单条渲染审计

第一条轨迹：

```text
results/preprocessing/checkpoint2/trajectory_000_rendered.txt
```

审计结果：

- 25 条 message。
- 16 个工具定义。
- 12 条 Assistant message。
- 7 次工具调用。
- 7 条工具结果。
- 不包含随机 `tool_call_id`。

## 7. Checkpoint 3：Tokenization 往返检查

核心代码：

```text
src/agentic_rl/preprocessing/tokenization.py
scripts/preprocessing/inspect_tokenized_trajectory.py
tests/preprocessing/test_tokenization.py
```

该阶段只做：

```text
Qwen 渲染文本 → input_ids → decode → 文本比较
```

明确不做：

- Labels。
- Loss mask。
- Truncation。
- Padding。
- 模型加载。

第一条轨迹结果：

| 指标 | 数值 |
|---|---:|
| 字符数 | 34,230 |
| Token 数 | 10,185 |
| `<|im_start|>` 数量 | 25 |
| `<|im_end|>` 数量 | 25 |

执行了两类一致性检查：

1. 渲染文本 encode 后再 decode，与原文本逐字符、逐字节完全相同。
2. “先渲染再 encode”与 `apply_chat_template(tokenize=True)` 的 Token ID 完全相同。

34,230 字符变成 10,185 Token 是正常的，平均约 3.36 字符/Token。数据以英文和 JSON 为主，BPE 可以用单个 Token 表示常见单词、带空格词片和重复 JSON 片段；精确 decode 已证明没有内容丢失。

## 8. Checkpoint 4：Assistant-only Labels

核心代码：

```text
src/agentic_rl/preprocessing/masking.py
src/agentic_rl/preprocessing/audit.py
scripts/preprocessing/inspect_assistant_labels.py
tests/preprocessing/test_masking.py
```

Labels 与 input IDs 等长：

```python
labels = [-100] * len(input_ids)
```

只在 Assistant target 区间复制原 Token ID：

```python
labels[start:end] = input_ids[start:end]
```

训练规则：

| 内容 | Label | Loss |
|---|---:|---:|
| System Prompt | `-100` | 否 |
| 工具定义 | `-100` | 否 |
| User 消息 | `-100` | 否 |
| Tool response | `-100` | 否 |
| `<|im_start|>assistant\n` | `-100` | 否 |
| Assistant 普通回复 | 对应 Token ID | 是 |
| Assistant `<tool_call>`、名称和参数 | 对应 Token ID | 是 |
| Assistant `<|im_end|>` | 对应 Token ID | 是 |
| Assistant 后的消息分隔换行 | `-100` | 否 |

Assistant 末尾的 `<|im_end|>` 参与 loss，因为模型需要学习停止当前回复或工具调用。Assistant 角色头不参与 loss，因为推理时 Chat Template 会提前提供 generation prompt。

不手动移动 labels。Hugging Face causal LM 在模型内部完成 logits 和 labels 的一步错位。

### 8.1 BPE 边界合并

部分 Assistant 文本以两个换行开头：

```text
模板 Assistant 头末尾：\n     → 应该 mask
Assistant 正文开头：\n\n     → 原本属于 target
```

Qwen tokenizer 会将连续三个换行合并为一个 Token：

```text
\n\n\n → ĊĊĊ
```

同一个 Token 不能一部分设为 `-100`、另一部分计算 loss。当前使用 fast tokenizer 的字符 offset mapping 判断边界：

- 完全处于 Assistant target 内的 Token 参与 loss。
- 完全处于上下文中的 Token 被 mask。
- 同时跨越 header 和 target 的 Token 整体 mask。
- 跨界被一并 mask 的正文字符会记录在 metadata 中，不静默忽略。

这里没有删除原始文本或 input ID。合并 Token 仍参与后续上下文，只是不要求模型预测它。

第一条轨迹审计结果：

| 指标 | 数值 |
|---|---:|
| 总 Token | 10,185 |
| 监督 Token | 1,290 |
| Mask Token | 8,895 |
| 监督比例 | 12.67% |
| Assistant 普通回复 | 5 |
| Assistant 工具调用 | 7 |

审计文件：

```text
results/preprocessing/checkpoint4/trajectory_000_label_summary.json
results/preprocessing/checkpoint4/trajectory_000_assistant_targets.txt
results/preprocessing/checkpoint4/trajectory_000_token_audit.tsv
```

## 9. Checkpoint 5：256 条全量临时扫描

核心脚本：

```text
scripts/preprocessing/scan_sft_dataset.py
```

256 条轨迹全部临时经过：

```text
Schema → Render → Tokenize → Labels
```

此时只保存统计，不保存正式 Token 数组。

扫描产物：

```text
results/preprocessing/checkpoint5/
├── dataset_overview.md
├── dataset_scan_summary.json
├── trajectory_metrics.csv
└── trajectory_metrics.jsonl
```

### 9.1 Task 和消息统计

| 指标 | 数值 |
|---|---:|
| 轨迹 | 256 |
| Task | 64 |
| 每 Task 轨迹数 | 2～8 |
| Assistant 区间 | 3,344 |
| Assistant 文本区间 | 1,280 |
| Assistant 工具调用区间 | 2,064 |
| Tool response | 2,064 |

### 9.2 Token 长度

| 长度区间 | 轨迹数 |
|---|---:|
| 4,097～8,192 | 29 |
| 8,193～12,288 | 187 |
| 12,289～16,384 | 38 |
| 超过 16,384 | 2 |

两条超长轨迹分别为 16,897 和 16,904 Token，只比 16K 多约 3%，且分别包含 4,058 和 5,995 个监督 Token，属于高价值长程轨迹。

目标模型配置中的 `max_position_embeddings=262144`。项目决定保留两条轨迹，将训练 `max_seq_length` 设置为：

```text
17408
```

没有对任何轨迹执行截断。

### 9.3 全量边界统计

| 指标 | 数值 |
|---|---:|
| 出现边界合并的轨迹 | 229 |
| 合并的 Assistant 区间 | 355 |
| 被一并 mask 的正文字符 | 710 |
| 非空白字符 | 0 |

355 处边界每处都只涉及两个前导换行，没有字母、汉字、数字或标点被忽略。因此当前保守 mask 规则被接受用于最终构建。

## 10. Checkpoint 6：正式 NumPy 数据集

核心代码：

```text
src/agentic_rl/preprocessing/storage.py
scripts/preprocessing/build_tokenized_dataset.py
tests/preprocessing/test_storage.py
```

构建脚本先读取 Checkpoint 5 的数据契约并验证原始 JSONL SHA-256。如果源文件在扫描后发生变化，正式构建会拒绝继续，要求重新扫描。

构建采用以下方式：

1. 根据全量扫描结果预分配 NumPy memory map。
2. 按原始 JSONL 顺序逐条重新渲染、Tokenize 和构造 labels。
3. 写入临时同文件系统目录。
4. 验证数组、metadata、offsets 和文件 hash。
5. 全部通过后原子发布为最终目录。
6. 目标目录已存在时拒绝覆盖，避免误删已有训练数据。

### 10.1 数组格式

`input_ids.npy`：

```text
dtype: int32
shape: [2637872]
```

`labels.npy`：

```text
dtype: int32
shape: [2637872]
ignore_index: -100
```

`offsets.npy`：

```text
dtype: int64
shape: [257]
```

没有保存单独的 `loss_mask.npy`，因为：

```python
loss_mask = labels != -100
```

没有保存 attention mask，因为当前文件没有 padding。训练时动态 collator 根据 batch 中的实际长度生成 attention mask。

### 10.2 Offsets 语义

所有轨迹按源 JSONL 顺序拼接到一维数组，offsets 保存累计边界：

```python
start = offsets[i]
end = offsets[i + 1]

trajectory_input_ids = input_ids[start:end]
trajectory_labels = labels[start:end]
```

第一条轨迹：

```text
offsets[0] = 0
offsets[1] = 10185
```

最后一项：

```text
offsets[-1] = 2637872
```

### 10.3 Metadata

`metadata.jsonl` 每行对应一条轨迹，记录：

- Dataset index。
- Trajectory ID 和 Task ID。
- 全局 offset 起止位置。
- Token、监督 Token 和 mask Token 数量。
- Assistant 文本和工具调用的局部 Token 区间。
- 边界合并时被 mask 的前导文本。
- 工具数量、Tool result 数量。
- 原始采样档位、temperature、seed、质量得分和选择来源。

### 10.4 Manifest 和完整性

`manifest.json` 记录：

- 数据格式名称与版本。
- 原始 JSONL 和 Checkpoint 5 summary 的 SHA-256。
- Tokenizer、Chat Template 和配置文件 SHA-256。
- 数组 dtype、shape 和 ignore index。
- 最大训练长度和无截断契约。
- Python、NumPy、Transformers 和 Tokenizers 版本。
- 最终各文件大小与 SHA-256。

Manifest 中的项目资源路径相对 `training/` 保存，便于整个项目同步到服务器后继续使用。

最终独立复读验证：

- 三个数组均可通过 `mmap_mode="r"` 打开。
- `input_ids` 和 `labels` shape 完全相同。
- Offsets 从 0 开始、严格递增，并以总 Token 数结束。
- 每个 label 只能是 `-100` 或同位置的 input ID。
- Metadata 行数与轨迹数、offsets 完全一致。
- 第一条最终 input IDs decode 后与 Checkpoint 2 渲染文本完全一致。
- 所有 manifest 文件 SHA-256 通过。
- 没有残留临时构建目录。

## 11. 训练侧读取方式

当前数据可以使用 NumPy memory map 按需读取，不需要一次性加载全部数据：

```python
import numpy as np

data_dir = "data/tokenized/retail_serial_v3_qwen3_4b_instruct_2507"

input_ids = np.load(f"{data_dir}/input_ids.npy", mmap_mode="r")
labels = np.load(f"{data_dir}/labels.npy", mmap_mode="r")
offsets = np.load(f"{data_dir}/offsets.npy", mmap_mode="r")

index = 0
start = offsets[index]
end = offsets[index + 1]

sample_input_ids = input_ids[start:end]
sample_labels = labels[start:end]
```

训练 Dataset 将 NumPy `int32` 切片转换成 PyTorch `long`。动态 collator 需要：

- `input_ids` 使用 Qwen `pad_token_id` 右侧 padding。
- `labels` 使用 `-100` padding。
- 根据真实长度生成 `attention_mask`。
- 不修改原始 labels，不手动 causal shift。
- 使用长度分桶减少长短轨迹混合造成的 padding 浪费。

## 12. 复现命令

在项目根目录执行：

```bash
cd "/path/to/agentic_rl"
```

同步本地预处理环境：

```bash
uv sync --project training --group preprocessing --group dev
```

检查一条 Qwen 渲染文本：

```bash
training/.venv/bin/python \
  training/scripts/preprocessing/inspect_rendered_trajectory.py \
  --index 0
```

检查单条 Tokenization 往返：

```bash
training/.venv/bin/python \
  training/scripts/preprocessing/inspect_tokenized_trajectory.py \
  --index 0
```

检查单条 Assistant-only labels：

```bash
training/.venv/bin/python \
  training/scripts/preprocessing/inspect_assistant_labels.py \
  --index 0
```

扫描全部 256 条：

```bash
training/.venv/bin/python \
  training/scripts/preprocessing/scan_sft_dataset.py
```

正式构建最终数据：

```bash
training/.venv/bin/python \
  training/scripts/preprocessing/build_tokenized_dataset.py
```

构建脚本默认拒绝覆盖已有目标目录。需要验证新的构建逻辑时，应指定新的输出目录，而不是直接覆盖当前正式数据：

```bash
training/.venv/bin/python \
  training/scripts/preprocessing/build_tokenized_dataset.py \
  --output-dir training/data/tokenized/retail_serial_v3_rebuild_check
```

运行测试：

```bash
training/.venv/bin/pytest -q training/tests
```

当前结果：

```text
25 passed
```

## 13. 当前边界和下一步

本阶段已经完成：

- OpenAI Chat 结构验证。
- Qwen3-4B-Instruct-2507 Chat Template 适配。
- 无损 Tokenization。
- Assistant-only labels。
- 256 条长度与边界全量审计。
- 最终 memory-mappable NumPy 数据集。
- Manifest、hash 和独立复读验证。

本阶段没有完成、需要进入训练框架阶段的内容：

- PyTorch NumPy Dataset。
- 动态 padding collator。
- 长度分桶 Sampler。
- Qwen3-4B-Instruct-2507 LoRA/QLoRA 模型加载。
- SFT 训练循环、梯度累积和 checkpoint 保存。
- W&B 训练监控。
- Validation 在线任务评测与 checkpoint 选择。
- 最长 16,904 Token 样本的显存冒烟测试。

因此，当前状态是“训练数据已经确定并可读取”，下一阶段是“训练框架实现和单 batch 显存验证”。
