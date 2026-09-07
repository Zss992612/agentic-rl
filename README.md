# ToolChain-RL

**长链路多工具智能体后训练系统**

面向 τ-bench Retail 场景中需要多轮用户交互与连续工具调用的任务，基于 **Qwen3-4B** 构建轨迹采集、监督微调（SFT）、强化学习（GRPO）与评测分析流程，探索长链路任务中的细粒度信用分配。

这是一个基于现有框架与任务环境开展的个人实践项目，重点是多轮训练适配、单卡训练优化及过程信号驱动的优势重分配。

## 核心工作

- **轨迹数据与 SFT**：使用 DeepSeek Flash 模拟用户与教师，在环境中执行工具并采集 512 条轨迹；经终局校验、语义审查及覆盖与多样性筛选，保留覆盖 64 个任务的 256 条样本。采用 LoRA 微调，并修复多轮 Chat Template 的训练／推理前缀错配。
- **多轮训练与显存优化**：基于 verl / vLLM 接入环境交互与工具调用，采用每批 4 个任务 × 每任务 8 条轨迹的采样配置；结合 Assistant-only 词表投影、分块计算、remove-padding 与动态 token 预算微批处理，减少冗余计算并控制显存开销。
- **细粒度信用分配**：构建数据库写入进度与查询事实覆盖的势函数，提取轮次级过程信号；通过有界长度补偿与加权中心化重分配成功轨迹优势，保持优势符号与轨迹内 token 加权均值，失败轨迹保留原始 GRPO 优势。该方案不需要额外训练 PRM 或 Critic。
- **评测与错误分析**：实现多次采样、checkpoint 评测及结果汇总，分析任务成功率、重复执行稳定性、工具错误与模型间成功／失败翻转样例。

## 实验结果

以下为同批评测结果：**40 个测试任务，每个任务 4 次采样，共 160 条轨迹**。模型 checkpoint 根据验证集表现选取。

| 模型 | 成功轨迹 | Pass¹ | Pass⁴ |
| --- | ---: | ---: | ---: |
| SFT | 106 / 160 | 66.25% | 37.50% |
| Vanilla GRPO · step 100 | 109 / 160 | 68.13% | 45.00% |
| Anchored Turn GRPO · step 80 | **114 / 160** | **71.25%** | 32.50% |

改进模型的 Pass¹ 较 SFT、Vanilla GRPO 分别提升 **5.00、3.13 个百分点**。与此同时，Pass⁴ 下降，说明平均成功率与重复执行稳定性存在取舍，不能将结果解释为全面提升。

这里的 Passᵏ 表示多次执行均成功的可靠性指标，不是“k 次中至少一次成功”的 pass@k。以上为有限任务与采样下的实验结果，并非统计显著性结论；测试集也用于后续失败案例分析，不应视为从未参与方法迭代的盲测集。

## 代码导航

| 内容 | 入口 |
| --- | --- |
| 数据采集与筛选 | [scripts/data](agentic-grpo/scripts/data) |
| 数据构建说明 | [数据构建记录](agentic-grpo/docs/data/retail_sft_data_construction_summary.md) |
| SFT 训练 | [scripts/sft](agentic-grpo/scripts/sft) |
| 多轮 rollout | [tau2_agent_loop.py](agentic-grpo/src/agentic_rl/rollout/tau2_agent_loop.py) |
| 环境与进度信号 | [envs/tau2_retail](agentic-grpo/src/agentic_rl/envs/tau2_retail) |
| 优势重分配算法 | [anchored_turn_grpo.py](agentic-grpo/src/agentic_rl/algorithms/anchored_turn_grpo.py) |
| 训练配置 | [configs/grpo](agentic-grpo/configs/grpo) |
| 评测与翻转样例分析 | [scripts/eval](agentic-grpo/scripts/eval) |
| 模板错配排查 | [排查记录](agentic-grpo/docs/2026-08-27_sft_train_inference_template_mismatch.md) |

## 运行说明

训练实验使用单张 80GB GPU，技术栈包括 Python、PyTorch、Transformers、PEFT、verl、vLLM 与 W&B。

当前仓库保留了实验配置与开发记录，**并非开箱即用的通用训练包**。运行前需要准备兼容的 GPU 依赖、Qwen3-4B 权重、任务环境及用户模拟器 API，并将配置中的本地路径替换为实际路径。

- 任务环境依赖本地 `tau2-bench/` checkout，该目录未随主仓库跟踪，需另行准备；本项目早期验证记录使用 `tau2==1.0.1`、提交 `363133a`。
- `verl/` 为本地依赖目录，不直接随主仓库提交；固定上游版本与项目适配补丁见 [patches/verl](patches/verl)。框架本身的通用能力不作为本项目原创成果。
- API 密钥通过本地环境变量配置，不应提交 `.env`、模型权重、原始运行日志或私人文件。

## 致谢

本项目基于 [τ-bench / τ²-bench](https://github.com/sierra-research/tau2-bench)、[verl](https://github.com/volcengine/verl)、[vLLM](https://github.com/vllm-project/vllm) 与 [Qwen3](https://github.com/QwenLM/Qwen3) 开展，感谢相关开源项目。第三方代码与数据遵循各自的许可及使用要求。
