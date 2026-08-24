# 2026-08-21 Retail 基线评测与远程环境进展总结

本文记录 2026-08-21 在 AutoDL 上完成 Qwen3-4B 系列 Retail 基线评测期间的实际进展、问题处理、测试结果，以及进入 SFT 阶段前仍需完成的工作。

## 1. 当前阶段结论

今天已经完成以下关键工作：

- 在 AutoDL A800 80GB GPU 上成功启动 vLLM 0.13.0，并以 OpenAI-compatible API 向 τ²-bench 提供本地模型服务。
- 跑通 Qwen3-4B non-thinking 的 Retail smoke、validation 和 test 流程。
- 完成一轮 40 题 Qwen3-4B non-thinking test。
- 完成一轮 40 题 Qwen3-4B-Instruct-2507 test。
- 验证评测 checkpoint、tmux 后台运行、GPU/KV Cache 监控和结果下载流程。
- 完成模型路径、ModelScope 缓存、进程管理、API Key、上下文长度、端口和代理等外围配置。

下一阶段可以进入 SFT，但开始训练前应先完成一次严格的本地与远端代码、配置和数据同步。

SSH 普通登录目前可用，但 VS Code/Codex 远程连接和 7897 反向代理隧道仍不够稳定。这是尚未关闭的基础设施问题，需要在 SFT 长任务开始前单独解决。

## 2. 当前远端运行结构

远端项目目录：

```text
/root/autodl-tmp/agentic_rl
```

当前环境分工：

```text
.venv-vllm/       vLLM、模型加载和 OpenAI-compatible API 服务
training/.venv/   τ²-bench 评测和后续训练侧脚本
```

模型缓存位置：

```text
/root/autodl-tmp/modelscope-cache/Qwen/Qwen3-4B
/root/autodl-tmp/modelscope-cache/Qwen/Qwen3-4B-Instruct-2507
```

服务约定：

```text
vLLM API:          http://127.0.0.1:8000/v1
本地 vLLM API Key: EMPTY
远程用户模拟器:    openai/deepseek-v4-flash
用户模拟器 API:    https://opencode.ai/zen/go/v1
```

`EMPTY` 只用于本地 vLLM 服务鉴权。真实的远程 API Key 保存在远端 `tau2-bench/.env` 中，不应复制到文档、聊天、Git 或公开结果中。

## 3. 今天处理的部署与配置问题

vLLM 本身的安装和部署过程比较顺利：在独立虚拟环境中安装完成、模型下载完成后即可正常启动服务，今天没有发现 vLLM 代码或推理引擎本身的 bug。

以下事项属于模型缓存布局、残留进程、环境变量、服务参数以及评测客户端之间的配置和使用问题，不应归因于 vLLM 自身缺陷。

### 3.1 模型已下载但 vLLM 再次下载

模型最初位于：

```text
/root/autodl-tmp/modelscope-cache/Qwen/Qwen3-4B
```

但用仓库 ID 启动时，ModelScope/vLLM 可能按另一套缓存布局寻找模型，从而再次下载。最终方案是向 vLLM 传入模型的绝对本地路径，同时使用稳定的 served model alias：

```text
本地加载路径:       /root/autodl-tmp/modelscope-cache/Qwen/Qwen3-4B
served model name: Qwen/Qwen3-4B
```

评测请求中的模型名应与 `--served-model-name` 一致，而不是与磁盘路径一致。

### 3.2 ModelScope 一直等待锁

日志反复出现：

```text
Still waiting to acquire lock ... Qwen___Qwen3-4B
```

原因是旧 vLLM/ModelScope 进程仍持有模型缓存锁。通过 `pgrep -af 'vllm|serve_vllm|EngineCore'` 检查并停止旧进程后恢复正常。

### 3.3 GPU 显存不足

旧的 vLLM EngineCore 仍占用 GPU，导致新服务按 `gpu_memory_utilization: 0.9` 启动时可用显存不足。清理旧进程后，Qwen3-4B 在 A800 80GB 上可以正常使用 0.9 配置。

### 3.4 `OMP_NUM_THREADS` 非法

启动时出现：

```text
libgomp: Invalid value for environment variable OMP_NUM_THREADS
```

该问题不是 vLLM 主故障。启动前执行以下命令即可消除：

```bash
unset OMP_NUM_THREADS
```

### 3.5 评测提示缺少 API Key

本地 vLLM 使用 `EMPTY`，但远程用户模拟器仍需要真实 API Key。评测环境没有读到 `OPENAI_API_KEY` 时，LiteLLM 会报 Missing credentials。

目前真实 Key 保存在：

```text
/root/autodl-tmp/agentic_rl/tau2-bench/.env
```

评测入口通过 `python-dotenv` 加载该文件。

### 3.6 模型不存在

当 vLLM 的 served model name 被设置为本地绝对路径，而评测端请求 `Qwen/Qwen3-4B` 时，API 返回：

```text
The model `Qwen/Qwen3-4B` does not exist
```

解决方案是在服务端明确区分 `model_path` 和 `served_model_name`。远端 `serve_vllm.py` 已增加这一支持，但该文件目前尚未同步回本地工作区。

### 3.7 上下文窗口不足

旧服务使用 `max_model_len: 16384`。Task 58 曾出现：

```text
input tokens: 14397
requested output: 2048
total: 16445 > 16384
```

正式评测将 vLLM 最大上下文提高到 32768，并保留 Agent `max_tokens: 2048`。修改服务参数后必须重启 vLLM 才能生效。

### 3.8 tmux 与断点续跑

采用独立 tmux 会话运行服务和评测：

```text
vllm             模型服务
eval-test         正式评测
kv-monitor        可选的 KV Cache 监控
```

τ²-bench 会在每条 simulation 完成后更新 `results.json`。中断后可以按 task/trial/seed 跳过已保存结果，但当前自定义入口原本每次生成新的时间戳目录，因此需要给 `run_retail_baseline.py` 增加 `--resume-run`，复用原 run name，并设置 `auto_resume=True`。

断点续跑只能从任务粒度继续；中断时尚未保存的单个任务需要从头执行。

### 3.9 误删结果恢复

AutoDL 数据盘回收站位于：

```text
/root/autodl-tmp/.Trash-0
```

误删的 test 结果仍存在于 `.Trash-0/files`，已确认可以将整个 run 目录移动回：

```text
/root/autodl-tmp/agentic_rl/tau2-bench/data/simulations/
```

使用终端 `rm` 删除的文件不会进入该回收站，因此重要代码、配置和结果仍需同步到本地或文件存储。

## 4. GPU 与并发观察

Qwen3-4B non-thinking test 后半段使用了：

```yaml
sampling:
  max_concurrency: 8
```

该参数表示最多同时运行 8 条完整 simulation，不表示始终有 8 个本地 vLLM 请求。每条 simulation 会在本地 Agent、环境工具和远程用户模拟器之间交替，因此实际 `num_requests_running` 会动态变化。

一次高负载快照：

```text
GPU:             NVIDIA A800 80GB PCIe
GPU-Util:        93%
Power:           256W / 300W
Temperature:     59C
Memory:          74325 MiB / 81920 MiB
vLLM requests:   running=4, waiting=0
KV Cache usage:  约 3.77%
```

结论：

- vLLM 在有请求时可以充分利用 GPU。
- 60W 与 230W 左右的功率来回变化，主要来自等待远程用户模拟器时的空档。
- `nvidia-smi` 显示约 74GB 占用主要是 vLLM 按 0.9 预留显存，并不表示 KV Cache 已使用 90%。
- 并发 8 在本次运行中没有造成 KV Cache 压力，也没有出现 vLLM 请求排队。
- 正式实验不应在同一个结果文件中途修改并发。后续应从新 run 开始固定配置。

## 5. 三个 test 结果

本地下载结果位于：

```text
~/Desktop/simulations/
```

### 5.1 汇总

| Run | 模型 | 完成数 | 成功数 | 观测 Pass^1 | 状态 |
|---|---|---:|---:|---:|---|
| `retail_qwen3_4b_non_thinking_test_20260821_221746` | Qwen3-4B non-thinking | 36/40 | 16 | 44.4% | 中断，不是完整正式结果 |
| `retail_qwen3_4b_non_thinking_test_20260821_230309` | Qwen3-4B non-thinking | 40/40 | 17 | 42.5% | 完整 |
| `retail_qwen3_4b_instruct_2507_test_20260821_232927` | Qwen3-4B-Instruct-2507 | 40/40 | 23 | 57.5% | 完整 |

第一条 36 题运行只能用于诊断，不应作为完整 test 分数。

### 5.2 Instruct-2507 与 non-thinking 的同题比较

在两个完整 40 题结果中：

```text
Qwen3-4B non-thinking:      17/40 = 42.5%
Qwen3-4B-Instruct-2507:     23/40 = 57.5%
绝对提升:                    +6 题 / +15.0 个百分点
```

逐题配对结果：

```text
两者都成功:                 11
仅 Instruct-2507 成功:      12
仅 non-thinking 成功:        6
两者都失败:                 11
```

共同失败 Task：

```text
18, 27, 32, 36, 38, 42, 56, 94, 97, 100, 102
```

Instruct-2507 相对完整 non-thinking run 新增成功的 Task：

```text
9, 12, 17, 39, 45, 60, 61, 62, 70, 71, 101, 111
```

Instruct-2507 相对回退的 Task：

```text
33, 51, 64, 79, 86, 90
```

### 5.3 工具使用与轨迹效率

| 指标 | non-thinking 完整 test | Instruct-2507 test |
|---|---:|---:|
| 工具错误消息总数 | 104 | 24 |
| 平均消息数 | 39.0 | 30.1 |
| 平均工具调用数 | 9.6 | 7.2 |
| `too_many_errors` | 3 | 0 |
| `max_steps` | 0 | 0 |
| 平均单任务 duration | 67.7 秒 | 53.9 秒 |

Instruct-2507 的主要优势不仅是成功率提高，还包括更短的交互轨迹、更少的重复工具调用和更少的工具错误。

duration 会受到并发和远程 API 状态影响，不能单独作为模型推理速度结论。

### 5.4 Pass^1 波动

两次 non-thinking 运行共同完成的 36 题中，两次都是 16 题成功，但只有 22 题结果一致，另外 14 题发生成功/失败翻转。

这说明单次 Pass^1 存在明显随机波动。当前结果适合做阶段性基线和模型方向判断，但不足以证明稳定的逐题能力差异。预算允许时应增加 trial 数。

## 6. `user_stop` 的正确解释

`user_stop` 是 τ²-bench 的合法终止类型，不等于失败原因。用户模拟器输出以下任一 token 时，当前框架都会结束用户侧对话并记录为 `user_stop`：

```text
###STOP###
###TRANSFER###
###OUT-OF-SCOPE###
```

Instruct-2507 的 40 条结果全部是 `user_stop`，其中 23 条成功、17 条失败，因此不能根据该字段直接判断用户模拟器或 Agent 谁有问题。

Instruct-2507 的 17 条失败均表现为最终数据库状态不匹配。具体归因需要检查：

- 最终写工具是否调用。
- 工具参数和目标对象是否正确。
- 是否在应继续处理时错误转人工。
- Agent 是否声称完成但实际上没有改变环境。
- 用户是否在只完成确认、尚未执行最终写操作时提前输出 `###STOP###`。

项目已经在 `training/docs/problem_log/data_issues.md` 中记录过用户模拟器提前 STOP 的已知案例。正式 benchmark 仍保留原始 reward；SFT/GRPO 数据审计应另外标记 `premature_user_stop`，避免错误归因。

## 7. 与 Qwen 官方结果的关系

Qwen 官方模型卡报告：

```text
Qwen3-4B non-thinking TAU2-Retail:   28.1%
Qwen3-4B-Instruct-2507 TAU2-Retail: 40.4%
官方模型间差值:                      +12.3 个百分点
```

本项目完整 test 的模型间差值为 +15.0 个百分点，提升方向和幅度与官方基本一致，但绝对分数不能直接比较。

当前项目与官方协议的主要差异：

- 本项目使用自定义选出的 40 个 Retail test Task。
- 用户模拟器为 `deepseek-v4-flash`。
- 当前仅运行一个 trial。
- 结果中的 `git_commit` 为 `unknown`，无法证明 benchmark 代码和任务版本与官方完全一致。
- 推理参数、Agent prompt、工具解析器和上下文配置可能不同。

因此项目报告应明确写为：

> 在项目固定的 40 题 Retail test split、DeepSeek-v4-flash 用户模拟器和单次采样设置下，Qwen3-4B-Instruct-2507 获得 57.5% Pass^1，Qwen3-4B non-thinking 获得 42.5%。该绝对分数不与官方结果直接横向比较，但模型间提升趋势与官方报告一致。

官方参考：

- [Qwen3-4B-Instruct-2507 模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [τ²-bench leaderboard submission 说明](https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md)

## 8. 开始 SFT 前必须完成的同步

当前本地与远端已经出现实质差异，不能直接开始正式训练。

已确认的差异包括：

- 远端存在 `training/scripts/eval/serve_vllm.py`，本地当前没有该文件。
- 本地 `retail_qwen3_4b_instruct_2507.yaml` 仍是 validation 2 题 smoke 配置，但远端已经运行了 40 题 test。
- 远端 `run_retail_baseline.py` 可能已经增加 `--resume-run`，本地版本尚未确认包含该修改。
- 远端保存了本次正式运行所用的本地模型路径、served model name、32768 context 和并发配置。
- 评测结果已下载到桌面，但尚未归档到项目内固定、可追踪的位置。

同步时应重点处理：

```text
training/scripts/eval/serve_vllm.py
training/scripts/eval/run_retail_baseline.py
training/configs/eval/*.yaml
training/configs/data/retail_split.yaml
training/data/sft/retail_serial_v3/sft_train_256.jsonl
tau2-bench 中项目实际使用的改动
完整 test results.json 及其运行配置
依赖锁文件和环境版本记录
```

同步原则：

1. 先进行 dry-run 或逐文件 diff，不直接用一侧覆盖另一侧。
2. `.env`、API Key、Codex token 和 SSH 凭据不得同步进 Git。
3. 保留原始 `results.json`，不要手工编辑。
4. 将 smoke、validation、test 配置拆成不同文件，避免反复修改同一个 YAML。
5. 为每个正式 run 固定模型、任务、trial、seed、并发和 context，并保存 git commit。
6. 同步完成后重新执行一次配置 dry-run 和 2 题 smoke，确认本地记录与远端实际运行一致。

## 9. SFT 阶段的建议入口

现有数据构建文档记录了 256 条正式 Retail SFT 轨迹：

```text
training/data/sft/retail_serial_v3/sft_train_256.jsonl
```

进入正式 SFT 前建议按以下顺序：

1. 完成代码、配置、数据和结果同步。
2. 冻结当前 Base test 和 Instruct-2507 test 作为基线。
3. 校验 256 条 JSONL 的条数、工具调用配对、聊天模板和训练目标 mask。
4. 明确本轮 SFT 的 base checkpoint、上下文长度、精度和显存策略。
5. 先执行少量 step 的 overfit/smoke run，检查 loss、保存和恢复。
6. 再启动正式 SFT。
7. 训练后先跑固定 validation，选定 checkpoint 后只跑一次正式 test。

SFT 数据构建细节见：

- `training/docs/data/retail_sft_data_construction_summary.md`
- `training/docs/data/sft_data_spec.md`

## 10. 尚未解决：SSH 与反向代理稳定性

当前状态：

- 普通 `ssh autodl` 可以连接远端。
- VS Code Remote SSH/Codex 连接放置一段时间后仍可能中断。
- 反向代理曾出现 `remote port forwarding failed for listen port 7897`。
- vLLM 使用 8000，反向代理使用 7897，两者不是同一端口，不存在直接冲突。
- 当前已将普通远程连接与代理隧道拆成 `autodl` 和 `autodl-proxy` 两个 Host，但稳定性问题尚未完成验证。

后续排查应分层进行：

1. 单独测试不带任何转发的普通 SSH 能否长时间保持。
2. 使用 `ssh -vvv -N autodl-proxy` 记录反向隧道断开时的客户端原因。
3. 确认 Mac 本地代理始终监听 `127.0.0.1:7897`。
4. 确认服务器 7897 只由一条 SSH 隧道监听，VS Code 使用的 `autodl` Host 中不得包含 `RemoteForward`。
5. 检查 Mac 睡眠、切换网络和代理软件重启是否会导致隧道失效。
6. 配置 SSH Key，避免自动重连受密码输入阻塞。
7. 原因明确后再考虑用 `autossh` 或 macOS `launchd` 自动维护唯一代理隧道。
8. 服务器可直接联网的任务不要强制依赖反向代理；ModelScope 下载和本地 vLLM 运行应优先使用服务器自身网络。

验收标准：

- 普通 SSH 和 VS Code Remote SSH 空闲至少两小时不掉线。
- 代理隧道断开后可以可靠重连，不残留占用 7897 的旧监听。
- VS Code 创建多条内部 SSH 会话时不再触发 7897 转发冲突。
- Codex、模型下载和评测所需网络路径彼此独立，单一路径失败不会影响本地 vLLM 服务。

现有连接方法和配置见 `training/docs/remote_codex_autodl_setup.md`。

## 11. 下一次工作的推荐顺序

```text
1. 备份远端代码、配置和正式结果
2. 对本地与远端执行逐文件 diff
3. 完成双向同步并排除所有 secret
4. 整理 smoke / validation / test 独立配置
5. 固化环境版本、运行命令和 git commit
6. 解决 SSH 基础连接与反向隧道稳定性
7. 对 SFT 数据和训练入口做 smoke 校验
8. 启动正式 SFT
9. validation 选 checkpoint
10. 使用固定协议进行 SFT test，并与 42.5% / 57.5% 基线比较
```
