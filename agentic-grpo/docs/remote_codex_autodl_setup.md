# 通过 SSH 在 AutoDL 服务器上运行 Codex

本文记录如何让本地 Mac 上的代理通过 SSH 转发到 AutoDL 服务器，并在服务器上安装、登录和运行 Codex。

## 1. 工作方式

```text
本地 Mac
├── VS Code / Codex 界面
├── SSH Host: autodl ──────────────────────────┐
└── 本地代理 127.0.0.1:7897                          │
          ▲                                              │
          │ SSH Host: autodl-proxy                        │
          │ 唯一一条 RemoteForward 7897                     │
          │                                              │
AutoDL 服务器                                             │
├── Codex CLI                                           │
├── 项目文件与终端 ◀──────────────────────────┘
└── 远端代理入口 127.0.0.1:7897

服务器访问外部服务时：
AutoDL 127.0.0.1:7897 → SSH 反向隧道 → Mac 代理 → OpenAI / Hugging Face
```

代码读取、文件修改和命令执行发生在服务器上；Mac 负责显示界面、建立 SSH 连接并提供网络代理。Codex 使用的模型运行在 OpenAI 云端，不占用 Mac 或服务器的 GPU。

VS Code/Codex 连接和代理隧道必须分开。VS Code Remote SSH 可能创建多条 SSH 会话，但服务器的 `127.0.0.1:7897` 只能由一条反向转发隧道监听。不要把 `RemoteForward 7897` 放在 VS Code 使用的 Host 中。

## 2. 前提条件

- Mac 上的代理软件已经启动。
- 本地 HTTP 或 Mixed 代理端口为 `7897`。
- 已获得服务器的 SSH 地址、端口和登录凭据。
- VS Code 已安装 Remote - SSH。

以下示例约定：

```text
本地代理端口：7897
远端代理端口：7897
VS Code/Codex Host 别名：autodl
代理隧道 Host 别名：autodl-proxy
```

如果端口或服务器地址发生变化，需要同步替换对应值。

## 3. 检查 Mac 本地代理

在 Mac 本地终端执行：

```bash
curl -x http://127.0.0.1:7897 -I https://huggingface.co
```

返回 HTTP 状态码说明本地代理可用。`Connection refused` 通常表示代理软件没有启动，或者端口不是 `7897`。

`127.0.0.1:7897` 是代理端口，不是网页，因此不需要在浏览器中直接打开。

## 4. 分离 VS Code/Codex 连接与代理隧道

在 Mac 的 `~/.ssh/config` 中为同一台 AutoDL 服务器配置两个 Host。两者使用相同的 `HostName`、`Port` 和 `User`，但职责不同：

```sshconfig
# 专供 VS Code Remote SSH 和 Codex 连接。
# 这里不得配置 RemoteForward 7897。
Host autodl
    HostName <服务器 SSH 地址>
    Port <服务器 SSH 端口>
    User root
    ServerAliveInterval 30
    ServerAliveCountMax 6
    TCPKeepAlive yes

# 只用于维持从服务器到 Mac 代理的反向隧道。
Host autodl-proxy
    HostName <服务器 SSH 地址>
    Port <服务器 SSH 端口>
    User root
    ServerAliveInterval 30
    ServerAliveCountMax 6
    TCPKeepAlive yes
    RemoteForward 7897 127.0.0.1:7897
    ExitOnForwardFailure yes
```

`ServerAliveInterval 30` 表示客户端每 30 秒发送一次 SSH 保活消息；`ServerAliveCountMax 6` 允许连续 6 次未收到响应后再判定连接失效。

`RemoteForward 7897 127.0.0.1:7897` 表示：

```text
服务器的 127.0.0.1:7897 -> Mac 的 127.0.0.1:7897
```

保存配置后，必须完全断开原来的 VS Code Remote SSH 窗口和旧 SSH 会话。已有会话不会自动加载新配置，并且可能仍然占用服务器的 `7897`。

然后在 Mac 的独立终端中启动唯一一条代理隧道：

```bash
ssh -N autodl-proxy
```

输入密码后，终端持续空白且不返回 shell 提示符是正常现象，表示隧道正在运行。该终端需要保持打开。

另开一个 Mac 终端验证普通 SSH 连接：

```bash
ssh autodl
```

VS Code Remote SSH 和 Codex/ChatGPT 桌面端都应选择 `autodl`，不要选择 `autodl-proxy`。OpenAI 官方远程连接说明会从 `~/.ssh/config` 读取具体 Host 别名，并要求先确认常规 `ssh <host>` 可用。

## 5. 验证远端代理

保持 Mac 上的 `ssh -N autodl-proxy` 运行，在服务器终端执行：

```bash
curl -x http://127.0.0.1:7897 -I https://huggingface.co --max-time 10
```

返回 HTTP 状态码表示反向转发成功。

如果服务器安装了 `ss`，也可以检查监听端口：

```bash
ss -lnt | grep 7897
```

精简镜像可能没有 `ss`，这种情况直接使用前面的 `curl` 测试即可。

## 6. 设置服务器代理变量

先在服务器当前终端临时设置代理：

```bash
export HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 https_proxy=http://127.0.0.1:7897
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
```

检查变量：

```bash
echo "$HTTP_PROXY"
```

正确结果必须是：

```text
http://127.0.0.1:7897
```

变量中不能出现 Markdown 链接格式，例如：

```text
[http://127.0.0.1:7897](http://127.0.0.1:7897)
```

随后测试不带 `-x` 的请求：

```bash
curl -I https://huggingface.co
```

确认临时配置正常后，可以编辑 `/root/.bashrc`，删除其中旧的或格式错误的代理配置，再加入：

```bash
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

编辑完成后，新建终端，或者执行：

```bash
source /root/.bashrc
```

不要反复向 `.bashrc` 追加同一组变量。

## 7. 安装 Codex CLI

在服务器执行 OpenAI 官方安装命令：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

安装程序会把 Codex 放入用户目录，并将对应 PATH 写入 `/root/.bashrc`。

如果安装后当前终端仍提示 `codex: command not found`，执行：

```bash
export PATH="/root/.local/bin:$PATH"
```

然后检查版本：

```bash
codex --version
```

## 8. 在无界面服务器上登录 Codex

优先使用设备码登录：

```bash
codex login --device-auth
```

终端会显示登录网址和一次性验证码。在 Mac 浏览器中打开该网址、登录 ChatGPT 并输入验证码。不要把验证码发到聊天或日志中。

登录完成后检查：

```bash
codex login status
```

如果设备码登录不可用，官方备用方案是先在有浏览器的本地电脑完成登录，再将 `~/.codex/auth.json` 复制到服务器同一路径。`auth.json` 包含访问令牌，应当像密码一样保护，不能提交到 Git 或发送到聊天中。

## 9. 在远程项目中使用 Codex

服务器必须同时满足：

```bash
command -v codex
codex login status
```

然后可以通过支持 SSH 项目的 Codex/ChatGPT 桌面端连接远程目录，或在 VS Code Remote SSH 窗口中使用相应的 Codex 工具。

远程运行时：

- 项目文件来自服务器。
- shell 命令在服务器执行。
- Python、PyTorch、vLLM 和 GPU 使用服务器环境与硬件。
- Codex 的模型请求通过 SSH 隧道转发到 Mac 代理，再访问外部服务。

每次开始远程工作时，推荐固定使用以下顺序：

1. 启动 Mac 代理软件，确认本地 `127.0.0.1:7897` 可用。
2. 在独立 Mac 终端运行 `ssh -N autodl-proxy`，并保持该终端打开。
3. 在 VS Code Remote SSH 或 Codex/ChatGPT 桌面端中连接 `autodl`。
4. 在服务器上用 `curl -x http://127.0.0.1:7897 -I https://huggingface.co --max-time 10` 确认代理。
5. 开始代码、下载或训练工作。

VS Code 重连时只会重建 `autodl` 连接，不会重复申请服务器的 `7897`。代理隧道由独立的 `autodl-proxy` 会话维持。

## 10. 常见问题

### `curl: (7) Failed to connect to 127.0.0.1 port 7897`

可能原因：

- Mac 代理没有启动。
- Mac 实际代理端口不是 `7897`。
- `ssh -N autodl-proxy` 没有运行或已经断开。
- `RemoteForward` 没有加入 `autodl-proxy` Host。

### `Error: remote port forwarding failed for listen port 7897`

这表示服务器端的 `127.0.0.1:7897` 已经被另一条 SSH 会话或其他进程占用。不是 Mac 本地代理端口冲突。

常见原因是把 `RemoteForward 7897` 配置在 VS Code 使用的 `autodl` Host 中，导致 VS Code 的多条 SSH 会话反复争抢同一个服务器端口。

处理方法：

1. 确认 `autodl` Host 中没有 `RemoteForward`。
2. 完全关闭旧的 VS Code Remote SSH 窗口和代理 SSH 终端。
3. 只启动一条 `ssh -N autodl-proxy`。
4. VS Code 只连接 `autodl`。

如果关闭旧会话后仍然报错，可以先忽略所有转发登录服务器：

```bash
ssh -o ClearAllForwardings=yes autodl
```

然后在服务器上检查端口：

```bash
ss -lntp | grep ':7897'
```

### VS Code 空闲一段时间后 SSH 断开

确认 `autodl` 和 `autodl-proxy` 都配置了：

```sshconfig
ServerAliveInterval 30
ServerAliveCountMax 6
TCPKeepAlive yes
```

这可以降低网络设备或云平台回收空闲 SSH 连接的概率。如果日志显示 `Connection refused` 或 `Connection closed by <server>`，还需要检查 AutoDL 实例是否停机、重启，或 SSH 地址与端口是否变化。

### `curl` 使用了带方括号和圆括号的代理地址

原因是网页或聊天中的链接被当成 Markdown 复制。环境变量中只能保留纯 URL：

```text
http://127.0.0.1:7897
```

### `bash: ss: command not found`

服务器镜像没有安装 `iproute2`。这不影响代理本身，使用下面的请求直接验证即可：

```bash
curl -x http://127.0.0.1:7897 -I https://huggingface.co --max-time 10
```

### 安装成功后仍然提示 `codex: command not found`

当前 shell 没有加载新 PATH。执行：

```bash
export PATH="/root/.local/bin:$PATH"
```

### SSH 断开后服务器无法联网

这是当前方案的正常特性：反向代理依赖独立的 `ssh -N autodl-proxy` 隧道。模型下载、API 调用或长时间任务需要保持该隧道稳定。VS Code 断开不一定会影响隧道，但关闭 `autodl-proxy` 终端会立即中断服务器对 Mac 代理的访问。

## 11. 官方参考

- [Codex CLI 安装](https://learn.chatgpt.com/docs/codex/cli#get-started-with-codex-cli)
- [Codex 认证与设备码登录](https://learn.chatgpt.com/docs/auth#login-on-headless-devices)
- [Codex SSH 远程连接](https://learn.chatgpt.com/docs/remote-connections#connect-to-an-ssh-host)
