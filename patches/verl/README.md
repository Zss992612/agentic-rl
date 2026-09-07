# verl 本地适配

补丁基于 https://github.com/verl-project/verl 的提交：

`bec9ef74768dd201881cd4e54cd0385e87caae27`

包含多轮 rollout 元数据、优势估计扩展、训练指标、Assistant-only LM Head、分块计算、LoRA 续训及相关测试的本地改动。框架原有实现仍归属上游项目，补丁不是独立训练框架。

在项目根目录执行（仅用于尚未准备 verl 的新环境）：

```bash
git clone https://github.com/verl-project/verl.git verl
git -C verl checkout bec9ef74768dd201881cd4e54cd0385e87caae27
git -C verl apply --check ../patches/verl/agentic-rl.patch
git -C verl apply ../patches/verl/agentic-rl.patch
```

已有本地修改的 checkout 不要重复应用补丁。依赖安装及训练配置需按 GPU 环境另行准备。
