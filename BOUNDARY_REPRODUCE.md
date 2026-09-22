# 第二轮时间局部控制复现

## 环境与前置数据

已验证环境：V100-SXM2-32GB，Python3.10，torch2.6.0+cu124，diffusers0.31.0，numpy2.2.6，Pillow12.3.0，einops0.8.2。未升级依赖，未训练权重。

研究机Python：`/root/miniconda3/bin/python`。脚本目录：`/root/viewdit/experiments/video`。结果目录：`/root/viewdit/results/video`。

Wan权重：`/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth`。
Cog权重与config：`/root/viewdit/weights/CogVAE`。
Cog safetensors官方SHA256：`a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e`。

Wan分支依赖上一轮`wan_real_pilot`的8个输入PT及其edited_z/base/edit字段，可用包内上一轮脚本重建。路径是研究环境固定路径，不是支持任意机器直接双击的发行版；换机器须统一ROOT/OUT/权重路径。

新内容归一化验证可从JPEG独立开始，不依赖Wan输入：运行`fetch_boundary_holdout.py`下载到脚本同级`boundary_holdout`，复制至研究机`/root/viewdit/data/boundary_holdout`。脚本逐文件核验manifest中的SHA256。仅下载4条视频前17帧，不包含模型权重。

## GPU串行执行

将包内脚本放入研究机脚本目录，避免与其它GPU任务并行。已有结果先另行备份或修改OUT，以免覆盖。

```bash
/root/miniconda3/bin/python /root/viewdit/experiments/video/wan_boundary_frontier.py --limit 2 --out /root/viewdit/results/video/wan_boundary_frontier_pilot
/root/miniconda3/bin/python /root/viewdit/experiments/video/wan_feasible_control.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/wan_local_rank.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_locality_probe.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/wan_transition_budget.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_norm_replay.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/wan_bridge_convergence.py --limit 2 --out /root/viewdit/results/video/wan_bridge_convergence_pilot
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_norm_holdout.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_affine_replay.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_native_moment_projection.py
```

预期结果条数依次18/24/2/12/32/8/6/16/8/8。必须同时检查`complete=true`及日志，不只检查文件是否存在。

## 直接使用参考统计编译

该模块为研究原型。要求同一参考、shape、时间分批配置、模型和dtype；每次decode前必须重置调用计数。用完恢复原始forward，不要在并发请求中共享这个被临时修改的模型。

```python
import torch
from cog_norm_replay import ReplayNorm
from cog_affine_replay import CompiledReferenceNorm

with torch.no_grad():
    recorder = ReplayNorm(vae.decoder)
    try:
        recorder.reset('capture')
        reference_video = vae.decode(reference_latent).sample
        moments = recorder.stats
    finally:
        recorder.close()

    compiled = CompiledReferenceNorm(vae.decoder, moments)
    try:
        compiled.reset()
        replay_reference = vae.decode(reference_latent).sample
        assert (replay_reference - reference_video).abs().max() < 1e-4
        compiled.reset()
        edited_video = vae.decode(edited_latent).sample
    finally:
        compiled.close()
```

`vae`、`reference_latent`和`edited_latent`必须由调用方提供且已经位于兼容设备。此片段是模块用法，不是从零加载模型的脚本；完整加载、编码、解码流程见`cog_affine_replay.py`。

此执行不等于标准decode。精确保护应相对`replay_reference`检验；相对标准reference允许FP32舍入差。任意块内边界、非因果注意力、前缀latent变化或条件分支变化均不由这个模块自动保证。

## 本地汇总与样例

把各组`stats.json`复制到本地并命名为`<组名>_stats.json`，与`summarize_boundary_round.py`放在同一级。运行：

```bash
python3 summarize_boundary_round.py
```

汇总脚本只用Python标准库；任一组未完成会报错。统计输出为`boundary_round_summary.json`。

在研究机运行`render_boundary_evidence.py`，将生成4组PNG与GIF到`results/video/boundary_evidence`。指标来自原始张量而非GIF。

## 测量注意事项

- 所有保护目标相对于同一VAE重建，不是未经压缩的原始像素；独立Cog验证直接使用新JPEG编码。
- 后缀优化计时、全视频decode计时、初始化encode计时不同，不能混用。
- 仿射编译时延使用热身后三次同步decode中位数；参考捕获和编译单独记录，不宣称小幅差异显著。
- 新内容是固定方法的验证集，不是可以无限调参又反复称作持出的集。
- 没有运行完整语义DiT编辑，没有完成盲评，不可从机制指标推断总体视频质量。
- 原始权重不在压缩包内；大PT留在研究机。包内MANIFEST列出分发文件SHA256，用于内容校验。
