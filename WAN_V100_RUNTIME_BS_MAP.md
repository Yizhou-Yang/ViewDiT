# Wan 在 V100 上的可运行配置定位

> 目的：为"在**有时域耦合**底座上检验『前缀校准 → 未来连带编辑』的『编辑-保持』取舍"定位可复用的 Wan 运行配置。
> 结论：历史在 V100 跑通并留存的 Wan 配置 = **官方 Wan2.1 VAE（因果解码器）**，正是所需的时域耦合底座。

## 关键判断

历史工作区所有 Wan 相关实验（`wan_state_probe` / `wan_real_pilot` / `wan_boundary_frontier` /
`wan_bridge_convergence` / `wan_tail_compensate` / `wan_feasible_control` / `wan_transition_budget` /
`wan_local_rank` / `wan_phase_audit` / `wan_baseline_audit`）**都不是**完整 Wan2.1 T2V/I2V DiT 生成，
而是基于 **官方 Wan2.1 VAE decoder 的逐帧因果解码干预**（操作 `_feat_map` 缓存）。

这**正是我们需要的时域耦合底座**：
- decoder 用 `CausalConv3d` + `CACHE_T=2` + `_feat_map` 逐帧缓存因果解码；
- 改前缀 latent → 经 `_feat_map` **连带影响**后续帧解码输出（真时域耦合）；
- 上一轮 Latte 冲突探针已证明：帧独立的底座无法检验"编辑-保持"取舍（构造性伪问题），
  必须移师这种因果耦合结构。

## V100 可运行配置（已核实就绪）

| 项 | 路径 / 值 | 状态 |
|---|---|---|
| VAE 权重 | `/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth`（507 MB） | ✅ 存在 |
| VAE 实现 | `/root/viewdit/experiments/video/wan_vae_official.py`（官方 `WanVAE`） | ✅ 存在 |
| 因果解码封装 | `wan_state_probe.py::stream()`（reset/blend `_feat_map`） | ✅ 存在 |
| 回调/后缀工具 | `wan_tail_compensate.py::suffix()` | ✅ 存在 |
| VAE 配置 | dim=96, z_dim=16, 因果 3D 卷积, cache_t=2（见 `wan_vae_config_review.txt`） | ✅ |
| 依赖 | torch(cu124) 2.6.0, einops 0.8.2 | ✅ |
| GPU | Tesla V100-SXM2-32GB，CUDA OK，当前空闲（33.7GB 可用） | ✅ |
| 历史基线结果 | `/root/viewdit/results/video/wan_*`（10 目录，含 stats/gif/png） | ✅ 可对照 |

## 推荐复用的运行方式

1. 以 `wan_state_probe.py` 的 `stream()` 为底座（`WanVAE(model).decode` 拆成逐帧因果步进）；
2. 在**前缀帧**上做 low-rank 边界干预（复用 Latte 冲突探针的 rank 全SVD 逻辑，改挂到 decoder 因果步进）；
3. 校验指标：前缀帧保持（RMS）+ 未来帧"编辑-连带"（suffix RMS/leak），
   且在此处才具备 Latte 缺失的"改前缀→影响未来"的真实耦合路径。

## 边界声明

- 该配置是 **VAE decoder 因果干预**，涉及真实时域耦合；但它不涉及 DiT 的 token 级语义编辑，
  因此"语义A 保持 ⊥ 语义B 编辑"的**构造仍需在 latent 上显式构造**（同一批帧上的正交目标），
  不能自动获得。这里只是提供了可运行的耦合底座，不改变需要显式构造正交冲突这一点。
- 该结论不改判 Latte P1/P2/P3 审查结论。