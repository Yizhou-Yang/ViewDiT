# CogVideoX-2b 组件组合筛选（V100，2026-09-24，持续更新）

## 设置
CogVideoX-2b，17 帧 480×720，DDIM-30，CFG 6，fp16 DiT。只改 DiT 采样过程，VAE 只用于解码测量。
自写采样循环与官方 pipeline 逐元素一致（max abs diff = 0.0，`verify.json`）。
speed = 同 prompt 同 seed 下 DiT 采样墙钟时间之比（相对 full）。质量参照：`full_perturb1e-2`（初始噪声加 1e-2 扰动，PSNR 31.1 / latent MSE 0.029）是"与 full 同分布、不可区分"的尺度。
指标：latent MSE、解码 PSNR（相对 full）、ΔCLIP（相对 full）、锐度比、运动比、时间差分误差。

代码：`cog_combo.py`（本地与远端 `/root/viewdit/experiments/video/`）；结果：`/root/viewdit/results/video/cog_combo_20260924/`，本地镜像 `v100_runs/results/video/cog_combo_20260924/`。

## 组件池（四个粒度）
| 代号 | 粒度 | 组件 | 已知来源 |
|---|---|---|---|
| BR | block | 后段块残差复用（warm w，每 k 步刷新） | Δ-DiT / FORA / 本项目 H20 L4 |
| OP_attn / OP_ffn | 算子 | 只复用 attention 输出（重算 FFN）/ 只复用 FFN | PAB / ToCa 类 |
| CF | 分支 | CFG 无条件分支复用 u=c+(u−c)_last | FasterCache |
| GI | 分支 | 后段关闭引导，只算条件分支 | guidance interval |
| OS | 步 | 整步跳过，v 或 x0 空间 Taylor-1 / 复用 | TaylorSeer / TeaCache |
| retro | 步 | 低权重回溯修正 | Z-Cache 类 |

## 第一轮（prompts 0–7，seed 3000，n=8）要点
- 单组件有效：BR_L4 1.83×/PSNR 29.95；BR_L6_w10 1.94×/28.58；OP_attn_L4 1.49×/30.56；GI_last8 1.14×/36.3；GI_last12 1.22×/32.7；OS 1.76×/23.6。
- 单组件无效：CF（1.27–1.40×，PSNR 16–21，锐度异常 1.4–2×）、GI_first2（前期关引导毁构图）、OP_ffn（加速小且更差）、BR_t1（块级 Taylor 不比零阶好）、所有减步（15 步 PSNR 14）。
- 有效组合：
  - **CB_L4+GI8：1.94×，PSNR 29.23，latent MSE 0.023 —— 比 BR_L4 更快，质量几乎不变，达到扰动尺度。**
  - **CB_OSx0+L4：2.41×，PSNR 23.2，ΔCLIP +0.54，锐度/运动比 1.12/1.08 —— 同 2.5× 下远胜减步 steps12（PSNR 13.3）。**
- 发现的组合规则（负面组合的根因）：CF 与块复用同步做会失败（CB_L4+CF 爆掉，PSNR 8.7；CB_attnL2+CF 完全崩坏 PSNR 3.8）。原因：CFG 复用步只跑条件分支，无条件分支的块缓存停在旧步；若之后在复用步使用它，就是跨多步的陈旧残差。规则：**分支级跳过的步不能作为块缓存刷新步**。

## 待完成
- 第二轮：遵守上述规则的 CF 组合、OS 各种变体、OS+L4+GI 三级堆叠。
- 第三轮：推高加速（O+B+GI 三级堆叠，目标 2.5–3×）。
- 确认轮：held-out prompts 8–15 × 2 seeds，对第一轮有效组合做复现。
