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

## 第二轮 screen2（prompts 0–7，seed 3000，n=8）要点
- 规则化 CF（C 放在早期 3,5,7,9，不与块刷新冲突）不再崩坏，但仍然有损：S2_L4+CFe 2.28×/PSNR 20.5，而 BR_L4 为 1.83×/29.95。早期 CFG 复用会改构图，因此 CF 从组件池中降级。
- **后段集中原则**：S2_OSlate+L4（O 只在第 11 步以后）为 2.15×/PSNR 30.2/MSE 0.019，质量已在扰动尺度（full_perturb1e-2 为 31.1），比 BR_L4 更快也更好。S2_L4+CFlate（C 只放在块复用步）为 2.04×/29.9，也在扰动尺度内。
- 约 2.9× 档：S2_L6+OS 2.88×/23.5/ΔCLIP +0.53；S2_OS+L4+GI8 2.89×/23.2。同档减步基线 steps11 只有 13.4，steps10 为 13.1，运动比掉到 0.6。组合明显优于减步。
- 约 3.2× 档：S2_OS3x0+Bhalf 3.24×/PSNR 20.2，运动比与锐度正常；S2_OS+L4+CFe+GI8 3.27×/17.2。同速 steps10 为 13.1。
- 无增益的变体：retro_w=0.25（23.27 对 23.2）、v1 外推（23.0）、attn 算子复用（23.65，但更慢）。
- 下一步：把 O 与 B 都挪到后段并加密（screen5），然后用 held-out 数据确认（confirm2）。

## 新组件（screen4 / learn，已排队）
- BU：非对称分支块复用。只让无条件分支复用块残差，条件分支每步重算，也就是 CF 的块级细粒度版本。
- odamp：阻尼 x0 外推。
- cmode=t1：对 CFG 差值 (u−c) 做 Taylor 外推。
- 小训练组件：闭环最小二乘拟合每步外推系数。在 prompts 0–3（seed 5000）上标定，在 prompts 8–15 上 held-out 测试。

## 待完成
- screen3（2.5–3× 堆叠）、screen5（后段密集）、confirm（第一轮胜者，held-out）、screen4（新组件）、learn（小训练）、confirm2（后段胜者，held-out），按此顺序排队。
