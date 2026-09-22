# Second-DiT Strong-Edit Boundary Probe (Latte / UCF101 / CFG)

日期：2026-09-22
目的：回应「V3 缺少第二个可跑的语义底座」——在 Latte（不同于 CogVideoX 的
第二个视频 DiT、类别条件、CFG 真实生成）上验证「紧凑边界状态在强语义编辑下
保持已确认前缀」的机制是否跨架构成立。

## 为什么用 Latte / UCF101（而不是 Wan）

- Wan2.1-T2V-1.3B 权重未下载，且 832×480 长视频配方是按 H20 设计的，V100-32GB
  跑不动。Latte `ucf101.pt` 权重已就绪、零下载成本，是真正的 class-conditional
  video DiT（extras=2），能立刻在 V100 上跑通。
- 用 CFG（cfg_scale=7.0，UCF101 官方默认）制造「强语义编辑」：在前缀由类别 A
  生成后、以同一噪声换类别 B 重新生成未来，作为语义翻转代理。这样编辑强度
  真实（区别于弱编辑/低秩压缩验证）。

## 协议（机制探针，非完整 benchmark）

- 模型：Latte-XL/2、ckpt `ucf101.pt`、num_classes=101、extras=2。
- 采样：DDIM 12 步、16 帧、CFG=7.0、seed=999。
- 前/后类别：A=14、B=95（字母序距离远的两个动作类，作为强语义差代理）。
- 保留前缀：p=8 帧（一半）。
- 三对照：none（编辑完全带走前缀）、full_oracle（存完整前缀）、
  boundary_rank{k}（存低秩边界状态 k=2,4,8,16,32,64）。
- 指标：latent 域前缀/未来 RMSE + payload 字节。像素解码 deferred。
- 记录：GPU 机 `/root/viewdit/results/video/latte_sem_cfg7/stats.json`，
  日志 `logs/latte_sem_cfg7.log`，脚本 `experiments/video/preset_latte_semantic_boundary.py`。

## 结果（cfg=7.0，edit_distance_ab = 1.49253）

| 方法 | prefix_latent_rms | payload_bytes |
|---|---:|---:|
| none | 1.4958 | — |
| full_oracle | 0 | 131072 |
| boundary_rank2 | 0.61555 | 8194 |
| boundary_rank4 | 0.5053 | 16388 |
| boundary_rank8 | 0.36553 | 32776 |
| boundary_rank16 | 0.21322 | 65552 |
| boundary_rank32 | 8.1678e-06 | 131104 |
| boundary_rank64 | 8.1678e-06 | 262208 |

## 解释（必须诚实区分「成立」与「边界」）

成立的部分：
- 强编辑强度较之前无 CFG 的弱编辑提升约 100 倍（edit_distance 0.0145 -> 1.49）。
- 该强度下 `none` 前缀被完全带走（RMSE≈1.49），说明「语义翻转确实威胁前缀」，
  是一个真实的强编辑场景。
- 紧凑边界状态在该强编辑跨 DiT 场景下，误差随 rank 干净单调下降，并逼近完整
  保真：rank32 把前缀误差从 1.496 压到 8e-6（相对缩小 >99.999%）。

边界的部分（不足以声称「显著」的地方）：
- rank32 的 payload（131104 B）已约等于 full_oracle（131072 B）。因为 Latte latent
  很小（32×32×4×16 帧），前缀 latent 本身只有约 131KB，低秩表示的「字节紧凑」优势
  在此不成立——它提供的是「格式/计算」优势，而非「字节压缩」优势。
- 仍是在 latent 域度量前缀 RMSE（解码前的中间量），没有证明「语义上保持住了、
  编辑做成了」的 pixel 视觉结果。
- 只用了一个类别对 + 一个 seed，是单点机制证据，不是统计显著性。

## 下一步（真正冲「novel+significant」的两个缺口）

1. 把「强语义编辑 + 紧凑状态保持」和**现有方法**（如普通像素梯度整体保护、固定
   注入、单区投影）在**同一条视频/同一 latent 可能输出**上对比，证明我们的「选择性
   释放/保持」在「编辑达成 vs 保护质量」权衡上明确更优——目前机制成立但无对比基线，
   novelty 撑不起来。
2. 换至少 2 个类别对 + 2 个 seed 复现，确认单调逼近曲线不是单点运气；并在条件允许时
   用 SD VAE 做一次 pixel decode，确认「保持住」的是语义内容而非空 latent。

## 状态

机制跨架构（CogVideoX + Latte）成立；尚不做「达到顶会 novel+significant」声明。
记录保留失败/边界，不包装成方法成功。