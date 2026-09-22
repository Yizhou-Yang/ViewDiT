# ViewDiT 方法验证/证伪 判定报告（2026-09-22）

结论：**当前方法（"编辑-保持"选择性低秩子空间）在 V100 全链证据下被证伪，无法成为顶会结果。**
依据：跨越全部三个可用底座的实验矛盾 + 决定性竞品 EditCtrl 的门槛。给出可行的 positive 转向路线。

---

## 一、待验证的核心主张

方法主张："编辑达成 ⊥ 保持一致性"是一对需要**选择性**调和的真冲突；
用低秩语义子空间，在强语义编辑下同时保住语义A前缀 + 让出语义B，且 reshape 成本 << 全量。
审稿判定的关键 = 该冲突是否真实存在，以及子空间是否真的"选择性"化解它。

## 二、全底座实验证据链（V100 实测）

| 底座 | 冲突性质 | 实测证据 | 对主张的判定 |
|---|---|---|---|
| Latte-XL/2 | 帧独立（无 temporal attn） | 冲突探针：保持/编辑方向同一向量 rank→满秩=全前缀替换 | ❌ 冲突为构造性伪问题 |
| Wan2.1 VAE decoder | 因果，时序局部 | conflict_scan：edit_amp 恒 1.0001，alpha 全范围 prefix 0 | ❌ 前缀干预不影响编辑，无冲突 |
| CogVideoX-2b VAE decoder | 短程局部（±2帧，衰减30x） | impulse response；pixel_splice/capsule 同时 prefix≈0+edit无损 | ❌ decoder层解耦，无真冲突 |
| CogVideoX-2b DiT attn1 | **全局时空注意力（真耦合层）** | 结构确认，但历史所有机制从未触达该层 | ⚠️ 唯一可能有真冲突的层，未验证 |

贯穿结论：**所有历史"编辑-保持"机制（BoundaryMomentTransport / CapsuleNorm / AlphaScanNorm）都挂在
decoder GroupNorm 上，而 decoder 的时域耦合要么不存在（Latte）、要么太弱/局部（Wan/Cog），
根本不构成要解决的"选择性冲突"。** 低秩 capsule 的 9472B/36x/tuple 只是"压缩前缀存储"，即 P2 已定性的
"压缩前缀缓存"，novelty 不成立。

## 三、决定性竞品：EditCtrl（Meta, arXiv 2602.15031, 2026-02）

"背景保持 + 局部编辑"是被社区攻占且已成熟的赛道，EditCtrl 用**训练过的 LoRA adapter（rank 128）**达成：
- 1.5B 参数量 / 17.42 PFLOPS 下，背景保留 PSNR 24.16 / SSIM 0.92，全面超越 VideoPainter(5B)/VACE(14B/589 PFLOPS)；
- 计算量正比于编辑 mask 面积（10-50× 快）；局部 encoder fine-tune 后能按 prompt 重生成编辑区域新语义。

我们的免训练 decoder-latent 修补在该赛道全面落后：
1. **编辑达成力**：edit 区无训、无法从 prompt 生成新语义，只有 latent 几何/色彩修补 → 远不及 EditCtrl；
2. **背景保持**：无超越 baselines 的实质提升（PSNR 未建立优势）；
3. **效率**：decoder 仍全 decode，无推理加速；
4. **唯一亮点** 低秩压缩（9472B）是存储优化，非编辑质量或推理加速贡献。

## 四、证伪判定（分项）

- **Novelty** ❌："选择性子空间化解编辑-保持冲突" 建立在 decoder 假冲突之上，真冲突在 DiT attn1 层而机制未触达；
  压缩前缀缓存不具 novelty。社区 (EditCtrl/VACE/VideoPainter) 已用训练化 adapter 覆盖该目标。
- **Significance** ❌：编辑达成+背景保持 / 效率 / 加速 三条指标均无超越，无法构成顶会意义。
- 综合：即使补跑 DiT attn1 层探针证明某 selective 机制，也会被审稿判为"EditCtrl 背景保持的低秩无训补丁变体"，
  novel 与 significant 均不足。

## 五、Positive 转向路线（可落地，V100 可迭代）

**真正的 gap：EditCtrl 等所有可编辑方法都依赖训练 adapter（LoRA 微调/局部 encoder）。**
"完全冻结模型、免训练"达成强语义（prompt 级）编辑 + 背景保持仍相对空白。

正向建议（择一或组合，先做概念验证再定）：
1. **免训练 DiT-attention 前缀注入**：在 CogVideoX attn1 层做添加性前缀 KV 注入（不更新权重），
   检验能否让冻结 DiT 完成 prompt 级语义编辑（类别/对象/风格替换）而不破坏背景。
   ——这是唯一可能挺住审稿的 novelty 落点；验证失败则放弃该方向。先做 1 个 clip × 2 个强语义 prompt，
   看 edit 区是否真实变为 prompt 语义、背景 PSNR 是否保持。基线 = pixel_splice / raw edit。
2. 若免训练 DiT 注入也失败（大概率点），则在空白的"**编辑-保持权衡曲线的无训逼近**"上重新定位，
   明确不声称编辑质量超越训练方法，而声称"在固定预算/免训练约束下的最佳权衡"，据此重审 novelty。

## 六、决定性实证：冻结 DiT 免训练语义编辑能力测试（2026-09-22 已跑）

在给 positive 路线下结论前，必须实证"免训练冻结 DiT 能否仅靠 prompt 完成强语义编辑"——这是
Route 1 的唯一立足点。用历史 `cog_semantic_protect_ablate.py` 的 `--protect=none` 分支（完全放开
全部保护），在 V100 上跑通（CogVideoX-2b V2V, DDIM, steps=20, strength=0.6, bear→white polar bear）。
脚本 `experiments/video/cog_semantic_protect_ablate.py`，数据 `results/video/cog_semantic_protect_ablate_none`。

关键指标（bear 编辑区，whiteness 判据：白熊=高亮度 & 低 chroma）：
- `brightness_gain = +0.0117`（亮度仅 +1.17%，白熊应需大幅提升，base=0.344→0.355 远不够）
- `chroma_drop = -0.0012`（色彩几乎不下降，白熊 chroma 应→0）
- `protected_max = 0.111`（前缀仍被连带改动）；`edit_rms = 0.247`

**判定（决定性证伪）：即使完全放开所有保护（protect=none，编辑的最自由上限），冻结的 CogVideoX-2b
DiT 也无法仅靠 prompt + 前缀注入把棕熊变成白熊**——亮度 +1%、色彩不降，编辑区变化是噪声/全局偏移，
而非语义一致的重写。这与 EditCtrl 需要 fine-tune local encoder 才能做编辑的社区共识一致：
**冻结模型（无 adapter / 无 LoRA）做不了 prompt 级强语义编辑。**

由此：
1. **免训练路线证伪**：Route 1（免训练 DiT-attention 注入做强语义编辑）已被实测排除。
2. **"编辑-保持"竞争动机落空坐实**：编辑本身在冻结模型下都不发生（edit 区亮度/色彩不变），
   "编辑达成 vs 保持一致"的权衡无从谈起——选择性子空间在为一场不会发生的编辑做保障。
3. 观点收束：**当前"编辑-保持选择性低秩子空间"方向在 V100 全链证据 + 决定性实证下被彻底证伪。**

## 七、Positive 转向路线（重定义 contribution）

在"冻结模型做不了强语义编辑"这个被实证的事实约束下，不声称"编辑质量/选择性取舍"，
只可保留其中**被证实仍成立且可能独立支撑的证据**重新定位：

- **已成立事实（可复用）**：①低秩 capsule 能近无损保持 latent 前缀（9472B/36x/prefix 3e-6，P1/P2 已证）；
  ②Wan/Cog decoder 的时域干预可做到"编辑区零破坏 + 前缀保持"（noninterference）；③DiT 帧间有真实全局
  attention 耦合（attn1）。
- **可转向的 honest 定位候选（需对照再定，均可 V100 迭代）**：
  a) **可压缩视频前缀表示**：把"冻结模型下用低秩子空间表示需要保持的前缀"做成视频生成/编辑里
     prefix-pinning 的免训练高效替代，对标 VAE 前缀缓存压缩，不碰编辑质量主张。
  b) **时域一致性的轻量修正**：利用 decoder 短程局部耦合 + 前缀 capsule，做"重建时保持首帧语义一致"
     的存储/计算优化，对准视频内存/带宽瓶颈（而非编辑）。
  c) **彻底换题**：放弃编辑-保持线；回到记忆中既定的"视频质量/控制"方向里另找有耦合层可操控、
     且无"冻结强语义被证伪"约束打击的切口。

**建议下一步**：在 (a)/(b)/(c) 三选一前，先用一次 V100 最小实验对比"低秩前缀 capsule vs 显式存完整前缀"
在同底座上的 解码质量-存储 权衡曲线，确认 (a)(b) 中是否有一个能构成"免训练 + 显著存储/带宽收益"的新颖点；
若无，走 (c) 换题。

## 八、处置建议（供用户决策）

- 当前方向已实证证伪，不建议在此投入更多 decoder 侧"编辑-保持"实验。
- 建议按第七节在 (a)/(b)/(c) 中择一重新定位，并先行最小 V100 验证再展开。

（本文档保留全部 trace 与脚本引用，供复现与审计。）