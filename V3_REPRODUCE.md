# 第三轮复现与状态说明

## 环境

V100-SXM2-32GB，torch2.6.0+cu124、diffusers0.31.0、Python3.10。未升级既有torch/diffusers等依赖，未训练或改变模型权重。后续确认缺少SentencePiece后，单独安装sentencepiece0.2.1至研究目录/root/viewdit/env_extra/cog_semantic，以进程级PYTHONPATH加载，不改全局环境。

本轮输入源码和统计位于本地工作区；远端脚本位于`/root/viewdit/experiments/video`，结果位于`/root/viewdit/results/video`。

## 已实现流程

- `cog_boundary_moment_transport.py`：全局参考、双参考、渐变双参考，4内容×2幅度×3方法，24条。
- `cog_online_boundary_norm.py`：单次编辑decode的参考/实时统计分区，8条开发结果。
- `cog_transport_validation.py`：8条新内容、33帧、2类扰动、2方法，32条独立验证。
- `cog_context_capsule.py`：224×128下边界统计/旧latent/像素残差存储比较，88条。
- `cog_context_capsule480.py`：480×720下同类比较，44条开发结果。
- `cog_capsule_compositor.py`：接受像素拼接作为输出，比较9种参考表示、2内容，18条。
- `cog_exact_memory.py`：删除时间核1卷积未使用缓存，2分辨率对照；编码和解码逐元素相同。
- `cog_semantic_noninterference.py`：完整CogVideoX2b文本条件编辑与两轮latent复用，是否完成以stats为准，不以脚本存在为准。

## 数据下载

`fetch_v3_validation.py`从公开DAVIS-Edit目录固定下载八条新内容前33帧、四条语义开发内容前49帧，共460JPEG。每条在manifest中记录SHA256。研究机目标目录`/root/viewdit/data/v3_validation_data`。

macOS归档可能包含`._`资源文件，读取脚本显式忽略点开头文件。文件数审计应依manifest，不能把隐藏元数据算作新样本。验证器会逐个验证manifest哈希。

## 官方模型

`fetch_cog_semantic_local.py`通过HuggingFace官方站下载CogVideoX2b完整权重和配置，校验官方LFS SHA256后保存verified_manifest。远端镜像403已记录，未绕过鉴权。

模型位于本地`models/CogVideoX-2b`及远端`weights/CogVideoX-2b`，不放入研究压缩包。远端语义队列再次按文件大小与哈希核验，避免清单先到但权重仍未传完时启动。

下载工具只负责研究输入，不能用其耗时当推理性能。大文件传输使用SFTP，未修改系统网络设置。

## 顺序运行

已有输出先备份或更改脚本OUT，以下命令会写入同名结果。GPU单任务串行，避免影响计时。

```bash
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_boundary_moment_transport.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_online_boundary_norm.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_transport_validation.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_context_capsule.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/miniconda3/bin/python /root/viewdit/experiments/video/cog_context_capsule480.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/miniconda3/bin/python /root/viewdit/experiments/video/cog_capsule_compositor.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/miniconda3/bin/python /root/viewdit/experiments/video/cog_exact_memory.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_semantic_noninterference.py --clips bear --frames 17 --steps 4 --strength 1 --rounds 1 --out /root/viewdit/results/video/cog_semantic_smoke
/root/miniconda3/bin/python /root/viewdit/experiments/video/cog_semantic_noninterference.py --clips bear --frames 49 --steps 50 --strength .6 --rounds 2 --out /root/viewdit/results/video/cog_semantic_bear49
```

语义模型采用FP32 T5生成embedding后卸载；FP16 DiT；FP32 VAE；DiT在VAE阶段卸载到CPU。显存改进只移除无用缓存并采用进程级allocator，不改变数学操作，已验证两分辨率逐元素一致。进程级配置不会永久改变系统。

## 时间和误差口径

参考状态payload仅计数组字节，不含共享模型/固定schema。真实系统还需shape、分批、层序等版本化配置。

Capsule合成恢复开销不包含原始edited decode、参考统计首次捕获；与旧latent重建基线比较同一恢复阶段。不能将其9.9秒误称总渲染时延，也不能将存储优势误称显存峰值优势。

双参考精度不可与单次版本成本混用。FP32统计精度不可与FP16统计大小混用。像素残差可完全保留编辑区，但必须针对每个候选重新生成；原latent参考和统计状态可复用于同一保护历史的未来替换。

压缩是简单zlib/低秩/量化基线，不代表最强学习式codec。尚无全局最优存储或信息论最小状态证明。

所有统计只在complete=true后进入正式汇总。运行中和失败组应单列。语义smoke不能用于语义质量结论。

## 失败审计

1. 公开镜像403，改本机官方源下载并校验。
2. 清单先传到造成缺失JPEG，等待条件改为全部文件大小匹配，实验内再核验哈希。
3. macOS归档资源文件误作JPEG，改为排除隐藏文件。
4. 480P FP32解码显存不足，进程级expandable_segments重试完成；另验证移除无用缓存获得更稳定显存空间。
5. 秩1矩阵字节view步长不合法，改NumPy按C顺序序列化，不改变payload字节定义。

这些都是实现/资源问题，不作为方法失败或成功的证据。原始失败日志保留在远端logs目录。

补充：真实语义命令需设置`PYTHONPATH=/root/viewdit/env_extra/cog_semantic`。初次T5缺少SentencePiece报错已保留。smoke中的capsule_compositor额外decode时延漏计为0，不能引用为零成本；后续源码已修正，若正在运行版本仍输出0则统一作为missing处理。

## 最终补测与紧凑文件

高强度语义补测已完成，使用相同命令改为`--strength .85 --rounds 1 --skip-roundtrip --out /root/viewdit/results/video/cog_semantic_bear49_strong`。实际42次有效去噪，跳过的往返指标为null。`cog_semantic_capsule_restore.py`验证真实生成输入上保存重载和前缀裁剪恢复，`capsule_codec.py`将状态保存为含版本、布局和哈希的VCAP格式。实际文件9450字节，统计精确往返。所有主要组共242条配置完成，另有2条codec验证。
