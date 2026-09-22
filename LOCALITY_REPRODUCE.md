# 本轮实验复现

日期：2026-09-21。代码依赖现有ViewDiT/Latte环境；Wan实验仅用独立VAE，不要求升级diffusers或安装Wan完整流水线。

## 文件与作用

- endpoint_stress.py：旧方案的拒绝/缩步长消融、三分区、轨迹pinning与像素/latent合成基线。
- wan_vae_official.py：来自Wan2.1官方仓库wan/modules/vae.py的未修改源码，保留原版权声明。
- wan_state_probe.py：官方解码复现、时域脉冲响应、首层/全状态恢复与像素拼接等价检查。
- wan_tail_compensate.py：生成视频上的因果后缀补偿，24次Adam，只改后续latent。
- wan_real_pilot.py：四个真实视频、两类局部编辑，固定24步补偿，保存全量案例和视频。
- wan_baseline_audit.py：同迭代数全latent优化、后缀重编码热启动、一次后续几何latent操作。
- summarize_locality.py：读取下载后的*_stats.json生成locality_summary.json，仅Python标准库。

## 环境

V100-SXM2-32GB；torch2.6.0+cu124；diffusers0.31.0；einops0.8.2；numpy2.2.6；Pillow12.3.0。Wan VAE使用FP32，无BF16或FlashAttention要求。脚本放在/root/viewdit/experiments/video，Python为/root/miniconda3/bin/python。

Latte首次启动因HF_HOME缺失而找不到已有VAE缓存，已通过单次命令设置HF_HOME=/root/viewdit/hf解决，没有重下Latte模型或改全局配置。

## 官方VAE与数据

权重：Wan-AI/Wan2.1-T2V-1.3B仓库的Wan2.1_VAE.pth，507609880字节。远端下载经hf-mirror传输，SHA256已与Hugging Face官方API的LFS摘要核对一致。默认路径/root/viewdit/weights/WanVAE/Wan2.1_VAE.pth。

源代码：https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py

权重：https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B

真实数据：https://huggingface.co/datasets/AlonzoLeeeooo/DAVIS-Edit

使用原始JPEGImages中的bear、blackswan、car-roundabout、camel，每个按文件名取前17帧，共68帧。目录接口在远端镜像返回403，实际改用本地官方站下载并逐文件校验SHA256后上传。没有绕过登录或权限限制。数据放到/root/viewdit/data/davis_locality_pilot/JPEGImages/<clip>。fetch_davis_pilot.py记录原下载尝试，但该镜像API当时不可用；可靠复现应从官方站按index.json下载并校验，或使用已有数据。

真实片段强制resize为224×128，保留大致16:9但不保持原分辨率；这是低分辨率机制pilot，不是正式高保真评测。

## 按序运行

```bash
HF_HOME=/root/viewdit/hf CONTROL_START=175 LATTE_STEPS=250 /root/miniconda3/bin/python /root/viewdit/experiments/video/endpoint_stress.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/wan_state_probe.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/wan_tail_compensate.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/wan_real_pilot.py
/root/miniconda3/bin/python /root/viewdit/experiments/video/wan_baseline_audit.py
```

state_probe需要此前endpoint_refine/seed40至43.pt；tail_compensate需要state_probe输出；baseline_audit需要real_pilot全部输出。所有脚本写入同名结果目录；complete=true才算该组完成。

## 指标解释

- 所有图像数值范围[-1,1]。保护目标是原视频经过同一VAE后的重建，不是未经压缩原始视频；原输入重建MSE单独记录。
- 几何编辑是latent横向差分；颜色编辑是原始视频矩形红通道变化再编码，之后只保留第2个latent的修改。不是文本编辑，也不是对象分割精准编辑。
- 17帧压缩为5个latent时刻；第2个latent的名义输出帧为0-based5至8；优化第3、4个latent恢复9至16帧。
- 核心区域保持来自因果解码结构，已测最大差为0；不能外推到任意双向VAE或跨压缩块的边界。
- 24步Adam即24次后缀前向与反向，不是24次DiT NFE。耗时还包括脚本末尾的验证解码，不包含首次VAE加载、源视频编码与所有参考缓存成本。
- 当前曲线保存每次更新前loss，报告最终结果另做解码，未按最优迭代选样。
- 全latent基线同为24步，但计算量和运行时不同，不能称等时延对比。
- full_state_reset实测等价于pixel_splice；这条基线不能被当作我们的独有贡献。
- 后续操作检查只是再施加小幅末尾latent扰动，不能叫做已验证完整DiT再编辑。

## 下载包范围

包含本轮脚本、报告、原始统计、少量GIF和索引，不包含模型权重、原始视频全数据或大型PT张量。官方代码遵循原仓库许可证；数据使用需遵循DAVIS及DAVIS-Edit来源条款。
