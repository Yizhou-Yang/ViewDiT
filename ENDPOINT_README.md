# Endpoint实验复现说明

这些脚本是现有ViewDiT/Latte项目的实验扩展，不是独立模型实现。需要已安装的Latte源码、ffs.pt和stabilityai/sd-vae-ft-ema本地权重；包内不分发权重。

## 实际验证环境

- GPU：Tesla V100-SXM2-32GB，SM70。
- Python执行器：/root/miniconda3/bin/python。
- torch 2.6.0+cu124，torchvision 0.21.0+cu124。
- diffusers 0.31.0，transformers 4.45.2，accelerate 1.0.1。
- numpy 2.2.6，Pillow 12.3.0，einops 0.8.2，timm 0.9.16。
- 依赖项目模块：run_xl_gate.py、tis_review.py及third_party/Latte。
- 不需要重新训练，未修改模型权重。

## 运行

将四个endpoint脚本放到同一个experiments/video目录，沿用现有项目依赖。不要直接覆盖已经有改动的同名文件。

```bash
CONTROL_N=2 CONTROL_SEED0=20 CONTROL_START=175 LATTE_STEPS=250 /root/miniconda3/bin/python /root/viewdit/experiments/video/endpoint_control.py
CONTROL_N=2 CONTROL_SEED0=30 CONTROL_START=175 LATTE_STEPS=250 /root/miniconda3/bin/python /root/viewdit/experiments/video/endpoint_native.py
CONTROL_N=4 CONTROL_SEED0=40 CONTROL_START=175 LATTE_STEPS=250 /root/miniconda3/bin/python /root/viewdit/experiments/video/endpoint_refine.py
CONTROL_START=175 LATTE_STEPS=250 /root/miniconda3/bin/python /root/viewdit/experiments/video/endpoint_validate.py
```

按顺序运行，避免抢占同一GPU。validate的种子50至53、8次反馈预算及两种新时间曲线固定在脚本中。四种脚本默认输出到/root/viewdit/results/video下各自的endpoint目录。control/native/refine支持脚本中声明的输出环境变量；validate路径固定在OUT常量。

stats.json会增量写入，只有complete=true表示该组实验完成。下载后按endpoint_control_stats.json、endpoint_native_stats.json、endpoint_refine_stats.json、endpoint_validate_stats.json命名，和summarize_endpoint.py放在同一目录。

```bash
python3 summarize_endpoint.py
```

## 看图

refine GIF从左到右：base、hard_direct、single_step_feedback、endpoint_feedback、target_composite。

validate GIF有英文列标题：base、hard_direct、fixed_feedback、safeguarded_feedback、target_composite。seed51_storyboard依次抽取0-based第3、4、10、11帧，覆盖双脉冲。

## 计费口径

每条尾段75次模型前向。两个validate反馈方法均在hard_direct初始结果上额外8条尾段，即600NFE，拒绝候选也计费。单独部署完整反馈流程：base250 + native75 + initial75 + feedback600 = 1000NFE，不含身份复现检查的额外75NFE。若base轨迹已有可复用则新增750NFE。所有方法对照同时运行的整组脚本另有重复计算和VAE解码开销，不能用单个反馈耗时替代全流程时间。

## 结果解释边界

这是合成latent目标，不是语义视频编辑。target_composite能直接达到零目标误差，因此不是已战胜合成基线。像素MSE使用解码图像[-1,1]数值范围。保留区零误差来自逐步回填与本模型逐帧2D VAE，不能自动推广到时域VAE。Anderson组合和接受/拒绝是标准数值技术，尚未通过单独消融拆分两者贡献。四个新种子不足以证明开放域统计显著性。
