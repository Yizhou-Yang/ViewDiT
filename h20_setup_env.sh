#!/bin/bash
# Reproducible H20 pod setup for ViewDiT cache-corrector study. Logs in /data/viewdit/logs.
set -x
ROOT=/data/viewdit
PY=$ROOT/env/py/bin/python
echo "torch==2.6.0+cu124" > $ROOT/env/constraints.txt
$ROOT/env/py/bin/pip install -c $ROOT/env/constraints.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124 -i https://mirrors.tencent.com/pypi/simple/ \
  diffusers==0.31.0 transformers==4.45.2 accelerate==1.0.1 sentencepiece safetensors huggingface_hub==0.25.2 \
  imageio imageio-ffmpeg pillow numpy==1.26.4 scipy peft==0.13.2
$PY -c "import torch,diffusers,transformers;print(torch.__version__,diffusers.__version__,transformers.__version__,torch.cuda.device_count())"
$ROOT/env/py/bin/pip freeze > $ROOT/env/pip_freeze.txt
export HF_ENDPOINT=https://hf-mirror.com
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-large-patch14', local_dir='$ROOT/weights/clip-vit-large-patch14', allow_patterns=['*.json','*.txt','model.safetensors'])"
echo ENV_DONE
