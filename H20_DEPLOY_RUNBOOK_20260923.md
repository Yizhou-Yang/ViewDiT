# H20 部署与运行手册（2026-09-23，12 小时租期 pod）

## 机器

- Pod：`ssh -p 36000 root@28.39.86.81`，2× NVIDIA H20（约 96 GB），驱动 535.247.01，CUDA 12.1 toolkit，384 核。
- 本机公钥已写入 pod `/root/.ssh/authorized_keys`，免密登录。密码不入仓库和文档。
- Pod 不支持 sftp/scp，只能用 `ssh ... 'cat > file' < file` 上传；rsync 可用。
- Pod 无法直连 V100 或 dev4，所有备份都经本机中转。
- Pod 预计 12 小时后回收，`/data` 不应视为持久存储。

## 目录（pod）

- `/data/viewdit/env/py`：venv，Python 3.9.16，torch 2.6.0+cu124，diffusers 0.31.0，transformers 4.45.2，精确版本见 `env/pip_freeze.txt`。
- `/data/viewdit/weights/CogVideoX-2b`：hf-mirror 下载，`THUDM/CogVideoX-2b`。
- `/data/viewdit/weights/clip-vit-large-patch14`：CLIP 评分。
- `/data/viewdit/experiments`：`h20_cache_corrector.py`、`h20_pipeline.sh`、`h20_setup_env.sh`、`h20_analyze.py`。
- `/data/viewdit/logs`：全部 stdout 日志，`pipeline.log` 记录阶段进度。
- `/data/viewdit/results`：`fit_B*/`（W、fit_log、fit_summary）、`eval/`、`score/`（含 GIF）、`diag/`、`run_meta.jsonl`（脚本 sha256 + 参数）。
- `/data/viewdit/latents`：评测 latent，较大，不入 git。

## 备份

- 本机 `h20_sync_loop.sh` 每 10 分钟：pod `/data/viewdit`（排除 weights、env、>300 MB）→ 本地 `h20_runs/pod_mirror/`，再 → dev4(`any4`) `/root/viewdit_h20_backup/`。同步状态写入 `h20_runs/sync.log`。
- 小文件（脚本、jsonl、summary、报告）另行提交到 GitHub `origin/main`。latent 与 GIF 不入 git。

## 复现

```bash
python3 -m venv /data/viewdit/env/py
/data/viewdit/env/py/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
bash /data/viewdit/experiments/h20_setup_env.sh
HF_ENDPOINT=https://hf-mirror.com python3 -c "from huggingface_hub import snapshot_download; snapshot_download('THUDM/CogVideoX-2b', local_dir='/data/viewdit/weights/CogVideoX-2b')"
cd /data/viewdit/experiments
CUDA_VISIBLE_DEVICES=0 ../env/py/bin/python h20_cache_corrector.py embed
nohup setsid bash h20_pipeline.sh &
../env/py/bin/python h20_analyze.py /data/viewdit/results
```

各阶段可断点续跑：eval/score 会跳过 jsonl 中已完成的 `(prompt, seed, method)`。

## 注意

- `pkill -f <pattern>` 会同时匹配发起该命令的 SSH 会话，已经误中断过一次安装；应改用 PID。
- 首次安装 `peft` 时会把 torch 升到 2.8，因此使用 `env/constraints.txt` 锁定 torch 2.6。
