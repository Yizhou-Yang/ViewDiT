# Real-video branch coupling gate

## Reproduce

Existing remote research environment: Python3.10, torch2.6.0+cu124, diffusers0.31.0, transformers4.45.2, scipy1.14.1, numpy and Pillow. SentencePiece is already installed in the project-specific `/root/viewdit/env_extra/cog_semantic`; no pip installation was performed. The actual interpreter is Python3.10, not the Python3.12 stated in an earlier unrelated pilot README.

Place `branch_coupling_gate.py` and `cog_exact_memory.py` in `/root/viewdit/experiments/video`. Use existing model `/root/viewdit/weights/CogVideoX-2b` and DAVIS frames `/root/viewdit/data/v3_validation_data`. The helper removes only unused time-kernel1 VAE caches, leaving arithmetic unchanged; no normalization/statistics editing method is applied.

```bash
PYTHONPATH=/root/viewdit/env_extra/cog_semantic /root/miniconda3/bin/python /root/viewdit/experiments/video/branch_coupling_gate.py --out /root/viewdit/results/video/branch_coupling_gate_reproduce
```

Choose a nonexistent output directory. Default execution generates 64 videos on CUDA; it is not a dry run. Text encoding is FP32; VAE encoding/decoding FP32; DiT FP16. CPU/GPU phase transfers keep only the active large module resident. Four clips, two manually specified target captions each, eight fixed seeds. 17 frames at 128x224 is substantially below native quality and is only a screening configuration. No model training or downloads occur in this script.

## Protocol and interpretation

See `BRANCH_COUPLING_PROTOCOL_20260922.md`, written before generation. Assignment is standard Hungarian matching, using decoded even-frame outer-border MSE. Odd-frame error is reported separately. Border is NOT a foreground segmentation mask; correlated odd/even frames are NOT independent test scenes. The same exact outputs are used for all pairings, so quality and diversity within each branch are unchanged by construction. Pairing never makes a failed edit correct. Finite-pool coupling is an ensemble baseline, not a free single-pair sampling method.

Outputs: stats.json, embeddings.pt, per-clip latent pools, decoded video pools and fixed first-seed contact sheets. All seeds are retained without quality filtering. No hidden test target, geometry or masks are passed to the generator. The pipeline accepts provided noised latents without a second add_noise, verified against installed prepare_latents source. DDIM eta0 is deterministic conditional on initial noise and prompt.

## Initial setup failures

First attempt stopped before generation because project SentencePiece was not on PYTHONPATH. Second stopped before generation because macOS AppleDouble `._*.jpg` resource files were included in the image enumeration. Fixed by using the existing dependency directory and excluding dot-prefixed files. Both failed directories were preserved; successful/restarted execution uses `branch_coupling_gate_20260922_run3`.

Do not describe these setup failures as method failures. Do not conflate results of this untrained V2V generator with a native instruction-trained editor such as Lucy Edit.

## Native spatial-resolution verification

`branch_coupling_native_gate.py` completed 8 outputs (bear/bus, two prompts, two seeds, 17 frames, 480x720) in `/root/viewdit/results/video/branch_coupling_native_gate_20260922_run2`. The first native attempt failed before sampling because automatic pipeline device inference selected CPU while the transformer was on CUDA. Both local gate scripts now explicitly return `self.transformer.device` from `_execution_device`. The low-resolution script was patched AFTER its completed run; its new hash does not describe old outputs. Its remote copy has not been overwritten by this local fix.

Native decoded outputs are finite and recognizable, unlike the low-resolution animal samples, but source preservation and instruction fidelity remain inadequate. Bear scenes alter background and pose; one panda retains brown fur. Bus color changes are partial, and the second seed barely distinguishes red/yellow. Both native assignments equal same-seed pairing, with zero additional benefit. This is only a two-scene development screen, not a statistical or universal negative result.

## Strength diagnostic and OOM recovery

`branch_strength_diagnostic.py` reuses native bus source latents and embeddings. It fixes seed20261001, CFG6, 50 configured steps and DDIM eta0, and samples red/yellow at strengths0.35/0.50. Strength0.75 outputs are reused as references. Four samplings completed; the original inline FP32 decode ran out of GPU memory. Saved `latents.pt` enables recovery WITHOUT re-generation:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/root/viewdit/env_extra/cog_semantic /root/miniconda3/bin/python /root/viewdit/experiments/video/branch_strength_decode.py --input /root/viewdit/results/video/branch_coupling_native_gate_20260922_run2 --out /root/viewdit/results/video/branch_strength_diagnostic_20260922
```

The recovery script uses a fresh FP32 VAE process and unchanged saved latents, no spatial tiling or decoder modification beyond the existing unused-cache patch. Read its final stats before claiming completion. A sweep on one known clip is development diagnosis, not held-out method evaluation.
