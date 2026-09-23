# Native-resolution amendment

Written after inspecting the low-resolution gate, before seeing native-resolution outputs.

The 64-output 128x224 gate completed. All four optimal assignments equal identical-seed pairing. However fixed bear and swan sheets are catastrophically malformed; all 16 bear samples at frame8 also lack valid target animals. Hence these numbers do not support a working editor or a semantic-coupling hypothesis. No claim that low resolution alone is the confirmed root cause yet.

## Follow-up

Use 480x720, still17 frames, same50/37 DDIM steps, strength.75, CFG6, same prompts and first two seeds. Fixed bear and bus, not chosen for prior positive outcomes; selected to cover identity replacement and colour edit. Eight outputs total. Change spatial resolution without changing model, algorithm, prompts or first two seeds. A GPU RNG tensor changes size, so identical numeric seed does NOT imply identical coordinate-wise noise between resolutions. Only compare quality descriptively, not a paired pixel effect.

This is not full native49-frame validation; it tests whether native spatial size restores usable generation in the current pipeline. If unsuccessful, do not continue training a coupling model on this protocol. If successful, inspect both seeds and colour/identity changes before interpreting any cross-branch difference. With two seeds and two development scenes, no inferential significance claim is possible.

The first native attempt failed before generating any sample due to automatic pipeline execution-device selection when VAE is on CPU and DiT on GPU. Installed DiffusionPipeline.device iterates module names and returns the first module device, so relying on this mixed-device state is unsafe. The native subclass explicitly returns transformer.device from _execution_device; no numerical sampling formula changed. Failed native directory is retained; resumed run writes to branch_coupling_native_gate_20260922_run2. Low-resolution successful results are still complete and unaffected; that process evidently selected CUDA and produced all64 outputs.

## Resource facts, not deployment claims

Existing CUDA interpreter is Python3.10. GPU V100-SXM2-32GB. Approximately401GiB disk free,36GiB available system memory at check. Lucy-Edit-Dev official HF file tree contains34.182GB of safetensor files (20.002GB transformer,11.362GB text encoder,2.819GB VAE). These are file sizes, not GPU memory estimates. Official example uses BF16 and81 frames480x832, but V100 lacks native BF16. Requires newer LucyEditPipeline than installed diffusers0.31; do not overwrite current environment. Need independent project environment, FP32 text/VAE or verified stable alternative, FP16 DiT with explicit offload, and correctness smoke test. No Lucy weights were downloaded or model run in this gate. Official model card identifies a non-commercial license, not unrestricted commercial use.
