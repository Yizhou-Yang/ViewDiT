# Interaction mechanism pilot

## Scope

This is a small Conv3D video-editing mechanism test on analytically ray-traced sphere/plane clips. It is NOT a CogVideoX/Wan training result, a reproduction of FactorMatte/AlbedoEdit/Generative Omnimatte, or a real-video benchmark.

Training uses source RGB, two visible object masks and image/time coordinates. The fixed three output slots mean remove A, remove B and remove both. Hidden geometry and ground-truth target regions are not inputs. All three operation combinations are seen during training; test split only generalizes scene parameters. It cannot establish compositional generalization or asset-family generalization.

Renderer uses actual 3D ray/sphere intersections and light visibility, Lambertian shading, four fixed directional lights, fixed orthographic camera, analytic plane texture and prescribed short trajectories. There is no indirect illumination, mirror transport, dynamics, anti-aliasing or collision resolution. Some randomized spheres may intersect. Do not describe this dataset as production-quality or complete physical counterfactual data. Interactions are deliberately simple and source-region effect masks are evaluation/auxiliary-training definitions, not learned semantic segmentations.

## Environment

Executed on Tesla V100-SXM2-32GB with Python 3.12, torch 2.6.0+cu124. Core scripts require torch and numpy; visualization requires Pillow. Reuse the existing research environment; no new dependency installation was used.

## Reproduce

Run from a directory containing the three Python scripts with the existing CUDA Python:

```bash
/root/miniconda3/bin/python /root/viewdit/experiments/video/interaction_gpu_pilot.py --out /root/viewdit/results/video/interaction_gpu_pilot_reproduce --steps 600
/root/miniconda3/bin/python /root/viewdit/experiments/video/interaction_gpu_pilot_v2.py --out /root/viewdit/results/video/interaction_gpu_pilot_v2_reproduce --steps 1800
/root/miniconda3/bin/python /root/viewdit/experiments/video/interaction_pilot_inspect.py --root /root/viewdit/results/video/interaction_gpu_pilot_reproduce --test-offset 200000
/root/miniconda3/bin/python /root/viewdit/experiments/video/interaction_pilot_inspect.py --root /root/viewdit/results/video/interaction_gpu_pilot_v2_reproduce --test-offset 300000
```

Scripts refuse an output directory already containing stats.json. All runs use 200 train scenes, 40 validation scenes, 80 test scenes, 5 frames at 48x64, batch 4, 3 initialization seeds and two learning rates. Checkpoint and learning-rate selection use validation changed-region MSE only. Test targets are never used in training or checkpoint selection.

V1 trains full-video paired MSE for 600 steps per configuration. V2 trains for 1800 steps per configuration, adding equal normalized changed/effects/keep region losses plus 0.1 interaction loss to BOTH models. V2 tests on new seeds starting at 300000 after V1's test set has been inspected. The test distribution itself remains the same sphere family; this is not external replication.

Both models have exactly 161097 trainable parameters and share the same backbone and initial zero output. Direct predicts (rA,rB,rBoth); interaction predicts (rA,rB,rAB) and outputs rBoth=rA+rB+rAB. Both emit all three edits in one pass. A linear output head makes these function classes equivalent; independently optimizing them with Adam is not invariant to this coordinate change.

## Outputs

Each remote directory contains stats.json, six selected model checkpoints, source/target examples and predictions. Inspection writes inspection.json and a contact sheet of the FIRST four test scenes (seed 11, middle frame), without selecting examples by quality.

Metrics are unclamped linear RGB MSE. Contact sheets clamp only for display. Effects are changed pixels on visible source floor; keep pixels are those with max RGB target-source change <=0.005. Temporal metric is adjacent-frame difference error, not perceptual flow consistency.

Bootstrap resamples paired training seeds and paired test scenes, 2000 repetitions. Three seeds are few; intervals are exploratory, not population guarantees. Multiple metrics are reported without familywise correction. A relative reduction below zero means the interaction model is worse.

## Reparameterization audit

For any trained interaction head, set WBoth=WA+WB+WAB and bBoth=bA+bB+bAB in the direct head, keeping the backbone unchanged. This converts the trained function without optimization. Conversely subtract the single heads from the direct both head. The inspection tests the converted function on every test scene.

Failure or equivalence of this linear binary prototype does not refute nonlinear shared entity modules, richer conditional architectures, diffusion priors, or the entire interaction-editing problem. Success of this pilot would also not establish top-conference novelty or significance.
