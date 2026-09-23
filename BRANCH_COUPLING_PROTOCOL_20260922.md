# Branch coupling gate, frozen before generation

2026-09-22. Purpose: test whether a new learned coupling is warranted before inventing a shared-memory architecture. No claim of novelty, trained editing capability or publication readiness.

## Hypothesis

Changing the dependence between edited video samples can improve cross-branch agreement without changing the empirical per-branch sample distribution. A trivial finite-pool assignment already has this property. Any later marginal-preserving learned coupling must beat identical seeds and justify itself against finite-pool optimal matching, including sampling cost.

## Fixed real-model experiment

Existing CogVideoX-2b, V100 32GB. FP32 T5 and VAE, FP16 DiT, mode encode real source clips, 17 frames 128x224, 50 configured DDIM steps, strength .75 (37 effective), CFG6, eta0. NO protected latent prefix, no altered decoder, no model updates. Four prior development clips bear/blackswan/bus/boat; not independent newly collected videos. Eight fixed seeds per branch, 20261001..20261008. Two prompts per clip differing in object appearance, not global lighting. 64 generated clips in total. Small resolution and short clips are an engineering diagnostic, not native model quality.

Branch prompts: polar bear vs panda; white swan vs pink swan; red bus vs yellow bus; red sailboat vs yellow sailboat. These are manually specified target descriptions, not learned instruction editing.

## Couplings, same exact output multiset

1. Identical seed index.
2. Uniformly random permutation, evaluated exactly by the all-pairs mean and by all 8! permutations for p-value diagnostics.
3. Hungarian matching on even-numbered frames, outer image border of 20% on all sides, decoded RGB MSE.
4. Hungarian matching with a different assignment per EVEN frame; evaluate temporal-switching risk separately if implemented. Not required for this gate.

Main score: MSE between matched branches on ODD frames, same outer border. Assignment never sees odd frames. This is temporal holdout within the SAME video, not independent test scenes, and the border is only a proxy for background. No foreground mask or correspondence oracle. Object entering border and legitimate appearance effects can invalidate the proxy. Report full-frame and central-region MSE separately; lower whole-frame difference does not imply better editing.

Controls: save the exact sampled tensors, verify every coupling uses a bijection; all empirical marginal statistics and quality are therefore unchanged by construction. Report within-branch diversity. Random/permutation baseline must be evaluated on the same pool, no cherry-picking. A coupling can preserve quality without creating quality; if the underlying samples fail the edit, the coupling is NOT a successful editor.

## Evidence gate

Descriptive only: four clips and correlated frames cannot establish population significance. Report per-clip relative reduction vs both identical-seed and random couplings, seed permutation p-values (not independent scene p-values), plus fixed contact sheets showing the first seed without quality selection. Use fixed frames 0,8,16 and show original, VAE reconstruction, and both branches. No post-hoc relabeling of colour histogram as semantic editing success.

If identical seeds already dominate assignment, shared-seed coupling is a strong baseline and no more elaborate coupling is justified here. If assignment helps, it establishes only that a standard optimal-matching baseline exploits pool-level dependence. It is NOT evidence that a novel learned coupling has been discovered. If edits/geometry fail visually, this gate is inconclusive for valid editing and must not motivate larger training from consistency numbers alone.

## Candidate positive route, not yet validated

Let fixed source/operation-conditioned Q_a satisfy Q_a Q_a^T=I and z~N(0,I), independent of the source conditioning. Sampling x_a=G(source,a,Q_a z) preserves each conditional marginal of a deterministic generator and changes cross-branch coupling. Q_a must NOT depend on sampled z unless that nonlinear map is separately proved measure preserving. The Gaussian invariance is standard, not novelty. Learned semantic, visibility-aware alignment is the unknown. It needs matched-data supervision, realistic spatial correspondence, strong same-seed/post-matching baselines, and unseen scenes/operations. A shared Q independent of a merely relabels the common noise and cannot change joint distribution. Diagonal signs/permutations and Procrustes are mandatory simple baselines.

An arbitrary joint model and Semantic Abduction are further baselines. Marginal fidelity is a population property, not a guarantee that each edited sample is correct or paired to the source's actual hidden scene. Under unobserved occlusion, ground-truth sample recovery is unidentifiable without additional evidence; evaluate a calibrated distribution, not unique hidden truth.
