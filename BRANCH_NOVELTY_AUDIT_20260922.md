# Branch consistency novelty audit

2026-09-22. This is a research gate, not a paper contribution claim.

## Findings grounded in retrieved primary content

Semantic Abduction (ICML2025, https://proceedings.mlr.press/v267/rasal25a.html; https://ar5iv.arxiv.org/html/2506.07883) uses a learned semantic variational posterior and spatial DDIM inversion, keeping semantic code fixed under interventions. A shared scene/identity code plus operation-conditioned diffusion is not by itself novel. Our video/occlusion scope differs from their evaluated image tasks, but an application change alone is insufficient.

Memory-V2V (https://arxiv.org/html/2601.16296; official https://dohunlee1.github.io/MemoryV2V/) explicitly treats previous outputs as consistency constraints, retrieves relevant edits, tokenizes and compresses memories. It evaluates multi-turn novel views and long text-guided video editing. We must not describe it as ordinary temporal context without cross-turn constraints.

Twin Rollouts (https://arxiv.org/html/2608.08982v1) formally shares the generated prefix and future exogenous noise across factual/counterfactual action streams. Primary HTML was downloaded and inspected. Its abstract explicitly states experiments are forthcoming; it is a framework note with a minimal illustration, not empirical SOTA. Self-generated rollouts have known exogenous draws; real observed video inversion is a different assumption. Still, 'noise-coupled counterfactual branching' cannot be claimed as first.

Coupled Diffusion Sampling (https://arxiv.org/html/2510.14981; https://coupled-diffusion.github.io/) concurrently runs a 2D editor and multi-view generator, with a coupling potential gradient. Primary equation7 and text explicitly describe tilting trajectories towards another variable. This is not the same construction as source-conditioned orthogonal noise transport, but 'joint sampling to improve consistency' is already covered. No reproduction performed here.

EDDY (https://arxiv.org/html/2605.06553v1) already studies altering particle dependence while preserving marginals using Fokker-Planck symmetries and antisymmetric fields; its objective is diversification. The primary text explicitly limits its exact guarantee under practical finite-difference/Hutchinson approximations. Switching repulsion to alignment cannot by itself be a contribution; cross-condition learned semantic transport and a cheaper exact construction would need actual new evidence.

Neural Diffusion Processes (ICML2023, https://arxiv.org/abs/2206.03992) learns distributions over functions through finite marginals with exchangeability-aware attention. Arbitrary-query joint generation and consistency claims need this literature as well, rather than introducing them as a new mathematical object.

Adjacent alternatives are crowded too: Robust Dreamer (https://arxiv.org/html/2605.30855v1) explicitly trains with prediction-corrupted latent memory; ReMind (https://ar5iv.labs.arxiv.org/html/2605.25333) learns recovery from non-local reliable anchors and interrupted/noisy observations; RECAP-Forcing (https://export.arxiv.org/abs/2608.26671) indexes memory by newly visible appearance. Generic robust memory/occlusion recovery is not a safe unclaimed pivot.

## Mathematical sanity check for a narrowed candidate

For each fixed source s and operation a, choose an orthogonal map Q(s,a) independent of the sampled z. If z~N(0,I), Qz has the same law. A deterministic frozen sampler G(s,a,Qz) therefore keeps its conditional output distribution exactly (ideal arithmetic), even if the sampler itself is approximate. Sharing z but changing Q across operations can change cross-operation dependence. Equal Q across all branches only relabels the shared noise and leaves their joint distribution unchanged.

These are elementary Gaussian symmetry facts, NOT new results. A sample-dependent Q(z) is NOT automatically measure preserving: in 1D Q(z)=sign(z) is pointwise orthogonal but Q(z)z=abs(z) is half-normal. No later implementation may invoke orthogonality alone if it chooses Q after inspecting the noise/output.

Finite-pool bijective reordering retains each branch's exact empirical sample multiset, independent of its matching cost. This is a standard assignment baseline, not a learned model or improved video. Preserving poor-quality marginals preserves poor quality too. Pool generation cost must be charged; it yields an ensemble of matched pairs, not a cheap single-pair service.

Minimizing cross-branch pixel differences can erase the requested difference. Valid agreement regions are determined by semantics/visibility, not merely shared image coordinates. Background under changed illumination, shadows and disocclusions can legitimately differ. The current outer-border proxy is an engineering screen, not a valid universal edit metric.

Real source videos need conditional anchoring. Known sampled diffusion noise is not proof of recovering physical exogenous variables; generic observational marginals do not identify cross-world coupling. Paired simulator or intervention data add assumptions/identification information and must be disclosed equally for baselines.

## Research decision before results

Do not train a new shared-memory or joint-output DiT merely because the phrases sound distinct. First evaluate same-seed, random, and fixed-pool assignment in the existing real-video generator. If same-seed already explains apparent improvement, stop the broad coupling variant. If standard assignment helps, only then investigate whether an amortized semantic transport can reproduce it on fresh seeds/unseen sources without pool search and without altering editing quality. Orthogonal Procrustes, diagonal signs, spatial permutations, shared-reference baselines, and an unrestricted equally supervised joint model are mandatory controls.

No method is yet demonstrated novel or significant. The accompanying GPU gate measures baseline headroom, not the performance of a proposed trained algorithm.
