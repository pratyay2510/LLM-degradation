# Reading List — Inverse Problems for Vision-Language Pipelines

Assembled for the CVPR-track project on **answer-space inference under ill-posed image degradation**.
Ordered by reading priority, not by topic. Each entry says *why* it matters to us and what to extract.

Status legend: `[ ]` unread · `[~]` skimmed · `[x]` read & notes taken

---

## Tier 0 — Read first. The theoretical spine.

These four establish why a single restored image is the wrong object to hand a downstream model.

- `[ ]` **Looks Too Good To Be True: An Information-Theoretic Analysis of Hallucinations in Generative Restoration Models** — Chen et al., NeurIPS 2024
  https://arxiv.org/abs/2405.16475
  *The most important paper on this list.* Proves an uncertainty–perception tradeoff: as perceptual quality improves, error entropy grows; perfect perceptual quality costs at least 2x the problem's inherent uncertainty. Our Gap-2 conjecture is the semantic analogue of this result. Extract: their Rényi-divergence formulation, and the exact statement of "inherent uncertainty."

- `[ ]` **The Perception-Distortion Tradeoff** — Blau & Michaeli, CVPR 2018
  https://arxiv.org/abs/1711.06077
  The foundation. Needed to argue precisely that no single restored image can be both faithful and on the natural-image manifold. Extract: the formal tradeoff curve and the proof sketch.

- `[ ]` **The Perception-Robustness Tradeoff in Deterministic Image Restoration** — Ohayon et al.
  https://arxiv.org/pdf/2311.09253
  Deterministic restorers that achieve high perceptual quality must be badly conditioned (huge Lipschitz constants). Direct ammunition for "do not use a point estimate."

- `[ ]` **Posterior-Mean Rectified Flow (PMRF): Towards Minimum MSE Photo-Realistic Image Restoration** — Ohayon et al.
  https://arxiv.org/pdf/2410.00418
  The strongest current attempt to get both ends of the tradeoff *in pixel space*. Our closest philosophical competitor — we need a paragraph distinguishing us (they optimize the image; we sidestep the image and estimate the answer). Extract: their exact claim and its limits.

---

## Tier 1 — The application gap we are filling.

The downstream/VLM side of the story. Read for positioning and for what is already claimed.

- `[ ]` **Demystifying the Visual Quality Paradox in Multimodal Large Language Models** — 2025
  https://arxiv.org/abs/2506.15645
  Documents that MLLM accuracy can *improve* on images humans consider degraded. This is the phenomenon we explain mechanistically. Read their VQ-TTT fix closely: they tune the model, we fix the estimator. Extract: their degradation suite and benchmark choices (useful protocol precedent).

- `[ ]` **UniRestore: Unified Perceptual and Task-Oriented Image Restoration Using Diffusion Prior** — CVPR 2025
  https://arxiv.org/html/2501.13134
  Task-oriented restoration state of the art. Note carefully: conditions on a *task class* (cls/det/seg), not an instance-level query, and always emits a single image.

- `[ ]` **TaskTok: Delving into Task Tokens for Task-driven Image Restoration**
  https://arxiv.org/html/2606.26615v1
  Selectively restores task-relevant tokens. Same two limitations as UniRestore. Together these two define the baseline family we must beat.

- `[ ]` **Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?**
  https://arxiv.org/pdf/2606.08063
  Has a harmful/beneficial answer-flip taxonomy — closest existing thing to our AFR metric. Cite as a starting point; show it measures a single sample and therefore cannot see posterior-induced instability.

- `[ ]` **HalluGen: Synthesizing Realistic and Controllable Hallucinations for Evaluating Image Restoration**
  https://arxiv.org/pdf/2512.03345
  Restoration-hallucination benchmark with spatial annotations. Our impossibility argument (no image-space detector can work) is aimed at this class of work. Read to make sure we characterize it fairly.

- `[ ]` **GLYPH-SR: High-Quality SR and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion**
  https://openreview.net/forum?id=GxPtLwLSOL
  Scene-text SR — the highest query/null-space-alignment special case, where fabrication is most obvious. Good source of a killer qualitative figure.

- `[ ]` **Understanding Degradation with Vision Language Model (DU-VLM)** — 2026 *(in repo: `papers/application/degradation-in-vlm.pdf`)*
  https://arxiv.org/pdf/2602.04565
  VLM estimates degradation physics, then acts as a zero-shot controller for a frozen diffusion model. This is our operator estimator when we go blind, and the bridge to the follow-up project.

---

## Tier 2 — Uncertainty, conformal prediction, and the guarantee machinery.

Everything needed to turn "we have a posterior over answers" into "we have a coverage guarantee."

- `[ ]` **Task-Driven Uncertainty Quantification in Inverse Problems via Conformal Prediction** — ECCV 2024
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07734.pdf
  *Our direct methodological ancestor.* Conformal intervals on a downstream task output computed from posterior samples. Limitation we extend: scalar/closed-set outputs only, not open-ended generation.

- `[ ]` **Conformal Language Modeling** — Quach et al., ICLR 2024
  https://arxiv.org/pdf/2306.10193
  The other half of the composition: risk control over unbounded free-text generation. Our contribution is composing this with the entry above across an inverse problem.

- `[ ]` **Conformal Risk Control** — Angelopoulos et al., ICLR 2024 + code
  https://github.com/aangelopoulos/conformal-risk
  We will use Learn-Then-Test essentially off the shelf. Read the code before the paper.

- `[ ]` **Detecting Hallucinations in Large Language Models Using Semantic Entropy** — Farquhar et al., Nature 2024
  https://www.nature.com/articles/s41586-024-07421-0
  Semantic-equivalence clustering — this is what makes "entropy over free-form answers" well defined. We import this machinery wholesale to define our semantic condition number.
  Cheaper prerequisite: Kuhn et al., *Semantic Uncertainty*, ICLR 2023 — https://arxiv.org/abs/2302.09664

- `[ ]` **Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs**
  https://arxiv.org/abs/2406.15927
  Approximates semantic entropy from a single generation's hidden states. Relevant to our efficiency story — the analogous trick may cut our sample budget.

- `[ ]` **Conformal Bounds on Full-Reference Image Quality for Imaging Inverse Problems**
  https://arxiv.org/pdf/2505.09528
  Same recipe (posterior sampling + conformal), applied to IQA rather than answers. Useful template for how to present the guarantee.

---

## Tier 3 — Sampler correctness. Read before trusting our own numbers.

- `[ ]` **PosteriorBench: From Point Estimates to Posterior Matching in Evaluating Generative Inverse Solvers**
  https://arxiv.org/html/2609.20794
  Do diffusion inverse solvers actually sample the posterior they claim? Largely no. Determines which solvers we can use and motivates the importance-weighting refinement. **Do not skip this one.**

- `[ ]` **A Survey on Diffusion Models for Inverse Problems** — Daras et al., 2024 *(in repo: `papers/base/inverse-diffusion-survey.pdf`)*
  https://arxiv.org/abs/2410.00083
  Our map of the solver landscape. Re-read specifically the latent-diffusion section — we need latent solvers at VLM resolution, and that is exactly where posterior fidelity is worst.

- `[ ]` **Taming Diffusion Models for Image Restoration: A Review** — Luo et al., Phil. Trans. A 2025 *(in repo: `papers/base/taming-review.pdf`)*
  https://arxiv.org/abs/2409.10353
  Broader restoration-side survey; good for related-work coverage and for the realistic-degradation taxonomy.

---

## Tier 4 — Solvers we will actually run (implementation references).

Pixel-space:
- `[ ]` **DPS — Diffusion Posterior Sampling for General Noisy Inverse Problems** — Chung et al., ICLR 2023 — https://arxiv.org/abs/2209.14687
- `[ ]` **DDRM — Denoising Diffusion Restoration Models** — Kawar et al., NeurIPS 2022 — https://arxiv.org/abs/2201.11793
- `[ ]` **DDNM — Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model** — Wang et al., ICLR 2023 — https://arxiv.org/abs/2212.00490
  *Read this one for the null-space decomposition specifically — it is the cleanest exposition of the geometry our bias analysis rests on.*
- `[ ]` **ΠGDM — Pseudoinverse-Guided Diffusion Models** — Song et al., ICLR 2023 — https://openreview.net/forum?id=9_gsMA8MRKQ

Latent-space (needed at VLM resolution):
- `[ ]` **PSLD — Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion** — Rout et al., NeurIPS 2023 — https://arxiv.org/abs/2307.00619
- `[ ]` **ReSample — Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency** — Song et al., ICLR 2024 — https://arxiv.org/abs/2307.08123

Blind / real-world (the application arm):
- `[ ]` **DiffBIR** — https://arxiv.org/abs/2308.15070
- `[ ]` **SUPIR — Scaling Up to Excellence** — CVPR 2024 — https://arxiv.org/abs/2401.13627
- `[ ]` **StableSR** — https://arxiv.org/abs/2305.07015

Regression baseline (the MMSE arm of our tradeoff table):
- `[ ]` **Restormer** — CVPR 2022 — https://arxiv.org/abs/2111.09881

---

## Tier 5 — Adjacent and framing. Skim for related work.

- `[ ]` **TReg — Regularization by Texts for Latent Diffusion Inverse Solvers** — ICLR 2025 *(in repo: `papers/application/Treg.pdf`)*
  https://arxiv.org/abs/2311.15658
  Text as a regularizer on the solution set. Read specifically to articulate the **prior vs. question** distinction — this is our defense against the circularity objection.

- `[ ]` **Deep Diffusion Image Prior (DDIP/D3IP) for Efficient OOD Adaptation** — Chung & Ye *(in repo: `papers/base/deep-diffusion-prior.pdf`)*
  https://arxiv.org/abs/2407.10641
  Test-time prior adaptation. Note: per-instance adaptation breaks conformal exchangeability — worth a remark in our paper.

- `[ ]` **Self-diffusion for Solving Inverse Problems** — NeurIPS 2025 *(in repo: `papers/application/self-diffusion.pdf`)*
  https://arxiv.org/abs/2412.18716
  Untrained-prior solver. Relevant if we want a no-pretrained-prior ablation.

- `[ ]` **DA-CLIP — Controlling Vision-Language Models for Multi-Task Image Restoration** — ICLR 2024
  https://arxiv.org/abs/2310.01018
  VLM features as restoration conditioning. Related work, opposite direction to ours.

- `[ ]` **Visual-Instructed Degradation Diffusion for All-in-One Image Restoration** — CVPR 2025
  https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Visual-Instructed_Degradation_Diffusion_for_All-in-One_Image_Restoration_CVPR_2025_paper.pdf

- `[ ]` **Selective "Selective Prediction": Reducing Unnecessary Abstention in Vision-Language Reasoning** — ACL Findings 2024
  https://arxiv.org/abs/2402.15610
  VLM abstention prior art; positions our risk–coverage results.

- `[ ]` **RobustVisRAG: Causality-Aware Vision-Based RAG under Visual Degradations**
  https://arxiv.org/pdf/2602.22013
  Degradation in multi-hop / retrieval pipelines. Adjacent application domain if we want a second vertical.

---

---
---

# PART II — Direction A & B core reading

After the pivot away from test-time ensembling. The project now rests on two claims:

- **A.** The inverse problem should be solved in the vision encoder's *token space*, not pixel space, because nothing about the answer is lost there (data processing equality) while the entropy to integrate over collapses.
- **B.** The uncertainty should come from a *linearized pushforward* of the posterior through the encoder — closed form, one backward pass — not from counting disagreements among samples.

Everything below is the base for those two claims. **This is now higher priority than Parts I Tier 1–5.**

---

## A1 — Does the encoder actually discard what degradation destroys?

*The load-bearing empirical premise of Direction A. If these papers say the encoder is highly sensitive to exactly the frequencies blur kills, Direction A is in trouble. Read these first, before writing any code.*

- `[ ]` **HAFI-VLM: A Frequency Perspective for Diagnosing and Enhancing Visual Perception in VLMs** — 2026
  https://pith.science/paper/2608.02124
  **Read this one first.** Introduces *spectral response rigidity*: the layerwise low/mid/high frequency energy profile of pretrained vision encoders is nearly invariant across images, tasks, and finetuning. Critically, the encoder is **not conditioned on the text query**, so evidence in attenuated bands is simply unavailable to the LLM. This is simultaneously the strongest support for our premise and the sharpest statement of the failure mode we exploit.

- `[ ]` **Fourier-VLM / Fourier Compressor: Frequency-Domain Visual Token Compression** — 2025
  https://arxiv.org/html/2508.06038v1
  Shows energy concentrates in low-frequency components across hidden dimensions; high-frequency content of visual *features* is semantically redundant and can be truncated with minimal loss. Direct quantitative evidence that the encoder's contraction directions overlap the degradation null space.

- `[ ]` **Semantic Richness or Geometric Reasoning? The Fragility of VLM Visual Invariance** — 2026
  https://arxiv.org/pdf/2604.01848
  The counterweight: where encoder invariance *breaks*. Read to find the regimes where Direction A fails, so we scope our claims honestly.

- `[ ]` **Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder** — 2024
  https://arxiv.org/pdf/2411.05195
  Information may be present in the encoder but not extracted/aligned. Matters because it separates "the tokens don't contain it" from "the LLM can't read it" — two very different diagnoses in our uncertainty decomposition.

- `[ ]` **Approximate Nullspace Augmented Finetuning for Robust Vision Transformers**
  https://arxiv.org/pdf/2403.10476
  Demonstrates a non-trivial nullspace for the patch embedding layer and studies encoder invariance to perturbations. This is the ViT-side null space that must be compared against the operator's null space — the core object of Direction B.

- `[ ]` **Intriguing Equivalence Structures of the Embedding Space of Vision Transformers**
  https://arxiv.org/pdf/2401.15568
  Which visually distinct inputs map to identical embeddings. Directly characterizes the many-to-one collapse we are relying on.

---

## A2 — Generative modeling in representation space

*Prior art for "generate features, not pixels." Establishes that this is a known-good move; our novelty is doing it conditioned on a measurement.*

- `[ ]` **RCG: Self-conditioned Image Generation via Generating Representations** — Li et al.
  https://arxiv.org/html/2312.03701v2
  The canonical reference. Generates in a self-supervised encoder's representation space, which is *far lower dimensional than pixel space yet rich in semantics*. That sentence is the seed of Direction A. Extract their argument for why representation-space generation is easier.

- `[ ]` **ReDi — jointly modeling VAE latents and DINO semantic features**
  (see survey below for pointer) — generates coherent image–feature pairs from noise. Relevant to the question of whether we need to model tokens alone or tokens plus enough latent to stay consistent.

- `[ ]` **SVG: Self-supervised Visual Generation with frozen DINO features + residual detail branch**
  Diffusion trained directly in a semantically structured latent space with a lightweight residual for fidelity. A good architectural template for our token-space posterior model.

- `[ ]` **Towards Controllable Image Generation through Representation-Conditioned Diffusion Models** — 2026
  https://arxiv.org/html/2605.27343

- `[ ]` **Representation Learning in Diffusion and Flow-based Models: An Application Aspect** — 2026 survey
  https://arxiv.org/html/2608.24068
  Use as the map of this subfield; read selectively.

- `[ ]` **Laminating Representation Autoencoders for Efficient Diffusion** — 2026
  https://arxiv.org/pdf/2602.04873

---

## A3 — Amortized posterior inference (how we get single-pass inference)

*This is what replaces the sampling loop. Train once offline, one forward pass at test time.*

- `[ ]` **Flow-based Generative Models for Amortized Bayesian Inference in Regression and Inverse PDE Problems** — 2026
  https://arxiv.org/html/2606.10370
  Conditional flow matching for amortized posteriors; no invertible architecture, no Jacobian determinants, near-real-time posterior sampling for unseen observations. **The most directly transferable recipe for our token-space posterior.**

- `[ ]` **Conditional Flow Matching for Physics-Constrained Inverse Problems with Finite Training Data** — 2026
  https://arxiv.org/html/2603.14135v3
  Important because our training data is finite and synthetic (we generate targets from a pixel-space solver). Read for their data-efficiency treatment.

- `[ ]` **Preconditioned One-Step Generative Modeling for Bayesian Inverse Problems in Function Spaces** — 2026
  https://arxiv.org/html/2603.14798v1
  One-step generation — the extreme end of what we want at inference. Also relevant: MeanFlow-style one-step transport.

- `[ ]` **ASPIRE: Iterative Amortized Posterior Inference for Bayesian Inverse Problems** — Inverse Problems 2025
  https://iopscience.iop.org/article/10.1088/1361-6420/adba3d
  Amortization that still refines per instance — the middle ground if pure amortization underfits.

- `[ ]` **Learning to Solve Bayesian Inverse Problems: Amortized Variational Inference with Gaussian and Flow Guides** — JCP 2024
  https://www.sciencedirect.com/science/article/abs/pii/S0021999124003668
  Gaussian *guide* = exactly our "mean + low-rank covariance" output format. Read for the variational objective.

---

## A4 — How much information do tokens actually carry? (invertibility)

*Bounds the whole enterprise. If tokens are near-invertible, token-space and pixel-space posteriors are equally hard and Direction A's entropy argument weakens.*

- `[ ]` **Implicit Inversion Turns CLIP into a Decoder** — 2025
  https://arxiv.org/html/2505.23161v1
  Reconstructs images from CLIP embeddings with no diffusion decoder, via a frequency-aware INR against the frozen encoder. Explicitly framed as *a tool for observing which features CLIP recognizes or disregards* — which is precisely the measurement we need.

- `[ ]` **unCLIP / Hierarchical Text-Conditional Image Generation with CLIP Latents** — Ramesh et al.
  https://arxiv.org/pdf/2204.06125
  The original diffusion decoder inverting a CLIP image encoder, producing *multiple* images per embedding. That one-to-many fan-out is the token-space posterior's support — read §on the decoder and the diversity it produces.

- `[ ]` **Moving Beyond Diversity: Visual Token Pruning as Subspace Reconstruction for Efficient VLMs** — 2026
  https://www.alphaxiv.org/abs/2606.18681
  Frames token selection as reconstruction in feature space. Useful subspace machinery and a sanity check on token redundancy.

---

## B1 — Linearized Laplace and pushforward uncertainty (closed-form UQ)

*Direction B's entire toolkit. The good news: this is mature, well-understood machinery that nobody has pointed at a degradation operator + vision encoder.*

- `[ ]` **Variational Linearized Laplace Approximation for Bayesian Deep Learning**
  https://arxiv.org/pdf/2302.12565
  Start here for the LLA formulation: linearize, push the covariance through the Jacobian, predictive covariance $J \Sigma J^\top + \sigma^2 I$. That expression *is* our $\Sigma_v$.

- `[ ]` **Scalable Linearized Laplace Approximation via Surrogate Neural Kernel** — 2026
  https://www.arxiv.org/pdf/2601.21835
  Scalability is our practical blocker (ViT-L Jacobians). Read for the sketching/kernel tricks.

- `[ ]` **Linearization Turns Neural Operators into Function-Valued Gaussian Processes**
  https://arxiv.org/pdf/2406.05072
  Clean treatment of linearized pushforward through an operator. Good template for how to *present* the derivation.

- `[ ]` **Generative Models and Bayesian Inversion Using Laplace Approximation**
  https://arxiv.org/pdf/2203.07755
  Laplace applied to a generative prior in an inverse problem: $g(z) \approx g(z_0) + J_{z_0}(z-z_0)$. The closest existing combination of our two ingredients.

- `[ ]` **Optimality of Sub-network Laplace Approximations** — 2026
  https://arxiv.org/pdf/2605.09075
  We only need the covariance over a subspace — this justifies doing Laplace on part of the network.

- `[ ]` **Reparameterization Invariance in Approximate Bayesian Inference**
  https://arxiv.org/pdf/2406.03334
  A known failure mode of Laplace methods. Read so a reviewer cannot ambush us with it.

- `[ ]` **Jacobian-Guided Anisotropic Noise Reshaping (task-sensitive vs task-insensitive subspaces)** — 2026
  https://arxiv.org/html/2605.16812
  Uses Jacobian analysis to split representation space into task-sensitive and task-insensitive subspaces, then treats them differently. Different application (differential privacy) but **the machinery is nearly exactly our identifiability measure** — read carefully and cite prominently.

- `[ ]` **Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology** — 2026
  https://arxiv.org/pdf/2605.14258
  Full spectral characterization of trained per-layer Jacobians in transformers. Tells us what structure to expect in $J$ before we compute it.

---

## B2 — Uncertainty that lives in the embedding, not in the output

*How to represent and consume a distribution over tokens. Prior art for "the embedding is a Gaussian."*

- `[ ]` **Intra-Class Probabilistic Embeddings for Uncertainty Estimation in VLMs** — WACV 2026
  https://openaccess.thecvf.com/content/WACV2026/papers/Lin_Intra-Class_Probabilistic_Embeddings_for_Uncertainty_Estimation_in_Vision-Language_Models_WACV_2026_paper.pdf
  Training-free, post-hoc, multivariate Gaussians over features. Closest methodological neighbor to our $\Sigma_v$ — note that their covariance comes from class statistics, ours from the measurement operator. That contrast is a clean novelty statement.

- `[ ]` **GroVE: Probabilistic Embeddings for Frozen VLMs via GPLVM** — UAI 2025
  https://arxiv.org/pdf/2505.05163
  Post-hoc probabilistic embeddings from a *frozen* VLM — matches our constraint exactly.

- `[ ]` **GeoFlowVLM: Geometry-Aware Joint Uncertainty for Frozen VLM Embeddings** — 2026
  https://arxiv.org/html/2605.13352v1
  Predictive Gaussian embedding whose covariance arises from uncertainty in the inferred latent. Structurally the same object we want; different source of uncertainty.

- `[ ]` **ProLIP: Probabilistic Language-Image Pre-Training** — 2024
  https://arxiv.org/pdf/2410.18857
  The pretraining-side version. Useful for the "extra axis of uncertainty in semantic space" framing.

---

## B3 — Task-relevant information as the right currency

*Theoretical framing that makes "solve it in semantic space" principled rather than expedient. This literature has been solving our problem in a different domain for years and nobody in vision restoration cites it.*

- `[ ]` **Task-Oriented Image Semantic Communication Based on Rate-Distortion Theory**
  https://arxiv.org/pdf/2201.10929
  Transmit task-relevant meaning, not pixels, under a noisy channel. **The channel is our degradation operator.** This is a whole adjacent field with our exact structure — mining it is cheap novelty-by-transfer and strengthens the framing enormously.

- `[ ]` **TOIB: Task-Oriented Orthogonalised Information Bottleneck for Distributed Semantic Communication** — 2026
  https://arxiv.org/pdf/2604.11053
  Orthogonal decomposition of task-relevant vs task-irrelevant information — the same split as our null-space/sensitive-subspace analysis.

- `[ ]` **Multi-Modal Multi-Task Semantic Communication: A Distributed Information Bottleneck Perspective** — 2025
  https://arxiv.org/html/2510.04000v1

- `[ ]` Tishby & Zaslavsky, *Deep Learning and the Information Bottleneck Principle* — for the DPI/sufficiency argument in its original form.
  https://arxiv.org/abs/1503.02406

---

## Still essential from Part I

Even after the pivot, do not skip these four:

- **Chen et al., Looks Too Good To Be True** (Tier 0) — the uncertainty–perception tradeoff we claim to escape.
- **Blau & Michaeli** (Tier 0) — the tradeoff itself.
- **DDNM** (Tier 4) — the cleanest null-space decomposition; Direction B's geometry is built on it.
- **PosteriorBench** (Tier 3) — our amortized model is trained against pixel-space solver outputs, so if those aren't real posterior samples we inherit the bias. This matters *more* after the pivot, not less.

And keep **Task-Driven Conformal UQ (ECCV'24)** + **Conformal Risk Control** from Tier 2 — the guarantee story is unchanged, it just now runs on an analytic covariance instead of a sample count.

---

# Reading order

**Week 1 — decide whether Direction A survives (5 papers)**
1. HAFI-VLM (A1) — spectral rigidity
2. Fourier-VLM (A1) — where feature energy lives
3. Approximate Nullspace Augmented Finetuning (A1) — the ViT null space
4. Implicit Inversion Turns CLIP into a Decoder (A4) — what tokens retain
5. DDNM (Part I, Tier 4) — the operator null space to compare against

Then run the alignment experiment. Everything downstream depends on its outcome.

**Week 2 — build the method**
6. RCG (A2) — representation-space generation
7. Flow-based Amortized Bayesian Inference (A3) — the single-pass recipe
8. Variational Linearized Laplace (B1) — the closed-form covariance
9. Jacobian-Guided Anisotropic Noise Reshaping (B1) — sensitive/insensitive subspace split
10. Intra-Class Probabilistic Embeddings (B2) — how to present a Gaussian over features

**Week 3 — framing, theory, and defense**
11. Chen et al., Looks Too Good To Be True
12. Blau & Michaeli
13. Task-Oriented Image Semantic Communication (B3)
14. PosteriorBench
15. Task-Driven Conformal UQ (ECCV'24)
