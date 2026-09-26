# Inverse-problem solvers coupled to text / CLIP / VLMs / MLLMs (2023 to Sep 2026): per-paper catalog and threat assessment for F1, F2 and F3

Scope: works that have an explicit inverse-problem formulation (y = A(x) + n with a known or estimated operator, data or measurement consistency, DPS/DDRM/DDNM/ΠGDM/PSLD/ReSample/DAPS/P2L/TReg/LATINO/flow-based solvers) and couple it to language or vision-language models in either direction. Search date: 22 Sep 2026. Abbreviations: DIS = diffusion inverse solver; DC = data consistency; "Multi-samp" = multiple posterior samples used downstream; "Evid" = hypothesis comparison or evidence computed.

Threat definitions used below:
- **F1**: answer identifiability. The MLLM's answer distribution or entropy over measurement-consistent posterior samples x ~ p(x|y), split into measurement-determined, prior-determined and undetermined.
- **F2**: hypothesis-conditioned evidence. Compute p(y | q, a) for each candidate answer a with a text-conditioned DIS, then choose by Bayes factor, with TReg-style text regularization applied symmetrically.
- **F3**: measurement-consistent self-recovery. Add DC guidance to a unified MLLM's generative self-restoration (for example the BAGEL rectified-flow generator in Robust-U1).

---

## KQ1: Per-paper catalog. Which works combine an explicit inverse-problem solver with text, CLIP, VLMs or MLLMs, and in which direction?

### Takeaway
Nearly all coupled work runs in one direction: text or VLM into the solver, through prompts, captions, prompt optimization, rewards or operator estimation. No verified paper sends measurement-consistent DIS posterior samples to a VLM, MLLM or CLIP zero-shot classifier to make an understanding decision. The nearest solver-to-downstream works use conventional classifiers (MRI, MNIST or CIFAR) or an MLLM as a quality judge.

### Cited Findings

#### Summary table (details and sources in the entries below)

| ID | Paper | Venue | Direction | Solver | Fwd op | DC | Multi-samp | Evid | F1 / F2 / F3 threat |
|---|---|---|---|---|---|---|---|---|---|
| A1 | TReg | ICLR 2025 (spotlight) | text→solver | latent DIS + CFG, CG/DPS DC, CLIP null-text negation | known | Y | N | N | none / **medium** / none |
| A2 | P2L | **ICML 2024** (not ECCV) | text→solver | latent DIS + prompt-embedding tuning | known | Y | N | N | none / low-med / none |
| A3 | LATINO-PRO | ICCV 2025 | text→solver | PnP with a latent consistency model, proximal DC, MML over the prompt | known | Y | N | partial (∇ log p(y\|c) only) | none / **med-high** / low |
| A4 | LADiBI | NeurIPS 2025 | text→solver (blind) | MPGD-style latent DIS + operator MLE | estimated | Y | N | N | none / low / low |
| A5 | Text-guided Explorable SR | CVPR 2024 | text→solver | DDNM/DPS/ΠGDM + T2I | known | Y | diverse samples, not consumed | N | low / low-med / none |
| A6 | MCS (text-prompted BFR) | arXiv 2511.14213 | text→solver | T2I posterior-guided sampling | constructed | Y | N | N | none / low-med / none |
| A7 | FlowDPS | ICCV 2025 | VLM caption→solver | flow (SD3) posterior sampling | known | Y | N | N | none / low / low-med |
| A8 | LD-RPS | ICCV 2025 | MLLM caption→solver | recurrent latent posterior sampling | learned (F-PAM) | Y (learned) | 3 seeds averaged | N | none / low / low-med |
| A9 | Inference-time search with side info | arXiv 2510.03352 (OpenReview; venue not verified) | text reward→solver | DPS / Blind-DPS / DAPS / MPGD + particle search | known/blind | Y | particles for selection | N | low / low-med / low |
| A10 | PDLS | DICTA 2025 (not A*) | text→solver | rectified flow + LQR dual steering | not verified | not verified | N | N | none / low / low |
| B1 | DU-VLM | arXiv 2602.04565 | VLM operator→solver | DDNM with VLM-predicted physics params | VLM-estimated | Y | N | N | low / low / low-med |
| B2 | VLU-Net | CVPR 2025 | CLIP→unrolled | deep unfolding, degradation-guided GD | implicit | partial | N | N | none / none / none |
| C1 | SUPIR | CVPR 2024 | VLM caption→restorer | SDXL restorer | none explicit | no likelihood | N | N | low / none / low |
| C2 | DA-CLIP | ICLR 2024 | CLIP→restorer | restoration net via cross-attention | none | N | N | N | none / none / none |
| C3 | GLYPH-SR | arXiv 2510.26339 | VLM/OCR→restorer | latent diffusion + TS-ControlNet | none | N | N | N | low / none / none |
| C4 | VLMIR | arXiv 2512.17292 / journal | CLIP→restorer | diffusion + cross-attention | none | N | N | N | none / none / none |
| C5 | TPGDiff | arXiv 2601.20306 | semantic prior→restorer | diffusion with triple priors | none | N | N | N | none / none / none |
| D1 | Task-driven UQ via conformal | ECCV 2024 | solver→classifier | posterior-sampling recovery | known (MRI) | Y | **Y** | N | **medium** / low / none |
| D2 | Classification-oriented adaptive sensing | arXiv 2609.21812 (18 Sep 2026) | solver→classifier | diffusion posterior sampling | known | Y | **Y** | class-mixture decomposition | **medium** / low-med / none |
| D3 | DiME evidence estimation | arXiv 2602.20549 | none (no text) | DAPS / PnP-DM | known | Y | Y (for evidence) | **Y (evidence, model selection)** | low / **high (estimator)** / none |
| D4 | Hallucination Score | arXiv 2507.14367 | restorer→MLLM judge | generative SR (not an IP solver) | none | N | not reported | N | low-med / none / none |
| D5 | High-level fidelity in SR | arXiv 2512.07037 | restorer→CLIP/BLIP/DINO metric | generative SR (not an IP solver) | none | N | N | N | low / none / none |
| E1 | Ravula et al. (pre-window) | NeurIPS 2021 | IP in CLIP feature space | supervised contrastive inversion | known | N/A | N | N | low / none / none |
| E2 | Align & Invert | arXiv 2511.16870 / OpenReview | DINOv2 features→solver | REPA inside a DIS | known | Y | N | N | none / none / low |
| F-a | Robust-U1 | arXiv 2606.08063 | MLLM self-restore→MLLM | BAGEL rectified flow | none | **N** | only in RL | N | n/a / n/a / **baseline** |
| F-b | CLEAR | arXiv 2604.04780 | unified model generation→understanding | unified MM model | not found | not found | not found | N | low / none / low-med |

#### A. Text or VLM shaping the prior or regularizer of a measurement-consistent solver

**A1. TReg: Regularization by Texts for Latent Diffusion Inverse Solvers.** Kim, J. et al. (Kim, Park, Chung, Ye, KAIST). ICLR 2025 spotlight. [arXiv 2311.15658](https://arxiv.org/abs/2311.15658); ICLR spotlight status per the [official repo](https://github.com/TReg-inverse/TReg).
- Direction: text→solver. Solver: latent DIS (LDPS variant) with DDIM + CFG. DC comes from conjugate-gradient optimization for linear problems, or optional DPS gradients for nonlinear ones. "Adaptive negation" optimizes the null-text embedding to minimize CLIP image-text similarity, so CLIP runs inside the sampler — [TReg HTML](https://arxiv.org/html/2311.15658).
- Fwd op: known (×16 bicubic SR, Gaussian deblur). DC: Y. Multi-samp: N. Evid: N — [TReg HTML](https://arxiv.org/html/2311.15658).
- Key finding: TReg includes experiments where the prompt deliberately differs from the true class of the measurement. The authors state that "the proper solution should satisfy 1) data consistency with measurement and 2) alignment with given prompt", and TReg attains low y-MSE together with high CLIP alignment to the *counterfactual* prompt. So different text conditions can each be made data-consistent. Prompts come from ground-truth class labels, or from alternative ImageNet classes chosen by LPIPS; no captioner is used. Appendix J concedes that a simple prompt may leave "residual symmetry". The paper never compares prompts or uses the measurement to choose between them — [TReg HTML](https://arxiv.org/html/2311.15658).
- Threat. F1: none (CLIP similarity is only an evaluation metric; there is no VLM decision). F2: **medium**. TReg is the machinery F2 would reuse, and its counterfactual-prompt experiment shows that y-MSE alone may not separate hypotheses. It computes no evidence or Bayes factor, so the decision rule stays open. F3: none.

**A2. P2L: Prompt-tuning Latent Diffusion Models for Inverse Problems.** Chung, H. et al. (Chung, Ye, Milanfar, Delbracio). **ICML 2024 (PMLR v235), not ECCV 2024 as the brief states** — [PMLR](https://proceedings.mlr.press/v235/chung24b.html); [arXiv 2310.01110](https://arxiv.org/abs/2310.01110).
- Direction: text→solver. The text embedding is optimized on the fly during reverse diffusion, by alternating minimization over prompt, latent and pixel. Latents are projected to stay in the encoder's range space. Fwd op: known (SR, deblur, inpainting). DC: Y. Multi-samp: N. Evid: N — [PMLR](https://proceedings.mlr.press/v235/chung24b.html); [arXiv abs](https://arxiv.org/abs/2310.01110).
- Motivation quoted in the literature: "determining an effective text prompt from degraded measurements is ambiguous", and P2L aims to find suitable prompts automatically — [arXiv abs](https://arxiv.org/abs/2310.01110).
- Threat. F1: none. F2: low-medium. It gives a point estimate of a continuous prompt, not a comparison between discrete hypotheses (the exact loss was not checked against the full text). F3: none.

**A3. LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization.** Spagnoletti, A. et al. (Spagnoletti, Prost, Almansa, Papadakis, Pereyra). ICCV 2025. [arXiv 2503.12615](https://arxiv.org/abs/2503.12615); [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Spagnoletti_LATINO-PRO_LAtent_consisTency_INverse_sOlver_with_PRompt_Optimization_ICCV_2025_paper.pdf).
- Direction: text→solver. Solver: zero-shot PnP with a Latent Consistency Model prior (about 8 NFEs). DC: proximal step prox_{δ g_y} with g_y = −log p(y|x), closed form via SVD for linear A. Fwd op: known. Multi-samp: N — [arXiv HTML](https://arxiv.org/html/2503.12615).
- **Prompt as empirical-Bayes hyperparameter.** ĉ(y) = argmax_c p(y|c), fitted by maximum marginal likelihood using SAPG and the Fisher identity ∇_c log p(y|c) = E_{x|y,c}[∇_c log p(x|c)]. Only the gradient is estimated; **p(y|c) itself is never computed**. The warm-start prompt is user-given (e.g. "a sharp photo of a dog") and no captioner is used. There is no hallucination analysis and no comparison between discrete candidate prompts; one ablation shows PRO corrects "incomplete or misleading prompts" — [arXiv HTML](https://arxiv.org/html/2503.12615).
- Threat. F1: none. F2: **medium-high**. This is the closest formal precedent: the text condition is scored by p(y|c). F2 still differs in three ways: it compares *values* of the evidence across discrete, question-derived hypotheses (a Bayes factor rather than MML optimization), applies regularization symmetrically, and outputs an answer. F3: low.

**A4. LADiBI: Blind Inverse Problem Solving Made Easy by Text-to-Image Latent Diffusion.** Dontas, M. et al. (Dontas, He, Murata, Mitsufuji, Kolter, Salakhutdinov). NeurIPS 2025. [arXiv 2412.00557](https://arxiv.org/abs/2412.00557); [NeurIPS page](https://neurips.cc/virtual/2025/125472).
- Direction: text→solver, blind. Positive and negative prompts, combined with CFG, encode priors on both the image and the operator. The operator A_φ is parameterized (a 61×61 kernel, or a small U-Net for JPEG). It is initialized by SDEdit/MPGD pseudo-supervision and then re-estimated by MLE every 5 steps. DC uses MPGD guidance on ‖y − A_φ(D(z))‖² plus LPIPS. Prompts are user-chosen (no VLM), and automatic prompt tuning is listed as future work. Hallucination is not discussed beyond generic model risks — [arXiv HTML v2](https://arxiv.org/html/2412.00557v2).
- Fwd op: estimated. DC: Y. Multi-samp: N. Evid: N.
- Threat. F1: none. F2: low. F3: low; it is a precedent for DC with an *estimated* operator inside a T2I latent generator.

**A5. Text-guided Explorable Image Super-resolution.** Gandikota, K.V. & Chandramouli, P. CVPR 2024. [arXiv 2403.01124](https://arxiv.org/abs/2403.01124); [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Gandikota_Text-guided_Explorable_Image_Super-resolution_CVPR_2024_paper.pdf).
- Direction: text→solver. It takes two routes: (i) modify T2I diffusion to enforce consistency with the LR input, and (ii) inject language into zero-shot DIS. The code implements text-guided DDNM, DPS and ΠGDM. The goal is diverse, semantically different reconstructions that stay data-consistent at large downsampling factors — [arXiv abs](https://arxiv.org/abs/2403.01124); [GitHub via search](https://github.com/KVGandikota/Text-guidedSR); [Semantic Scholar](https://www.semanticscholar.org/paper/Text-Guided-Explorable-Image-Super-Resolution-Gandikota-Chandramouli/3324358091a949e9c79e64267fee318d2cde04f5).
- Fwd op: known. DC: Y. Multi-samp: diverse samples shown to users, not consumed by any model. Evid: N.
- Threat. F1: low. F2: low-medium; it produces hypothesis-conditioned consistent reconstructions but scores none of them. F3: none.

**A6. MCS: Measurement-Constrained Sampling for Text-Prompted Blind Face Restoration.** Li, W. et al. (Li, Zhang, Gao, Guo, Ma). arXiv only, Nov 2025 (venue not verified). [arXiv 2511.14213](https://arxiv.org/abs/2511.14213).
- Direction: text→solver. It builds an inverse problem by applying controlled degradations to coarse restorations, then runs posterior-guided sampling inside T2I diffusion. A "Forward Measurement" keeps results aligned with the input; a "Reverse Measurement" creates projection spaces so that solutions can align with *various prompts*. The target is the one-to-many nature of extreme-LQ faces — [arXiv abs](https://arxiv.org/abs/2511.14213).
- Fwd op: constructed/synthetic. DC: Y (to the constructed measurement). Multi-samp: N. Evid: N.
- Threat. F1: none. F2: low-medium; it again shows that several prompts are consistent with one input, without scoring them. F3: none.

**A7. FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems.** Kim, J. et al. ICCV 2025. [arXiv 2503.08136](https://arxiv.org/abs/2503.08136); [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_FlowDPS__Flow-Driven_Posterior_Sampling_for_Inverse_Problems_ICCV_2025_paper.html).
- Direction: VLM caption→solver. Prompts are fixed for AFHQ/FFHQ, extracted with **DAPE** (a degradation-aware prompt extractor) for DIV2K, and taken from **LLaVA** captions for high-resolution results. DC: 3 gradient steps on ‖y − A·D(z)‖² with an SD3 flow prior. The authors note that "defining an appropriate text prompt solely based on the given measurement can be challenging". There is no prompt-effect or hallucination study — [arXiv HTML](https://arxiv.org/html/2503.08136).
- Fwd op: known. DC: Y. Multi-samp: N. Evid: N.
- Threat. F1: none. F2: low. F3: **low-medium**. It shows that DPS-style DC already works in a rectified-flow latent generator, the same family as BAGEL's image head. F3's novelty therefore has to come from the *unified-MLLM self-recovery loop*, not from "DC plus rectified flow".

**A8. LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling.** Li, H. et al. (Li, Wang, Huang, Huang, Wang, Chu). ICCV 2025. [arXiv 2507.00790](https://arxiv.org/abs/2507.00790); [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_LD-RPS_Zero-Shot_Unified_Image_Restoration_via_Latent_Diffusion_Recurrent_Posterior_ICCV_2025_paper.pdf).
- Direction: MLLM caption→solver. An MLLM annotates the LQ image to produce prompts that act as a semantic prior. The forward operator is **learned**: F-PAM aligns intermediate results with the degraded image through distance and quality losses. Recurrent posterior sampling re-noises the previous restoration to γT. The authors acknowledge "strong hallucinations" early in generation and use recurrence to counter them. Results are averaged over 3 seeds — [arXiv HTML](https://arxiv.org/html/2507.00790v1).
- DC: Y (against a learned operator). Multi-samp: seed averaging only. Evid: N.
- Threat. F1: none. F2: low. F3: low-medium; it is a precedent for MLLM text plus DC-like guidance in task-blind restoration.

**A9. Inference-Time Search Using Side Information for Diffusion-Based Image Reconstruction.** Farahbakhsh, M. et al. (Farahbakhsh, Kunde, Kalathil, Narayanan, Chamberland, Texas A&M). arXiv, v3 May 2026. An [OpenReview forum](https://openreview.net/forum?id=qj1EL0oT1n) exists, but venue and decision could not be verified (verification wall). [arXiv 2510.03352](https://arxiv.org/abs/2510.03352).
- Direction: text reward→solver. Base solvers are DPS, Blind DPS, DAPS and MPGD, with DC kept by the base solver. Search is SMC-like: Greedy Search, or Recursive Fork-Join Search over N = 4–8 particles, with weights ∝ exp(r/τ). Rewards: **ImageReward** for text side information, AdaFace identity similarity for reference images, and NMI for MRI contrasts. CLIPScore is used to assess semantic alignment. The paper includes reward-robustness ablations but no explicit analysis of text-induced hallucination — [arXiv HTML v3](https://arxiv.org/html/2510.03352v3).
- Fwd op: known, or blind via Blind-DPS. Multi-samp: particles used only for selection. Evid: N.
- Threat. F1: low. F2: low-medium; it is a soft "text as reward" alternative that F2 should cite and contrast with a likelihood ratio. F3: low.

**A10. PDLS: Prompt-Guided Dual Latent Steering for Inversion Problems.** Wu, Y. et al. (Wu, Liu, Zhao, Wu). DICTA 2025 oral, not an A* venue. [arXiv 2509.18619](https://arxiv.org/abs/2509.18619).
- A training-free rectified-flow method. A structural path preserves source integrity and a prompt-guided semantic path steers semantics, both solved in closed form as an LQR optimal-control problem. Tasks: Gaussian and motion deblur, SR and inpainting on FFHQ-1K and ImageNet-1K. Prompt source and the exact DC mechanism **could not be verified** from the abstract — [arXiv abs](https://arxiv.org/abs/2509.18619).
- Threat: low for F1, F2 and F3.

Other latent or flow DIS that accept text conditioning but add no VLM logic (background only): PSLD, NeurIPS 2023 ([arXiv 2307.00619](https://arxiv.org/html/2307.00619)); STSL / "Beyond First-Order Tweedie", which also does text-guided editing of corrupted images ([project page](https://stsl-inverse-edit.github.io/)); FLAIR, NeurIPS 2025, flow-based variational solver with hard DC ([arXiv 2506.02680](https://arxiv.org/abs/2506.02680)); RLSD, ICLR 2025, a repulsive multimodal variational posterior ([arXiv 2406.16683](https://arxiv.org/abs/2406.16683)).

#### B. VLM estimates the forward operator (blind), then a measurement-consistent solver runs

**B1. DU-VLM: Understanding Degradation with Vision Language Model.** arXiv, 4 Feb 2026; authors not retrieved. [arXiv 2602.04565](https://arxiv.org/abs/2602.04565).
- It recasts degradation understanding as hierarchical structured prediction (type → parameter keys → continuous values), trained by SFT plus RL with structured rewards and CoT anchors — [arXiv abs](https://arxiv.org/abs/2602.04565).
- As a "zero-shot controller", it injects the predicted parameters D̂ into **pre-trained DDNM**. It uses explicit physics forward models, e.g. haze I_d = I_c·t + A(1 − t) with t = e^{−βd}, plus blur, low-light and SR. The inverse is described as "variational inference regularized by the diffusion prior". Evaluation uses PSNR, SSIM and LPIPS, with NIQE, BRISQUE and CLIP-IQA for real-world data; there is no VQA and no posterior samples go downstream — [arXiv HTML](https://arxiv.org/html/2602.04565).
- Fwd op: VLM-estimated. DC: Y (DDNM range-null projection). Multi-samp: N. Evid: N.
- Threat. F1: low. F2: low. F3: low-medium. It is the clearest "VLM → operator → DC solver" precedent. Any F3 variant that estimates an operator will have to cite it, but DU-VLM restores for pixel quality, not for the MLLM's own understanding.

**B2. VLU-Net: Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks.** Zeng et al. CVPR 2025. [arXiv 2503.16930](https://arxiv.org/html/2503.16930); [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zeng_Vision-Language_Gradient_Descent-driven_All-in-One_Deep_Unfolding_Networks_CVPR_2025_paper.pdf).
- An unrolled network. CLIP, fine-tuned on degradation image-text pairs, supplies degradation features that drive a Degradation-guided Gradient Descent Module in each of K stages. The operator is implicit or learned, not a known A — [arXiv HTML](https://arxiv.org/html/2503.16930).
- Threat: none for F1, F2 and F3.

Related (A4, A8): LADiBI encodes the operator prior with *text* (not a VLM) and fits it by MLE; LD-RPS learns the operator with MLLM-captioned prompts. Non-solver VLM degradation modeling, e.g. MVLR (ICAC 2025), uses VLM CoT to retrieve degradation prototypes into an encoder-decoder with no diffusion and no DC — [arXiv 2511.16998](https://arxiv.org/abs/2511.16998).

#### C. VLM/CLIP-conditioned restorers without an explicit inverse-problem formulation (listed known papers)

- **C1. SUPIR**, Yu, F. et al., CVPR 2024, [arXiv 2401.13627](https://arxiv.org/abs/2401.13627). Text-prompt-guided restoration with a "restoration-guided sampling method to suppress the fidelity issue" (abstract). LLaVA captions are reported by a later paper — [SRSR 2510.22534](https://arxiv.org/html/2510.22534). Whether restoration-guided sampling uses an explicit likelihood or operator **was not verified** from full text; from the abstract it is a fidelity pull rather than a y = A(x) + n likelihood. Threat: low for F1, F2 and F3.
- **C2. DA-CLIP**, Luo, Z. et al. (Luo, Gustafsson, Zhao, Sjölund, Schön), ICLR 2024, [arXiv 2310.01018](https://arxiv.org/abs/2310.01018). A controller adapts the frozen CLIP image encoder to predict HQ content embeddings and degradation features ("a natural classifier for different degradation types"), fed by cross-attention into a restoration network. The abstract mentions no forward operator or DC; the backbone was not re-verified. Threat: none.
- **C3. GLYPH-SR**, [arXiv 2510.26339](https://arxiv.org/abs/2510.26339); an [OpenReview forum](https://openreview.net/forum?id=GxPtLwLSOL) exists but venue and decision were **not verifiable**. VLM-guided latent diffusion with an OCR-guided TS-ControlNet and a "ping-pong" scheduler that alternates text-centric and scene-centric guidance. It improves OCR F1 by up to +15.18 pp on SVT, CTW1500 and CUTE80 at ×4 and ×8 — [arXiv HTML](https://arxiv.org/html/2510.26339). No DC. Threat: low; it uses a downstream recognizer (OCR) as its fidelity metric, which is a single-sample precedent for task-level evaluation.
- **C4. VLMIR**, Yang, C., Dong, R., Lam, K.-M., [arXiv 2512.17292](https://arxiv.org/abs/2512.17292), with a journal version on [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0262885626001988) (journal name not verified). LQ and HQ caption embeddings are aligned with a cosine loss and LoRA, a degradation predictor separates degradation from content embeddings, and cross-attention conditions the diffusion model. No DC. Threat: none.
- **C5. TPGDiff**, [arXiv 2601.20306](https://arxiv.org/abs/2601.20306), Jan 2026. Semantic (distillation-driven extractor), structural and degradation priors injected into a diffusion restorer; 30.60 dB / 0.908 SSIM on all-in-one. No DC; whether the semantic prior is distilled from a VLM **was not verified** — [arXiv HTML](https://arxiv.org/html/2601.20306). Threat: none.

#### D. Solver (or restorer) outputs consumed by a downstream model: the solver→understanding direction

- **D1. Task-Driven Uncertainty Quantification in Inverse Problems via Conformal Prediction.** Wen, J., Ahmad, R., Schniter, P. ECCV 2024. [arXiv 2405.18527](https://arxiv.org/abs/2405.18527); [ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07734.pdf). The recovered image feeds a downstream soft-output classifier. Conformal intervals guaranteed to contain the task output of the true image are built, and for posterior-sampling recovery they are locally adaptive. Measurements are collected in rounds until task uncertainty is acceptable. Demonstrated on accelerated MRI (a medical vertical, noted as out of scope) — [arXiv abs](https://arxiv.org/abs/2405.18527). Multi-samp: **Y**. Threat. F1: **medium**, because "downstream-task uncertainty induced by measurement plus recovery" is claimed, though for a classifier, with no MLLM or QA and no split between measurement-determined and prior-determined. F2: low. F3: none.
- **D2. Classification-oriented adaptive sensing via posterior sampling.** Enttsel, A., Rousselot, M., Corlay, V. arXiv, **18 Sep 2026**, eess.SP. [arXiv 2609.21812](https://arxiv.org/abs/2609.21812). Calibrated soft-classifier outputs on diffusion posterior samples estimate **within-class and between-class uncertainty**, motivated by the closed-form posterior covariance of a class-conditional GMM. This drives the choice of the next sensing direction. MNIST and CIFAR-10; no VLM or CLIP — [arXiv abs](https://arxiv.org/abs/2609.21812). Threat. F1: **medium**. It posted four days before this search, and it is the closest "semantic posterior uncertainty from posterior samples" work; it uses a classifier on toy datasets, with no MLLM and no three-way split. F2: low-medium (class-mixture reasoning). F3: none.
- **D3. DiME: Sample-efficient evidence estimation of score-based priors for model selection.** Wang, F. & Bouman, K.L. (Caltech). arXiv, v2 30 Apr 2026. [arXiv 2602.20549](https://arxiv.org/abs/2602.20549). It estimates log p(y|M) = E[log p(y|x0)] − KL(p(x0|y) ‖ p(x0)), with the KL term obtained by integrating the squared likelihood score over diffusion time using DAPS or PnP-DM samples. It "consistently selects the correct prior from a set of 10 diffusion models trained on MNIST digits given a single noisy measurement" under Gaussian and Fourier phase retrieval, and compares priors for M87* black-hole imaging. There are no text-conditioned priors, and the work is framed as Bayesian model selection — [arXiv HTML](https://arxiv.org/html/2602.20549). Threat. F2: **high for the estimator component**, since evidence-based "which hypothesis explains y" with DIS already exists, amounting to classification by evidence over 10 class priors. F2's novelty must lie in text- or question-conditioned hypotheses, the symmetric text regularization, and the VLM/VQA decision. F1: low. F3: none.
- **D4. Hallucination Score: Towards Mitigating Hallucinations in Generative Image Super-Resolution.** Ren, W. et al. arXiv. [arXiv 2507.14367](https://arxiv.org/abs/2507.14367). An MLLM (GPT-4o primarily, Qwen2.5-VL also tested) is prompted to score hallucinatory elements of SR outputs, and the score aligns with human judgments. HS proxies serve as differentiable rewards to fine-tune diffusion SR models — [arXiv abs](https://arxiv.org/abs/2507.14367); [HTML](https://arxiv.org/html/2507.14367.pdf). The restorers are not explicit IP solvers, and per-sample variation was not reported. Threat. F1: low-medium (an MLLM judging restoration hallucination is claimed). F2: none. F3: none.
- **D5. Evaluating and Preserving High-level Fidelity in Super-Resolution.** Rocafort, J.M. et al. (Rocafort, Su, Vazquez-Corral, Gomez-Villa). arXiv, Dec 2025. [arXiv 2512.07037](https://arxiv.org/abs/2512.07037). The dataset has 723 images with at least 12 human binary annotations each of semantic change. Fidelity is measured as cosine similarity of fine-tuned CLIP, BLIP, DINOv2 or PE-core embeddings between SR and GT, with up to about 10% PLCC gain over IQA metrics. Evaluated models: StableSR, SeeSR, PASD, SwinIR, BSRGAN, with **no DPS-type solvers**. Adding the metric as a loss to SeeSR improves both fidelity and perception — [arXiv HTML](https://arxiv.org/html/2512.07037v1). Threat. F1: low. The embedding-similarity metric needs GT and is not VQA.
- **D6. (Background) Diffusion-driven test-time adaptation.** DDA ("Back to the Source") projects corrupted inputs back toward the source domain using ILVR low-frequency guidance, then classifies — [ResearchGate](https://www.researchgate.net/publication/373312682_Back_to_the_Source_Diffusion-Driven_Adaptation_to_Test-Time_Corruption). CODE is an efficient corruption editor for TTA — [Springer, ECCV-2024 LNCS](https://link.springer.com/chapter/10.1007/978-3-031-72943-0_11). These use a conventional classifier, not a VLM, and single samples. Threat: low.

#### E. Inverse problems in encoder feature space

- **E1. Inverse Problems Leveraging Pre-trained Contrastive Representations.** Ravula, S. et al. (Ravula, Smyrnis, Jordan, Dimakis). NeurIPS 2021, **pre-window**. [arXiv 2110.07439](https://arxiv.org/abs/2110.07439). Given A(x) with known A, it recovers the **CLIP representation R(x)** via a contrastively trained student. Linear probes on the robust representations beat end-to-end supervised baselines under blur, noise and masking — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/hash/498f940d9b933c529b06aa96d18f7eda-Abstract.html). It is supervised, deterministic, with no posterior and no VLM QA. Threat: low for F1 (a precedent for "inverse problem in VLM-encoder space").
- **E2. Align & Invert: Solving Inverse Problems with Diffusion and Flow-based Models via Representation Alignment.** [arXiv 2511.16870](https://arxiv.org/abs/2511.16870); [OpenReview PDF](https://openreview.net/pdf?id=n69u66Zwnv) (venue not verified; the search snippet gives a May 2026 date that conflicts with the Nov-2025 arXiv ID, probably a revision). It adds REPA between DIS/flow internals and **DINOv2** features at inference, and interprets it as minimizing a divergence in DINOv2 embedding space. Integrated into several solvers for SR, box inpainting, and Gaussian and motion deblur — [arXiv abs](https://arxiv.org/abs/2511.16870). DINOv2 is vision-only, not a VLM. Threat: none for F1 and F2; low for F3.

#### F. Unified-MLLM self-restoration (F3 context)

- **F-a. Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?** Tang et al. [arXiv 2606.08063](https://arxiv.org/abs/2606.08063). The fetch tool reported "AAAI", which is **unverified and doubtful** given the June-2026 posting. Built on BAGEL: stage-I SFT on ImageNet-C with a rectified-flow loss conditioned on the corrupted input, then Flow-GRPO with semantic and pixel rewards. **No explicit measurement or forward-operator term at inference.** G stochastic trajectories are used only for RL advantage; a single recovered image is produced at inference. The authors concede that recovery is limited under severe corruption and depends on paired data — [arXiv HTML](https://arxiv.org/html/2606.08063v1). This is F3's baseline, not a threat; it confirms that the DC gap is open.
- **F-b. CLEAR: Unlocking Generative Potential for Degraded Image Understanding in Unified Multimodal Models.** Hao, X. et al. [arXiv 2604.04780](https://arxiv.org/pdf/2604.04780). It uses the generation branch of a unified model to aid degraded-image understanding (MMBench, MMVet, CVBench, R-Bench). The fetched text showed **no DC or forward operator**, but this is only partially verified. Threat. F3: low-medium; it is a second "unified generation for degraded understanding" paper that competes for the framing, not for DC.

#### Out-of-scope items encountered (noted only)
- Visual Quality Paradox (restore-then-MLLM with non-IP restorers; VQ-TTT) — [arXiv 2506.15645](https://arxiv.org/html/2506.15645).
- DocIntent, a document-VQA agentic restoration framework with an answerability loop — [arXiv 2608.29037](https://arxiv.org/html/2608.29037).
- Vision-Language Controlled Deep Unfolding for joint *medical* restoration and segmentation — [arXiv 2601.23103](https://arxiv.org/pdf/2601.23103).
- SR-Diff, VLM semantic embeddings for diffusion restoration (journal, no DC) — [Connection Science](https://www.tandfonline.com/doi/full/10.1080/09540091.2026.2656975).
- Test-Time Preference Optimization for IR (AAAI 2026): diffusion inversion of an initial restoration plus preference-ranked candidates; reward model unspecified in the abstract — [arXiv 2511.19169](https://arxiv.org/abs/2511.19169).
- Difficulty-adaptive RL with an MLLM-IQA reward for diffusion restoration — [arXiv 2511.01645](https://arxiv.org/html/2511.01645v1).
- ReasonX, MLLM-as-judge for intrinsic decomposition — [arXiv 2512.04222](https://arxiv.org/html/2512.04222v1).

### Inferences
- The text/VLM→solver direction is crowded. Prompts come from users (TReg, LADiBI, LATINO-PRO warm start), are optimized (P2L, LATINO-PRO), come from VLM captions (FlowDPS via DAPE/LLaVA, LD-RPS via an MLLM, SUPIR via LLaVA), or enter as rewards (2510.03352). None of these solvers returns an answer to a question.
- The solver→understanding direction is nearly empty for VLMs. Every solver→downstream work found uses a conventional classifier (D1, D2, D6), evidence-based prior selection (D3), or an MLLM as a *quality judge* of single outputs (D4).
- The venue for P2L in the brief should be corrected to ICML 2024.

### Gaps
- DU-VLM author list, and whether DDNM is used strictly or relaxed for nonlinear haze and low-light models, were not retrieved.
- Venue and decision status of 2510.03352, GLYPH-SR and Align & Invert could not be verified (OpenReview verification wall).
- SUPIR's restoration-guided sampling and DA-CLIP's backbone were not verified from full text.
- PDLS's prompt source and DC mechanism were not verified.
- CLEAR's method details (base model, any consistency term) are only partially verified.
- A Semantic Scholar "cited-by" sweep of TReg, P2L, DPS, PSLD, DAPS, DDNM and DU-VLM was **not performed**: no Semantic Scholar API tool was used, and web search stood in. Low-citation 2026 arXiv papers may have been missed.

---

## KQ2 (critical): Does any paper feed measurement-consistent posterior samples into a VLM, MLLM or CLIP zero-shot classifier for an understanding decision, and compare answers across samples? Does any evaluate restoration by VQA-based semantic fidelity?

### Takeaway
**No verified paper does this.** Across roughly 30 targeted queries, no work passes DPS/DDRM/DDNM/ΠGDM/PSLD/DAPS/flow-solver posterior samples to a VLM, MLLM or CLIP zero-shot classifier to produce an answer, caption or class. None measures answer variation across posterior samples, and none runs VQA- or TIFA-style semantic-fidelity evaluation of IP-solver outputs. The nearest neighbours replace the VLM with a conventional classifier (ECCV 2024 task-driven conformal UQ; the 18 Sep 2026 classification-oriented adaptive sensing paper) or measure VLM answer entropy under generic image perturbations rather than posterior samples (Visual Semantic Entropy, ECCV 2026).

### Cited Findings
- Where CLIP appears in text-conditioned DIS, it serves as an *evaluation or regularization* signal: TReg reports CLIP image-text similarity and uses CLIP inside adaptive negation, with no downstream decision — [TReg HTML](https://arxiv.org/html/2311.15658). 2510.03352 uses ImageReward as the particle reward and CLIPScore for alignment assessment — [arXiv HTML v3](https://arxiv.org/html/2510.03352v3).
- Posterior samples feeding a *classifier*, with uncertainty on the task output: Wen, Ahmad and Schniter (ECCV 2024) build conformal intervals on classifier outputs from posterior-sampling recovery and stop sensing adaptively (MRI) — [arXiv 2405.18527](https://arxiv.org/abs/2405.18527). Enttsel et al. (arXiv, 18 Sep 2026) run a calibrated classifier on diffusion posterior samples to split within-class from between-class uncertainty for adaptive sensing (MNIST, CIFAR-10) — [arXiv 2609.21812](https://arxiv.org/abs/2609.21812).
- VLM answer entropy under *image perturbations*, not posterior samples: Visual Semantic Entropy (Huy et al., ECCV 2026 per the arXiv listing) perturbs only the image and clusters answers into semantic prototypes. It finds VLMs overconfident on ambiguous inputs and that text-paraphrase methods capture "prompt sensitivity rather than visual ambiguity" — [arXiv 2606.31407](https://arxiv.org/abs/2606.31407). HEDGE applies controlled visual perturbations with entailment- or embedding-based clustering for VQA hallucination detection, with no restoration or generative resampling — [arXiv 2511.12693](https://arxiv.org/abs/2511.12693).
- MLLM judgments on restorations exist only as *quality or hallucination scoring of single SR outputs*: Hallucination Score (GPT-4o / Qwen2.5-VL) — [arXiv 2507.14367](https://arxiv.org/abs/2507.14367). The high-level fidelity metric uses fine-tuned CLIP, BLIP or DINOv2 embeddings with human labels, and includes no IP solvers — [arXiv 2512.07037](https://arxiv.org/html/2512.07037v1).
- Downstream-recognizer fidelity is reported only for non-IP restorers: GLYPH-SR reports OCR F1 gains — [arXiv 2510.26339](https://arxiv.org/html/2510.26339). The TextSR paper notes that StableSR and SUPIR "show significant reductions in legibility with respect to OCR accuracy and consequently lead to inaccurate VQA performance" — [TextSR 2505.23119](https://arxiv.org/html/2505.23119).
- Pixel-space theory relevant to answer identifiability, all without VLMs: posterior uncertainty decomposed into intrinsic ambiguity (forward-operator-driven) versus estimation uncertainty — [arXiv 2605.15050](https://arxiv.org/abs/2605.15050). Necessary and sufficient conditions and computable bounds for hallucination depending only on the forward model — [arXiv 2605.13146](https://arxiv.org/abs/2605.13146). Hallucination entering DIS through the prior-side update (RPU) — [arXiv 2606.02331](https://arxiv.org/abs/2606.02331).
- Caution for any multi-sample protocol: posterior samples are heavy-tailed, so most samples share the same semantics, and "meaningful diversity" requires post-processing (Cohen et al., ICLR 2024) — [arXiv 2310.16047](https://arxiv.org/abs/2310.16047). Higher reconstruction accuracy "does not necessarily imply better posterior consistency" across DIS (score-KSD) — [arXiv 2602.04189](https://arxiv.org/abs/2602.04189).

### Inferences
- The central F1 construction (posterior samples of an IP solver → MLLM answers → answer entropy, split into measurement-determined, prior-determined and undetermined) appears **unclaimed** as of 22 Sep 2026. Adjacent claims are: (a) task-output uncertainty from posterior samples with classifiers (D1, D2), and (b) VLM answer entropy from image perturbations (VSE). A paper must cite both and distinguish itself by (i) measurement-consistent posterior samples rather than arbitrary perturbations, (ii) an open-vocabulary MLLM QA rather than a fixed classifier, and (iii) the prior-versus-measurement attribution.
- Two technical risks that reviewers will raise, each backed by a cited paper. Few DIS samples may under-represent rare semantic modes (Cohen et al.), and DIS posteriors may be miscalibrated (score-KSD paper). F1 should include a calibration check, e.g. on synthetic problems with known posteriors.

### Gaps
- It cannot be ruled out that some 2026 workshop paper or non-indexed preprint samples a DIS and queries a VLM. Semantic Scholar cited-by sweeps were not run.
- The exact perturbation family in VSE (noise, crops or generative) was not retrieved; if it is generative resampling, the overlap with F1 grows.

---

## KQ3: Which works use text or VLMs to shape the prior or regularizer of a measurement-consistent solver? How do they handle hallucinated content? Does anyone compare competing text hypotheses via likelihood ratio, evidence or Bayes factor?

### Takeaway
Text shapes the prior in TReg, P2L, LATINO-PRO, LADiBI, Text-guided Explorable SR, MCS, FlowDPS, LD-RPS, 2510.03352 and PDLS. None explicitly guards against text-induced confirmation bias beyond generic remedies such as recurrence, null-text negation or reward ablations. **No paper computes a likelihood ratio or Bayes factor between competing text hypotheses given y.** LATINO-PRO comes closest by maximizing p(y|c) over a continuous prompt, estimating only its gradient. DiME computes evidence values for unconditional priors.

### Cited Findings
- TReg shows counterfactual prompts (differing from the true class) can yield data-consistent solutions (low y-MSE plus CLIP alignment to the prompt), and it never chooses between prompts. It acknowledges residual ambiguity ("residual symmetry") for simple prompts — [TReg HTML](https://arxiv.org/html/2311.15658).
- LATINO-PRO sets ĉ(y) = argmax_c p(y|c) by SAPG using the Fisher identity. p(y|c) is never evaluated, discrete candidate prompts are never compared, and hallucination is not discussed — [LATINO-PRO HTML](https://arxiv.org/html/2503.12615).
- LD-RPS admits "strong hallucinations" early in sampling and mitigates them by recurrent re-sampling; its prompts come from an MLLM reading the degraded image — [LD-RPS HTML](https://arxiv.org/html/2507.00790v1).
- FlowDPS takes prompts from DAPE or LLaVA and runs no prompt-effect or hallucination ablation — [FlowDPS HTML](https://arxiv.org/html/2503.08136).
- LADiBI takes prompts from the user and names only generic generative-model risks — [LADiBI HTML](https://arxiv.org/html/2412.00557v2).
- 2510.03352 ablates reward robustness and notes perceptual gains do not guarantee PSNR/LPIPS/SSIM gains, with no text-hallucination analysis — [arXiv HTML v3](https://arxiv.org/html/2510.03352v3).
- Multiple prompts, multiple consistent solutions, no scoring: Text-guided Explorable SR — [arXiv 2403.01124](https://arxiv.org/abs/2403.01124); MCS — [arXiv 2511.14213](https://arxiv.org/abs/2511.14213).
- Evidence as a decision rule, without text: DiME selects among 10 MNIST class priors from one noisy measurement by estimated evidence — [DiME HTML](https://arxiv.org/html/2602.20549).
- Text-conditional *likelihood* as a zero-shot classifier on **clean** images: Diffusion Classifier (Li et al., ICCV 2023) picks the class prompt with the best denoising ELBO — [arXiv 2303.16203](https://arxiv.org/html/2303.16203v3). Clark & Jaini show T2I diffusion models are zero-shot classifiers (NeurIPS 2023) — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/b87bdcf963cad3d0b265fcb78ae7d11e-Paper-Conference.pdf).

### Inferences
- F2's ingredients all exist separately:
  1. text-conditioned DIS: TReg, FlowDPS;
  2. prompt scoring by p(y|c): LATINO-PRO, gradient only;
  3. evidence estimators for DIS: DiME;
  4. text-conditional likelihood classification on clean x: Diffusion Classifier.
- **The combination is unclaimed**: evidence p(y|q,a) for discrete answer hypotheses derived from a VQA question, with symmetric text regularization and a Bayes-factor decision. The obvious reviewer objection will be "LATINO-PRO + DiME + Diffusion Classifier". F2 should position itself as measurement-level diffusion classification with open-vocabulary hypotheses generated from (q, a), and report the flat-evidence (undetermined) regime.
- TReg's counterfactual-prompt experiment is direct evidence that under strong degradation several hypotheses reach similar y-MSE. Evidence ratios may therefore be close to 1. That is a feature for F2 (it flags "undetermined") but could make the paper's headline accuracy weak unless the degradation levels are graded.

### Gaps
- No paper was found that computes p(y|c) values, rather than gradients, for text-conditioned latent diffusion or flow priors. Whether DiME's estimator transfers to latent, text-conditioned priors with CFG is untested in the literature found.
- Whether any 2026 paper extends Diffusion Classifier to degraded inputs via a likelihood p(y|c) was searched but not found. Absence is not proof.

---

## KQ4: Which works estimate the forward operator with a VLM (blind inverse problems) and then solve with data consistency?

### Takeaway
DU-VLM (arXiv 2602.04565) is the clearest case: a VLM predicts physical degradation parameters that are injected into DDNM. LADiBI (NeurIPS 2025) encodes the operator prior through text prompts, not a VLM, and fits the operator by MLE inside a latent DIS. LD-RPS (ICCV 2025) learns an alignment operator while an MLLM supplies prompts. Other VLM degradation-understanding works do not use a DC solver.

### Cited Findings
- DU-VLM: hierarchical, CoT-anchored prediction of degradation type, parameter keys and continuous values; SFT plus RL with structured rewards; "zero-shot controller" injecting D̂ into pre-trained DDNM with explicit physics models (haze, blur, low-light, SR) — [arXiv abs](https://arxiv.org/abs/2602.04565); [HTML](https://arxiv.org/html/2602.04565).
- LADiBI: a parameterized operator (61×61 kernel, or a small U-Net for JPEG) is initialized from SDEdit/MPGD pseudo-estimates and refined by MLE with L1 and LPIPS; DC via MPGD on ‖y − A_φ(D(z))‖² — [arXiv HTML v2](https://arxiv.org/html/2412.00557v2).
- LD-RPS: the F-PAM learnable module models the latent-to-image and degraded-to-clean gap; MLLM-generated prompts — [arXiv HTML](https://arxiv.org/html/2507.00790v1).
- 2510.03352 runs Blind DPS as one of its base solvers under text-reward search — [arXiv HTML v3](https://arxiv.org/html/2510.03352v3).
- VLM degradation modeling *without* a DC solver: MVLR (ICAC 2025) retrieves prototypes into an encoder-decoder — [arXiv 2511.16998](https://arxiv.org/abs/2511.16998). DA-CLIP's degradation embeddings condition a restoration network — [arXiv 2310.01018](https://arxiv.org/abs/2310.01018). VLU-Net drives an unrolled gradient step with CLIP degradation features — [arXiv 2503.16930](https://arxiv.org/html/2503.16930).

### Inferences
- For F1 and F2 under real (unknown) degradations, a DU-VLM-style estimator plus a DIS is a defensible pipeline, but it adds operator-misspecification error. None of these works propagates operator uncertainty into downstream semantic uncertainty. That is a potential sub-contribution for F1 (marginalizing over the estimated operator).
- For F3, a VLM-estimated operator gives the missing "A" needed to add DC to BAGEL's generator. DU-VLM is the precedent to cite; no work closes that loop *inside* the same unified MLLM.

### Gaps
- DU-VLM's handling of nonlinear models inside DDNM, which strictly handles linear A, was not verified.
- No work was found that has a VLM output a *distribution* over operators, or that evaluates downstream VQA after VLM-estimated-operator restoration.

---

## KQ5: Does any paper solve inverse problems in VLM-encoder feature or token space, or treat the VLM answer as the quantity of interest in Bayesian inversion (goal- or task-oriented Bayesian inversion)?

### Takeaway
Feature-space inversion exists but predates the window or uses vision-only encoders. Ravula et al. (NeurIPS 2021) recover CLIP representations from A(x); Align & Invert (2025/26) regularizes DIS with DINOv2 alignment. Task-oriented Bayesian inversion exists for classifiers (ECCV 2024; arXiv Sep 2026) and for linear-Gaussian QoIs (classical goal-oriented inversion). **No paper treats a VLM or MLLM answer as the QoI of a Bayesian inverse problem.**

### Cited Findings
- Recovering the CLIP representation R(x) from A(x) with known A, via contrastive student training; linear-probe classification beats end-to-end baselines under blur, noise and masking (NeurIPS 2021) — [arXiv 2110.07439](https://arxiv.org/abs/2110.07439); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2021/hash/498f940d9b933c529b06aa96d18f7eda-Abstract.html).
- REPA with DINOv2 inside DIS/flow solvers, interpreted as variational divergence minimization in DINOv2 embedding space — [arXiv 2511.16870](https://arxiv.org/abs/2511.16870).
- Task-centred UQ that constructs intervals on a downstream classifier's output from posterior samples — [arXiv 2405.18527](https://arxiv.org/abs/2405.18527). Semantic (class) posterior uncertainty steering sensing — [arXiv 2609.21812](https://arxiv.org/abs/2609.21812).
- Classical goal-oriented Bayesian inversion: optimal low-rank approximations of the QoI posterior for linear-Gaussian problems — [arXiv 1607.01881](https://arxiv.org/html/1607.01881).
- CLIP embedding inversion with an implicit neural representation (decoding, not a degraded-measurement IP) — [arXiv 2505.23161](https://arxiv.org/html/2505.23161v1).

### Inferences
- "Goal-oriented Bayesian inversion where the QoI is an MLLM answer to question q" is an **unclaimed framing** that unifies F1 (the posterior predictive of the answer) and F2 (evidence per answer). The classifier-based works (D1, D2) and the linear-Gaussian goal-oriented literature are the natural citations for it.

### Gaps
- No work was found solving an IP directly in a VLM *token* space, e.g. inferring LLaVA or Qwen-VL visual tokens from y. It may exist under different wording (e.g. "feature restoration for MLLMs"), but that falls under non-IP restorers, which are out of scope.

---

## KQ6: Threat assessment for the user's formulations F1, F2 and F3

### Takeaway
- **F1: overall LOW-MEDIUM threat.** The core pipeline is unclaimed. The closest neighbours are classifier-based task UQ (ECCV 2024; 18 Sep 2026 arXiv) and VLM visual-perturbation entropy (ECCV 2026).
- **F2: overall MEDIUM threat.** Every ingredient exists separately: TReg for text-conditioned DIS, LATINO-PRO for MML over p(y|c), DiME for evidence values with DIS, and Diffusion Classifier for text-likelihood classification on clean images. The specific VQA-hypothesis Bayes-factor decision was not found.
- **F3: overall LOW-MEDIUM threat.** Robust-U1 and CLEAR have no DC. DC inside rectified-flow latent generators is established (FlowDPS). No work adds DC to a unified MLLM's self-restoration.

### Cited Findings
- **F1 nearest neighbours.**
  - Posterior samples → classifier → conformal task uncertainty (ECCV 2024, MRI) — [arXiv 2405.18527](https://arxiv.org/abs/2405.18527).
  - Diffusion posterior samples → calibrated classifier → within/between-class uncertainty (arXiv, 18 Sep 2026) — [arXiv 2609.21812](https://arxiv.org/abs/2609.21812).
  - Image-only perturbations → VLM answer clusters → entropy (ECCV 2026) — [arXiv 2606.31407](https://arxiv.org/abs/2606.31407).
  - Pixel-space split of intrinsic ambiguity from estimation uncertainty — [arXiv 2605.15050](https://arxiv.org/abs/2605.15050).
  - Hallucination bounds from forward-operator ill-posedness — [arXiv 2605.13146](https://arxiv.org/abs/2605.13146).
- **F2 nearest neighbours.**
  - Counterfactual-prompt data-consistent reconstructions (TReg, ICLR 2025) — [TReg HTML](https://arxiv.org/html/2311.15658).
  - argmax_c p(y|c) via the Fisher identity (LATINO-PRO, ICCV 2025) — [HTML](https://arxiv.org/html/2503.12615).
  - Evidence p(y|M) with DAPS/PnP-DM samples for prior selection, 10 MNIST class priors (DiME) — [HTML](https://arxiv.org/html/2602.20549).
  - Text-conditional ELBO classification (ICCV 2023) — [arXiv 2303.16203](https://arxiv.org/html/2303.16203v3).
  - Text as reward over DIS particles — [arXiv 2510.03352](https://arxiv.org/html/2510.03352v3).
- **F3 nearest neighbours.**
  - Robust-U1: BAGEL self-recovery with a rectified-flow SFT loss and Flow-GRPO; no measurement term at inference; single output — [HTML](https://arxiv.org/html/2606.08063v1).
  - CLEAR: unified-model generation for degraded understanding; no DC found — [arXiv 2604.04780](https://arxiv.org/pdf/2604.04780).
  - FlowDPS: DPS-style DC in an SD3 rectified-flow latent — [HTML](https://arxiv.org/html/2503.08136).
  - DU-VLM: a VLM supplies the operator to a DC solver — [HTML](https://arxiv.org/html/2602.04565).
  - LADiBI: operator MLE plus latent DC — [HTML](https://arxiv.org/html/2412.00557v2).

#### Per-paper threat lines (compact)
| Paper | F1 | F2 | F3 |
|---|---|---|---|
| TReg | none: CLIP used only as a metric | **medium**: machinery for F2; shows multiple prompts fit y; no evidence/BF | none |
| P2L | none | low-med: point-estimates the prompt from y | none |
| LATINO-PRO | none | **med-high**: MML over p(y\|c), the formal precursor; no evidence values or discrete answers | low |
| LADiBI | none | low | low: estimated-operator DC precedent |
| Text-guided Explorable SR | low | low-med: hypothesis-conditioned consistent SR without scoring | none |
| MCS | none | low-med: same as above for faces | none |
| FlowDPS | none | low | low-med: DC in rectified-flow latents is known |
| LD-RPS | none | low | low-med: MLLM prompts plus learned-operator DC |
| 2510.03352 | low | low-med: text as reward, not likelihood | low |
| PDLS | none | low | low |
| DU-VLM | low | low | low-med: VLM→operator→DC precedent |
| VLU-Net | none | none | none |
| SUPIR / DA-CLIP / GLYPH-SR / VLMIR / TPGDiff | none-low: no DC; GLYPH-SR uses OCR F1 single-sample | none | none-low |
| Task-driven conformal UQ (ECCV'24) | **medium**: task-output uncertainty from posterior samples (classifier, MRI) | low | none |
| Classification-oriented adaptive sensing (Sep'26) | **medium**: semantic posterior uncertainty via classifier on DIS samples | low-med | none |
| DiME | low | **high (estimator)**: evidence-based hypothesis selection with DIS exists | none |
| Hallucination Score | low-med: MLLM judges restoration hallucination | none | none |
| High-level fidelity SR | low | none | none |
| Visual Semantic Entropy (ECCV'26) | **medium**: VLM answer entropy under image perturbations | none | none |
| Ravula 2021 / Align & Invert | low: feature-space IP precedent | none | none / low |
| Robust-U1 | n/a | n/a | baseline: gap confirmed (no DC) |
| CLEAR | low | none | low-med: competing unified-model framing, no DC found |
| Meaningful Diversity (ICLR'24), score-KSD, RPU, CHEM | low: cautions on sampling and calibration | none | none |

### Inferences
- The most defensible headline contribution is **F1 combined with F2 under a "goal-oriented Bayesian inversion with an MLLM answer as QoI" framing** (KQ5). F1 gives the posterior predictive over answers. F2 gives per-answer evidence, recovering the "undetermined" case when Bayes factors are near 1, as TReg's counterfactual-prompt results suggest will happen under heavy degradation.
- The 18 Sep 2026 classification-oriented sensing preprint and ECCV-2026 VSE show that this space is being approached from both sides (IP→classifier and VLM→perturbation entropy). Priority risk is real on a months-long horizon.
- For F3, the reviewer objection will be "FlowDPS already does DC in rectified flows". The answer must show what DC adds specifically to *self-recovery for understanding*, e.g. answer accuracy and the measurement-determined fraction, not only PSNR.

### Gaps
- The Robust-U1 venue ("AAAI") reported by the fetch tool is unverified and likely wrong for a June-2026 arXiv paper.
- Robust-U1 inference latency was not found in the main text (it may be in its appendix).
- CLEAR's base model and any consistency mechanism are unverified.
- No Semantic Scholar cited-by sweep was run for TReg, P2L, DPS, PSLD, DAPS, DDNM or DU-VLM. Recommended as the next verification step before submission.
