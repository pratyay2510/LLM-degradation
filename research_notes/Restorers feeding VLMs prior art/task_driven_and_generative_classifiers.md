# Task-driven image restoration, posterior-sampling-to-task precedents, and generative classifiers through a forward operator (2023 – Sep 2026)

Scope note: this sweep covers (1) task-driven / machine-perception restoration (TDIR), (2) works that push diffusion posterior samples into a downstream classifier/detector/segmenter to obtain task-level uncertainty or decisions, (3) generative classifiers and class-/model-conditional evidence computed on degraded measurements, and (4) goal-oriented / decision-aware Bayesian inversion. MLLM-specific restore-then-answer pipelines are out of scope and only flagged when encountered.

Venue verification legend used below: [arXiv-comment] = venue stated in the arXiv comment or journal-ref line; [proceedings] = proceedings/journal page located (CVF, NeurIPS, PMLR, ECVA/Springer, IOP, IEEE DOI); [S2-venue] = venue field from the Semantic Scholar API only (weaker); [unverified] = could not confirm.

Formulation labels used in threat ratings: F1 = answer identifiability via MLLM answer entropy over measurement-consistent posterior samples; F2 = hypothesis-conditioned evidence p(y | q, a) via a text-conditioned diffusion inverse solver, decided by Bayes factor; F3 = data-consistency guidance on a unified MLLM's own generative self-restoration.

---

## Key Question 1: Which TDIR methods use (or don't) a forward operator, data consistency, or posterior sampling, and do any propagate posterior uncertainty to the task output?

### Takeaway
Every verified top-venue TDIR method found (UniRestore CVPR 2025, EDTR ICCV 2025, TaskTok ECCV 2026, DORA TPAMI 2026) is a supervised restoration network that may borrow a pretrained diffusion prior but does not model an explicit forward operator, does not enforce measurement/data consistency, and does not draw or propagate posterior samples; the only work that touches sampling stochasticity in TDIR (Noise-Free One-Step LoRA, arXiv 2026) treats it as a nuisance to be removed rather than a signal. "Restore-or-not" utility prediction exists only as arXiv-only work as of Sep 2026 (TaskGuard, SafeRestore).

### Cited Findings (per-paper catalog)

**UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior** — I-Hsiang Chen et al. — CVPR 2025 (Highlight) [arXiv-comment: "Accepted by CVPR2025 (Highlight)"] — arXiv 2501.13134, https://arxiv.org/abs/2501.13134
- Downstream task/model: classification and segmentation heads ("task-oriented image restoration", TIR) alongside perceptual restoration (PIR).
- Restorer type: Stable-Diffusion prior adapted through autoencoder encoder features (Complementary Feature Restoration Module + task feature adapter), supervised.
- Forward operator: N (per abstract; degradations are weather/blur/noise handled blindly). Data consistency: N. Posterior samples propagated downstream: N. Evidence/hypothesis comparison through operator: N.
- Key finding: a single diffusion-prior restorer can be steered by task-feature adapters to serve both human-perceptual and machine-task objectives; "diffusion prior ... images are often unsuitable for TIR scenarios" motivates the task adapter — https://arxiv.org/abs/2501.13134
- Threat: F1 none (no sampling, no uncertainty); F2 none; F3 low (shows a generative prior can be adapted to machine tasks, but no measurement consistency).
- Citation-graph note: the Semantic Scholar citers of UniRestore (54 as of Sep 2026) include TaskTok, EDTR, TaskGuard, SafeRestore, Noise-Free One-Step LoRA, DORA, FDIR, and PixRestore — this list was used as the TDIR seed set — https://api.semanticscholar.org/graph/v1/paper/arXiv:2501.13134/citations

**Exploiting Diffusion Prior for Task-driven Image Restoration (EDTR)** — Jaeha Kim, Junghun Oh, Kyoung Mu Lee — ICCV 2025 [arXiv-comment: "Accepted to ICCV 2025"; CVF PDF located: https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Exploiting_Diffusion_Prior_for_Task-driven_Image_Restoration_ICCV_2025_paper.pdf] — arXiv 2507.22459, https://arxiv.org/abs/2507.22459
- Downstream task/model: high-level vision tasks under multiple complex degradations (detection/segmentation/classification per abstract framing "high-level vision tasks").
- Restorer type: pre-restoration network followed by a pretrained diffusion prior run with a small number of denoising steps, trained with task loss.
- Forward operator: N. Data consistency: N (consistency is via pixel-space pre-restoration, not a measurement model). Posterior samples propagated downstream: N. Evidence through operator: N.
- Key finding: naively combining the diffusion prior with TDIR fails to restore task-relevant details; EDTR "effectively harnesses the power of diffusion prior to restore task-relevant details" via pre-restoration plus few-step denoising — https://arxiv.org/abs/2507.22459
- Threat: F1 none; F2 none; F3 low (generative prior for tasks, no DC).

**TaskTok: Delving into Task Tokens for Task-driven Image Restoration** — Hongjae Lee, Sojung Kang, Jaeseong Yu, Seung-Won Jung — ECCV 2026 [arXiv-comment: "ECCV 2026"; proceedings page not yet checkable] — arXiv 2606.26615, https://arxiv.org/abs/2606.26615
- Downstream task/model: high-level vision tasks (TDIR); analysis of latent token space of a generative-prior restorer.
- Restorer type: generative-prior latent restorer that selectively refines only task-relevant latent tokens ("index-wise specialization").
- Forward operator: N. Data consistency: N. Posterior samples downstream: N. Evidence: N.
- Key finding: "task-relevant cues are unevenly distributed across the token sequence"; refining a subset of tokens suffices for TDIR and avoids "semantic alteration by indiscriminately updating all latent tokens" — https://arxiv.org/abs/2606.26615
- Threat: F1 low (its observation that full-latent regeneration semantically alters content is an argument for measurement-anchoring, adjacent to F1's prior-determined bucket); F2 none; F3 low.

**Breathing New Life into Small Object Detection with Detection-Oriented Rectification (DORA)** — (first author not captured in the sweep) — IEEE TPAMI 2026 [proceedings: DOI 10.1109/TPAMI.2026.3704810; PubMed 42308073] — no arXiv ID found.
- Downstream task/model: small object detection. Restorer type: "degradation-then-rectification" framework that learns a degradation basis set, forms a degradation-conditioned prompt, and does task-oriented rectification with a contrastive alignment of rectified embeddings to detection-friendly exemplars — https://pubmed.ncbi.nlm.nih.gov/42308073/
- Forward operator: learned degradation basis (not a physical operator). Data consistency: N. Posterior samples: N. Evidence: N.
- Threat: F1/F2/F3 none.

**Noise-Free One-Step LoRA for Task-Driven Image Restoration with Diffusion Priors** — Jaeha Kim, Kyoung Mu Lee — arXiv-only (Jul 2026), venue [unverified] — arXiv 2607.25390, https://arxiv.org/abs/2607.25390
- Key finding (directly relevant to F1): "diffusion-based restoration is inherently stochastic, as the sampling process depends on a random noise term, which can undermine task consistency"; a deterministic, noise-free one-step pass with LoRA adaptation "substantially improve[s] TDIR" — https://arxiv.org/abs/2607.25390
- Forward operator: N. DC: N. Posterior samples downstream: N (explicitly eliminates sampling). Evidence: N.
- Threat: F1 medium as a counter-argument (they document that sample-to-sample stochasticity changes task outputs, i.e., the very phenomenon F1 measures, but frame it as a defect to remove rather than an identifiability signal); F2 none; F3 low.

**TaskGuard: Task-Conditioned Restoration Utility for Risk-Aware Object Detection** — Vung Pham — arXiv-only (Sep 2026), venue [unverified; comment line gives page counts only] — arXiv 2609.08011, https://arxiv.org/abs/2609.08011
- Downstream task/model: object detection with frozen restorer + frozen detector. Restorer type: post-hoc controller (not a restorer) that predicts "restoration utility": "should the restoration be used or should the original observation be preserved?" — https://arxiv.org/abs/2609.08011
- Forward operator: N. DC: N. Posterior samples: N. Evidence: N (uses "task-conditioned evidence" in the sense of detector-sensitivity features, not likelihood evidence).
- Key finding: "a visually improved image need not improve the downstream task"; exact regional counterfactuals reveal within-image utility heterogeneity.
- Threat: F1 low (shares the premise that restoration can hurt the task, but no posterior/uncertainty); F2 none (the word "evidence" is not Bayesian); F3 low.

**SafeRestore: Detector-Relative Risk Certificates for Selective Industrial Image Restoration** — Shaoliang Yang, Jun Wang — arXiv-only (Sep 2026) [unverified] — arXiv 2609.03475, https://arxiv.org/abs/2609.03475
- Downstream: industrial defect detector. Formulates restoration as a selective action over {measured image, five restored candidates, review}, with exact binomial certificates on "positive-conditional evidence-loss incident rate" — https://arxiv.org/abs/2609.03475
- Forward operator / DC / posterior samples / evidence through operator: N / N / N / N.
- Companion arXiv 2607.17401 ("Does Super-Resolution Preserve Defect Evidence?") reports that "Reconstruction fidelity and inspection utility diverge: the two learned reconstruction models attain the highest structural similarity yet detect fewer defect pixels than bicubic interpolation" — https://arxiv.org/abs/2607.17401
- Threat: F1 low; F2 none; F3 low.

**Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior** — Ruoyu Feng et al. — NeurIPS 2025 [arXiv-comment: "Accepted by NeurIPS 2025"] — arXiv 2511.22549, https://arxiv.org/abs/2511.22549
- Adjacent (compression, not restoration): generative-prior decoder with a Semantic Consistency loss to serve both machine tasks and human perception; cites Diffusion Classifier. Forward operator: the codec (Y, learned); DC: N; posterior samples downstream: N; evidence: N.
- Threat: F1/F2 none; F3 low.

**Restore, Assess, Repeat (RAR)** — I-Hsiang Chen et al. — CVPR 2026 [arXiv-comment: "Accepted by CVPR2026"] — arXiv 2603.26385, https://arxiv.org/abs/2603.26385
- "Restore-or-not"-adjacent but quality-driven, not task-driven: integrates IQA into an iterative latent restore/verify loop. No operator, no DC, no posterior sampling, no downstream task. Threat: none for F1–F3.

**How far have we gone in Generative Image Restoration?** — Xiang Yin et al. — CVPR 2026 Findings [arXiv-comment: "Accepted by CVPR 2026 Findings"; note this is a Findings track, not the main proceedings] — arXiv 2603.05010, https://arxiv.org/abs/2603.05010
- Large-scale study of generative restoration; identifies the failure-mode shift from "detail scarcity (under-generation)" to "detail quality and semantic control (preventing over-generation)" and evaluates "semantic correctness" — https://arxiv.org/abs/2603.05010
- Threat: F1 low (semantic over-generation is exactly the prior-determined content F1 wants to separate); F2/F3 none.

**Context (pre-window, cited by all TDIR papers):** Rethinking Image Restoration for Object Detection — Shangquan Sun et al. — NeurIPS 2022 [proceedings: https://proceedings.neurips.cc/paper_files/paper/2022/hash/1cac8326ce3fbe79171db9754211530c-Abstract-Conference.html]; reports "the existing image restoration methods cannot improve the object detector performance and sometimes even reduce the detection performance" and trains restoration with a targeted adversarial pseudo-GT. URIE (Son et al., ECCV 2020) is the canonical "restoration for recognition" precedent; its arXiv ID was not verified in this sweep.

**Other arXiv-only TDIR/analysis items encountered (not catalogued in detail):** FDIR (2608.00111; three-way fidelity/perception/machine-preference tradeoff for compression restoration, says generative approaches "hallucinate plausible but factually incorrect textures that degrade ... downstream task accuracy" — https://arxiv.org/abs/2608.00111); Degradation-Aware AIR via Latent Prior Encoding (2509.17792; annotates 3,000 degraded test images and benchmarks restorers with YOLOv12 detection — https://arxiv.org/abs/2509.17792); LL-Bench (2606.02535); Task-Guided Prompting for RS restoration (IEEE TGRS 2026, not a listed venue).

### Inferences
- The TDIR literature's "task loss on restorer output" paradigm is orthogonal to inverse-problem machinery: none of the verified papers has a likelihood term, so none can express "measurement-determined vs prior-determined" content. F1's decomposition therefore has no TDIR precedent; the closest conceptual relatives are the restoration-utility papers (TaskGuard, SafeRestore) which ask "did restoration help the detector?" post hoc, without a posterior.
- The Noise-Free One-Step LoRA finding that sampling stochasticity changes task outputs is the strongest empirical support (and the strongest rhetorical competitor) for F1: a reviewer could ask why not just make the restorer deterministic; the answer must be that determinism hides ambiguity rather than resolving it.

### Gaps
- Could not retrieve full papers for UniRestore/EDTR/TaskTok to confirm the exact task heads and whether any ablation uses multiple samples; "forward operator N / DC N" is from abstracts and the method descriptions in search summaries.
- TPAMI DORA first author and arXiv ID not captured.
- No verified top-venue "restore-or-not" decision paper was found in 2023–2026; both candidates are arXiv-only.

---

## Key Question 2: Which works compute task-level uncertainty or decisions from posterior samples (classification entropy over samples, conformal sets over task outputs)? Domains?

### Takeaway
The direct precedent is Wen, Ahmad, Schniter (ECCV 2024): conformal intervals on a soft-output classifier's output computed over diffusion/CGAN posterior samples in accelerated MRI, with a multi-round acquisition rule; its TMLR 2025 sequel bounds full-reference IQ metrics the same way. Beyond that, the "posterior samples to task" idea appears in adaptive sensing (AdaSense ECCV 2024, reconstruction-variance criterion) and, in arXiv-only form eight days ago, a classification-driven variant that decomposes posterior variance into within-class and between-class terms using classifier outputs on diffusion posterior samples (2609.21812) — the closest thing to F1's entropy-over-samples found.

### Cited Findings (per-paper catalog)

**Task-Driven Uncertainty Quantification in Inverse Problems via Conformal Prediction** — Jeffrey Wen, Rizwan Ahmad, Philip Schniter — ECCV 2024 [arXiv-comment: "European Conference on Computer Vision, 2024"; ECVA PDF: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07734.pdf; Springer chapter 10.1007/978-3-031-73027-6_11] — arXiv 2405.18527, https://arxiv.org/abs/2405.18527
- Downstream task/model: "soft-output classification" (real-valued task output) applied to reconstructed images. Restorer type: posterior-sampling image recovery (the framework is recovery-agnostic; "for posterior-sampling-based image recovery, we construct locally adaptive prediction intervals").
- Forward operator: Y (imaging inverse problem, accelerated MRI in the experiments per the PMC/ECVA versions). Data consistency: Y (inherent to the recovery). Posterior samples propagated downstream: Y — the task function is evaluated on each posterior sample and a conformal interval is calibrated on the resulting spread. Evidence through operator: N.
- Key finding: "we use conformal prediction to construct an interval that is guaranteed to contain the task output from the true image up to a user-specified probability, and we use the width of that interval to quantify the uncertainty contributed by measurement-and-recovery"; additionally, "we propose to collect measurements over multiple rounds, stopping as soon as the task uncertainty falls below an acceptable level" — https://arxiv.org/abs/2405.18527
- Threat: F1 HIGH — this is task-level uncertainty from posterior samples with a "measurement-and-recovery contributed" interpretation and an acquisition-stopping rule; F1 differs in (a) discrete answers from an MLLM rather than a scalar soft output, (b) entropy/three-way split instead of conformal interval, (c) natural images/VQA rather than MRI. F2 low. F3 none.
- Citation graph: the Semantic Scholar citations endpoint for this paper returned HTTP 429 on every attempt, so follow-ups could not be enumerated (see Gaps).

**Conformal Bounds on Full-Reference Image Quality for Imaging Inverse Problems** — Jeffrey Wen, Rizwan Ahmad, Philip Schniter — Transactions on Machine Learning Research, May 2025 [arXiv journal-ref line] — NOT on the user's venue list — arXiv 2505.09528, https://arxiv.org/abs/2505.09528
- Combines conformal prediction with approximate posterior sampling to bound PSNR/SSIM/LPIPS of a recovery without ground truth; denoising and accelerated MRI — https://arxiv.org/abs/2505.09528
- Forward operator Y; DC Y; posterior samples propagated to a (metric) function Y; evidence N. Threat: F1 medium (same machinery, but the "task" is an IQ metric, not a semantic decision); F2/F3 none.

**Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling (AdaSense)** — Noam Elata, Tomer Michaeli, Michael Elad — ECCV 2024 [arXiv-comment; ECVA PDF https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10059.pdf] — arXiv 2407.08256, https://arxiv.org/abs/2407.08256
- Downstream: none (reconstruction). Uses zero-shot diffusion posterior sampling; "By sequentially sampling from the posterior distribution, we can quantify the uncertainty of each possible future linear measurement" — https://arxiv.org/abs/2407.08256
- Forward operator Y; DC Y; posterior samples used for a decision (which measurement to take next) Y, but the decision is reconstruction-driven; evidence N.
- Threat: F1 low-medium (establishes "posterior samples → decision" in a natural-image CS setting); F2 none; F3 none.

**Classification-oriented adaptive sensing via posterior sampling** — Andriy Enttsel, Maxime Rousselot, Vincent Corlay — arXiv-only (submitted 18 Sep 2026) [unverified] — arXiv 2609.21812, https://arxiv.org/abs/2609.21812
- Downstream: classification (MNIST, CIFAR-10). "Using calibrated soft classifier outputs, we estimate these uncertainty terms [within-class and between-class] from diffusion posterior samples and propose a classification-oriented criterion for selecting the dominant sensing direction" — https://arxiv.org/abs/2609.21812
- Forward operator Y (linear CS); DC Y; posterior samples propagated to a classifier Y; evidence N.
- Threat: F1 HIGH (classifier outputs over diffusion posterior samples, explicitly separating class-relevant from class-irrelevant posterior uncertainty — a two-way version of F1's three-way split, on toy datasets, framed as sensing design); F2 low; F3 none. arXiv-only; see Appendix.

**From Posterior Sampling to Meaningful Diversity in Image Restoration** — Noa Cohen, Hila Manor, Yuval Bahat, Tomer Michaeli — ICLR 2024 [arXiv-comment] — arXiv 2310.16047, https://arxiv.org/abs/2310.16047
- Argues posterior sampling "is commonly of limited practical value because of the heavy tail of the posterior distribution" (the sky-inpainting example: samples are "dominated by (practically identical)" cloud fills) and proposes diversity-seeking sampling instead — https://arxiv.org/abs/2310.16047
- Forward operator Y; DC Y; downstream task N; evidence N.
- Threat: F1 medium — a reviewer will ask whether entropy over i.i.d. posterior samples under-represents rare-but-plausible answers; F1 should discuss diversity-seeking sampling as an alternative sample set. F2 none; F3 none.

**Looks Too Good To Be True: An Information-Theoretic Analysis of Hallucinations in Generative Restoration Models** — Regev Cohen, Idan Kligvasser, Ehud Rivlin, Daniel Freedman — NeurIPS 2024 [proceedings: https://proceedings.neurips.cc/paper_files/paper/2024/hash/2847d43f17410c5beb25b2736c3ae778-Abstract-Conference.html; OpenReview 85tu7K06i3] — arXiv 2405.16475, https://arxiv.org/abs/2405.16475
- Proves an uncertainty–perception tradeoff: "the global minimal uncertainty in generative models grows in tandem with perception" and "attaining perfect perceptual quality entails at least twice the inherent uncertainty of the restoration problem" — https://proceedings.neurips.cc/paper_files/paper/2024/hash/2847d43f17410c5beb25b2736c3ae778-Abstract-Conference.html
- Forward operator Y (theory over degradations); DC N/A; downstream task N; evidence N.
- Threat: F1 medium (theoretical backing that perceptually perfect restorers necessarily inject prior-determined content — supports the motivation for F1's split); F3 medium (a unified MLLM's self-restoration aims at perceptual output and hence at maximal hallucination; F3's DC guidance is the fix). F2 none.
- Citers (Semantic Scholar, 25): Hallucination-Aware Diffusion Sampling via Robust Prior Updates (arXiv 2606.02331, separates "prior update" from "measurement-conditioning step" in DPS-type solvers and shows hallucinated content enters through the prior side — https://arxiv.org/abs/2606.02331); CHEM conformal hallucination metric (arXiv 2512.09806); Hallucination Score for GSR (arXiv 2507.14367); Trustworthy SR via Generative Pseudoinverse (arXiv 2505.12375); Trautmann et al. (MLMI@MICCAI 2025, below). None of these is a listed venue.

**The Perception-Robustness Tradeoff in Deterministic Image Restoration** — Guy Ohayon, Tomer Michaeli, Michael Elad — ICML 2024 [proceedings: PMLR v235, pp. 38599–38638, https://proceedings.mlr.press/v235/ohayon24a.html] — arXiv 2311.09253
- Proves that a deterministic restorer approaching perfect perceptual quality and perfect measurement consistency must have Lipschitz constant going to infinity, hence is "necessarily more susceptible to adversarial attacks" — https://proceedings.mlr.press/v235/ohayon24a.html
- Threat: F3 medium (F3's DC guidance on a deterministic-looking MLLM self-restoration lands exactly in this regime; the paper motivates stochastic/posterior outputs instead); F1 low; F2 none.

**Posterior-Mean Rectified Flow (PMRF)** — Guy Ohayon, Tomer Michaeli, Michael Elad — ICLR 2025 [arXiv-comment] — arXiv 2410.00418, https://arxiv.org/abs/2410.00418
- Constructs the MMSE-optimal estimator under a perfect-perceptual-index constraint (posterior mean then rectified-flow transport). No downstream task, no posterior samples propagated. Threat: F1 low (a restorer output alternative to sampling), F2 none, F3 low.

**Mind the Detail: Uncovering Clinically Relevant Image Details in Accelerated MRI with Semantically Diverse Reconstructions (SDR)** — Jan Nikolas Morshuis, Christian Schlarmann, Thomas Küstner, Christian F. Baumgartner, Matthias Hein — MICCAI 2025 [arXiv-comment; Springer chapter 10.1007/978-3-032-04937-7_34] — NOT a listed venue but directly on point — arXiv 2507.00670, https://arxiv.org/abs/2507.00670
- Generates multiple reconstructions "with enhanced semantic variability while all of them are fully consistent with the measured data", then evaluates with an object detector trained on fastMRI+: SDR "significantly reduces the chance of false-negative diagnoses (higher recall) and improves mean average precision" — https://arxiv.org/abs/2507.00670
- Forward operator Y; DC Y; measurement-consistent samples propagated to a detector Y; evidence N.
- Threat: F1 HIGH in spirit (measurement-consistent semantically diverse samples fed to a downstream detector to expose decision ambiguity; medical domain, detector not MLLM); F3 low; F2 none.

**Evaluating structural uncertainty in accelerated MRI: are voxelwise measures useful surrogates?** — Luca L. C. Trautmann et al. — MLMI@MICCAI 2025 [S2-venue] — not a listed venue — arXiv 2503.10527
- Uses ensembles of reconstruction models with segmentation as the downstream task and shows "voxel level uncertainty does not provide insight into morphological uncertainty" — https://arxiv.org/abs/2503.10527
- Threat: F1 low-medium (task-level vs pixel-level uncertainty argument in the medical domain).

**PosteriorBench** — Jiachen Yao et al. (Anandkumar group) — arXiv-only (17 Sep 2026) [unverified] — arXiv 2609.20794, https://arxiv.org/abs/2609.20794
- Benchmark of distributional accuracy of generative inverse solvers on four physics inverse problems (Darcy, Poisson, CCS, light transport) against reference posteriors; documents "mode collapse, overconfident uncertainty, or averaging incompatible solutions" — https://arxiv.org/abs/2609.20794
- No downstream task. Threat: F1 low (a caution: the sample set F1 uses may not be posterior-faithful; F1 should cite when discussing solver choice); F2 low; F3 none.

**Posterior Information Dynamics of Diffusion Models for Linear Inverse Problems** — Xiangming Meng — arXiv-only (Aug 2026) [unverified] — arXiv 2608.21709
- Information-theoretic account of when measurement information enters reverse diffusion; for a uniform empirical prior on n samples "the measurement reduces the remaining explanation budget from H(I) = log n to the conditional Shannon entropy H(I | r)" — https://arxiv.org/abs/2608.21709
- Threat: F1 low-medium (conceptual: entropy of "which explanation" given the measurement is F1's quantity, but at the image-index level with no semantic task).

**User-defined Event Sampling and Uncertainty Quantification in Diffusion** — OpenReview sdhcjMzhHN — [unverified: OpenReview blocked automated fetch; venue and content not confirmed]. Flagged because the title suggests event-level (rather than pixel-level) uncertainty from diffusion samples.

### Inferences
- Within listed venues, the only verified "posterior samples → task-level uncertainty" work is Wen et al. ECCV 2024 (scalar soft-output classifier, conformal interval). No listed-venue paper computes an entropy/agreement statistic over discrete task outputs of posterior samples on natural images; the two closest (SDR, classification-oriented sensing) are MICCAI and arXiv respectively.
- F1's novelty therefore rests on (i) discrete MLLM answers, (ii) the three-way measurement-/prior-/undetermined split, and (iii) natural-image VQA; the underlying "spread of a task function over p(x|y)" primitive is established and must be cited (Wen 2024; Wen 2025 TMLR; Elata 2024; Morshuis 2025).

### Gaps
- Semantic Scholar citation graph for 2405.18527 was unavailable (HTTP 429 on three attempts), so 2025–2026 follow-ups to Wen et al. in listed venues could not be enumerated; a manual check of Schniter/Ahmad 2025–2026 publications is advised.
- Domain of Wen et al.'s experiments (accelerated MRI) is from the PMC/ECVA listing context, not re-read from the paper body.
- OpenReview page sdhcjMzhHN could not be fetched.

---

## Key Question 3: Has anyone computed class-conditional or text-conditional evidence of a degraded measurement, p(y | c) = ∫ p(y | x) p(x | c) dx, with a conditional diffusion prior, for classification or hypothesis testing?

### Takeaway
Two verified top-venue results together cover most of F2's mechanism, but neither is F2: (a) Noised Diffusion Classifiers (NeurIPS 2024) compute exactly p(y | c) for the special operator "identity plus Gaussian noise" via an ELBO on the noisy input, decided by Bayes' rule; (b) DiME (ICLR 2026) estimates the model evidence p(y | M) of a diffusion prior through an arbitrary (nonlinear, ill-conditioned) forward operator by integrating over the time-marginals of a posterior sampler, and selects among priors by that evidence. No paper found combines the two — a text/hypothesis-conditioned prior, a general forward operator, and Bayes-factor decision among semantic hypotheses.

### Cited Findings (per-paper catalog)

**Your Diffusion Model is Secretly a Zero-Shot Classifier (Diffusion Classifier)** — Alexander C. Li et al. — ICCV 2023 [arXiv-comment "In ICCV 2023"; CVF PDF https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf] — arXiv 2303.16203
- Class-conditional ELBO of the clean input x under each text/class condition, argmax over c; "density estimates from large-scale text-to-image diffusion models ... can be leveraged to perform zero-shot classification" — https://arxiv.org/abs/2303.16203
- Forward operator: N (input is the clean image; the only "degradation" is the diffusion noise schedule). DC: N. Posterior samples: N. Evidence/hypothesis comparison: Y but through the identity operator only.
- Reported robustness: "better robustness to distribution shift than competing discriminative classifiers" and effective robustness on ImageNet-A, but not on ImageNetV2/ObjectNet (per the paper's own text surfaced in search) — https://arxiv.org/pdf/2303.16203v3
- Threat: F2 HIGH as the foundational "text-conditional evidence → Bayes decision" recipe; F2 is this recipe pushed through a forward operator y = A(x) + n. F1 none. F3 none.
- Citation sweep (385 citers scanned; keyword filtered): no citer computes class-conditional evidence of a degraded measurement through a general forward operator. Items examined and ruled out below.

**Text-to-Image Diffusion Models are Zero-Shot Classifiers** — Kevin Clark, Priyank Jaini — NeurIPS 2023 [S2-venue only] — arXiv 2303.15233 — same identity-operator recipe; threat as above.

**Robust Classification via a Single Diffusion Model (RDC)** — Huanran Chen et al. — ICML 2024 [arXiv-comment "Accepted by ICML 2024"; journal-ref "ICML 2024"] — arXiv 2305.15241, https://arxiv.org/abs/2305.15241
- Generative classifier from a pretrained diffusion model; "RDC first maximizes the data likelihood" (a likelihood-maximization purification step) then classifies by conditional ELBO; targets adversarial robustness, not measurement degradation.
- Forward operator N; DC N; posterior samples N; evidence Y (identity operator). Threat: F2 medium (establishes generative-classifier robustness under perturbed inputs); F1/F3 none.

**Diffusion Models are Certifiably Robust Classifiers (Noised Diffusion Classifiers, EPNDC/APNDC)** — Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, Jun Zhu — NeurIPS 2024 [arXiv-comment "Accepted by NeurIPS 2024"; proceedings PDF https://proceedings.neurips.cc/paper_files/paper/2024/file/59a3444d39b97ba01a17994f938e1ccc-Paper-Conference.pdf] — arXiv 2402.02316
- "generalizes diffusion classifiers to classify Gaussian-corrupted data by deriving evidence lower bounds (ELBOs) for these distributions, approximating the likelihood using the ELBO, and calculating classification probabilities via Bayes' theorem"; EPNDC is the "mathematically rigorous version that calculates the likelihood of noisy data", APNDC an ensemble-like approximation — https://arxiv.org/abs/2402.02316
- Forward operator: Y, but restricted to y = x + σε (additive Gaussian noise, matched to the diffusion marginal). DC: implicit (the noisy input is treated as a diffusion time-marginal). Posterior samples: N. Evidence/hypothesis comparison through operator: Y (class-conditional evidence of the noisy measurement).
- Key numbers: over 80% / 70% certified robustness on CIFAR-10 at ℓ2 radii 0.25 / 0.5 using "a single off-the-shelf diffusion model without any additional data" — https://arxiv.org/abs/2402.02316
- Threat: F2 HIGH — this is p(y | c) of a degraded measurement under a class-conditional diffusion prior, decided by Bayes' rule; F2 must position itself as the extension to general (blur/downsampling/masking/JPEG) operators, text hypotheses, and open-vocabulary answers, and note that for Gaussian noise F2 reduces to EPNDC. F1 none; F3 none.

**Sample-efficient evidence estimation of score based priors for model selection (DiME)** — Frederic Wang, Katherine L. Bouman — ICLR 2026 [arXiv-comment "ICLR 2026"; project page https://imaging.cms.caltech.edu/dime/] — arXiv 2602.20549, https://arxiv.org/abs/2602.20549
- "evaluating the model evidence p(y | M) under different models M that specify the prior and then selecting the one with the highest value"; "DiME, an estimator of the model evidence of a diffusion prior by integrating over the time-marginals of posterior sampling methods"; uses "only a handful of posterior samples (e.g., 20)"; validated on "highly ill-conditioned, non-linear inverse problems, including a real-world black hole imaging problem" — https://arxiv.org/abs/2602.20549
- Forward operator: Y (general, nonlinear). DC: Y (posterior sampling). Posterior samples: Y (used to estimate evidence, not fed to a task). Evidence/hypothesis comparison through operator: Y — Bayes-factor model selection between diffusion priors given y.
- Threat: F2 HIGH — supplies the estimator F2 needs; the difference is that M indexes which prior/dataset, not which text hypothesis about the answer to a question; a text-conditioned prior p(x | q, a) is formally "a model M_a", so F2 can be described as DiME with hypothesis-indexed priors and must cite it as the evidence machinery. F1 low; F3 none.

**Bayesian model selection and misspecification testing in imaging inverse problems only from noisy and partial measurements** — Tom Sprunck, Marcelo Pereyra, Tobias Liaudat — arXiv-only (v3 May 2026) [unverified] — arXiv 2510.27663, https://arxiv.org/abs/2510.27663
- Unsupervised model selection/misspecification detection from y alone via "Bayesian cross-validation and data fission, a randomized measurement splitting technique", "compatible with any Bayesian imaging sampler, including diffusion and plug-and-play samplers" — https://arxiv.org/abs/2510.27663
- Forward operator Y; DC Y; posterior samples Y (for predictive scoring); hypothesis comparison through operator Y (between models, not semantic hypotheses).
- Threat: F2 medium-high (an alternative, evidence-free selection criterion — predictive scoring on held-out fission of y — that a reviewer could ask F2 to compare against); F1 low; F3 none.

**Generative Classifiers Avoid Shortcut Solutions** — Alexander C. Li, Ananya Kumar, Deepak Pathak — ICLR 2025 [arXiv-comment] — arXiv 2512.25034 — diffusion/AR generative classifiers "model all features, both core and spurious" and win on five distribution-shift benchmarks — https://arxiv.org/abs/2512.25034. Identity operator; threat F2 low (motivational).

**Your VAR Model is Secretly an Efficient and Explainable Generative Classifier** — Yi-Chung Chen, David I. Inouye, Jing Gao — ICLR 2026 [arXiv-comment] — arXiv 2510.12060 — AR generative classifier; identity operator; threat F2 low.

**Noise Matters: Optimizing Matching Noise for Diffusion Classifiers** — Yanghao Wang, Long Chen — NeurIPS 2025 [S2-venue only; arXiv comment empty] — arXiv 2508.11330 — reduces the variance of diffusion-classifier scores from random noise draws — https://arxiv.org/abs/2508.11330. Threat F2 low (variance-reduction trick reusable for F2's evidence estimates).

**Diffusion Classifiers Understand Compositionality, but Conditions Apply** — Yujin Jeong et al. — NeurIPS 2025 Datasets & Benchmarks [arXiv-comment] — arXiv 2505.17955 — benchmark of zero-shot diffusion classifiers on compositional prompts. Threat F2 low (evidence that text-conditioned ELBO scores discriminate compositional hypotheses, which F2's question-conditioned hypotheses will be).

**Conditional Diffusion Models are Medical Image Classifiers that Provide Explainability and Uncertainty for Free** — Gian Mario Favero et al. — MIDL 2025 [arXiv-comment] — not a listed venue — arXiv 2502.03687 — majority voting over diffusion-classifier reconstructions yields uncertainty. Identity operator. Threat F2 low, F1 low.

**Membership Inference on Text-to-Image Diffusion Models via Conditional Likelihood Discrepancy** — NeurIPS 2024 [S2-venue] — arXiv 2405.14800 — hypothesis test using conditional-vs-unconditional likelihood gaps; identity operator; threat none (noted as the nearest "likelihood-ratio test with a text-conditioned diffusion model").

**What happens to diffusion model likelihood when your model is conditional?** — Mattias Cross, Anton Ragni — workshop/arXiv (Sep 2024) [unverified] — arXiv 2409.06364 — reports that conditional DM likelihoods behave "surprisingly" (properties captured by conditional likelihoods are poorly understood) — https://arxiv.org/abs/2409.06364. A caution for F2's use of text-conditioned evidence.

**Direct Image Classification from Fourier Ptychographic Microscopy Measurements without Reconstruction** — arXiv 2505.05054 [unverified venue; not fetched] — discriminative classification directly on measurements (no generative prior); reported gains of 10–12% on CIFAR-10 from using all 25 measurements vs one (search summary). Threat none (no evidence computation).

**CNS-Bench** — Olaf Dünkel et al. — ICCV 2025 [arXiv-comment] — arXiv 2507.17651 — diffusion-generated continuous nuisance shifts for classifier robustness; not a generative classifier. Threat none; useful as a degradation benchmark.

### Inferences
- The pieces of F2 exist separately: text-conditional evidence with identity operator (Diffusion Classifier, ICCV 2023), class-conditional evidence of a Gaussian-noised measurement (NDC, NeurIPS 2024), and evidence of a general-operator measurement under a diffusion prior with Bayes-factor selection (DiME, ICLR 2026). Combining them for hypothesis-indexed text-conditioned priors over general degradations with VQA-style answer sets was not found in any listed venue or on arXiv through Sep 2026.
- A careful related-work paragraph should state F2 = "DiME-style evidence with the prior swapped for a hypothesis-conditioned text-to-image prior, reducing to NDC when A = I with Gaussian noise and to Diffusion Classifier when there is no degradation."
- Sprunck–Pereyra–Liaudat's predictive-scoring alternative is the most natural baseline a reviewer would request for F2's Bayes-factor.

### Gaps
- Semantic Scholar filtering of the 385 Diffusion Classifier citers was by title keyword; a citer with an uninformative title that computes evidence through an operator could have been missed.
- Did not verify whether NDC/EPNDC has been extended to non-Gaussian operators in 2025–2026 (searches "diffusion classifier compressed measurements", "generative classifier degraded measurement", "likelihood ratio test diffusion prior" returned nothing on point).
- No search hit for "zero-shot CLIP classification of posterior samples"; the query returned only zero-shot restoration solvers (LD-RPS ICCV 2025 uses MLLM semantic priors for restoration — out of scope, noted below).

---

## Key Question 4: Is there "decision-aware" or "goal-oriented" Bayesian inversion where the quantity of interest is a downstream decision rather than the image (SIAM / Inverse Problems literature)?

### Takeaway
Goal-oriented inversion in the applied-math literature targets low-dimensional scalar/functional quantities of interest (regularization parameters, expansion coefficients, PDE outputs) and optimal experimental design for those QoIs; no 2023–2026 SIAM-Imaging or Inverse-Problems paper was found whose QoI is a semantic decision/classification computed through a diffusion prior. Task-adapted reconstruction (Adler et al., Inverse Problems 2022) is the closest framework and predates the window.

### Cited Findings
- **Uncertainty quantification for goal-oriented inverse problems via variational encoder-decoder networks** — Babak Maboudi Afkham, Julianne Chung, Matthias Chung — Inverse Problems, 2024 [proceedings: IOP page fetched, https://iopscience.iop.org/article/10.1088/1361-6420/ad5373]. QoIs are "(1) optimal regularization parameters for tomography reconstruction, and (2) expansion coefficients defining conductivity fields in hydraulic tomography ... scalar functionals or low-dimensional summaries ... not downstream classifications"; VED networks map observations directly to QoI with posterior-predictive sampling. Forward operator Y; DC N (bypasses reconstruction); posterior samples propagated to QoI Y; evidence N. Threat: F1 low (establishes "uncertainty of a downstream quantity, not the image" as a first-class object), F2/F3 none.
- **Task adapted reconstruction for inverse problems** — Jonas Adler et al. — Inverse Problems 38(7), 2022 [IOP: https://iopscience.iop.org/article/10.1088/1361-6420/ac28ec] — context only (pre-window): formalizes reconstruction and task "as appropriate estimators in statistical estimation problems" and trains them end-to-end; the search summary notes the field "increasingly considers embedding inverse problems within broader, interconnected workflows".
- **Goal oriented optimal design of infinite-dimensional Bayesian inverse problems using quadratic approximations** — Journal of Scientific Computing 2025 (not a listed venue) — arXiv 2411.07532 — sensor placement minimizing posterior variance of a goal functional (G_q-optimality) — https://arxiv.org/abs/2411.07532. Context for "acquire measurements to resolve a downstream quantity" (relates to Wen 2024's multi-round acquisition and AdaSense).
- **Goal-oriented optimal approximations of Bayesian linear inverse problems** — Spantini et al. (SIAM, 2017) — context only, pre-window — https://arxiv.org/abs/1607.01881.

### Inferences
- The applied-math "goal-oriented" line supplies vocabulary (QoI, goal functional) but no precedent for semantic decisions; F1 and F2 can be framed as goal-oriented inversion where the goal is a discrete answer and the prior is a generative model, without conflicting with existing SIAM/IP results.

### Gaps
- SIAM Journal on Imaging Sciences 2023–2026 was not searched issue-by-issue; only keyword searches were run. A targeted SIIMS table-of-contents scan for "decision", "classification", "hypothesis" would close this.

---

## Appendix: arXiv-only, unverified venue — five most threatening (as of 23 Sep 2026)

1. **Classification-oriented adaptive sensing via posterior sampling** — Enttsel, Rousselot, Corlay — arXiv 2609.21812 (18 Sep 2026). Classifier outputs over diffusion posterior samples decomposed into within-/between-class uncertainty. Threat: F1 high. https://arxiv.org/abs/2609.21812
2. **Bayesian model selection and misspecification testing in imaging inverse problems only from noisy and partial measurements** — Sprunck, Pereyra, Liaudat — arXiv 2510.27663 (v3 May 2026). Selection among Bayesian imaging models from y alone, compatible with diffusion samplers. Threat: F2 medium-high (baseline). https://arxiv.org/abs/2510.27663
3. **Noise-Free One-Step LoRA for Task-Driven Image Restoration with Diffusion Priors** — Kim, Lee — arXiv 2607.25390 (Jul 2026). Documents that sampling stochasticity changes task outputs and removes it. Threat: F1 medium (counter-framing). https://arxiv.org/abs/2607.25390
4. **TaskGuard** — Pham — arXiv 2609.08011 (Sep 2026). Restore-or-not utility prediction for detection. Threat: F1 low, F3 low. https://arxiv.org/abs/2609.08011
5. **Hallucination-Aware Diffusion Sampling for Inverse Problems via Robust Prior Updates** — Jin et al. — arXiv 2606.02331 (Jun 2026). Separates prior-side from measurement-side content in DPS-type solvers ("measurement-conditioned hallucination"). Threat: F1 medium (a solver-level notion of prior-determined content), F3 low. https://arxiv.org/abs/2606.02331

Also arXiv-only and lower threat: PosteriorBench (2609.20794), SafeRestore (2609.03475), Posterior Information Dynamics (2608.21709), FDIR (2608.00111), CHEM (2512.09806).

## Out-of-scope items encountered (for other researchers)
- LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling — ICCV 2025 (CVF PDF located) — uses MLLM-extracted semantic priors to guide posterior sampling (text/VLM-coupled solver).
- Universal Image Restoration via Internalized Chain-of-Thought Reasoning (arXiv 2606.17557), EvoIR-Agent (2605.22208), Detect in Any Scene (2605.31174), RobustVisRAG (2602.22013), ImIR (ACCV 2026, 2609.25267): MLLM/agentic restoration lines, surfaced from the UniRestore citation graph.
- Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in LVLMs — ICLR 2025 [S2-venue] — arXiv 2502.06130 — uses diffusion-model likelihood feedback inside an LVLM decoder (cites Diffusion Classifier); relevant to the MLLM-side researchers.

## Summary threat matrix (verified venues only)
- F1: HIGH — Wen et al. ECCV 2024 (conformal task-output intervals over posterior samples); MEDIUM — Meaningful Diversity ICLR 2024, Looks Too Good NeurIPS 2024; LOW — AdaSense ECCV 2024, TaskTok ECCV 2026, CVPR 2026 Findings GIR study.
- F2: HIGH — Diffusion Classifier ICCV 2023, Noised Diffusion Classifiers NeurIPS 2024, DiME ICLR 2026; MEDIUM — RDC ICML 2024; LOW — Generative Classifiers Avoid Shortcuts ICLR 2025, VAR classifier ICLR 2026, Noise Matters NeurIPS 2025, Compositionality NeurIPS 2025 D&B.
- F3: MEDIUM — Perception-Robustness ICML 2024, Looks Too Good NeurIPS 2024; LOW — UniRestore CVPR 2025, EDTR ICCV 2025, PMRF ICLR 2025.
