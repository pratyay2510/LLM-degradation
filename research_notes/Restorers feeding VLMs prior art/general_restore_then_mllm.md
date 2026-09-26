# General-domain "restore-then-understand" pipelines and MLLM corruption-robustness benchmarks (Jan 2024 – Sep 2026)

Scope: natural images/video; restorer (any kind) placed in front of a VLM/MLLM for VQA / captioning / MCQ / video QA / grounding, plus robustness benchmarks that include (or conspicuously omit) a restore-then-answer baseline. Agentic restoration (JarvisIR, AgenticIR, RestoreAgent, 4KAgent, Restore-R1, HybridAgent, TIR-Agent, EvoIR-Agent, PaAgent, Q-Agent, MoA-VR), unified-model self-recovery (Robust-U1 method itself, CLEAR), document/medical/RS/driving verticals, task-driven restoration for classifiers, and inverse solvers coupled with text (TReg, P2L, LATINO-PRO, DU-VLM, SUPIR, DA-CLIP) are covered by other researchers and are only cross-referenced here.

Method note on verification: WebSearch budget was exhausted early, so venues were verified by (a) the arXiv "Comments"/"Journal-ref" line fetched with curl, (b) CVF Open Access query pages fetched with curl (WACV 2025/2026, CVPR 2025/2026, ICCV 2025), (c) OpenReview API v2 search, (d) the PMLR copyright line inside the PDF. dblp and AAAI OJS blocked API access; Semantic Scholar rate-limited heavily (only R-Bench, Visual-Quality-Paradox, CoTAM, Robust-R1 and Robust-U1 citation lists were retrieved). Anything not verified from a proceedings page or the arXiv comment line is marked UNVERIFIED.

Threat legend (per the user's formulations): F1 = answer identifiability via MLLM answer entropy over measurement-consistent posterior samples; F2 = hypothesis-conditioned evidence p(y | q, a) via text-conditioned diffusion inverse solver + Bayes factor; F3 = data-consistency guidance added to a unified MLLM's own generative self-restoration with an internal-vs-external comparison.

---

## KQ1. Which accepted top-venue papers run a restoration/enhancement model before an MLLM/VLM and report downstream understanding accuracy? (restorer, MLLM, benchmark, helped/hurt)

### Takeaway
Only five accepted top-venue papers actually place an external restorer in front of an MLLM and report understanding accuracy (Robust-U1 appendix, Latif et al. WACV 2026, ClearText-Video ECCV 2026, RobustVisRAG CVPR 2026, and Res-Bench [AAAI 2026, venue UNVERIFIED]); in every one of them the restorer is a plain feed-forward or one-step diffusion model, no forward operator or data consistency is used, and the result is that restoration gives marginal, non-monotonic, or negative gains (except super-resolution for resolution loss). No accepted paper propagates restoration uncertainty to the answer.

### Cited Findings — per-paper catalog (accepted, venue verified unless marked)

**P1. Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?** — Jiaqi Tang, Jianmin Chen, Youyang Zhai, Wei Wei, Runtao Liu, Mengjie Zhao, Xiangyu Wu, Qingfa Xiao, Qifeng Chen (HKUST/NWPU et al.)
- Venue: ICML 2026 — verified by arXiv comment "Accepted by ICML 2026" and the PDF copyright line "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026" — [arXiv 2606.08063](https://arxiv.org/abs/2606.08063)
- Restorer(s) used as external baselines (Appendix B.1, Table 8): DFPIR (all-in-one, CVPR 2025), EVSSM (deblurring, CVPR 2025), MambaIRv2 (denoising, CVPR 2025), BiLaLoRA (dehazing, CVPR 2026) — all regression-type restorers; applied "as a preprocessing step before Qwen2.5-VL-7B" on R-Bench (MCQ/VQA/CAP at low/mid/high severity) — [arXiv PDF](https://arxiv.org/pdf/2606.08063)
- Downstream MLLM: Qwen2.5-VL-7B (external pipeline); BAGEL is the unified base for Robust-U1 itself.
- Numbers (Table 8, Overall): all-in-one DFPIR 0.5511; deblurring EVSSM 0.4581; denoising MambaIRv2 0.5459; dehazing BiLaLoRA 0.5371; Robust-U1 0.7398. Per-task, e.g., VQA-high: DFPIR 0.3503 vs Robust-U1 0.6934; CAP-high: DFPIR 0.6336 vs Robust-U1 0.7640 — [arXiv PDF](https://arxiv.org/pdf/2606.08063)
- Paper's stated explanation: "specialized modules (deblurring, denoising, dehazing) require knowing the degradation type and tend to fail under unknown or compound corruptions" and "even the all-in-one restoration model is optimized for perceptual quality rather than downstream understanding" — [arXiv PDF](https://arxiv.org/pdf/2606.08063)
- Explicit inverse-problem formulation: N (no operator, no data consistency, no posterior sampling; recovery is SFT + RL with SSIM + CLIP rewards). Posterior/uncertainty propagated to answer: N. Answerability/identifiability measured: N.
- Key finding for user: the "restoration → understanding" pipeline with SOTA regression restorers underperforms internal self-recovery by ~0.19 overall on R-Bench. Note the table does NOT include a "Qwen2.5-VL-7B with no restoration" row, so whether external restoration helped or hurt vs. raw degraded input cannot be read from Table 8 alone (gap).
- Threat: F3 HIGH (this is exactly the internal-vs-external comparison F3 needs, minus data consistency; the user's F3 must show DC guidance beats plain Robust-U1 self-recovery); F1 LOW (no posterior samples/entropy); F2 LOW.

**P2. Enhancing Vision Language Corruption Robustness using Cross-Distribution & Prompted Denoisers** — Sameer Shafayet Latif, Sadab Shiper, K. M. Rahiduzzaman Kiran, Md Farhan Ishmam, Md Azam Hossain, Abu Raihan Mostofa Kamal, Md Hamjajul Ashmafee (Islamic Univ. of Technology / Utah / Alberta)
- Venue: WACV 2026 — verified on CVF Open Access ([CVF page](https://openaccess.thecvf.com/content/WACV2026/html/Latif_Enhancing_Vision_Language_Corruption_Robustness_using_Cross-Distribution__Prompted_Denoisers_WACV_2026_paper.html); [PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Latif_Enhancing_Vision_Language_Corruption_Robustness_using_Cross-Distribution__Prompted_Denoisers_WACV_2026_paper.pdf)). No arXiv ID found.
- Benchmark: VLSRB — 18 visual + 18 textual corruption functions; source distribution VQAv2, target distribution DARE (count/order/trick/VCR sub-tasks).
- Restorer: "Visual DeNoiser (VDN)": a Mixture-of-Experts of Corruption-Specific Visual Denoisers (CSVD), one per corruption type, trained with MSE to the clean image using DnCNN, BRDNet and DRUNet backbones, routed at inference by a Visual Corruption Routing Network (VCRN); plus a zero-shot LLM textual denoiser (TDN) — [CVF PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Latif_Enhancing_Vision_Language_Corruption_Robustness_using_Cross-Distribution__Prompted_Denoisers_WACV_2026_paper.pdf)
- Downstream VLMs: LLaVA-v1.6-7B, InstructBLIP-7B, Janus-Pro-7B, Gemini 2.0 Flash (VQA accuracy, Clean / Corr / Corr+VDN / Corr+TDN / Corr+VDN+TDN).
- Result: "overall accuracy gain of up to 5.5%" and "9% in certain categories", but "Denoising text or both modalities yields the highest performance gains, whereas visual-only denoising produces marginal improvements"; e.g., LLaVA-v1.6-7B overall: clean 40.40, corrupted 28.52, +VDN 29.63 (+1.11), +TDN 35.73 (+7.21); "Gemini 2.0 Flash experiences consistent degradation across all categories" when the visual denoiser is added; "denoising sometimes introduces excessive blurring or artifacts, especially when applied to complex corruptions" — [CVF PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Latif_Enhancing_Vision_Language_Corruption_Robustness_using_Cross-Distribution__Prompted_Denoisers_WACV_2026_paper.pdf)
- Explicit inverse-problem formulation: N (corruption-type classification + per-type CNN denoiser; no operator/likelihood). Uncertainty propagated: N. Identifiability: N.
- Threat: F1 LOW, F2 LOW, F3 LOW (feed-forward denoisers; result actually supports the premise that naive restoration is insufficient).

**P3. ClearText-Video (CTVid): A Large-Scale Text-Centric Video Dataset Bridging Video Restoration and Scene-Text Enhancement** — Jinlong Li (corresponding), Jiaming Ding, Dingfu Lu et al. (OPPO US AI Center)
- Venue: ECCV 2026 — arXiv comment "This paper is accepted by 2026 Proceedings of the European Conference on Computer Vision" ([arXiv 2608.28784](https://arxiv.org/abs/2608.28784)); ECVA page not yet checked (proceedings not yet online at search time) — treat as accepted-per-authors.
- Data: 4,639 real text-rich egocentric videos, 550K+ frames, 1.6M scene-text annotations, 220K+ spatial/temporal QA pairs; each High-Quality (HQ) video has content-matched Degraded-Quality (DQ: 4x bicubic downsampling; synthetic blur via 16x RIFE interpolation + temporal fusion of 65 frames) and Restored-Quality (RQ) variants; tasks: Text-Centric Video Restoration and Multi-Quality VideoQA — [arXiv HTML](https://arxiv.org/html/2608.28784)
- Restorers producing RQ: DOVE (one-step diffusion video SR), MIMO-UNet+ (deblurring), S3Diff (diffusion image SR) — [arXiv HTML](https://arxiv.org/html/2608.28784)
- Downstream MLLMs (16): GPT-5.4, GPT-5.4-mini, Claude-Sonnet-4.6, Gemini-2.5-pro/flash, InternVL2.5-8B, InternVL3-8B, Llama3.2-11B, Llama3-LLaVA-Next-8B, LLaVA-OneVision, MiniCPM-o 2.6, MiniCPM-V 4.5, Phi-4-multimodal, Qwen3-VL-8B, Qwen2.5-VL-7B (+SFT variant).
- Result: spatial QA, averaged over 16 MLLMs, LR lowers accuracy by 3.14 pts and blur by 6.05 pts from HQ; restoration gives partial, non-monotonic recovery, e.g., Gemini-2.5-pro HQ 71.67 → DQ-LowRes 65.00 → RQ-S3Diff 70.00; Claude-Sonnet-4.6 DQ-Blur 60.00 → RQ-MIMO 70.00; Qwen3-VL-8B HQ 50.56 → DQ-LR 45.43 → RQ-DOVE 47.70; paper states "Restoration does not yield a uniformly monotonic improvement" — [arXiv HTML](https://arxiv.org/html/2608.28784)
- Explicit inverse-problem formulation: N at inference (the degradation operators are known because they are synthetic, but restorers are generic feed-forward/one-step-diffusion models, not measurement-consistent). Uncertainty propagated: N. Identifiability: N.
- Threat: F1 LOW–MEDIUM (only accepted video restore-then-VideoQA benchmark; it provides paired HQ/DQ/RQ data one could reuse, but no posterior/entropy analysis); F2 LOW; F3 LOW.

**P4. RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations** — I-Hsiang Chen, Wei-Ting Chen et al. (National Taiwan Univ. / Microsoft)
- Venue: CVPR 2026 — arXiv comment "Accepted by CVPR2026" ([arXiv 2602.22013](https://arxiv.org/abs/2602.22013)); CVF page not checked.
- Restore-then-answer baseline: a "Two-Stage" pipeline that runs PromptIR (all-in-one regression restorer) before VisRAG (MiniCPM-V 2.0 retriever, MiniCPM-V 2.6 generator) on the Distortion-VisRAG dataset (367,608 question–document pairs, 12 synthetic + 5 real distortions: blur, noise, low light, shadow, brightness, saturation, resolution).
- Numbers (Table 3, synthetic): VisRAG 77.57 MRR@10 / 50.40 Top-1; Two-Stage (PromptIR) 77.78 / 50.56; RobustVisRAG 80.11 / 58.22. Real-world: VisRAG 56.47 / 42.99; Two-Stage 53.59 / 40.42 (restoration HURTS); RobustVisRAG 63.82 / 55.39. Authors: "the restoration step may distort clean images and does not ensure downstream robustness under degraded conditions" — [arXiv HTML](https://arxiv.org/html/2602.22013)
- Inverse-problem formulation: N. Uncertainty propagated: N. Identifiability: N. (Documents rather than natural scenes — borderline vertical; included because the baseline is a general-domain restorer.)
- Threat: LOW for all three.

**P5. Res-Bench: Benchmarking the Robustness of Multimodal Large Language Models to Dynamic Resolution Input** — Chenxu Li, Zhicai Wang, Yuan Sheng, Xingyu Zhu, Yanbin Hao, Xiang Wang (USTC / HFUT)
- Venue: Semantic Scholar venue field says "AAAI Conference on Artificial Intelligence"; arXiv comment only says "23 pages"; AAAI OJS search returned nothing — **UNVERIFIED (treat as AAAI 2026 pending check)** — [arXiv 2510.16926](https://arxiv.org/abs/2510.16926)
- Content: 14,400 samples, 12 resolution levels (112–1344 px), 8 MLLMs (GPT-4o, Gemini 1.5-Pro, Qwen2.5-VL, Kimi-VL, LLaVA-OneVision, InternVL-2.5, MiniCPM-o-2.6, mPLUG-Owl3).
- Restore-then-answer: off-the-shelf DiffIR (diffusion-prior SR) vs white padding for 224/448 px inputs; Table 3: 224→448 padding +0.003 accuracy, 224→448 SR +0.028, 448→896 SR +0.026; conclusion "SR-enhanced images significantly outperform their original low-resolution counterparts and also surpass the performance of the simple padded images" — [arXiv HTML](https://arxiv.org/html/2510.16926)
- Inverse-problem formulation: N. Uncertainty: N. Identifiability: N.
- Threat: LOW (SR helps for a pure downsampling operator; no posterior reasoning).

**P6. Reversing the Flow: Generation-to-Understanding Synergy in Large Multimodal Models (G2U)** — Tong et al. (BUPT)
- Venue: arXiv comment "Accepted by CVPR 2026 Findings" ([arXiv 2605.15792](https://arxiv.org/abs/2605.15792)). Note: "CVPR Findings" is a secondary track; not in CVF main proceedings — flag as **not main-track**.
- Mechanism: a unified model (BAGEL-7B primary; also BLIP3-o, Janus-Pro, MetaQuery, Show-o2, TokenFlow-XL, LlamaFusion) performs "controlled generative acts" — enhancement prompts (deblurring, denoising, inpainting, colorization, saturation, exposure adjustment) and expansion prompts (outpainting, viewpoint change) — and the generated image is concatenated with the original as input for answering; +1.6% on R-Bench, +1.8% MMBench, +4.2% HallusionBench, +1.2% MMStar — [arXiv HTML](https://arxiv.org/html/2605.15792)
- Inverse-problem formulation: N (no data consistency; "generative fidelity bounds perceptual gain"). Uncertainty: N. Identifiability: N.
- Threat: F3 MEDIUM (internal self-enhancement fed back to the same unified model, training-free; F3 must position DC guidance as the missing ingredient); F1/F2 LOW.

**P7. Benchmarking and Enhancing VLM for Compressed Image Understanding (CompressVLMBench)** — Tsinghua AIR / Beihang (lead authors not captured; repo bblgbr)
- Venue: ICML 2026 — arXiv comment "The paper is accepted by ICML 2026" ([arXiv 2512.20901](https://arxiv.org/abs/2512.20901)); OpenReview [NK1ZC7pNmF](https://openreview.net/forum?id=NK1ZC7pNmF); GitHub README says "[ICML 2026]" ([repo](https://github.com/bblgbr/CompressVLMBench)).
- Content: 11 codecs (JPEG, HM, VTM; ELIC, TCM, MLICpp; HiFiC, MS-ILLM, DiffEIC, RDEIC, StableCodec), 9 VLMs (Qwen 1–32B, InternVL3 1–8B, Janus-Pro 1–7B), 7 tasks (POPE, GQA, COCO-Caption, OCRBench, MMBench, MME, SEEDBench), >1M compressed images.
- Key conceptual result: performance gap decomposed into an "Information Gap" (irreversible loss in compression) and a "Generalization Gap" (VLM failure to adapt), with the claim that "only the generalization gap can be mitigated"; fix is a conditional vision-encoder adaptor (codec id + bitrate via RoPE, distilled to uncompressed features), +10–30% — [arXiv HTML](https://arxiv.org/html/2512.20901)
- Restorer in front of VLM: N (no artifact-removal model; adaptation is inside the encoder). Inverse-problem: N. Uncertainty: N. Identifiability: partially — the information-gap notion is a dataset-level proxy for "what the measurement no longer determines", but it is not measured per question or via posterior samples.
- Threat: F1 MEDIUM (closest accepted conceptual precedent for separating measurement-determined vs. unrecoverable information; F1 should cite and contrast: per-question, posterior-based, not codec-level); F2/F3 LOW.

**P8. When MLLMs Meet Compression Distortion: A Coding Paradigm Tailored to MLLMs (CoTAM)** — Jinming Liu, Zhaoyang Jia, Jiahao Li, Bin Li, Xin Jin, Wenjun Zeng, Yan Lu (SJTU / EIT / MSRA)
- Venue: ICLR 2026 Poster — verified on OpenReview (venue field "ICLR 2026 Poster", [forum YDRoTtmXu1](https://openreview.net/forum?id=YDRoTtmXu1)); arXiv comment empty — [arXiv 2509.24258](https://arxiv.org/abs/2509.24258)
- Content: shallow-CLIP-attention importance map for bit allocation + a "multi-level fidelity decoder" (decoded image as prior + lightweight transformer latent adapter with multi-level feature loss); MLLMs LLaVA-1.5-7B/13B, LLaVA-OneVision-7B, InternVL2-8B; benchmarks MME, TextVQA, POPE, SeedBench, VQAv2, MMMU, MMBench, Video-MME; 35.99% bitrate saving over ELIC at equal MLLM performance — [arXiv HTML](https://arxiv.org/html/2509.24258)
- Restorer: the decoder-side adapter is a learned feature-level enhancement, not a pixel restorer. Inverse-problem: N. Uncertainty: N. Threat: LOW.

**P9. Bridging Compressed Image Latents and Multimodal Large Language Models (ComNeck)** — Chia-Hao Kao, Cheng Chien, Yu-Jen Tseng, Yi-Hsin Chen, Alessandro Gnutti, Shao-Yuan Lo, Wen-Hsiao Peng, Riccardo Leonardi
- Venue: ICLR 2025 — arXiv comment "Accepted by ICLR 2025" ([arXiv 2407.19651](https://arxiv.org/abs/2407.19651))
- Content: lightweight transform-neck + surrogate loss adapts compressed latents to MLLM vision tasks without decoding to pixels; no reconstruction/restoration stage. Inverse-problem: N. Threat: NONE/LOW.

**P10. Unified Coding for Both Human Perception and Generalized Machine Analytics with CLIP Supervision (UG-ICM)** — Kangsheng Yin, Quan Liu, Xuelin Shen, Yulin He, Wenhan Yang, Shiqi Wang
- Venue: AAAI 2025 — arXiv comment "publised to AAAI 2025" ([arXiv 2501.04579](https://arxiv.org/abs/2501.04579))
- Downstream evaluation is ResNet101 / YOLOv3 / Mask-RCNN / DeepLabv3+, not an MLLM ([arXiv HTML](https://arxiv.org/html/2501.04579)). Listed only to close the "coding-for-machines" thread. Threat: NONE.

**P11. Image Quality Assessment: From Human to Machine Preference (Machine Preference Database, MPD)** — Chunyi Li, Yuan Tian, Xiaoyue Ling, Zicheng Zhang, Haodong Duan, Haoning Wu, Ziheng Jia, Xiaohong Liu, Xiongkuo Min, Guo Lu, Weisi Lin, Guangtao Zhai (SJTU / Shanghai AI Lab / Moonshot / NTU)
- Venue: CVPR 2025 — verified on CVF ([CVF page](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.html))
- Content: proposes "IQA for Machine Vision"; 30 distortion/processing types (including enhancement ops such as Gaussian denoise, CNN denoise, brighten, compression) → 30,000 images; machine preference scored by consistency of downstream tasks: Yes-or-No, MCQ, VQA, Captioning with 15 LMMs (InternVL2, InternLM-XComposer2, LLaVA-1.5, LLaVA-Next, LLaVA-OneVision, Mantis, MiniInternVL, mPLUG-Owl3, Ovis, Phi-3.5, Pangea, Qwen2-VL, Yi-1.5, Qwen1.5-VL, VisualGLM, Monkey) plus SEG/DET/RET with specialist models; 2.25M preference annotations — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.pdf)
- Restorer-in-front pipeline: only as processing ops inside the distortion set, not an evaluated restoration method. Inverse-problem: N. Uncertainty: N. Threat: LOW (relevant as "image quality for machine perception" prior art; it establishes that human-perceptual quality and MLLM task consistency diverge).

**P12. Leveraging Vision-Language Models to Select Trustworthy Super-Resolution Samples Generated by Diffusion Models** — Cansu Korkmaz, A. Murat Tekalp, Zafer Doğan (Koç Univ.)
- Venue: IEEE TCSVT 2025 — arXiv comment "accepted to IEEE Transactions on Circuits and Systems for Video Technology" and journal-ref "IEEE TCSVT 2025" ([arXiv 2506.20832](https://arxiv.org/abs/2506.20832))
- Content: explicitly frames SR as "an ill-posed inverse problem with many feasible solutions"; generates 100 samples per LR image from a pretrained latent diffusion SR model with different seeds; VLMs (BLIP-2, GPT-4o) are prompted with structured queries ("What is the number?", artifact/quality questions) to rank samples; top-ranked samples are averaged; validated on digit/character recognition (MNIST, Urban100 text) and a Trustworthiness Score (CLIP similarity + edge SSIM + wavelet artifact term) — [arXiv HTML](https://arxiv.org/html/2506.20832)
- Explicit inverse-problem formulation: PARTIAL — multiple samples from a diffusion SR prior are treated as candidate solutions, but there is NO measurement-consistency step (no y = A(x)+n projection, no DPS/DDNM), and the diffusion SR model is a conditional generator, not a posterior sampler in the DPS sense. Uncertainty propagated to answer: N (VLM output is used to select an image, not to compute answer entropy). Identifiability: N.
- Threat: F1 MEDIUM (nearest precedent for "VLM judgments over many diffusion SR samples of one measurement"; distinct because samples are not measurement-consistent and VLM disagreement is used for selection, not identifiability); F2 LOW–MEDIUM (asks the VLM a hypothesis question per sample, but never scores p(y | a)); F3 NONE.

**P13. Robust-R1: Degradation-Aware Reasoning for Robust Visual Understanding** — Jiaqi Tang et al. (HKUST)
- Venue: AAAI 2026 Oral — arXiv comment "Accepted by AAAI2026 Oral" ([arXiv 2512.17532](https://arxiv.org/abs/2512.17532)); Semantic Scholar venue field AAAI.
- No restore-then-answer baseline: compares to Qwen2.5-VL-3B, Gemma3-4B, InternVL-4B and robust-CLIP-type methods (TeCoA, RobustCLIP, RobustLLaVA) only; method is SFT + RL with rewards r_deg (accurate degradation-parameter perception) and r_len; degradations across acquisition/transmission/environment/post-processing; benchmarks R-Bench and MMMB/MMStar/RealWorldQA with synthetic corruption at 25/50/100% — [arXiv HTML](https://arxiv.org/html/2512.17532)
- Inverse-problem: N (degradation type/intensity is estimated as text, never used as an operator). Uncertainty: N. Identifiability: N.
- Threat: F1 LOW–MEDIUM (it reasons explicitly about "what the degradation removed" in text; F1 differs by measuring it through posterior samples); F2/F3 LOW.

**P14. Event-MLLM: Learning to See through Illumination Extremes with Event Streaming in MLLMs** — Baoheng Zhang, Jiahui Liu, Gui Zhao, ..., Xiaojuan Qi, Hayden Kwok-Hay So (HKU)
- Venue: CVPR 2026 — arXiv comment ([arXiv 2603.27558](https://arxiv.org/abs/2603.27558)); CVF page not checked.
- Fuses event streams with RGB under 17 brightness rates; compares to "general-purpose, illumination-adaptive, and event-only baselines"; I could not confirm whether an LLIE-preprocessing baseline is included (gap). Restorer: N (sensor fusion). Threat: LOW.

**P15. Latent Denoising Improves Visual Alignment in Large Multimodal Models** — Dhruvesh Patel et al.
- Venue: ACM MM 2026 — arXiv comment ([arXiv 2604.21343](https://arxiv.org/abs/2604.21343))
- Trains the LMM to denoise saliency-masked/noised projected visual tokens back to clean teacher features; reports reduced degradation under ImageNet-C-style corruptions. Not a pixel restorer; no inverse formulation. Threat: LOW.

### Cited Findings — accepted benchmarks/methods that do NOT include a restore-then-answer baseline (for KQ2 cross-reference)
- **MLLM-IC: Benchmarking Multimodal Large Language Models Against Image Corruptions** — Xinkuan Qiu, Meina Kan, Yongbin Zhou, Shiguang Shan; ICCV 2025 (verified on CVF: [page](https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_Benchmarking_Multimodal_Large_Language_Models_Against_Image_Corruptions_ICCV_2025_paper.html)); 40 corruption types x 34 low-level capabilities in a 3-level hierarchy; grep of the main-paper and supplemental PDFs finds no restoration/enhancement/denoising baseline (only references to restoration datasets as corruption sources) — [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Qiu_Benchmarking_Multimodal_Large_Language_Models_Against_Image_Corruptions_ICCV_2025_paper.pdf)
- **Revisiting Visual Corruptions in LVLMs: A Shape–Texture Perspective on Model Failures** — Xinkuan Qiu, Meina Kan, Zhenliang He, Yongbin Zhou, Shiguang Shan; CVPR 2026 (verified on CVF: [page](https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_Revisiting_Visual_Corruptions_in_LVLMs_A_Shape-Texture_Perspective_on_Model_CVPR_2026_paper.html)); no restoration baseline found in the PDF text.
- **R-Bench: Are your Large Multimodal Model Robust to Real-world Corruptions?** — Chunyi Li et al. (SJTU); IEEE JSTSP 2025 (OpenReview public-article record and dblp entry "IEEE J. Sel. Top. Signal Process. 2025": [OpenReview QHRhtRabNJ](https://openreview.net/forum?id=QHRhtRabNJ)); 33 corruption dimensions, 7 pipeline stages, 3 severities, 20 LMMs; "no restore-then-answer experiments are included" — [arXiv 2410.05474](https://arxiv.org/html/2410.05474). It is the benchmark on which Robust-U1's external-restorer comparison (P1) is run.
- **Scaling Test-Time Robustness of VLMs via Self-Critical Inference Framework** — Kaihua Tang et al.; CVPR 2026 (verified on CVF: [page](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Scaling_Test-Time_Robustness_of_Vision-Language_Models_via_Self-Critical_Inference_Framework_CVPR_2026_paper.html)); no restoration stage found in the PDF text.
- **The Perceptual Observatory: Characterizing Robustness and Grounding in MLLMs** — Tejas Anvekar et al. (ASU); WACV 2026 (verified on CVF: [page](https://openaccess.thecvf.com/content/WACV2026/html/Anvekar_The_Perceptual_Observatory_Characterizing_Robustness_and_Grounding_in_MLLMs_WACV_2026_paper.html)); no restoration stage found.
- **Visual Robustness Benchmark for VQA (VRB)** — Md Farhan Ishmam et al.; WACV 2025 (verified on CVF: [page](https://openaccess.thecvf.com/content/WACV2025/html/Ishmam_Visual_Robustness_Benchmark_for_Visual_Question_Answering_VQA_WACV_2025_paper.html)); mentions "visual denoising strategies" only as future work; the follow-up is P2.
- **EgoNight** — Deheng Zhang et al.; ICLR 2026 (arXiv comment, [arXiv 2510.06218](https://arxiv.org/abs/2510.06218)); nighttime egocentric VQA with day–night aligned videos; 10 MLLMs; "does not evaluate LLIE or restoration preprocessing" — [arXiv HTML](https://arxiv.org/html/2510.06218)
- **Robust Test-time Video-Text Retrieval: Benchmarking and Adapting for Query Shifts** — ICLR 2026 (arXiv comment, [arXiv 2604.20851](https://arxiv.org/abs/2604.20851)); retrieval under query shifts, not restoration (content not examined further).

### Inferences
- Across P1–P5 the empirical pattern is consistent: regression restorers (DFPIR, EVSSM, MambaIRv2, BiLaLoRA, PromptIR, DnCNN-family) give at best marginal gains and often hurt on real degradations; generative/diffusion SR (DiffIR, S3Diff, DOVE) helps when the operator is pure downsampling. This is the "restoration is not a cure-all" story and no accepted paper turns it into a measurement-consistency argument.
- The Robust-U1 Table 8 result (external 0.55 vs internal 0.74) will be the reviewer's reference point for F3; the user's F3 needs both a DC-guided variant and an external DPS-style restorer on the same R-Bench protocol to separate "internal vs external" from "consistent vs inconsistent".

### Gaps
- Robust-U1 Table 8 lacks the no-restoration Qwen2.5-VL-7B row, so the helped/hurt direction vs. raw degraded input is not extractable without re-running (the main Table 1 may contain it; not extracted).
- ECCV 2026 (ClearText-Video) and CVPR 2026 (RobustVisRAG, Event-MLLM, G2U "Findings") proceedings pages were not checked; acceptance rests on the arXiv comment line.
- Res-Bench's AAAI 2026 status is unverified.

---

## KQ2. Do MLLM robustness benchmarks include a restore-then-answer baseline? Findings?

### Takeaway
Most accepted corruption benchmarks (R-Bench, MLLM-IC, Qiu CVPR 2026, EgoNight, VRB, Perceptual Observatory) do not include any restoration baseline; the ones that do (VLSRB/WACV 2026, ClearText-Video/ECCV 2026, Res-Bench [unverified AAAI]) plus the arXiv-only DarkQA and Visual-Quality-Paradox all report that off-the-shelf restoration is unstable, severity-dependent, or harmful, with SR for downsampling the only reliable win.

### Cited Findings
- Benchmarks WITH a restoration baseline and their finding: VLSRB (P2) — visual denoising "marginal", Gemini 2.0 Flash degrades; ClearText-Video (P3) — "not uniformly monotonic"; Res-Bench (P5) — DiffIR SR +0.028; RobustVisRAG (P4) — PromptIR two-stage hurts on real degradations (56.47→53.59 retrieval, 42.99→40.42 generation) — sources as in KQ1.
- arXiv-only benchmarks with a restoration baseline (see Appendix): DarkQA — DarkIR, RetinexFormer, ZeroDCE, RUAS before LLaVA-1.6-7B, LLaVA-OneVision-8B, InternVL3.5-8B/30B, Qwen3-VL-8B/32B, GPT-4o; "significant accuracy improvement at more severe low-light levels (L4 and L5), performance decreases at moderate levels (L1–L3)" — [arXiv 2512.24985](https://arxiv.org/html/2512.24985); Visual Quality Paradox — NAFNet/MWFormer (main text) and SUPIR/DiffBIR/DA-CLIP (appendix) before LLaVA-v1.5-7B, LLaVA-v1.6-Mistral-7B, Qwen2.5-VL-3B on MathVista, MMMU, ScienceQA, TextVQA, MME; "in some cases ... the restored images even lead to worse performance than the directly degraded inputs (like ScienceQA and TextVQA for LLaVA-v1.5-7B)"; e.g., Qwen2.5-VL-3B ScienceQA 75.71 clean, 73.90 +Gaussian noise, 37.92 after NAFNet restoration (Table 1) — [arXiv PDF 2506.15645](https://arxiv.org/pdf/2506.15645)
- Benchmarks WITHOUT: R-Bench (JSTSP 2025), MLLM-IC (ICCV 2025), Qiu shape–texture (CVPR 2026), EgoNight (ICLR 2026), VRB (WACV 2025), Perceptual Observatory (WACV 2026), plus arXiv-only Bench-C ([arXiv 2511.19032](https://arxiv.org/html/2511.19032): 849 MCQs, 19 corruptions x 5 severities, 13 LVLMs, "no restoration or denoising preprocessing baselines"), SpaceDG ([arXiv 2605.22536](https://arxiv.org/html/2605.22536): 9 physically simulated degradations rendered inside 3DGS, 25 MLLMs, "does not evaluate any restore-then-answer pipeline"), MMCBench (arXiv comment "Technical report", [arXiv 2401.11943](https://arxiv.org/abs/2401.11943)), IC-Bench (Pattern Recognition 2026 per Semantic Scholar citation list of R-Bench; not a listed top venue; content not examined).
- Robust-U1 citation-graph check: Semantic Scholar lists only one citing paper (Remember-R1, ACM MM 2026, unrelated) as of the query, so no follow-up work has yet extended its external-restorer comparison — [S2 API, arXiv:2606.08063 citations].

### Inferences
- The field's benchmarks treat restoration as an afterthought; a paper that makes the restorer a posterior sampler and reports answer-distribution statistics would be the first accepted work to do so on R-Bench/MLLM-IC-style data.

### Gaps
- IC-Bench (Pattern Recognition) and "Hard to Read, Easy to Jailbreak" (ACL 2026, [arXiv 2605.07250]) were not examined; both are outside the venue filter.
- Video corruption benchmarks: no accepted video-LLM corruption benchmark with a restoration baseline other than ClearText-Video was found (searched arXiv API: "video" + "corruption"/"degradation" + "video large language"/"video-language" + "robustness").

---

## KQ3. Does any such work use a known/estimated operator, data consistency, or posterior sampling (multiple restorations) to propagate uncertainty to the answer?

### Takeaway
No — among all accepted papers found, none feeds measurement-consistent posterior samples (DPS/DDRM/DDNM/PSLD/ReSample/DAPS, PnP) to an MLLM, none uses a forward operator at inference, and none propagates restoration uncertainty into the answer; the closest fragments are (i) TCSVT 2025 Korkmaz et al. (100 unconstrained diffusion-SR samples ranked by a VLM), (ii) physically-modeled forward operators used only to synthesize benchmark data (DarkQA RAW pipeline, SpaceDG 3DGS engine), (iii) Robust-R1's text-space degradation-parameter estimation, and (iv) CompressVLMBench's information-gap/generalization-gap decomposition.

### Cited Findings
- Diffusion restorers used before MLLMs in the literature are conditional generators, not measurement-consistent samplers: SUPIR, DiffBIR, DA-CLIP (Visual Quality Paradox, [arXiv 2506.15645](https://arxiv.org/pdf/2506.15645)); DiffIR (Res-Bench, [arXiv 2510.16926](https://arxiv.org/html/2510.16926)); DOVE, S3Diff (ClearText-Video, [arXiv 2608.28784](https://arxiv.org/html/2608.28784)). None applies y = A(x)+n consistency.
- Korkmaz et al. (TCSVT 2025): "Super-resolution (SR) is an ill-posed inverse problem with many feasible solutions consistent with a given low-resolution image ... diffusion models generate a diverse set of SR images, but selecting the most trustworthy solution from this set remains a challenge" — 100 seeds, VLM ranking, averaging; no data-consistency projection — [arXiv 2506.20832](https://arxiv.org/abs/2506.20832)
- DarkQA's forward operator (used for data synthesis only): sRGB → linear → Bayer RAW → photon-shot + read + row-pattern + quantization noise → EV drop → simplified ISP → sRGB — [arXiv 2512.24985](https://arxiv.org/html/2512.24985); "No mention of inverse-problem formulations, data-consistency constraints, posterior sampling, or uncertainty propagation."
- SpaceDG's "physically grounded degradation synthesis engine that embeds degradation formation process into 3D Gaussian Splatting rendering" (defocus, distortion, motion blur, haze, water droplets, low light, over-exposure, JPEG, low-res) — [arXiv 2605.22536](https://arxiv.org/abs/2605.22536)
- Robust-R1 estimates degradation type and parameters in the reasoning chain (reward r_deg), but "contains no uncertainty quantification or inverse-problem formulations" — [arXiv 2512.17532](https://arxiv.org/html/2512.17532)
- CompressVLMBench: performance gap = Information Gap (irreversible) + Generalization Gap (mitigable); "only the generalization gap can be mitigated" — [arXiv 2512.20901](https://arxiv.org/html/2512.20901)
- Robust-U1 (ICML 2026) rewards SSIM to ground truth + CLIP similarity; no operator or DC — [arXiv 2606.08063](https://arxiv.org/pdf/2606.08063)
- Uncertainty-in-VQA work exists but without restoration: Variational VQA (TMLR 2026, outside filter) uses variational-Bayes posterior samples over weights for selective prediction — [arXiv 2505.09591](https://arxiv.org/abs/2505.09591); Robust-TO (arXiv-only) attaches "a calibrated reliability score" per frame/tool for video reasoning — [arXiv 2606.26904](https://arxiv.org/abs/2606.26904)
- arXiv API sweeps for "posterior sampling" + "vision-language", "diffusion posterior sampling" + "language model", "data consistency" + "multimodal large language", "inverse problem" + "vision-language" (2024–2026) returned no paper feeding posterior samples to an MLLM for understanding (results: Consistency Regularised Gradient Flows for Inverse Problems [2605.07907], Hypothesis Testing in Imaging Inverse Problems [2505.22481], DA-CLIP CVPRW 2024 [2404.09732], a MICCAI 2025 cardiac-MR uncertainty-propagation paper [2507.12945], and "Does AI Understand Imaging? ... Agentic AI for Computational Imaging" [2607.07189]) — export.arxiv.org API queries, Sep 2026.

### Inferences
- "Hypothesis Testing in Imaging Inverse Problems" (arXiv 2505.22481, not examined, venue unknown) is the one title that sounds adjacent to F2 (Bayes-factor over hypotheses given y); it should be read by whoever covers inverse-solver literature, since it may formalize p(y | hypothesis) without any MLLM.
- The user's F1/F2 have no direct accepted precedent; the "threat" comes from adjacent framings (information gap; VLM-over-samples selection) rather than from any pipeline that does the same thing.

### Gaps
- Could not examine arXiv 2505.22481 (Hypothesis Testing in Imaging Inverse Problems) or 2607.07189 (agentic computational imaging benchmark) for venue/content.
- Semantic Scholar keyword search was rate-limited for most restoration/MLLM queries; coverage of 2024–2025 journal papers (TIP/TCSVT/TMM) that might feed restored images to CLIP-style VLMs is therefore incomplete.

---

## KQ4. Low-level-vision-for-MLLM: SR for MLLM input, image quality for machine perception, MLLM-oriented enhancement, coding-for-machines targeting MLLM/VLM. Any inverse-problem formulations?

### Takeaway
The accepted "coding/quality for MLLM" line (ComNeck ICLR 2025, CoTAM ICLR 2026, CompressVLMBench ICML 2026, MPD CVPR 2025, Res-Bench SR experiment) uniformly adapts encoders/decoders or measures machine preference; none formulates the decoder as an inverse problem with data consistency or reports uncertainty.

### Cited Findings
- ComNeck (ICLR 2025): adapts compressed latents to MLLM vision tasks with a transform-neck and surrogate loss; three modes (frozen codec, joint human-machine, machine-only); no pixel reconstruction — [arXiv 2407.19651](https://arxiv.org/abs/2407.19651)
- CoTAM (ICLR 2026 Poster): 35.99% bitrate saving on ELIC at equal MLLM performance; decoder "combines decompressed image features with learned semantic enhancements via adapter fusion"; "No inverse-problem formulation or data consistency mechanisms" — [arXiv 2509.24258](https://arxiv.org/html/2509.24258); [OpenReview](https://openreview.net/forum?id=YDRoTtmXu1)
- CompressVLMBench (ICML 2026): single conditional adaptor gives +10–30% across codecs/bitrates; no restoration model — [arXiv 2512.20901](https://arxiv.org/html/2512.20901)
- MPD (CVPR 2025): IQA-for-machines with 15 LMMs on YoN/MCQ/VQA/CAP; notes that images "subjectively satisfactory to humans" may not be "applicable to the machines downstream tasks such as detection and question answering, vice versa" — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.pdf)
- Res-Bench (AAAI 2026, unverified): DiffIR SR before MLLMs improves accuracy +0.026–0.028 — [arXiv 2510.16926](https://arxiv.org/html/2510.16926)
- UG-ICM (AAAI 2025): CLIP-supervised unified codec; downstream = ResNet/YOLO/Mask-RCNN/DeepLab, not MLLM — [arXiv 2501.04579](https://arxiv.org/html/2501.04579)
- Outside filter / unverified but on-topic: Prompt-Guided Prefiltering for VLM Image Compression (ICME 2026; 25–50% bitrate reduction at equal VQA accuracy) — [arXiv 2604.00314](https://arxiv.org/abs/2604.00314); Tell Codec What Worth Compressing (VCIP 2024 per Semantic Scholar) — [arXiv 2408.08575](https://arxiv.org/abs/2408.08575); LL-ICM: Image Compression for Low-level Machine Vision via LVLM (arXiv-only) — [arXiv 2412.03841](https://arxiv.org/abs/2412.03841); Exploring Multimodal Knowledge for Image Compression via Large Foundation Models (IEEE TIP 2025 per Semantic Scholar; VLM→codec direction; not examined).
- Chain-of-Zoom (NeurIPS 2025 Spotlight) uses a VLM to prompt a diffusion SR model (VLM→restorer direction) and does not evaluate downstream understanding — [arXiv 2505.18600](https://arxiv.org/abs/2505.18600); excluded per the direction rule.

### Inferences
- The "information gap vs generalization gap" framing (CompressVLMBench) is the most likely reviewer comparison for F1; F1's differentiator is per-question, posterior-based identifiability instead of aggregate codec-level loss.

### Gaps
- The TIP 2025 compression paper and Hallucination Score for generative SR ([arXiv 2507.14367], venue unknown; uses MLLM-style scoring of SR hallucinations) were not examined in depth.

---

## KQ5. Video: restore-then-video-LLM pipelines?

### Takeaway
ClearText-Video (ECCV 2026) is the only accepted work found that runs video restorers (DOVE, MIMO-UNet+, S3Diff) before video-capable MLLMs and reports VideoQA accuracy; EgoNight (ICLR 2026) benchmarks night video QA without any enhancement baseline; everything else (Robust-TO tool orchestration, BLMSP bitstream-corrupted video, RO-Bench counterfactual videos) is arXiv-only.

### Cited Findings
- ClearText-Video results and restorers: see P3 — [arXiv 2608.28784](https://arxiv.org/html/2608.28784)
- EgoNight: ~32.8% (synthetic) and ~25.0% (real) day→night drops; no LLIE baseline — [arXiv 2510.06218](https://arxiv.org/html/2510.06218)
- Robust-TO (arXiv-only): "frontier video reasoning models can suffer 15-30%p accuracy drops" under motion blur, glare, occlusion; per-frame reliability-relevance scores and confidence-cost GRPO; agentic (covered by the agentic researcher) — [arXiv 2606.26904](https://arxiv.org/abs/2606.26904)
- BLMSP (arXiv-only): bitstream-native semantic priors injected into "video restoration, captioning, and human pose estimation" backbones for bitstream-corrupted video — [arXiv 2608.21837](https://arxiv.org/abs/2608.21837)
- RO-Bench (arXiv-only): counterfactual edited videos (style/object/background), not physical corruption — [arXiv 2510.08936](https://arxiv.org/abs/2510.08936)
- Video coding for MLLMs (arXiv-only): CMVC "When Video Coding Meets MLLMs" — [arXiv 2408.08093](https://arxiv.org/abs/2408.08093); Visual Token Coding for Video MLLMs — [arXiv 2608.28008]; Artic AI-oriented RTC — [arXiv 2602.12641].

### Inferences
- Video restore-then-VideoLLM is nearly empty at top venues; a video instance of F1 (posterior samples from a video inverse solver → answer entropy) has no precedent.

### Gaps
- NTIRE 2026 Bitstream-Corrupted Video Restoration challenge report ([arXiv 2604.06945]) not examined for any VideoLLM evaluation.

---

## KQ6. MLLM-based IQA papers (Q-Bench, Q-Instruct, DepictQA, Co-Instruct, LL-Bench) — only if they evaluate restoration outputs against downstream understanding

### Takeaway
Only MPD (CVPR 2025) evaluates processed/distorted images through downstream LMM understanding tasks; the MLLM-IQA family otherwise scores quality, and LL-Bench (arXiv-only) scores restored images with an MLLM-based metric rather than through downstream QA.

### Cited Findings
- MPD (CVPR 2025): downstream-task consistency (YoN/MCQ/VQA/CAP with 15 LMMs) as the machine-preference signal over 30 processing types — [CVF page](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.html)
- LL-Bench (arXiv-only, no venue): 2,469 degraded images, 28,919 restored images from 10 large generative models + 21 conventional restorers, 152,020 pairwise human preferences, and an MLLM-based LL-Score; evaluation is human-preference alignment, not downstream understanding — [arXiv 2606.02535](https://arxiv.org/abs/2606.02535)
- The awesome list github.com/ChunmingHe/awesome-multimodal-large-language-models-in-low-level-vision lists Robust-U1 as its only "restoration → MLLM understanding" entry and "explicitly lacks comprehensive coverage of downstream MLLM evaluation papers" — [GitHub](https://github.com/ChunmingHe/awesome-multimodal-large-language-models-in-low-level-vision)

### Inferences
- None of the IQA-MLLM line threatens F1–F3.

### Gaps
- Q-Bench/Q-Instruct/DepictQA/Co-Instruct were not re-examined individually (pre-2024 or pure IQA); no evidence they evaluate downstream understanding of restored images.

---

## Threat summary table (accepted papers)

| Paper | Venue (how verified) | Restorer → MLLM | Inverse formulation | Uncertainty→answer | F1 | F2 | F3 |
|---|---|---|---|---|---|---|---|
| Robust-U1 (Tang et al.) | ICML 2026 (arXiv comment + PMLR line) | DFPIR/EVSSM/MambaIRv2/BiLaLoRA → Qwen2.5-VL-7B (App. B.1) | N | N | low | low | **high** — the internal-vs-external baseline F3 must beat |
| Latif et al. | WACV 2026 (CVF) | DnCNN/BRDNet/DRUNet MoE → LLaVA-1.6/InstructBLIP/Janus-Pro/Gemini | N | N | low | low | low |
| ClearText-Video | ECCV 2026 (arXiv comment) | DOVE/MIMO-UNet+/S3Diff → 16 MLLMs (VideoQA) | N | N | low–med | low | low |
| RobustVisRAG | CVPR 2026 (arXiv comment) | PromptIR → MiniCPM-V VisRAG | N | N | low | low | low |
| Res-Bench | AAAI 2026 (UNVERIFIED) | DiffIR SR → 8 MLLMs | N | N | low | low | low |
| G2U "Reversing the Flow" | CVPR 2026 Findings (arXiv comment; non-main track) | BAGEL self-enhancement concat → BAGEL | N | N | low | low | **medium** |
| CompressVLMBench | ICML 2026 (arXiv comment + OpenReview) | none (encoder adaptor) | N (info-gap decomposition) | N | **medium** (conceptual) | low | low |
| CoTAM | ICLR 2026 (OpenReview) | decoder adapter → LLaVA/InternVL | N | N | low | low | low |
| ComNeck | ICLR 2025 (arXiv comment) | latents → MLLM | N | N | none | none | none |
| Korkmaz et al. | IEEE TCSVT 2025 (arXiv journal-ref) | 100 diffusion SR samples ranked by VLM | partial (ill-posed framing, no DC) | N | **medium** | low–med | none |
| Robust-R1 | AAAI 2026 Oral (arXiv comment) | none | N (text-space degradation params) | N | low–med | low | low |
| MPD | CVPR 2025 (CVF) | processing ops → 15 LMMs | N | N | low | low | low |
| MLLM-IC, Qiu CVPR26, R-Bench, VRB, Perceptual Observatory, EgoNight, Tang CVPR26, Event-MLLM, Latent Denoising, UG-ICM | verified as listed above | none | N | N | none/low | none | none |

---

## Appendix: arXiv-only, unverified venue (5 most threatening)

1. **Demystifying the Visual Quality Paradox in Multimodal Large Language Models** — Shuo Xing, Lanqing Guo, Hongyuan Hua, Seoyoung Lee, Peiran Li, Yufei Wang, Zhangyang Wang, Zhengzhong Tu (TAMU/UT Austin/Toronto/NTU) — [arXiv 2506.15645](https://arxiv.org/abs/2506.15645) (v1 only, comment "18 pages"; OpenReview search returns only the dblp CoRR record; NeurIPS 2025 / ICLR 2026 virtual-site searches returned no listing) — **no accepted venue found**. Restorers: NAFNet (denoise/deblur), MWFormer (snow/fog), SUPIR (blur), DiffBIR (noise), DA-CLIP (snow/fog) before LLaVA-v1.5-7B, LLaVA-v1.6-Mistral-7B, Qwen2.5-VL-3B on MathVista/MMMU/ScienceQA/TextVQA/MME; restoration sometimes worse than degraded input; proposes VQ-TTT (learnable blur-kernel interpolation + shallow LoRA, entropy-minimization at test time). Inverse: N. Uncertainty: N. Threat: F1 MEDIUM (establishes that the "restored image" is not the object of interest; F1 can cite it as motivation), F2/F3 LOW.
2. **DarkQA** — Yohan Park, Hyunwoo Ha, Wonjun Jo, Tae-Hyun Oh (KAIST/POSTECH) — [arXiv 2512.24985](https://arxiv.org/abs/2512.24985) (comment "submitted to the IEEE"; Semantic Scholar venue field says IEEE Robotics and Automation Letters 2025 — RA-L is not on the user's list; publisher page not checked). Physics-based RAW forward model; DarkIR/RetinexFormer/ZeroDCE/RUAS before 7 VLMs; LLIE helps only at severe levels. Threat: F1 MEDIUM (explicit, parameterized forward operator — but used only to build the benchmark; a reviewer may ask why the user does not use its noise model), F2/F3 LOW.
3. **Allegory of the Cave: Measurement-Grounded Vision-Language Learning (PRISM-VL)** — Kepeng Xu, Li Xu, Gang He, Wenxin Yu (Xidian/SWUST) — [arXiv 2605.11727](https://arxiv.org/abs/2605.11727). Argues that ISP rendering is a lossy forward operator and moves the VLM interface to RAW-derived linear XYZ measurements with camera-metadata conditioning; +4.46 pts LLM-judge accuracy over Qwen3-VL-8B on a low-light/HDR benchmark; no restoration, no posterior. Threat: F1 LOW–MEDIUM (shares the "reason from the measurement, not the rendering" framing; no identifiability analysis).
4. **Confidence-Aware Tool Orchestration for Robust Video Understanding (Robust-TO)** — [arXiv 2606.26904](https://arxiv.org/abs/2606.26904). Per-frame trustworthiness and calibrated tool reliability scores weighted in reasoning; agentic (cross-ref agentic researcher). Threat: F1 LOW (uncertainty-aware but not restoration-based).
5. **SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation** — Xiaolong Zhou, Yifei Liu, ..., Le Ma (Shanghai AI Lab) — [arXiv 2605.22536](https://arxiv.org/abs/2605.22536). Physically grounded degradation engine in 3DGS, 25 MLLMs, SFT gains; no restoration baseline. Threat: LOW (but a ready-made source of paired clean/degraded data with known operators for F1 experiments).

Also seen (lower threat, arXiv-only): Bench-C ([2511.19032]) no restoration; ETCHR question-conditioned image editor decoupled from the understanding model ([2605.23897]); BLMSP bitstream-corrupted video ([2608.21837]); VisualDeltas preference learning from quality perturbations ([2603.07272]); Understanding Degradation with VLM / DU-VLM ([2602.04565], covered by the inverse-solver researcher); LL-Bench ([2606.02535]); Hallucination Score for generative SR ([2507.14367]); BRUCE scientific-VLM corruption escalation ([2608.07742], "Submitted to IEEE Access"). Outside the venue filter but on-topic: IC-Bench (Pattern Recognition 2026), "Hard to Read, Easy to Jailbreak" (ACL 2026), DiffCAP diffusion purification for VLMs (TMLR 2026, adversarial not corruption, [2506.03933]), Variational VQA (TMLR 2026, [2505.09591]), VL-UR adverse-weather restoration (ICME 2025, [2504.08219]), Prompt-Guided Prefiltering (ICME 2026, [2604.00314]).
