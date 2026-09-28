# Brief: Where does license-plate reading break inside a multimodal LLM under noise?

You are an autonomous research engineer. This document is your whole brief, and no one will answer questions while you work. By the end of this session you will have done the following, without stopping for input:

1. Found out what the cluster offers.
2. Built the Python environment and downloaded the models and data.
3. Written a modular experiment suite.
4. Tested it.
5. Run the full experiment through Slurm.
6. Produced a written report with figures that answers the scientific questions in §2.

Read the whole brief before you act. §1 explains how to operate, §2 why the experiment exists, §3 the constraints you must never break, §4–§7 what to build, §8 how to run it, and §9 what "done" means.

---

## 1. How to operate

**Work to the end without pausing.** Do not ask for confirmation. When a decision comes up, first try to answer it by inspecting the system (§4). If that isn't possible, pick the most defensible option, record it in `src/DECISIONS.md` (the decision, the alternatives, why you chose it, and how to change it), and continue. The user will read `DECISIONS.md` afterwards and override anything they disagree with through the flag blocks.

**Something is only done once it has run.** A module is not finished because it exists. It is finished when a script has executed it and the log ends in `STATUS: OK`. Run a smoke test on every stage before its full run, and read the logs yourself.

**Report results honestly.** Never fabricate, interpolate or smooth results to make a story cleaner. A null result is a result: if CLIP turns out not to be the bottleneck, or steering doesn't work, say so plainly. If a stage fails and you can't fix it, the report says which stage failed and why.

**Hard stops.** These are the only situations where you stop pursuing a path. In each case, finish everything that doesn't depend on it, then state the blocker exactly in the final message.
- You cannot create or write to `/p/vast1/dutta5/LLM-deg` or any fallback location under `/p/vast1/dutta5`.
- No GPU can be reached, through either `sbatch` or an interactive allocation.
- A dataset's license forbids research use, and every alternative listed in §6.1 is also unusable.

**Respect the login node.** Do not run model inference on the login node. Downloads and CPU-only work (environment builds, data parsing, noise generation, report building) are fine there. All GPU work goes through `sbatch` or an interactive allocation (`salloc`/`srun`).

**Long jobs.** Submit the full pipeline as a Slurm dependency chain (§8), then check on it with `squeue`/`sacct` and the log files, and fix and resubmit anything that fails. If your session is going to end before the jobs finish, make sure the following are in place, and say so in the final message:
- `src/scripts/run/run_all.sh` can relaunch everything.
- `src/scripts/run/status.sh` shows how far the pipeline has got.
- The report stage can be rerun on its own.

**Parallelism.** If you can delegate to sub-agents, you may use them to write independent modules in parallel, such as the degradations, the evaluation metrics and the adapters. You own integration and testing.

---

## 2. Scientific goal

When an MLLM sees a clean license-plate image, it reads the plate correctly: the image and the model's text knowledge are meaningfully aligned. Adding Gaussian noise of known strength σ eventually makes the model give partial, incomplete or hallucinated plates, and in the end fail completely. We want to know **where and why the alignment breaks**, and **whether the model can be steered back**.

The questions:
1. **Behaviour.** How do accuracy, token probabilities and run-to-run variability change with σ, for two open models and one fixed prompt?
2. **Location of the break.** Does the vision encoder (CLIP or ViT) fail to produce a usable representation? Or does the connector or LLM fail to read a representation that still carries the characters? Or does the LLM's prior overwrite weak visual evidence late in the stack?
3. **Emergence layer.** At which LLM layer does the correct answer first become decodable? How does that layer shift, or stop existing, as σ grows?
4. **Degradation as steering.** If the degraded activations are moved back toward the clean region of activation space, does the model answer correctly? Where is the point past which nothing helps because the information is no longer in the measurement?

The design is deliberately small and controlled: **10 images**, **1 isolated degradation** (additive Gaussian noise of known σ), **1 fixed prompt**, **2 models**, and **1 greedy run plus 5 sampled runs** per condition.

### Hypotheses the analysis must distinguish

| ID | Hypothesis | What would show it |
|----|-----------|--------------------|
| H1 | **Encoder failure** ("CLIP is bad") | Encoder features drift far from clean at the σ where accuracy drops. CLIP's image↔text alignment for the true string collapses. Linear probes cannot decode the characters from the encoder. Patching in the clean encoder output restores the answer, but steering on the LLM side does not. |
| H2 | **Alignment or readout failure** | The characters can still be decoded from the encoder or projector, but the logit lens never surfaces them in the LLM. Steering on the LLM side, or projecting onto a clean subspace, restores the answer. |
| H3 | **Prior takeover** | The correct token rises to rank 1 in the middle layers, then a late layer overwrites it. The failures are fluent plates of the right shape. |
| H4 | **Measurement limit** | No non-oracle intervention recovers the answer, and probes fail at every layer. This is the point past which the model can't be helped. |

Every analysis stage exists to produce evidence for or against these. The report assigns every (model, image, σ) cell to one of the regimes in S9.

---

## 3. Hard constraints (from the user; never violate)

1. **All code goes under `src/`**: Python, bash, configs and tests. The only things outside `src/` are the run outputs `logs/` and `results/`, which sit in the working directory.
2. **Everything runs through bash scripts.** Each script opens with an **editable flag block** covering Slurm resources and experiment settings, so the user can change later runs by editing only that block. Each script works in `MODE=slurm` (it submits through `sbatch`) and in `MODE=interactive` (it runs directly on an interactive GPU that is already allocated).
3. **The code is modular**, split into packages and subpackages (§5.1). No monolithic scripts.
4. **Every run writes a log file into the working directory.** The working directory is the directory the script was launched from: `$SLURM_SUBMIT_DIR` under Slurm, `$PWD` interactively. Logs go in `<WORKDIR>/logs/<stage>/<run_id>.log`.
5. **Everything large lives under `/p/vast1/dutta5/LLM-deg`**: environments, model weights, all caches (HF, torch, pip, conda), datasets, and large tensors such as activations. It is on the same filesystem as the workspace. Its layout is modular (§5.3), and nothing large may end up in `$HOME`. The user wrote the path without a leading slash, so treat `/p/vast1/dutta5/LLM-deg` as the intended path and record that assumption in `DECISIONS.md`.
6. **Open-source Hugging Face models only.** Use LLaVA plus one other model (§6.2), and license-plate data with ground-truth strings (§6.1).

---

## 4. Phase A: discover the environment

Run these checks, adapting as needed. Write what you find to the "Environment" section of `src/DECISIONS.md`, then build every flag default from the findings.

| Need | How to find it |
|------|------------|
| Slurm partitions, GPU partitions, time limits | `sinfo -s`, `sinfo -o "%P %G %l %D %N"`, `scontrol show partition` |
| Account or bank | `sacctmgr -nP show assoc user=$USER format=account,partition,qos` |
| GPU vendor, model and memory (decides CUDA or ROCm wheels) | `srun -p <gpu-partition> -A <acct> -N1 --gres=gpu:1 -t 5 nvidia-smi` (or `rocm-smi`); `sinfo -o "%G"` |
| Interactive-allocation syntax | Try `salloc --help` and check for site wrappers (e.g. `mxterm`) in `$PATH`. Note what works. |
| Internet on compute nodes | `srun ... curl -sI https://huggingface.co` |
| Modules | `module avail python cuda rocm 2>&1` |
| Storage | Check `/p/vast1/dutta5` exists and is writable, then check free space and quota (`df -h`, `lfs quota` if present). You need about 60 GB. |
| Repo state | `git status`, current branch. Do not commit unless §9 says to. |

If a check fails (for example, `srun` is refused on the login node), try the other routes, then choose the most conservative default and document it.

---

## 5. Architecture

### 5.1 Layout under `src/`

Treat this layout as required. You may add modules, but do not merge packages.

```
src/
├── README.md                     # how to run everything; flag reference; output locations
├── DECISIONS.md                  # environment findings + every judgment call you made
├── pyproject.toml                # makes `llmdeg` installable (pip install -e src)
├── requirements.txt              # direct deps, pinned after a successful install
├── requirements.lock.txt         # pip freeze of the working env
├── configs/
│   ├── paths.yaml                # LLMDEG_ROOT + derived dirs (single source of truth for bash AND python)
│   ├── models/{llava15_7b,qwen25vl_7b,clip_l336}.yaml
│   ├── data/openalpr.yaml
│   ├── degradations/gaussian_noise.yaml
│   ├── prompts/plate_read.yaml   # THE fixed prompt
│   └── experiments/{s1_select,s3_behavior,s4_vision,s5_llm_layers,s6_patching,s7_steering,s8_probes,s9_report}.yaml
├── llmdeg/
│   ├── utils/        {paths,config,logging,seeding,io}.py
│   ├── data/         sources/{openalpr,hf_plates}.py, preprocess.py, text_norm.py, selection.py, manifest.py, synthetic.py
│   ├── degradations/ base.py, registry.py, gaussian_noise.py, {blur,downsample,jpeg}.py (registered stubs), quality.py
│   ├── models/       base.py (VLMAdapter), llava.py, qwen_vl.py, clip.py, registry.py
│   ├── inference/    generate.py, score.py
│   ├── eval/         string_metrics.py, failure_taxonomy.py, variability.py
│   ├── probing/      hooks.py, capture.py, drift.py, clip_alignment.py, logit_lens.py, image_token_lens.py, attention.py, linear_probe.py
│   ├── intervention/ patching.py, steering.py, clean_subspace.py
│   ├── restoration/  denoise.py
│   ├── analysis/     aggregate.py, regimes.py, stats.py, plots/
│   └── cli/          s1_select.py … s9_report.py   (thin: parse config → call library)
├── tests/            pytest: CPU tests always; GPU tests marked and run inside a job
└── scripts/
    ├── common/   env.sh, launch.sh, job_wrapper.sh
    ├── setup/    00_init_root.sh, 01_create_env.sh, 02_download_models.sh, 03_download_data.sh
    └── run/      s1_select.sh … s9_report.sh, run_all.sh, smoke_all.sh, status.sh
```

The rules:
- `cli/` modules contain no science.
- Every entry point runs as `python -m llmdeg.cli.<stage> --config <yaml> --set a.b=c ...`.
- The resolved config is always saved into the run directory.
- Python contains no hard-coded paths. All paths come from `paths.yaml` and environment variables.

### 5.2 Bash, launching and logging

Every `src/scripts/run/sX_*.sh` has this shape. Fill in the defaults from Phase A.

```
# ================= EDIT ME =================
MODE=slurm                 # slurm | interactive
PARTITION=<from Phase A>
ACCOUNT=<from Phase A>
TIME=02:00:00
GPUS=1  CPUS=8  MEM=64G
MODELS="llava15_7b qwen25vl_7b"
SIGMAS=auto                # "auto" = read the analysis σ set chosen in S3; or explicit list
N_SAMPLES=5  TEMPERATURE=0.7  N_NOISE_SEEDS=1
SMOKE=0  OVERWRITE=0
DEPENDENCY=""              # optional sbatch --dependency string
EXTRA_SET=""               # free-form --set overrides
# ===========================================
source "$(dirname "$0")/../common/env.sh"
source "$(dirname "$0")/../common/launch.sh"
```

- `#SBATCH` lines cannot expand variables, so `launch.sh` passes resources as `sbatch` flags (`--partition --account --time --gres --cpus-per-task --mem --job-name --output --error --dependency`). It runs `mkdir -p` on the log directories before submitting, because Slurm silently drops output when the directory is missing. It prints the job ID.
- Slurm stdout and stderr go to `<WORKDIR>/logs/slurm/%x_%j.{out,err}`.
- `env.sh` does the following:
  - Resolves the repo root and `LLMDEG_ROOT`, and asserts that it matches `paths.yaml`.
  - Exports `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`, `PIP_CACHE_DIR`, `CONDA_PKGS_DIRS`, `XDG_CACHE_HOME`, `TMPDIR` (all under `LLMDEG_ROOT/cache`) and `PYTHONPATH`.
  - Loads any modules the site needs and activates the environment.
  - Sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when `OFFLINE=1` (the default for GPU jobs).
- **Run ID** has the form `<stage>__<model|all>__<YYYYmmdd-HHMMSS>__<jobid|int>`.

| Output | Location |
|--------|----------|
| Log | `<WORKDIR>/logs/<stage>/<run_id>.log`. Python writes to both stdout and this file, and bash `tee`s into it as well. |
| Small results (csv/parquet/json/png, resolved config, run header) | `<WORKDIR>/results/<stage>/<run_id>/`, plus a `latest` symlink per stage |
| Large tensors | `$LLMDEG_ROOT/runs/<stage>/<run_id>/`, with a `large ->` symlink in the results dir |

Every log starts with a **run header**: run ID, command line, resolved config, git hash and dirty flag, host, GPU, and the torch/transformers/CUDA versions. It ends with `STATUS: OK`, or with a traceback followed by `STATUS: FAILED`.

### 5.3 `LLMDEG_ROOT` layout

`00_init_root.sh` creates this tree and is safe to run more than once:

```
/p/vast1/dutta5/LLM-deg/
├── envs/llmdeg/          ├── tools/            (micromamba if used)
├── cache/{hf,torch,pip,conda_pkgs,xdg,tmp}/
├── models/<org>__<name>/ (explicit snapshots; loaded by local path, never by hub id at run time)
├── datasets/raw/<source>/        (+ LICENSE copy + DATASET_CARD.md)
├── datasets/processed/<source>/  (canvases, degraded images, manifests)
└── runs/<stage>/<run_id>/
```

### 5.4 Environment (`01_create_env.sh`)

`01_create_env.sh` has flags `ENV_TOOL=venv|micromamba`, `PYTHON_MODULE`, `CUDA_MODULE`, `TORCH_INDEX_URL` (CUDA or ROCm, set from Phase A) and `ENV_PREFIX`.

- **Required packages:** torch, transformers (a version recent enough for Qwen2.5-VL; pin it and record it, because internal module paths change between versions), accelerate, safetensors, huggingface_hub, pillow, numpy, scipy, pandas, pyarrow, pyyaml, scikit-image, scikit-learn, rapidfuzz, matplotlib, pytest.
- **Optional packages:** qwen-vl-utils, and deepinv for the DRUNet denoiser. If deepinv won't install, fall back to scikit-image non-local means and record that.
- **After installing:** freeze the environment to `requirements.lock.txt`, then check it inside a GPU job: `torch.cuda.is_available()`, a bf16 matmul, and imports of every package.
- **Hooks:** use plain PyTorch forward hooks, not TransformerLens.

### 5.5 Shared engineering rules

- **Determinism.** Seed Python, NumPy and torch. Noise seeds are a hash of `(image_id, σ, noise_seed_idx)`. Sampling seeds are 0 to N_SAMPLES−1. Running the same configuration twice must give byte-identical degraded images and identical greedy outputs, and you must test this.
- **Resumability.** Each stage skips work units whose outputs already exist unless `OVERWRITE=1`, and checkpoints every N units.
- **Assert, never fall back silently.** Assert canvas size, image-token count, that the processor did not crop, layer counts, and that the eval and pool sets do not overlap.
- **Batch size 1 for every hooked forward pass.** Cast captured tensors to fp16 on the CPU, and do all lens and probe math in fp32.
- **Smoke mode.** Every stage accepts `SMOKE=1` (2 images, 3 σ values, 1 sampled run, 2 layers) and finishes in about 5 minutes.
- **`.gitignore`.** Add `logs/`, `results/**/large`, `*.safetensors` and `__pycache__/`.

---

## 6. Data and models

### 6.1 Data

**Primary: the OpenALPR end-to-end benchmark** (`github.com/openalpr/benchmarks`). Use `endtoend/us` (about 222 images) and `endtoend/eu`. Each image has a plate bounding box and a ground-truth plate string in Latin alphanumerics.
- `03_download_data.sh` runs on the login node. It sparse-checks-out `endtoend/` only into `datasets/raw/openalpr/` and copies the repo's LICENSE file next to it.
- Read the license and write `DATASET_CARD.md` with the license, where the data came from, and the intended use (non-commercial research on model robustness).
  - If the license permits research use, proceed.
  - If the license is ambiguous, proceed, but flag it prominently in `DECISIONS.md` and in the final message.
  - If the license forbids research use, switch to the fallback.
- The annotation format looks like `filename x y w h PLATE` in one `.txt` file per image. Check the real files before you write the parser, and assert that 100% of them parse.

**Fallback: `prithivMLmods/Number-Plate-Recognition`** on Hugging Face. Its card says Apache-2.0; check that. It may not have bounding boxes. If so, crop the plate with a detector or use full images, and record the confound. **Do not use CCPD**: its plates start with a Chinese province character, which brings in a language and tokenizer confound.

**Preprocessing** (`data/preprocess.py`, deterministic):
1. Crop to the plate's bounding box plus a 25% context margin on each side (a config value), clamped to the image edges.
2. **Pad to a square** using the crop's mean colour, keeping the plate centred.
3. Resize to **336×336** with bicubic interpolation and antialiasing.
4. Save as **PNG**. Never use JPEG anywhere in the pipeline.
5. Record the plate's bounding box in canvas coordinates, and map it onto each model's image-token grid.
6. Normalise the ground truth: uppercase it and remove everything outside `[A-Z0-9]`. Keep the raw string as well.

### 6.2 Models

| Key | HF repo | Role |
|-----|---------|------|
| `llava15_7b` | `llava-hf/llava-1.5-7b-hf` | The vision tower is **CLIP ViT-L/14-336**, so it tests directly whether CLIP is the problem. The rest is a 2-layer MLP projector and Vicuna-7B with 32 layers. |
| `qwen25vl_7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | A strong OCR model with a different ViT and a 2×2 patch merger. It is the contrast case. |
| `clip_l336` | `openai/clip-vit-large-patch14-336` | The standalone CLIP including its text tower. Use it in S4 to test alignment in the joint embedding space. Check whether its vision weights are **identical** to LLaVA's vision tower by comparing tensors, and record the result. |

- `02_download_models.sh` runs on the login node. It uses `hf download` (or `huggingface-cli download`) with `--local-dir $LLMDEG_ROOT/models/<org>__<name>`.
- It also records the commit SHA of each model.
- Load the models in bf16 with `sdpa`. Use `attn_implementation="eager"` only for the attention-capture pass in S5.

**Fixed input geometry.** This matters for comparing σ across models.
- Both models receive **the same 336×336 PNG**.
- **LLaVA-1.5:** `CLIPImageProcessor` resizes the short side and then **centre-crops**. On a non-square image that would cut off the ends of the plate, which is why the canvas is padded to a square first. Assert that the processed tensor is 336×336 and that, after de-normalising, it matches the canvas. Assert that there are **576** image tokens (a 24×24 grid).
- **Qwen2.5-VL:** set `min_pixels = max_pixels = 336*336`. 336 is 12×28, so this gives a 24×24 grid of patches, which the merger reduces to **144** image tokens (a 12×12 grid). Assert the count.
- LLaVA passes **CLIP's penultimate-layer patch features** to the LLM (`vision_feature_layer=-2`, with the CLS token dropped). S4 must report LLaVA's actual input (layer −2 patch features) and CLIP's joint embedding separately, and must not conflate them.

**The `VLMAdapter` interface** (`models/base.py`) is the only place model-specific code may live:

```
build_inputs(image_uint8, prompt, target_text=None)   # chat template; optional teacher-forced target
image_token_positions(inputs); target_token_positions(inputs)
image_grid_shape()                                     # (24,24) LLaVA, (12,12) Qwen
vision_blocks(); vision_feature_layer_used(); projector(); llm_blocks(); final_norm(); lm_head()
generate(inputs, mode=greedy|sample, seed, max_new_tokens) -> text, ids, per-token logprobs, top-5 alternatives
```

Do not rely on module paths you remember. The paths inside transformers models move between versions: for example `model.language_model.model.layers` has become `model.model.language_model.layers`, and `model.visual` has become `model.model.visual`. On first load, print the model tree to the log and write the adapter against what is actually there. Add a GPU test that asserts the layer counts and image-token counts.

**Fixed prompt** (`configs/prompts/plate_read.yaml`, used for every model and condition, applied through each model's chat template, `max_new_tokens=16`):

> "What is the license plate number in this image? Answer with only the characters on the plate."

---

## 7. Pipeline stages and automatic gates

Each stage has a **gate**. Evaluate the gate yourself from the stage's outputs, and record the result in the stage's `gate.json`. When a gate fails, apply the remedy given for it and document it. Do not continue past a failed gate without a remedy.

### S0: Setup (login node, no GPU)
Run `00_init_root.sh`, then `01_create_env.sh` (the GPU sanity check inside it runs as a short job), then `02_download_models.sh` and `03_download_data.sh`.
**Gate:** the environment imports and sees a GPU inside a job; all 3 models are present with their SHAs recorded; the data parses completely; the license is recorded.

### S1: Data preparation and choosing the 10 images (GPU)
1. **Candidate pool:** every parsed crop whose plate is at least 40 px tall on the canvas.
2. **Clean greedy inference:** run both models on the whole pool.
3. **Eligible images:** those that **both models read exactly right** after normalisation. This is the "good image, good alignment" premise.
4. **Pick 10** from the eligible set with a seeded greedy diversity heuristic. Spread them across region (US/EU), plate length, character mix (make sure confusable glyphs such as 0/O, 1/I, 8/B and 5/S appear) and plate contrast.
5. **Freeze the manifest** as `datasets/processed/openalpr/manifest_eval10.json`, holding the IDs, ground truth, bounding boxes and each model's clean outputs. Copy it into the results directory.
6. **Clean pool:** every other pool image. It is used in S7 and S8 and must never overlap the eval-10. Assert this.

**Gate:** 10 images, 100% clean greedy accuracy for both models, zero overlap with the pool.
**Remedy:**
- First, loosen the size threshold.
- Then add the Brazil subset.
- Then use the fallback dataset.

### S2: Generating the degraded images (CPU)
- **Noise model:** `x_noisy = clip(round(x + N(0, σ²)), 0, 255)`, in uint8 pixel units, i.i.d. per pixel and per channel, applied to the whole canvas.
- **Initial σ grid:** 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100. `N_NOISE_SEEDS` defaults to 1. This keeps the noise draw separate from the model's own sampling randomness.
- **Output:** PNGs go to `.../degraded/gaussian_noise/sigma_<σ>/seed_<k>/<id>.png`. PSNR and SSIM (whole canvas and plate region) go to `quality.csv`, and a contact sheet of images × σ goes alongside.
- **Stubs:** blur, downsample and JPEG are registered as stubs that raise `NotImplementedError`. Adding more isolated degradations later should need only a config change.

**Gate:** 10 × |σ| × seeds files exist; regenerating them produces identical bytes; PSNR falls monotonically as σ rises.

### S3: Behavioural sweep (GPU; one job per model, run in parallel)
**Runs.** For each image × σ × noise seed, do **1 greedy run** and **5 sampled runs** (temperature 0.7, top_p 1.0, seeds 0–4). For each run, record the raw and normalised output, the token IDs, the logprob of each token, and the top-5 alternatives at each step.

**Teacher-forced scoring.** Score log p(target | image, prompt), where the target is **the model's own clean greedy output** (correct by S1, and in the model's natural formatting). Record the sequence logprob, the minimum token probability, and the probability of each character (map tokens back to characters, and record tokens that span several characters).

**Metrics:**
- Exact match, character accuracy (1 − normalised Levenshtein distance), CER and length error.
- **Correct-prefix length**: how far the answer gets before its first error. This captures partial answers.
- **Variability**: the number of unique answers, the agreement rate with the majority answer, the mean pairwise normalised edit distance, and the answer entropy.
- **Failure taxonomy**: correct / partial / confusable substitution / fluent-wrong (hints at H3) / truncated / refusal / other.
- **Calibration**: the AUROC of mean token probability as a predictor of correctness.

**Derived:**
- σ₉₀, σ₅₀ and σ₁₀ per model: the interpolated σ where mean character accuracy crosses 0.9, 0.5 and 0.1.
- **Moderate band = [σ₉₀, σ₁₀]**, per model.
- **Analysis σ set:** {0, every grid point in the band, one severe σ above σ₁₀}. Write it to `analysis_sigmas.json`, and have S4–S8 read it when `SIGMAS=auto`.

**Figures:**
- Accuracy, character accuracy and sequence logprob vs σ, with a thin line per image and the mean.
- Variability vs σ.
- Stacked bars of failure types vs σ.
- A heatmap of character errors by position.

**Gate:** every model has at least 3 grid points inside its moderate band; the greedy run at σ=0 is 100% correct and matches S1.
**Remedy (apply automatically, then rerun S2 and S3 for the new σ values only):**
- If a model is still above 0.5 at σ=100, extend the grid (120, 150, 200, 255).
- If the band covers fewer than 3 points, add points inside it.

Record every change to the grid.

### S4: Vision-side representation ("is CLIP bad?")
For each model, image and analysis σ, capture every vision-block output (all patch tokens) and the projector output.
1. **Drift at each layer:** the cosine similarity between clean and noisy, over all patches and over plate patches only, and linear CKA across the 10 images.
2. **Normalised drift:** `d(clean_i, noisy_i^σ) / mean_{j≠i} d(clean_i, clean_j)`. A value above 1 means the image has lost its identity at that layer. Show this as a layer × σ heatmap.
3. **Alignment in CLIP's joint space** (CLIP/LLaVA only): compare the noisy canvas against the text `"a license plate that reads 'XXXX'"` for the true string and about 50 distractors. The distractors are every single-character substitution drawn from the confusable set, plus random plates of the same length. Report the rank and softmax probability of the true string vs σ.
4. Mark LLaVA's layer (−2) on every plot.

**Gate:** drift at σ=0 is exactly zero, as a sanity check; the CLIP true-string rank at σ=0 is recorded, and if it isn't top-5, record that as a finding rather than a bug.

### S5: LLM-side layers ("where does the answer emerge?")
Teacher-force the target and capture the residual stream at every LLM layer.
1. **Logit lens on answer positions.** Compute `softmax(W_U · final_norm(h_ℓ,t))` for each layer ℓ and target position t. Record the probability, rank, entropy and top-5 tokens.
   - **Emergence layer E** is the first ℓ where the correct token is rank 1 and stays rank 1 at every later layer. Also record **E_p50**, where the probability reaches 0.5 and stays there. Use NaN if the token never emerges.
   - **Late-override detector (H3):** the correct token reaches rank 1 in some middle layer but is not rank 1 at the final layer. Record the layer where it is lost and the token that replaced it.
   - Also run the lens on the model's **own wrong greedy output** at σ values where it fails.
2. **Image-token lens.** Apply the same lens to the image-token positions. For each layer, record the fraction of plate-region tokens whose top-5 decoded tokens include one of the plate's characters, together with its spatial map. This shows when the visual tokens start to mean the characters.
3. **Attention** (eager). For each layer, from the target positions, record the attention mass on all image tokens and on plate-region tokens. Store it averaged over heads and per head, with overlay maps for a few layers.

The report must state that logit-lens readings at early layers are unreliable for Llama-family models. Put more weight on ranks and on how E shifts across σ than on absolute layer numbers.

**Gate:** at σ=0, E exists (is not NaN) for at least 80% of the (image, position) pairs for each model. If it doesn't, check the norm/unembed wiring before concluding anything.

### S6: Causal activation patching ("where is the break?")
Pair a clean run with a noisy run of the same image. The token layouts are identical by construction.
**Metric:** `R = (logp_patched − logp_noisy) / (logp_clean − logp_noisy)`, plus exact match of the patched greedy output. Skip cells where `logp_clean − logp_noisy` is below a small ε, and record them.

**Sites (patch one at a time, then let the noisy forward pass continue):**
1. The output of each vision block. Sweep every layer.
2. The projector output.
3. The LLM residual stream at layer ℓ, **image-token positions only**. Sweep every layer.
   - When interpreting this, note that text positions before ℓ have already attended to the noisy image. If R falls with depth, that shows where the noisy evidence has already been absorbed.
4. Variants of sites 1 and 3 that patch **plate-region tokens only**.
5. **Reverse patching**, noisy into clean: which layers are enough on their own to break the model?

**Output:** the curve R(ℓ) for each model and σ, and the earliest and latest recovering layer for each image and σ.
**Gate:** patching the whole projector output gives R ≈ 1 (up to numerical error), because all image information enters the LLM through it. This confirms the harness is correct. σ=0 cells are skipped because R is undefined there.

### S7: Degradation as a steering problem ("put it in a clean space")
Estimate everything **without the test image**, using leave-one-image-out over the eval-10 or the S1 clean pool. Using the test image's own clean activations would be the S6 oracle.
1. **Mean-difference steering.** Compute `v_s(σ) = mean(h_noisy − h_clean)` at site s. Try a global version (averaged over positions) and a per-position version. Apply `h ← h − α·v` with α ∈ {0.5, 1, 1.5, 2}, sweeping over sites and α.
2. **Clean-subspace projection.** Fit PCA to **clean** pool activations at site s (image-token positions). Then set `h ← μ + P_kP_kᵀ(h − μ)`, with k chosen to explain 90, 95 or 99% of the variance.
3. **Controls.** Apply a random direction with the same norm as v. Also apply each intervention to the clean inputs: the loss in clean accuracy must stay within a tolerance you record, or the method is marked invalid.
4. **Image-space baseline.** Denoise with known σ (DRUNet via deepinv, falling back to NL-means) and rerun the model. This is the "restore the measurement first" route and ties the experiment to the broader inverse-problem project.
5. **Oracle.** The best clean patch from S6, as an upper bound.

**Output:** a recovery table with one row per (model, σ, method, site, α/k), and a figure of recovery vs σ for {none, best steering, best projection, denoiser, oracle}.
**Gate:** the random-direction control recovers much less than mean-difference steering; if it doesn't, report that steering is not specific.

### S8: Linear-probe decodability ("is the information still there?"). This stage is required, and it runs after S7.
- **Training data:** clean-pool crops (never the eval-10), augmented with noise across the σ grid. If fewer than 200 usable pool crops exist, add **synthetic plates** (`data/synthetic.py`: PIL, an OFL-licensed monospace font, random `[A-Z0-9]` strings with the same lengths as the real plates, the same canvas pipeline). Record the ratio of real to synthetic.
- **Features at site s:** take the plate-region image tokens, split them into L horizontal bins where L is the plate length (train one probe per length, or restrict to the most common length), and mean-pool each bin.
- **Probe:** multinomial logistic regression (36 classes) for each (site, slot), with the L2 penalty chosen by cross-validation on the training data.
- **Evaluate** on the eval-10 at each σ. The key figure is probe character accuracy for each layer against the model's own character accuracy, as a function of σ.

Say plainly that the probes are trained on small data and are indicative only.

### S9: Aggregation and report (CPU)
Join everything on `(model, image_id, σ, noise_seed)`, then assign each cell a regime. Thresholds go in the config and are reported.

| Regime | Rule |
|--------|------|
| **A: Correct** | greedy exact match |
| **B: Misaligned but recoverable** | wrong; LLM-side steering or projection reaches R ≥ 0.5; probe can decode it |
| **C: Present but unreadable** | wrong; probe can decode it at the encoder or projector; no non-oracle LLM-side method reaches R ≥ 0.5 |
| **D: Lost in the encoder, recoverable from the measurement** | wrong; probe fails in the features; the image-space denoiser recovers the answer |
| **E: Measurement limit** | nothing non-oracle recovers it; probe fails everywhere |

Generate `results/REPORT.md` with the figures linked. It covers the following, in order:
1. **Setup:** models with their SHAs, library versions, the dataset and its license, the σ grid with any adaptive changes, the prompt, seeds, git hash and cluster details.
2. **Behaviour:** the behavioural curves, σ₅₀ per model, the failure taxonomy and variability.
3. **Is CLIP bad?** Answer yes, no or partly, backed by the drift and alignment evidence.
4. **Emergence:** how E shifts with σ, and any late overrides.
5. **Patching:** the restoration curves and where the break happens.
6. **Recovery:** steering and projection vs the denoiser vs the oracle.
7. **Regime map:** an image × σ grid coloured by regime, for each model. This is the headline figure.
8. **Verdict on H1–H4 for each model:** each with its supporting evidence and caveats (n = 10, so wide bootstrap CIs over images; the reliability of the logit lens; probe data size).
9. **Deviations:** every departure from this brief and every failed gate.

Report results per image, not just means.

---

## 8. Run order

1. **Phase A** (§4): write the Environment section of `DECISIONS.md`.
2. **S0 setup:** build and download.
3. **Write the code**, in the order the stages depend on each other. Write the CPU pytest tests as you go and run them. They cover the noise's determinism and value range, text normalisation, the string metrics, the failure taxonomy, config overrides and path resolution.
4. **GPU tests** inside a short job: adapter asserts, the no-crop check, token counts, a hook capture/patch round-trip (patching a layer with its own activations must leave the logits unchanged to within 1e-3).
5. **Run `smoke_all.sh`**, which runs every stage with `SMOKE=1`. Fix problems until every log ends in `STATUS: OK`.
6. **Run `run_all.sh`**, which submits the full chain with `--dependency=afterok`:
   - S1 runs first.
   - S2 (CPU) runs next.
   - S3 runs as one job per model, in parallel.
   - The S3 gate and remedy runs as a short CPU job that can resubmit S2/S3 for extra σ values.
   - S4, S5 and S6 run per model, in parallel.
   - S7, then S8, then S9 run last.

   `run_all.sh` prints the job IDs and writes them to `logs/run_all_<timestamp>.jobs`.
7. **Monitor the jobs** with `status.sh` (which wraps `squeue`, `sacct` and the log tails). Fix failures and resubmit from the failed stage; resumability means completed units are skipped.
8. Once S9 finishes, **read `REPORT.md` critically.** Check that the figures match the tables and that every claim is backed by a number.

**Compute sizing** (for the `TIME` defaults; one GPU with 24 GB or more per job is enough for a 7B model in bf16):
- S3 is about 1.4k short generations in total, which takes minutes.
- S6 is roughly 20k forward passes per full sweep, a few GPU-hours at most.
- Captured activations take about 150 MB per LLaVA forward pass. Store only image-token and target positions, and delete `runs/**/tmp` when finished.

---

## 9. Definition of done

- [ ] `src/` matches §5.1. `src/README.md` explains how to rerun any stage by editing its flag block. `src/DECISIONS.md` covers the environment findings and every judgment call.
- [ ] Nothing large is in `$HOME`: all caches, the environment, weights and data are under `LLMDEG_ROOT`. Check this with `du` on `~/.cache` before and after.
- [ ] Every stage has run in full (or its failure is documented), and each has a log in `<WORKDIR>/logs/` ending in `STATUS: OK`.
- [ ] Every gate's outcome is recorded in its stage's `gate.json`.
- [ ] `results/REPORT.md` exists, with every figure rendered and a per-model verdict on H1–H4.
- [ ] CPU tests pass. GPU tests passed inside a job, and their log is kept.
- [ ] Commit the code (not the outputs) on a new branch `exp/plate-degradation`. Do not push.

## 10. Final message to the user

Keep it short and factual:
1. What ran, and what didn't.
2. Where the report, logs and large outputs are.
3. The headline answer for each model: where the break is (H1–H4), the σ₅₀ values, and whether steering or projection helped compared with the denoiser.
4. The decisions they may want to override, with links to `DECISIONS.md` entries: partition, account, license status, any σ-grid changes, and any fallbacks used.
5. Any jobs still running, and the command to check on them.
