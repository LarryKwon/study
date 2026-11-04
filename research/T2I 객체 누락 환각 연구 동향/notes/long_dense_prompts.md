# Object/Detail Omission in T2I under Long, Dense, Detail-Heavy Prompts (2023–2026)

> Scope note: these notes cover failures under long/dense prompts (many objects, attributes, relations, paragraph-length text). Numbers come from the primary arXiv papers where I could fetch them. Secondary summaries are marked as such. The search budget was about 15 calls, so several of the requested methods are listed under Gaps without verified numbers.
>
> **Newest trend (2025–2026):**
> 1. Benchmarks have moved from ~80-token "dense" prompts (DPG-Bench, 2024) to 280–680-token paragraph prompts with per-entity checks (DetailMaster, ICML 2026; LongBench-T2I, 2025).
> 2. DPG-Bench is saturating. Top open and commercial models cluster at about 85–88 overall, and some authors now call it unreliable for frontier models.
> 3. Even frontier models (GPT Image-1, Gemini, FLUX) omit about 40–60% of described characters on long prompts.
> 4. The two main remedies are agentic plan-then-generate-then-verify pipelines (Plan2Gen, T2I-Copilot) and learned compositional decomposition of long prompts (PRISM, Apr 2026).
> 5. The 2026 frontier leaders (Nano Banana Pro, FLUX.2) now lose points mainly to miscounting and geometry rather than wholesale omission. Weaker systems still omit elements.

## Q1. How does fidelity degrade with more objects, attributes, or sentences? Position effects, token limits, encoders, attention dilution

### Takeaway
On paragraph-length prompts (avg ~285 tokens), even the best 2025 models render only about 35–60% of described characters correctly. Accuracy falls steadily as prompt length grows, and the main failure is omitting whole entities. When an entity does appear, about 90% of its attributes are correct. CLIP-only (77-token) models collapse, and T5/LLM encoders help a lot. However, no T2I paper I found isolates a "later-mentioned objects are dropped more" position effect. The evidence for it is only indirect, from text-encoder retrieval studies.

### Cited Findings
- **DetailMaster (Jiao et al., arXiv:2505.16915, ICML 2026)**
  - Per-category character generation accuracy on 4,116 prompts (avg 284.89 tokens):
    - GPT Image-1: object 59.41 / animal 48.04 / person 40.40
    - Gemini 2.0 Flash: 55.44 / 47.84 / 34.23
    - FLUX.1-dev: 51.47 / 45.83 / 34.91
    - SD3.5 Large: 48.56 / 46.20 / 32.95
    - SDXL: 24.41 / 29.54 / 16.73
    - SD1.5: 20.79 / 27.69 / 13.89
    - — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - Characters that are generated get most of their attributes right. For example, GPT Image-1 scores 91.77 / 92.31 / 94.10 attribute accuracy on object / animal / person, and FLUX is about 89–91. So the bottleneck is entity omission more than attribute errors. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - Spatial and relational detail is weaker still. The best character-location accuracy is 53.92% (GPT Image-1), and the best entity-relationship accuracy is 63.07%. Global scene attributes (background, light, style) are nearly solved for modern models at about 95–98%. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - Accuracy on character attributes, locations, and relationships drops consistently as CLIP-tokenized prompt length grows. The buckets are <250, 250–300, 300–350, 350–400, and >400 tokens. Exact per-bucket values appear only in the paper's Fig. 3. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - Attribute accuracy also drops for characters that are present (Appendix N), so length hurts binding as well as presence. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - The paper attributes the failures to:
    - **CLIP context limits:** CLIP's 77-token limit cripples SD1.5 and SDXL.
    - **Encoder vs. training:** T5-based DeepFloyd IF does better (object 31.01), but the authors argue that long-prompt training matters more than raw context size.
    - **Hypothesized causes:** weak encoders that miss syntactic dependencies, and attribute leakage. These are hypotheses, not causally validated.
    - — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - The paper does not analyze position effects. Its nine-region partition is only a labeling device. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
- **PRISM (Huang, Lin, Du, arXiv:2604.18258, Apr 2026)**
  - Bins DetailMaster prompts into <200 / 200–300 / 300–400 / 400–500 / >500 tokens. LongAlign does well under 300 tokens but drops by up to 30% above 500 tokens. — [arXiv:2604.18258](https://arxiv.org/html/2604.18258v1)
  - Restates DetailMaster's finding that "more than half of the described characters are omitted" even by FLUX-Dev. Its character-presence numbers: FLUX-Dev 42.02%, Qwen-Image 40.46%, SD-3.5 39.01%. — [arXiv:2604.18258](https://arxiv.org/html/2604.18258v1)
- **Counting as a dense-prompt sub-failure**
  - T2ICountBench reports that accuracy falls off as the requested object count grows, across state-of-the-art diffusion models. It also reports that LLM prompt refinement does not fix this. — [arXiv:2503.06884](https://arxiv.org/abs/2503.06884) (abstract-level)
  - "Demystifying Numerosity in Diffusion Models" argues that the initial noise often determines object count more than the text does, and that scaling alone won't fix counting. — [arXiv:2510.11117](https://arxiv.org/html/2510.11117v1) (abstract-level)
- **Position-effect evidence (indirect, retrieval only)**
  - A study of multimodal embedding models finds the text encoders favor early segments: moving a segment to the beginning consistently improves retrieval accuracy. CLIP-ViT-B/16 shows a weaker but present early-position bias. — [arXiv:2511.11216](https://arxiv.org/pdf/2511.11216)
- **Hard truncation:** CLIP-encoder SD models truncate at 77 tokens, and everything after the cutoff is ignored. This is practitioner guidance, not a measured study. — [Medium: 10 Image Prompting Mistakes](https://roblaughter.medium.com/10-image-prompting-mistakes-and-how-to-avoid-them-244f972d0c2a)
- **Concept loss in final encoder layers:** A 2026 paper argues the final text embedding drops concept-level information that is still present in intermediate encoder layers. Injecting intermediate representations at early denoising steps recovers some dropped concepts. — [arXiv:2606.30262](https://arxiv.org/pdf/2606.30262) (search-snippet level)
- **ParaDiffusion's motivation:** it explicitly cites the 77-token CLIP limit of SD and DALL-E 2 as the barrier to paragraph-to-image generation. — [arXiv:2311.14284](https://arxiv.org/html/2311.14284v3)

### Inferences
- The dominant long-prompt failure in modern models is entity-level omission. Global style and lighting are nearly solved, and attributes of rendered entities are about 90% correct. This suggests omission happens at the "which entities get allocated a region" stage (layout/attention allocation), not at attribute rendering.
- Encoder capacity (T5/LLM vs. CLIP) is necessary but not sufficient. Long-prompt training data, and possibly decomposition, matter as much as context length.
- A rigorous causal test of the position effect is an open, publishable gap. One design: permute the clause order of DetailMaster prompts and measure per-entity presence.

### Gaps
- No T2I paper found that measures omission rate as a function of an entity's position in the prompt, for example whether later-mentioned objects are dropped more.
- DetailMaster's per-length-bucket numbers (Fig. 3) could not be extracted as text.
- I found no direct study of "attention dilution" (cross-attention mass per token vs. number of entities) in long-prompt T2I within budget.
- I did not verify Long-CLIP's "effective CLIP length ≈ 20 tokens / 248-token context" claims against the primary source (Long-CLIP, Zhang et al., ECCV 2024, arXiv:2403.15378). The search only confirmed that its authors report about 20% gains in long-caption retrieval. — [summary in search of arXiv 2311.14284 / Long-CLIP](https://arxiv.org/abs/2311.14284)

## Q2. Benchmarks for long/dense prompts and their metrics

### Takeaway
There are three generations of benchmarks:
1. **Dense ~80-token prompts with VQA-based scoring:** DPG-Bench (2024).
2. **MLLM-judged multi-dimension benchmarks:** OneIG, LongBench-T2I.
3. **Paragraph-length (~285–680 token) prompts with per-entity detection plus MLLM verification:** DetailMaster, ICML 2026.

Rubric-based MLLM judging (an LLM writes atomic criteria, an MLLM checks each one) is the newest evaluation pattern.

### Cited Findings
- **DPG-Bench (ELLA paper, Hu et al., arXiv:2403.05135, 2024)**
  - 1,065 dense prompts, each describing multiple objects with several attributes and relationships. — [arXiv:2403.05135](https://arxiv.org/abs/2403.05135)
  - Scoring is automatic: questions generated from a prompt graph are answered by mPLUG-large. Sub-scores are Global / Entity / Attribute / Relation / Other. — [arXiv:2403.05135](https://arxiv.org/abs/2403.05135); summary via [themoonlight review](https://www.themoonlight.io/review/ella-equip-diffusion-models-with-llm-for-enhanced-semantic-alignment)
  - LongBench-T2I lists DPG-Bench's average prompt length as 84, vs. 15 for DrawBench and 10 for T2I-CompBench. — [arXiv:2505.24787](https://arxiv.org/html/2505.24787v1)
- **LongBench-T2I (Zhou et al., "Draw ALL Your Imagine", arXiv:2505.24787, May 2025)**
  - 500 prompts with an average length of 683 (the paper's "Long" column; unit unspecified). — [arXiv:2505.24787](https://arxiv.org/html/2505.24787v1)
  - Built by expanding an Objects365 seed object through LLM sketch and complex-instruction steps, followed by human review. — [arXiv:2505.24787](https://arxiv.org/html/2505.24787v1)
  - Nine dimensions: Object, Background, Color, Texture, Lighting, Text, Composition, Pose, Special Effects. — [arXiv:2505.24787](https://arxiv.org/html/2505.24787v1)
  - Judges are Gemini-2.0-Flash and InternVL3-78B. The authors state that CLIPScore cannot capture complex-instruction nuances. Code: github.com/yczhou001/LongBench-T2I. — [arXiv:2505.24787](https://arxiv.org/html/2505.24787v1)
- **DetailMaster (arXiv:2505.16915, ICML 2026)**
  - 4,116 prompts (plus an 800-prompt mini set), averaging 284.89 tokens. Length distribution:

    | Token bucket | Prompts |
    |---|---|
    | 100–200 | 285 |
    | 200–300 | 2,399 |
    | 300–400 | 1,151 |
    | >400 | 281 |

    — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - Sources are DOCCI and Localized Narratives captions, expanded by Qwen2.5-VL-72B and checked by humans. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - Four dimensions: Character Attributes, Character Locations (nine-region structured), Scene Attributes (background, lighting, style), and Entity Relationships. Annotations: 8,597 characters, 37,165 attributes, 6,910 locations, 18,526 relationships. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - Metric: a character counts as present only if both the Qwen2.5-VL-7B MLLM judge and the YOLOE-11L detector confirm it (BLIP selects boxes). Attributes and relations are then checked by the MLLM. The authors ran a robustness check with InternVL3-9B and say rankings are preserved. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - Data, curation code, and evaluation tools are in the modelscope/data-juicer repo. — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/62789); [arXiv](https://arxiv.org/html/2505.16915v1)
  - The ICML poster abstract mentions 19 generators, while the arXiv main table has 12 (7 general-purpose + 5 long-prompt). The counts differ between versions. — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/62789)
- **OneIG-Bench:** reported as a 0–1 "Overall" score in EN and ZH. Example figures are in Q4. — [Qwen-Image report, arXiv:2508.02324](https://arxiv.org/html/2508.02324)
- **LongText-Bench:** measures long text-rendering accuracy (EN/ZH), which is the text-rendering side of long prompts. — [arXiv:2508.02324](https://arxiv.org/html/2508.02324)
- **ParaPrompts / ViLG-300 (ParaDiffusion, arXiv:2311.14284, IJCV 2025)**
  - The ParaImage dataset has ParaImage-Big (generated captions) and ParaImage-Small (a few thousand human-written captions). Descriptions run up to about 400 words; the abstract says up to 512 words, so the versions differ.
  - Evaluation uses human-rated text faithfulness.
  - — [arXiv:2311.14284](https://arxiv.org/html/2311.14284v3)
- **Image-Description-Prompt benchmark (Abdoli et al., arXiv:2608.14976, Aug 2026)**
  - Uses the 48 hardest prompts from the DataSeeds.AI dataset, selected by automated complexity scoring.
  - GPT-5.4-Pro writes a weighted rubric of atomic criteria for each prompt, and Gemini 3.1 Pro Preview judges each criterion.
  - The authors argue that average-case benchmarks understate the gaps between systems.
  - — [arXiv:2608.14976](https://arxiv.org/abs/2608.14976)
- **PRISM-Bench (with the FLUX-Reason-6M dataset, arXiv:2509.09680, 2025):** a reasoning-oriented T2I benchmark. Not to be confused with the PRISM decomposition method (arXiv:2604.18258). — [arXiv:2509.09680](https://arxiv.org/pdf/2509.09680)
- **Qwen-Image-Bench (arXiv:2605.28091, 2026):** an 18-model "generation to creation" leaderboard (Quality, Aesthetics, Alignment, Real-world Fidelity, Creative). — [HF dataset](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench); [arXiv:2605.28091](https://arxiv.org/html/2605.28091v1)
- **DPG-Bench reliability:** a 2026 paper notes that GenEval and DPG-Bench do not reliably track the actual capabilities of modern T2I models. — per search summary of [BizGenEval arXiv:2603.25732](https://arxiv.org/pdf/2603.25732) / [LLaDA-Image arXiv:2609.03796](https://arxiv.org/pdf/2609.03796) (the exact attributing paper is unclear from the snippet)

### Inferences
- Metrics have evolved from embedding similarity (CLIPScore), to VQA (DPG, mPLUG), to MLLM judges (LongBench-T2I, OneIG), to detector-plus-MLLM per-entity verification (DetailMaster), to LLM-authored per-prompt rubrics (2608.14976).
- For omission specifically, DetailMaster's detector-plus-MLLM "character presence" is currently the closest thing to a standardized object-recall metric for long prompts.
- DPG-Bench is saturating (top scores 85–88, entity sub-scores about 90–92) while DetailMaster entity presence stays at 35–60%. This shows that roughly 80-token prompts no longer stress modern encoders. Omission research should report on DetailMaster or LongBench-T2I.

### Gaps
- I did not verify details for T2I-CompBench++ complex split, GenAI-Bench, DenseT2I, or DOCCI-based T2I evaluation (DOCCI is confirmed only as a source for DetailMaster prompts) within budget.
- Whether LongBench-T2I's "683" length is in words or tokens is not stated in the fetched text.

## Q3. Methods addressing long/dense prompt omission

### Takeaway
There are four families of methods:
1. **Stronger, longer text encoders:** LLM-as-encoder (ELLA, ParaDiffusion, LLM4GEN) and long-CLIP variants or segment-level encoding (LongAlign).
2. **Dense recaptioned training data.**
3. **Prompt decomposition:** layer- or region-wise (LLM Blueprint, Plan2Gen) or learned compositional (PRISM, 2026).
4. **Agentic plan, generate, verify, and correct loops:** SLD, GenArtist, T2I-Copilot, Plan2Gen.

On DetailMaster, the 2023–2024 SD1.5/SDXL-based long-prompt methods give only modest absolute gains (character presence about 22–35%). They are far below what the newer backbones reach without any special method.

### Cited Findings
- **ELLA (arXiv:2403.05135, 2024)**
  - Adds a Timestep-Aware Semantic Connector between a frozen LLM and a frozen U-Net.
  - On DPG-Bench, ELLA-SDXL scores 80.23 vs. DALL-E 3's 83.50 (secondary summary).
  - — [arXiv:2403.05135](https://arxiv.org/abs/2403.05135); [summary](https://www.themoonlight.io/review/ella-equip-diffusion-models-with-llm-for-enhanced-semantic-alignment)
- **ParaDiffusion (Wu et al., arXiv:2311.14284, IJCV 2025)**
  - Uses Llama-2 as the text encoder with LoRA alignment, trained on ParaImage.
  - Claims up to 45% higher human-rated text faithfulness than SDXL and DeepFloyd IF on ViLG-300 and ParaPrompts.
  - — [arXiv:2311.14284](https://arxiv.org/html/2311.14284v3)
- **LongAlign (Liu et al., arXiv:2410.11817, ICLR 2025)**
  - Segment-level encoding: the long text is split into segments that are encoded separately and merged, which bypasses CLIP's limit.
  - Decomposed preference optimization: the CLIP preference score is split into text-relevant and text-irrelevant parts and reweighted to reduce overfitting.
  - Fine-tuned SD1.5 in about 20 hours and beat PixArt-α and Kandinsky 2.2 on long-text alignment.
  - — [arXiv:2410.11817](https://arxiv.org/html/2410.11817v2); [ICLR 2025](https://iclr.cc/virtual/2025/poster/31132)
- **Long-prompt methods on DetailMaster**
  - Object / location / relation accuracy:

    | Method (base) | Object | Location | Relation |
    |---|---|---|---|
    | ELLA (SD1.5) | 34.50 | 15.92 | 16.64 |
    | LongAlign (SD1.5) | 32.95 | 14.89 | 19.17 |
    | ParaDiffusion (SDXL) | 35.25 | 20.32 | 25.50 |
    | LLM Blueprint (SD1.5) | 21.41 | 18.44 | 11.60 |
    | LLM4GEN (SD1.5) | 21.75 | 9.22 | 7.44 |
    | Base SD1.5 | 20.79 | 8.68 | 5.88 |

  - These methods help their bases but are far below FLUX.1-dev (51.47 object). — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
  - DetailMaster also finds that long-prompt-optimized models score lower on style than their baselines, which the authors attribute to a realism bias in the benchmark. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
- **PRISM: learned compositional prompt decomposition (Huang, Lin, Du, arXiv:2604.18258, Apr 2026; 2026 trend)**
  - **How it works:**
    - A light module splits the long-prompt encoding into N components (default 4).
    - The frozen T2I model predicts noise for each component, and the predictions are summed as an energy-based conjunction.
    - The module is trained only with the diffusion loss, without decomposition labels.
    - For T5 encoders it uses a Q-former; for Qwen2.5-VL encoders it uses LoRA.
    - — [arXiv:2604.18258](https://arxiv.org/html/2604.18258v1)
  - **DetailMaster character presence:**

    | Backbone | PRISM | Baselines |
    |---|---|---|
    | Qwen-Image | 46.84% (PRISM-Qwen) | Qwen-Image 40.46%, FLUX-Dev 42.02% |
    | SD1.5 | 28.21% | LongAlign 25.88%, SD1.5 19.12% |

    PRISM gains about 7.4% on average over baselines on prompts above 500 tokens. — [arXiv:2604.18258](https://arxiv.org/html/2604.18258v1)
  - Naive sentence splitting does much worse (14.01% presence), which shows that learned decomposition matters. Larger N gives larger gains. Inference is about 2× slower. — [arXiv:2604.18258](https://arxiv.org/html/2604.18258v1)
  - Image quality vs. Qwen-Image: VQAScore 86.21 vs. 85.02; HPSv3 12.05 vs. 8.56. — [arXiv:2604.18258](https://arxiv.org/html/2604.18258v1)
- **Plan2Gen (LongBench-T2I paper, arXiv:2505.24787, 2025)**
  - A training-free agent. An LLM splits the instruction into background, midground, and foreground layers, then each layer is generated and validated iteratively.
  - Highest overall average on LongBench-T2I: 3.73 (Gemini judge) and 3.41 (InternVL3 judge), narrowly above GPT-4o (3.70 / 3.39). GPT-4o still has the best Object score (3.94 vs. 3.76).
  - Refinement beyond 3 steps per layer slightly degrades results (over-correction or semantic drift).
  - Better plan-to-instruction consistency correlates with better images.
  - — [arXiv:2505.24787](https://arxiv.org/html/2505.24787v1)
- **SLD: Self-correcting LLM-controlled Diffusion (Wu, Lian, Gonzalez, Li, Darrell; arXiv:2311.16090; CVPR 2024)**
  - A closed loop: an LLM controller detects and fixes errors through latent-space operations, without retraining.
  - Reported DALL-E 3 overall score of 52.3% rising to 76.8% with SLD on its numeracy, attribute, and spatial evaluation (spatial 71 → 85).
  - — [arXiv:2311.16090](https://arxiv.org/html/2311.16090v1); [GitHub](https://www.github.com/tsunghan-wu/SLD)
- **GenArtist (NeurIPS 2024)**
  - An MLLM agent that breaks generation into a tree of generation, editing, and self-correction steps, with position-aware tool execution (detector-derived boxes).
  - Scores 0.588 VQAScore overall vs. 0.766 for FLUX1.1-pro in T2I-Copilot's comparison.
  - — [GenArtist review](https://liner.com/review/genartist-multimodal-llm-as-an-agent-for-unified-image-generation); [T2I-Copilot supp.](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Chen_T2I-Copilot_A_Training-Free_ICCV_2025_supplemental.pdf)
- **T2I-Copilot (Chen et al., arXiv:2507.20536, ICCV 2025)**
  - A training-free multi-agent system: input interpreter, generation engine, and quality evaluator.
  - Reports a +38% VQAScore gain over GenArtist at 1.41× higher cost (authors' own claim).
  - — [ICCV 2025 supp.](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Chen_T2I-Copilot_A_Training-Free_ICCV_2025_supplemental.pdf); [alphaXiv](https://www.alphaxiv.org/overview/2507.20536v2)
- **Prompt refinement and counting:** T2ICountBench concludes that LLM prompt refinement does not fix counting failures, so prompt rewriting alone is not a remedy for numerosity omissions. — [arXiv:2503.06884](https://arxiv.org/abs/2503.06884)
- **Intermediate-layer text representations:** injecting intermediate encoder states into early denoising steps recovers some dropped concepts. — [arXiv:2606.30262](https://arxiv.org/pdf/2606.30262)

### Inferences
- **Trajectory:** 2023–24 work grafted LLM encoders or segment encoders onto SD1.5/SDXL. Large DiT backbones with T5/LLM/VLM encoders (FLUX, SD3.5, Qwen-Image) absorbed most of those gains. In 2025–26, the frontier moved to:
  - (a) structured decomposition on top of strong backbones (PRISM);
  - (b) training-free agentic loops with MLLM verifiers (Plan2Gen, T2I-Copilot).
- Agentic pipelines can match or exceed GPT-4o on long-instruction averages, but not on object presence. They also have diminishing returns past about 3 refinement steps.
- Learned decomposition beats naive sentence splitting by a wide margin (29% vs. 14% presence on the SD1.5 ablation). So "just split the prompt" is not a solution by itself.

### Gaps
- I did not verify numbers within budget for:
  - LaVi-Bridge
  - LLM4GEN's own reported results (only its DetailMaster row is verified)
  - Lumina-Next/Lumina-Image 2.0 encoder design
  - Playground v3, Kolors, SANA
  - Long-CLIP in generation
  - DALL-E 3 and PixArt recaptioning ablations
  - Region-wise generation (RPG, LMD)
  - Multi-turn editing approaches
- Agentic methods have not been reported on DetailMaster's per-entity presence metric, so their effect on omission specifically is unknown.

## Q4. Frontier/commercial model performance on dense-prompt benchmarks

### Takeaway
On DPG-Bench, the 2025 leaders are tightly clustered:

| Model | DPG-Bench overall |
|---|---|
| Qwen-Image | 88.32 |
| Seedream 3.0 | 88.27 |
| Z-Image | 88.14 |
| Lumina-Image 2.0 | 87.20 |
| GPT Image 1 [High] | 85.15 |
| FLUX.1-dev | 83.84 |

On harder paragraph-length benchmarks, GPT Image-1 / GPT-4o lead 2025-era comparisons, but they still omit about 40–60% of characters. By late 2026, rubric-based tests put Nano Banana Pro (84.8) and FLUX.2 (82.3) well ahead of Hunyuan 3.0 (63.3).

### Cited Findings
- **DPG-Bench (Qwen-Image tech report, arXiv:2508.02324, Table 3)**

  | Model | Global | Entity | Attribute | Relation | Other | Overall |
  |---|---|---|---|---|---|---|
  | Qwen-Image | 91.32 | 91.56 | 92.02 | 94.31 | 92.73 | 88.32 |
  | Seedream 3.0 | 94.31 | 92.65 | 91.36 | 92.78 | 88.24 | 88.27 |
  | Lumina-Image 2.0 | — | — | — | — | — | 87.20 |
  | HiDream-I1-Full | — | — | — | — | — | 85.89 |
  | GPT Image 1 [High] | — | 88.94 | — | — | — | 85.15 |
  | Janus-Pro-7B | — | — | — | — | — | 84.19 |
  | SD3 Medium | — | — | — | — | — | 84.08 |
  | FLUX.1-dev | — | 90.00 | — | — | — | 83.84 |
  | PixArt-Σ | — | — | — | — | — | 80.54 |
  | SDXL | — | — | — | — | — | 74.65 |
  | SD1.5 | — | — | — | — | — | 63.18 |

  — [arXiv:2508.02324](https://arxiv.org/html/2508.02324)
- **Z-Image (arXiv:2511.22699):** reports Z-Image third on DPG-Bench at 88.14, trailing Seedream 3.0 and Qwen-Image. — [arXiv:2511.22699](https://arxiv.org/pdf/2511.22699) (search snippet; table not fetched)
- **OneIG-Bench Overall (EN / ZH)**

  | Model | EN | ZH |
  |---|---|---|
  | Qwen-Image | 0.539 | 0.548 |
  | GPT Image 1 [High] | 0.533 | 0.474 |
  | Seedream 3.0 | 0.530 | 0.528 |
  | HiDream-I1 | 0.477 | — |
  | SD3.5 Large | 0.462 | — |
  | FLUX.1-dev | 0.434 | — |
  | BAGEL | 0.361 | — |

  — [arXiv:2508.02324](https://arxiv.org/html/2508.02324)
- **LongText-Bench (long text rendering, EN / ZH)**

  | Model | EN | ZH |
  |---|---|---|
  | GPT Image 1 [High] | 0.956 | 0.619 |
  | Qwen-Image | 0.943 | 0.946 |
  | Z-Image | 0.935 | 0.936 |
  | Seedream 3.0 | 0.896 | 0.878 |
  | FLUX.1-dev | 0.607 | 0.005 |

  — [arXiv:2508.02324](https://arxiv.org/html/2508.02324); [arXiv:2511.22699](https://arxiv.org/html/2511.22699)
- **DetailMaster (paragraph prompts):** GPT Image-1 is best overall (object 59.41, location 53.92, relation 63.07), ahead of Gemini 2.0 Flash, FLUX.1-dev, and SD3.5 Large. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
- **PRISM paper (DetailMaster character presence):** Qwen-Image 40.46%, FLUX-Dev 42.02%, SD-3.5 39.01%. — [arXiv:2604.18258](https://arxiv.org/html/2604.18258v1)
- **LongBench-T2I (Gemini judge, Object / Avg)**

  | Model | Object | Avg |
  |---|---|---|
  | GPT-4o | 3.94 | 3.70 |
  | Gemini-2.0 | 3.60 | 3.49 |
  | Gemini-2.0 w/ CoT | 2.79 | 2.96 |
  | Infinity-8B | 3.36 | 3.44 |
  | FLUX.1-dev | 3.24 | 3.14 |
  | Janus-Pro-7B | 2.67 | 2.60 |

  CoT prompting hurt Gemini-2.0 on this benchmark. — [arXiv:2505.24787](https://arxiv.org/html/2505.24787v1)
- **Image-description rubric benchmark (Aug 2026)**
  - Scores: Gemini 3 Pro Image ("Nano Banana Pro") 84.8, FLUX.2 82.3, Ideogram 3.0 65.7, Hunyuan 3.0 63.3. — [arXiv:2608.14976](https://arxiv.org/abs/2608.14976)
  - Leaders lose points mainly to object miscounting and geometric artifacts. Trailing systems garble text. Ideogram 3.0 frequently omits requested elements. — [arXiv:2608.14976](https://arxiv.org/abs/2608.14976)
- **Qwen-Image-Bench overall (2026):** Seedream 4.0 56.21, FLUX.2 Pro 54.57, GPT Image 1 54.07, HunyuanImage 3.0 50.81 (rank 14). — [HF Qwen-Image-Bench](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench); [ModelScope](https://www.modelscope.cn/datasets/Qwen/Qwen-Image-Bench)
- **BizGenEval (arXiv:2603.25732), hard / easy overall:** Nano-Banana-Pro 76.7 / 93.7, Nano-Banana-2.0 68.5 / 92.5, HunyuanImage-3.0 13.0 / 40.1, FLUX.2-Pro 1.6 / 21.1. — [arXiv:2603.25732](https://arxiv.org/pdf/2603.25732) (search snippet)
- **Human-preference Elo (Alibaba AI Arena, reported in Z-Image):** Imagen 4 Ultra Preview 1048, gemini-2.5-flash-image-preview 1046, Seedream 4.0 1039, Qwen-Image 1008, GPT Image 1 986. — [arXiv:2511.22699](https://arxiv.org/html/2511.22699)

### Inferences
- Model rankings flip depending on benchmark difficulty. For example, GPT Image 1 is below Qwen-Image and Seedream 3.0 on DPG but best on DetailMaster and LongBench-T2I objects. This reinforces that DPG-Bench no longer discriminates long-prompt omission ability.
- In 2026 the Gemini 3 Pro Image ("Nano Banana Pro") and FLUX.2 generation appears to lead on dense, compositional prompts. The evidence is a single 48-prompt study plus BizGenEval, so confidence is moderate.

### Gaps
- I found no primary DPG-Bench or DetailMaster numbers for Seedream 4.0, HunyuanImage 3.0, FLUX.2, Nano Banana / Nano Banana Pro, Imagen 4, or gpt-image-1.5 within budget. The tech reports for those models should be checked.
- Secondary sources conflict on Nano Banana 2.0's overall score in the Qwen-Image-Bench / LLaDA-Image tables (59.82 in one table whose caption refers to LongText-Bench). It is unverified.

## Q5. What remains unsolved

### Takeaway
Entity omission on paragraph-length prompts is still the core unsolved problem: best-case character presence is about 40–60%. Spatial placement of many entities (≤54%), relations (≤63%), exact counts, long-prompt robustness above 400–500 tokens, and text rendering inside dense scenes remain open. Causal and mechanistic understanding (position effects, attention allocation) is largely missing.

### Cited Findings
- **Entity omission:**
  - Top models reach only about 50% on detail-level dimensions.
  - Person entities are hardest. Best person accuracy is 40.40% (GPT Image-1), and FLUX.1-dev reaches 34.91%.
  - — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
- **Very long prompts:**
  - Performance drops sharply beyond about 500 tokens; LongAlign loses up to 30%.
  - Even PRISM's best presence is 46.84%.
  - — [arXiv:2604.18258](https://arxiv.org/html/2604.18258v1)
- **Counting:**
  - Accuracy degrades with larger requested counts, and prompt refinement does not help. — [arXiv:2503.06884](https://arxiv.org/abs/2503.06884)
  - Initial noise dominates count. — [arXiv:2510.11117](https://arxiv.org/html/2510.11117v1)
  - Frontier leaders in 2026 still lose most points to miscounting. — [arXiv:2608.14976](https://arxiv.org/abs/2608.14976)
- **Text rendering in long prompts:**
  - "Text and Symbol" is the hardest LongBench-T2I dimension, mostly below 2.5/5 on the Gemini judge. — [arXiv:2505.24787](https://arxiv.org/html/2505.24787v1)
  - FLUX.1-dev's Chinese LongText score is 0.005. — [arXiv:2508.02324](https://arxiv.org/html/2508.02324)
  - Trailing 2026 systems garble text. — [arXiv:2608.14976](https://arxiv.org/abs/2608.14976)
- **Agentic loops:** they plateau or degrade beyond about 3 refinement steps per layer. — [arXiv:2505.24787](https://arxiv.org/html/2505.24787v1)
- **Causes not validated:** DetailMaster leaves causal validation of attribute leakage and encoder syntax failures open. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2)
- **Evaluator dependence:** DetailMaster notes that a different MLLM evaluator may inflate scores. LongBench-T2I's InternVL3 and Gemini judges give different absolute scores. — [arXiv:2505.16915v2](https://arxiv.org/html/2505.16915v2); [arXiv:2505.24787](https://arxiv.org/html/2505.24787v1)

### Inferences
- Open research directions:
  - position-controlled omission studies;
  - per-entity attention or region allocation analysis as entity count grows;
  - combining learned decomposition (PRISM) with verifier loops;
  - evaluating agentic systems on per-entity presence rather than averaged judge scores.
- Small objects and scenes with more than 10 entities are likely the hardest regime. No source found here reports results stratified by entity count or object size for long prompts, so this is inferred rather than shown.

### Gaps
- I found no benchmark that stratifies omission by number of entities (for example, 5 / 10 / 20 objects) or by object size within long prompts.
- I found no frontier-lab disclosure of how their models handle long prompts internally, such as encoder context length or recaptioning density, for Nano Banana, gpt-image-1, Seedream 4, or FLUX.2.
