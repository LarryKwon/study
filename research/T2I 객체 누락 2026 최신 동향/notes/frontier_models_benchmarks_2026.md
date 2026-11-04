# 2026 Frontier T2I Models and Benchmarks, as They Bear on Omission, Dense/Long Prompts, Conflicting/Counterfactual/Negation Prompts, and Extra Objects

Scope note: researched 2026-11-04. Tags: **[verified]** means I read the claim on the primary page (an arXiv HTML page or official repo). **[unverified]** means it comes from a search snippet or secondary press/aggregator only. Already in the earlier report and not repeated in detail: GenEval 2 (2512.16853), NegT2IBench (2610.03084), Image-Description-Prompt benchmark (2608.14976), Scaling text conditioning (2607.29679), SenseNova-U1.5 GenEval table. **New arXiv IDs in these notes:** 2610.26850 (GenOlympiad), 2605.28091 (Qwen-Image-Bench), 2605.10730 (Qwen-Image-2.0 report), 2609.03796 (LLaDA-Image), 2608.28206 (NumBench). Seen in search but not read: 2605.25347 (ERNIE-Image report), 2609.00525 (GenScale).

**Main gap (stated up front):** I found **no** 2026 source that reports GPT-Image-2, Nano Banana Pro/2, Seedream 4.5/5, FLUX.2, HunyuanImage 3 or Imagen 4 on GenEval 2, DetailMaster, LongBench-T2I, T2I-CompBench++, TIIF-Bench, OneIG or NegT2IBench in fetchable text. The frontier closed models show up on newer holistic or hard benchmarks (Qwen-Image-Bench, GenOlympiad, UniGenBench++ leaderboard images, arenas). They do not show up on the older compositional ones.

## (a) 2026 frontier/open model releases and what they claim about dense/long/compositional prompt following

### Takeaway
The 2026 frontier is GPT-Image-2 (Apr 2026) → GPT-Image-2.5 (Sunburst/Flare variants, autumn 2026), Nano Banana 2 (Gemini 3.1 Flash Image, Feb 2026) and Nano Banana Pro, Seedream 5.0 (Lite/Pro), FLUX.2 (pro/max/flex/dev/klein) and Qwen-Image-2.0 (May 2026). Their claims about "reduced omission" and "complex multi-layered prompts" are mostly marketing or human-eval statements with no released numbers. The one detailed open tech report (Qwen-Image-2.0) names a VLM condition encoder, an RL-trained prompt enhancer and multi-reward GRPO as the levers.

### Cited Findings
- **GPT-Image-2 / ChatGPT Images 2.0.** Released April 21–22, 2026, with API access from day one. The OpenAI post (as quoted) calls it "a step change in detailed instruction following, placing and relating objects accurately, and rendering dense text". It supports up to 8 coherent images per prompt and aspect ratios from 3:1 to 1:3. "Thinking" outputs are reserved for paid tiers. **[unverified, secondary press; no OpenAI primary page fetched]** — [EdTech Innovation Hub](https://edtechinnovationhub.com/news/chatgpt-images-20-is-here-and-it-finally-handles-text-slides-and-multilingual-content); [IT Brief](https://itbrief.news/story/openai-launches-chatgpt-images-2-0-with-api-access); [Framia (date)](https://framia.converge.ai/page/en-US/news/gpt-image-2-release-date)
- **GPT-Image-2.5** exists and is evaluated in Oct 2026 papers (see GenOlympiad below). Arena listings show "gpt-image-2.5-sunburst" and "Flare" variants. **[verified that it was evaluated in 2610.26850; release details unverified]** — [GenOlympiad](https://arxiv.org/html/2610.26850v1)
- **Nano Banana 2 = Gemini 3.1 Flash Image**, launched Feb 26, 2026, and now the default in the Gemini app, Search AI Mode and Lens. Google claims better prompt following and good handling of "complex, multi-layered prompts", plus grounding in Gemini world knowledge and real-time web search. It debuted #1 on the Artificial Analysis T2I board with Elo 1,272, against GPT Image 1.5 (high) at 1,268. It costs about $67 per 1k images, roughly half the price of Nano Banana Pro. **[unverified, press; no model card fetched]** — [Neowin](https://www.neowin.net/amp/google-launches-nano-banana-2-pro-quality-at-flash-speeds/); [OfficeChai](https://officechai.com/ai/nano-banana-2); [SiliconANGLE](https://siliconangle.com/?p=732297)
- **Qwen-Image-2.0 Technical Report** (arXiv 2605.10730, 11 May 2026) **[verified]** — [arXiv](https://arxiv.org/html/2605.10730)
  - Condition encoder: frozen **Qwen3-VL**. Backbone: MMDiT (MSRoPE, QK-RMSNorm, SwiGLU). VAE: f16c64.
  - **Prompt Enhancer** initialized from Qwen3.5-9B, trained with SFT and then GRPO using MLLM visual-consistency and aesthetic rewards plus rule-based text constraints.
  - **RLHF with five reward models** (aesthetics, image-text alignment, portrait, instruction following, visual consistency) via GRPO. DMD distillation to 4 NFE. No separate reasoning module; CoT traces are used only as supervision for the prompt enhancer.
  - Claims instructions of **up to 1K tokens** for slides, posters, infographics and comics, and better instruction following that **"reduc[es] concept omission, compositional failure, and hallucinated content"**. This claim rests only on human evaluation, with **no numbers** given.
  - The only external number: LMArena T2I Elo **1168, #9 globally** (accessed 22 Apr 2026). No GenEval, DPG or OneIG tables appear in the HTML. Prompt-rewriting status in its comparisons is not stated.
- **Seedream 5.0 (Lite/Pro).** The vendor/host page says it "follows instructions precisely with dramatically reduced hallucination". This is marketing language with no measured results. **No Seedream 5 tech report found.** **[unverified]** — [fal.ai](https://fal.ai/seedream-5.0)
- **FLUX.2, HunyuanImage 3.x, Z-Image, Emu 3.5, BAGEL/Janus successors, SenseNova-U1.5.** I found no 2026 tech-report claims in this pass. A search for "Seedream 5.0 technical report / HunyuanImage 3.1 / FLUX.2 technical report" returned only third-party comparisons. **[gap]** — [search results e.g. gradually.ai](https://www.gradually.ai/en/ai-models/seedream-5-pro/)
- **LLaDA-Image** (arXiv 2609.03796, 3 Sep 2026) is a fully open masked-diffusion-LLM image generator. On GenEval it scores 0.85 overall but **counting 0.53** (Turbo 0.42). For comparison, Qwen-Image scores counting 0.89 and Seedream 3.0 scores 0.91. The comparison excludes results with prompt enhancement, explicit reasoning or test-time rewriting. **[verified]** — [arXiv](https://arxiv.org/html/2609.03796)

### Inferences
- The shared 2026 recipe is a VLM/LLM text encoder, an RL-trained prompt rewriter, and multi-reward GRPO with an "instruction following" reward. On that basis, omission is now treated as an RL-reward target rather than an architectural one. This rests on Qwen-Image-2.0's explicit description. For closed models it is an inference.
- Closed vendors' "thinking" modes (GPT-Image-2 paid tiers, Nano Banana with Gemini grounding) mean that benchmark numbers for "GPT-Image-2" or "Nano Banana" may silently include reasoning or rewriting. Papers rarely state which mode they used.

### Gaps
- No OpenAI system card or model card for GPT-Image-2/2.5, and no Google model card for Nano Banana Pro/2, was fetched. Architecture (AR vs diffusion, rewriter) is not confirmed. One source (MindStudio) claims GPT-Image-2 is native to the GPT-4o architecture, but nothing else corroborates it.
- No tech reports found for Seedream 5, FLUX.2, HunyuanImage 3.x, Z-Image or SenseNova-U1.5 in this pass. ERNIE-Image (2605.25347) includes Seedream 4.5 and Qwen-Image comparisons but was not read.

## (b) Newest 2026 benchmarks/leaderboards: current rankings and weakest failure categories

### Takeaway
On holistic and arena boards, GPT-Image-2/2.5 lead, followed by Nano Banana Pro/2. On the hardest constraint-satisfaction benchmark (GenOlympiad, Oct 2026), Nano Banana Pro leads the T2I models with only 23.9/100 strict. **Quantity/counting and counterfactual (prior-violating) instructions are near-floor for every model**: the best T2I quantity score is 17.1 strict and counterfactual is ≤27.8. Strong LLM rewriting roughly doubles scores. Real-world fidelity, physical logic and contact interaction are the weakest pillars on Qwen-Image-Bench.

### Cited Findings
**GenOlympiad** (arXiv 2610.26850, 19 Oct 2026) **[verified]** — [arXiv](https://arxiv.org/html/2610.26850v1)
- 801 prompts on 9 axes: quantity and precision, latent computation, physical consistency, spatial/3D, text fidelity, symbolic systems, long-range consistency, **counterfactual instruction following**, and world-knowledge grounding. Average prompt length is 199 words / 254 tokens (median 222). Strict scoring requires all constraints to be met; relaxed scoring gives weighted partial credit.
- **Judge: Gemini 2.5 Flash.** Pairwise agreement with GPT-5.6 and GPT-4.1 is above 96%.
- Overall strict / relaxed scores:

| Model | Strict | Relaxed |
|---|---|---|
| Nano Banana Pro | 23.9 | 40.0 |
| Seedream 5.0 Pro | 15.1 | 28.4 |
| GPT-Image-2.5 | 14.9 | 29.6 |
| Nano Banana 2 | 12.4 | 25.4 |
| GPT-Image-2 | 10.2 | 23.4 |
| FLUX.2 pro | 6.3 | 15.0 |
| GPT-Image-1.5 | 6.0 | 15.8 |
| Seedream-4.5 | 3.0 | 10.6 |
| FLUX.2 dev | 3.5 | 10.3 |
| Qwen-Image 20B | 2.2 | 5.7 |

- Strict scores on selected axes (Counterfactual / Quantity / Long-range consistency):

| Model | Counterfactual | Quantity | Long-range |
|---|---|---|---|
| Nano Banana Pro | 27.8 | 17.1 | 24.4 |
| Seedream 5.0 Pro | 24.1 | 10.4 | 13.7 |
| GPT-Image-2.5 | 23.0 | 13.2 | 9.8 |
| GPT-Image-2 | 14.0 | 5.2 | 8.8 |
| Nano Banana 2 | 13.4 | 6.8 | 15.9 |
| FLUX.2 pro | 6.0 | 2.7 | 1.9 |
| Seedream-4.5 | **0.0** | 5.7 | 1.8 |
| Qwen-Image 20B | **0.0** | 3.1 | 1.6 |

- **Prompt-rewriting effect:** GPT-5.6 Sol rewriting on GPT-Image-2.5 raises strict overall from **14.9 to 33.0**, and counterfactual from 23.0 to 33.0. Spatial *drops* from 18.4 to 12.5. PromptEnhancer-32B gives "nearly no overall improvement" (15.1).
- "Coding models" (GPT-6 Astra, GPT-5.6 Sol) were prompted to emit an HTML document rendered by JavaScript. They score highest: Astra 40.5 strict, counterfactual 38.4, quantity 40.8. Their aesthetics are lower (4.61 vs 5.51 for GPT-Image-2.5).
- **Counterfactual failure = reversion to priors**, as noted in the appendix: "The model tends to draw the compass that it has seen many times, with the red needle pointing away from the sun"; "The model tends to let the winning team celebrate" (for a prompt where the losing team should celebrate). Nano Banana Pro's weakest axes are world knowledge (10.5) and spatial (11.2).
- No analysis of score vs object count or prompt length is given. About 20 tasks were not scored for open models because of context-length limits.

**Qwen-Image-Bench** (arXiv 2605.28091, 27 May 2026; Alibaba Qwen) **[verified]** — [arXiv](https://arxiv.org/html/2605.28091v1); [HF dataset](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench)
- 1,000 bilingual prompts, 500 short and 500 long (long = above 235 characters EN / 70 characters ZH). Five pillars, 23 L2 and 56 L3 facets. The prompts were LLM-drafted (Qwen3-Max, ChatGPT-5.2) and then curated by artists.
- **Judge: Q-Judger**, a Qwen3.6-27B fine-tuned on more than 130k human-labelled pairs. Spearman ρ with human rankings is 0.92. Note the possible conflict of interest: a vendor-built benchmark and judge.
- Overall (ZH-column values from Table 2) with Alignment pillar: GPT Image 2 **64.69** (Alignment 65.85); Nano Banana 2.0 59.82 (62.40); GPT Image 1.5 59.65 (61.72); Nano Banana Pro 59.45 (61.25); Qwen Image 2.0 Pro 57.84 (59.28); Seedream 5.0 57.22 (58.90); Seedream 4.5 56.78 (57.31); FLUX 2 Max 55.33 (57.35); FLUX 2 Pro 54.57 (57.01); Imagen 4.0 Ultra 51.99 (54.02); HunyuanImage 3.0 50.81 (52.00); Imagen 4.0 50.29 (51.64); Qwen Image 49.23 (50.72).
- LLaDA-Image's reproduction gives EN overall scores of GPT-Image 2 65.23, Nano-Banana 2.0 59.59, Nano-Banana Pro 59.33, Seedream 5.0 57.80, FLUX.2 Max 56.14, HunyuanImage 3.0 51.35, Imagen 4.0 51.08 and Z-Image 50.52 — [LLaDA-Image](https://arxiv.org/html/2609.03796) **[verified]**
- **Weakest facets ("systemic ceilings")**, best-model scores: Anatomical Fidelity 20.9, Physical Logic 31.9, **Objects 41.9**, Animals 42.6, Contact Interaction 43.3. Real-world Fidelity is the lowest pillar for every model. Alignment facets (Quantity, 2D Space, Composition Relationship) sit around 40–60 across the industry, but **per-model facet values are not given**. There is no short-vs-long or object-count breakdown.
- The authors note Seedream 4.5 beats the newer Seedream 5.0 on Quality and Aesthetics (54.41 vs 52.55 on Quality).

**UniGenBench++** (2510.18701; repo updated through 2026) **[verified for repo news; scores unverified]** — [GitHub](https://github.com/CodeGoat24/UniGenBench)
- Added in Feb 2026: GPT-4o-1.5 (likely GPT Image 1.5), Seedream-4.5, and FLUX.2 klein/pro/flex/max. Added in Nov 2025: Nano Banana Pro, FLUX.2-dev, Z-Image. **GPT-Image-2 and HunyuanImage 3 are not listed.** The leaderboards for EN short, EN long and ZH are images only, so scores could not be extracted.
- Judge: Gemini 2.5 Pro, or the offline UniGenBench-EvalModel (qwen3vl-32b-v1 recommended; the 72B variant reaches about 94% agreement with Gemini 2.5 Pro).
- Per snippet: GPT-4o-1.5 is #1 overall and Nano Banana Pro is #2. **[unverified, search snippet / benchmarklist mirror]** — [benchmarklist](https://benchmarklist.com/benchmarks/unigenbench/)

**T2I-CoReBench**: the only 2026 citation found covers GPT-Image-1 and Nano Banana, not GPT-Image-2 or Seedream 5. It reports that multi-instance composition is stronger in Nano Banana. **[unverified snippet]** — [arXiv 2607.29679 (cites it)](https://arxiv.org/pdf/2607.29679)

**Arena leaderboards** (all **[unverified]**: aggregator snapshots, and live pages were not fetched)
- LMArena/arena.ai, Sep 24, 2026 snapshot: gpt-image-2.5-sunburst 1424 (preliminary), Flare 1401, GPT Image 2 1383 — [flami.pro](https://flami.pro/en/blog/lmarena-text-to-image-leaderboard)
- Snapshot from Oct 26 data: Sunburst #1, Flare #2, MAI-Image-2.6 #4, Nano Banana 2.1 #5, Reve 2.1 #7. Seedream 5.0 Pro is #14 (1250) and FLUX.2 [max] #28 (1161) — [teamday.ai](https://www.teamday.ai/blog/best-ai-image-models-2026)
- Artificial Analysis AA-Image-T2I v2.1 FAQ: GPT Image 2.5 Sunburst (max) 1223, Flare (max) 1207, GPT Image 2 (high) 1189 — [Artificial Analysis](https://artificialanalysis.ai/image/leaderboard/text-to-image)
- The tech-insider article quotes an older AA Elo of 1,339 for GPT Image 2, which is inconsistent with the AA FAQ figures. A wavespeed post dated "December 2026" is future-dated and conflicting, so it should be ignored.
- Arena Elo measures overall preference, not omission. The two boards' Elo scales are not comparable.

**Older compositional benchmarks with 2026 open models only** (LLaDA-Image tables; no rewriting) **[verified]** — [arXiv 2609.03796](https://arxiv.org/html/2609.03796)
- GenEval overall / counting / position:

| Model | Overall | Counting | Position |
|---|---|---|---|
| Qwen-Image | 0.87 | 0.89 | 0.76 |
| Qwen-Image-2512 | 0.75 | **0.57** | **0.47** |
| HunyuanImage 3.0 | 0.72 | **0.48** | **0.42** |
| Z-Image | 0.84 | 0.78 | 0.62 |
| Z-Image Turbo | 0.82 | 0.77 | 0.65 |

- Single-object scores are 0.99–1.00 and two-object scores 0.92–0.99 for all of these. DPG-Bench sits at 84.9–88.6 with entity subscores of 89.6–92.5.
- LongText-Bench EN/ZH: GPT-Image 1 [High] 0.956/0.619, Qwen-Image-2512 0.956/0.965, Z-Image 0.935/0.936.
- CVTG-2K word accuracy for LLaDA-Image falls from 0.892 with 2 text regions to 0.857 with 5.

### Inferences
- The two primary-interest axes both persist at the frontier. **Counterfactual instructions** sit at ≤28% strict for the best T2I model, and the failure mode the paper names is prior reversion. **Quantity** sits at ≤17%. Both are far below arena-implied quality. This supports framing conflict and counterfactual failure as a "dominant prior wins" omission or overwrite.
- Rankings are benchmark-dependent. GPT-Image-2 leads Qwen-Image-Bench and the arenas, but on GenOlympiad it trails Nano Banana Pro, Seedream 5.0 Pro and Nano Banana 2. A likely cause is the different mix of constraint satisfaction vs aesthetics/quality, along with different judges (Q-Judger vs Gemini 2.5 Flash). A Gemini-family judge scoring a Gemini-family generator is a possible bias, but this is not tested.
- The GenEval regression of Qwen-Image-2512 relative to Qwen-Image (0.87→0.75, counting 0.89→0.57) suggests that aesthetic/realism-focused updates can trade off compositional fidelity. HunyuanImage 3.0 shows the same weak counting and position pattern. This is an inference from the reported numbers, not a claim the authors make.
- Strong LLM rewriting (GPT-5.6) more than doubles GenOlympiad strict scores, while a weaker 32B enhancer does nothing. This means any 2026 comparison must state the rewriter, consistent with GenEval 2's earlier finding.

### Gaps
- **No fetchable 2026 numbers** for GPT-Image-2/2.5, Nano Banana Pro/2, Seedream 4.5/5, FLUX.2, HunyuanImage 3 or Imagen 4 on GenEval 2, DetailMaster, LongBench-T2I, T2I-CompBench++, TIIF-Bench, OneIG or NegT2IBench. TIIF-Bench search returned nothing. The OneIG leaderboard was not checked because of the tool-call budget.
- UniGenBench++ per-model scores are image-only on GitHub. The live HF Space was not fetched.
- The appendix and remaining text of GenOlympiad (about 22k characters) were not read. There may be per-axis relaxed scores and more failure examples.
- Qwen-Image-Bench has a short/long split but publishes no long-vs-short breakdown and no per-model facet values (e.g., Quantity, Objects).

## (c) Independent evaluations of omission rate vs number of objects / prompt length on 2026 models

### Takeaway
Only counting-specific work (NumBench, Aug 2026) measures performance against requested count on commercial 2026 systems: accuracy "declines sharply", with all methods weak above 50 objects. Even there, the abstract does not separate omission from extra objects. No 2026 study found reports **per-object omission rate vs object count or prompt length** for GPT-Image-2, Nano Banana Pro, Seedream 5 or FLUX.2.

### Cited Findings
- **NumBench: Diagnosing Counting Failures in T2I Models** (arXiv 2608.28206, 28 Aug 2026). It evaluates five commercial systems, two open models and two specialized counting methods; the abstract does not name the models. Performance "declines sharply with requested count", and all methods are weak above 50 objects. Count range has the largest effect, followed by layout and composition, and grid guidance is the best layout. A 14,400-image human study validates automated evaluation up to count 50. A process model predicts a "near-quadratic collision deficit" at low occupancy. The abstract does not split under-count from over-count. **[verified abstract only]** — [arXiv](https://arxiv.org/abs/2608.28206)
- **GenOlympiad** gives prompt-length statistics (median 222 tokens) but **no correlation of score with length or constraint count** **[verified]** — [arXiv](https://arxiv.org/html/2610.26850v1)
- **CVTG-2K** text-region scaling (LLaDA-Image word accuracy 0.892 at 2 regions → 0.857 at 5) is the only "accuracy vs number of elements" curve in the 2026 open-model reports read. It covers text regions, not objects **[verified]** — [arXiv 2609.03796](https://arxiv.org/html/2609.03796)
- Already in the earlier report and only listed here for completeness: the 2608.14976 frontier failure analysis (miscounting dominates for Nano Banana Pro and FLUX.2; Ideogram 3.0 omits elements) and the GenEval 2 atomicity collapse (pre-GPT-Image-2 models).

### Inferences
- The closest proxy for "omission vs object count on 2026 frontier models" is still counting benchmarks (NumBench, GenOlympiad Quantity axis). These conflate omission and over-generation into a single "wrong count" signal.
- GPT-Image-2/2.5 and Nano Banana Pro have not been run on GenEval 2, DetailMaster or NegT2IBench in any public source found. Running them, with rewriting/thinking on and off, would itself be a novel, low-cost contribution.

### Gaps
- NumBench model names, per-count accuracy curves, and the under-count vs over-count split need the full paper (not read).
- No 2026 independent evaluation found that measures **unrequested extra objects** on frontier models, except NegT2IBench, which is already covered and did not include the frontier closed models as far as the earlier notes show.
- Whether closed-model API benchmark runs had vendor-side prompt rewriting or "thinking" enabled is generally unreported (e.g., GenOlympiad does not state whether GPT-Image-2 ran in thinking mode).
