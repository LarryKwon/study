# HF Daily Papers + recent arXiv (May–Nov 2026): T2I prompt faithfulness for objects named in the prompt

Method and verification legend (read first):
- Data pulled 2026-11-04 from the HF Papers JSON API (the same data behind huggingface.co/papers): every daily page from 2026-05-01 to 2026-11-04 via `https://huggingface.co/api/daily_papers?date=YYYY-MM-DD` (188 days, **5,415 featured papers**), plus about 56 semantic searches via `https://huggingface.co/api/papers/search?q=...`. **317** of the 5,415 featured papers are image-generation papers (keyword filter on title and abstract; titles containing "video" excluded).
- "Featured" = appeared on a Daily Papers date page; the date given is that page's date. "Up" = HF upvote count on 2026-11-04. "Not featured" = HF has a paper page (it shows up in search) but it never appeared on a daily page.
- **[V]** = verified against the abstract text returned by the HF API (numbers quoted from the abstract). **[V-title]** = only the title/existence was checked on arxiv.org/abs. **[I]** = my inference. Nothing below comes from training memory.
- OLD = arXiv ID is already in the earlier report (`T2I 객체 누락 환각 연구 동향.md` or its notes/). NEW = not in it.
- The arXiv export API returned HTTP 429 (rate limit), so the arXiv complement is partial. I used HF search (which also indexes non-featured arXiv papers) and direct arxiv.org/abs checks instead.
- Each paper's source is `https://huggingface.co/papers/<id>` unless marked arXiv.

## Q1. Which object-faithfulness-relevant T2I papers did HF Daily Papers feature May–Nov 2026, and how much attention did they get?

### Takeaway
Papers that directly target object-level faithfulness (omission, binding, counting, counterfactual) get little attention on HF. When featured they typically have 2–30 upvotes. The core omission and negation papers from the earlier report were **never indexed on HF at all**. The highly upvoted image papers (50–177 upvotes) are foundation-model tech reports, unified models, RL or on-policy distillation recipes, reward models and agentic systems. These report faithfulness only as GenEval/DPG/TIIF/WISE numbers.

### Cited Findings

**A. Directly on-topic (object presence, binding, counting, counterfactual, faithfulness evaluation)**

| arXiv | Title | HF status (date, up) | Org | One-line summary | Key numbers | Interest | Old/New |
|---|---|---|---|---|---|---|---|
| 2605.00273 | When Do Diffusion Models learn to Generate Multiple Objects? | Featured 2026-05-04, 7 | – | A controlled synthetic dataset framework ("mosaic": spatial relations, attribution, counting) is used to train diffusion models from scratch and find where multi-object failures come from | Scene complexity matters more than concept imbalance; counting is "uniquely difficult" with little data; compositional generalization collapses as more combinations are held out [V] | Omission (multi-object), counting, binding | NEW — [src](https://huggingface.co/papers/2605.00273) |
| 2606.24548 | Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning (CF-World) | Featured 2026-06-24, 11 | – | 3-level benchmark: factual → explicit counterfactual → implicit counterfactual (rules that contradict real-world priors) | New metrics Prior Resistance Rate (PRR) and Reasoning Retention Rate (RRR). All models degrade sharply from factual to counterfactual. Authors attribute this to world knowledge and appearance being "tightly coupled patterns", so models fall back on common co-occurrences [V] | Conflicting / prior-violating | OLD — [src](https://huggingface.co/papers/2606.24548) |
| 2606.03715 | Text-to-Image Models Need Less from Text Encoders Than You Think | Featured 2026-06-09, 11 | – | A "bag of position-tagged words" embedding with no prompt context guides DiT T2I as well as the full embedding | Quality and text fidelity are "on par" with the full embedding. Conclusion: complex linguistic structure is decoded by the image model, not the text encoder [V] | Mechanism for binding/omission (where composition is resolved) | NEW — [src](https://huggingface.co/papers/2606.03715) |
| 2607.19139 | Text Template Tokens Are Implicit Semantic Registers in Diffusion Transformers | Featured 2026-07-22, **73** | RTP-LLM | Causal interpretability of large DiTs: structural/template tokens become image-to-text attention sinks that causally maintain object identity | Pruning the heads that attend most to prompt tokens removes 20% of attention FLOPs for a 1.4-point GenEval drop. Prompt semantics go into image latents first and are then "read back" into template tokens [V] | Mechanism for object presence (MM-DiT internals) | NEW — [src](https://huggingface.co/papers/2607.19139) |
| 2610.06844 | Learning to Read the Contextual Tokens in Diffusion Transformers | Featured 2026-10-07, 8 | Tel Aviv University | A bottleneck maps MM-DiT contextual text tokens into a frozen LLM so the LLM can be asked about the image while it forms | Scene info is decodable early in denoising, even with an empty prompt. More readable tokens go with higher human preference. Proposes "Contextual Alignment" training [V] | Mechanism; could be a probe for omission [I] | NEW — [src](https://huggingface.co/papers/2610.06844) |
| 2607.29679 | Scaling Properties of Text Conditioning in Visual Generation | Featured 2026-08-03, 30 | ByteDance Seed | Converged diffusion loss scales with the amount of *structured* language in the prompt | Loss falls about linearly with GPG and follows a power law with ED. Structured prompts plus a trained prompter improve diffusability and promptability [V] | Dense/long prompts | OLD — [src](https://huggingface.co/papers/2607.29679) |
| 2606.24849 | IV-CoT: Implicit Visual Chain-of-Thought for Structure-Aware T2I | Featured 2026-06-25, 19 | Ant Group | Splits the conditioning queries into structural queries (a latent plan, supervised by sketches during training) then semantic queries | Targets counts, spatial relations, attribute binding and layout; "superior" GenEval and T2I-CompBench results (no number in abstract) [V] | Counting, binding | NEW — [src](https://huggingface.co/papers/2606.24849) |
| 2605.12495 | AlphaGRPO: Self-Reflective Multimodal Generation in UMMs via Decompositional Verifiable Reward | Featured 2026-05-13, 34 | – | GRPO on AR-diffusion UMMs; the reward (DVReward) uses an LLM to split the request into atomic verifiable questions, judged by an MLLM | Gains on GenEval, TIIF, DPG, WISE, and on GEdit with no editing training [V] | Dense prompts (per-requirement reward) | NEW — [src](https://huggingface.co/papers/2605.12495) |
| 2605.08043 | SCOPE: Structured Decomposition and Conditional Skill Orchestration for Complex Image Generation | Featured 2026-05-11, 8 | – | Keeps track of "semantic commitments" (each requirement in the prompt) through grounding, generation and verification; failing them is named the "Conceptual Rift". Repairs violated commitments | New Gen-Arena benchmark (human-annotated) with an entity-first metric, EGIP: 0.60. WISE-V 0.907, MindBench 0.61 [V] | **Omission** (entity-gated pass rate) | NEW — [src](https://huggingface.co/papers/2605.08043) |
| 2605.14876 | Unlocking Complex Visual Generation via Closed-Loop Verified Reasoning (CLVR) | Featured 2026-05-15, 2 | USTC | Planning ↔ diffusion loop with step-level visual verification; Proxy Prompt RL; inference cut to 4 NFEs per step | Close to proprietary models (no number in abstract) [V] | Complex/dense prompts | OLD — [src](https://huggingface.co/papers/2605.14876) |
| 2610.03617 | DEPICT: Scoring T2I Alignment by Answer Agreement | Featured 2026-10-06, 1 | Sword | Training-free metric: compares VQA answers from the image with answers from the caption alone, instead of assuming every answer should be YES | **Negation accuracy 19% → 88%**. Explicitly targets "missing objects, swapped attributes, miscounts, ignored negations". Beats all training-free metrics [V] | Negation, omission (evaluation) | NEW — [src](https://huggingface.co/papers/2610.03617) |
| 2607.21606 | TILT: Compositional Generation with a Model-Intrinsic Reward | Featured 2026-07-28, 2 | – | Training-free test-time tilting: treats a compositional failure as an overlap between the joint and single-concept distributions; the reward favors samples where *all* concepts appear together | Better compositional alignment on T2I-CompBench prompts (no number in abstract) [V] | **Omission** (joint presence) | NEW — [src](https://huggingface.co/papers/2607.21606) |
| 2605.28615 | Compositional T2I via Region-aware Bimodal DPO (BiDPO) | Featured 2026-06-02, 0 | – | DPO over image *and* text preferences plus region-level guidance; new BiComp preference set | Binding, relations, counting; "substantial" improvement (no number in abstract) [V] | Binding, counting | NEW — [src](https://huggingface.co/papers/2605.28615) |
| 2606.23835 | ABACUS: Unified Model for Count Understanding and Count-Faithful Generation | Featured 2026-06-26, 5 | – | 3B UMM; cycle-consistent GRPO in which the understanding branch critiques the count in generated images | SOTA on 7 counting benchmarks [V] | Counting | NEW — [src](https://huggingface.co/papers/2606.23835) |
| 2606.31924 | InstanceControl | Featured 2026-07-03, 9 | – | VLM-built instance correspondences for multi-instance ControlNet-style control, with no instance labels | Reduces attribute confusion between instances [V] | Binding | NEW — [src](https://huggingface.co/papers/2606.31924) |
| 2605.21343 | OcclusionFormer: Z-Order for Layout-Grounded Generation | Featured 2026-05-21, 8 | FudanCVL | SA-Z dataset; models occlusion order via volume rendering when layout boxes overlap | Addresses entangled textures where boxes overlap [V] | Multi-object (overlap) | NEW — [src](https://huggingface.co/papers/2605.21343) |
| 2606.06113 | Structured Defect Grounding (SDG) for T2I Feedback | Featured 2026-06-12, 16 | Kolors / Kuaishou | Each defect is a (location, type, reason, importance) tuple; SDG-30K dataset; BoxFlow-GRPO spatial rewards | The SDG detector beats leading proprietary VLMs at grounding defects [V] | Instance-level verification | NEW — [src](https://huggingface.co/papers/2606.06113) |
| 2606.00377 | Score-Control for Hallucination Reduction in Diffusion Models | Featured 2026-06-04, 1 | – | Links "hallucination" (samples outside the data support) to score smoothness / Lipschitz constant; Variance-Guided Score Modulation | Note: "hallucination" here means implausible samples, *not* unrequested objects [V] | Hallucination (different definition) | NEW — [src](https://huggingface.co/papers/2606.00377) |
| 2610.22602 | Stuck on Shuffle: Attribute-Object Binding in VLMs (SABER) | Featured 2026-10-19, 19 | – | Binding benchmark for *retrieval embedders* across all 4 image/text pairings, up to 10 objects | Large binding failures in CLIP-style and MLLM embedders. Fine-tuning on text-only binding data also improves negation benchmarks [V] | Binding (evaluator/encoder side, not T2I) | NEW — [src](https://huggingface.co/papers/2610.22602) |
| 2606.24256 | Trimming the Long-Tail of Visual World Modeling Evaluation (Tailor-Bench) | Featured 2026-06-30, 30 | UIUC | Regular / Unconventional / **Impossible** (attribute-violating tool) scenarios for image and video world models | Probes "constraint awareness" under prior violation [V] | Prior-violating | NEW — [src](https://huggingface.co/papers/2606.24256) |
| 2610.02959 | TerraVis: World-Grounded Visual Consistency in T2I | Featured 2026-10-09, 1 | AIML (Adelaide) | Taxonomy of 18 world-consistency violations at object, interaction and scene level; MLLM workflow | [V] | Plausibility (opposite of counterfactual) | NEW — [src](https://huggingface.co/papers/2610.02959) |

**B. On HF but never featured (low visibility), on-topic**

| arXiv | Title | HF up | Summary / numbers | Interest | Old/New |
|---|---|---|---|---|---|
| 2608.03357 | Can T2I Models Draw from the Right Frame of Reference? (FoR-T2I) | 23 | 1,200 prompt pairs; across 22 models, accuracy is 41.8% lower when the relation is described via an object's orientation (FoR) than in camera coordinates; best FoR accuracy 44.3%; VLM-gated rewriting 25.0 → 29.2% [V] | Spatial binding | NEW — [src](https://huggingface.co/papers/2608.03357) |
| 2606.31711 | Arena-T2I Hard: Faithfulness with Dependency-Aware Checklist | 1 | 310 real arena prompts × about 30 yes/no constraints each; best closed model 0.855; 33-pp spread over 11 systems; arena Elo does *not* predict faithfulness; DAG checklist reward + GDPO on SD3.5-M / FLUX.1-dev [V] | **Dense prompts / omission** | NEW — [src](https://huggingface.co/papers/2606.31711) |
| 2608.29210 | Imag-Eval | 0 | 1,140 prompts / 8,842 rules; difficulty is driven by "number of grounded rules and their binding to instances" rather than prompt length; covers missing parts and floating objects [V] | **Dense prompts / omission** | OLD — [src](https://huggingface.co/papers/2608.29210) |
| 2609.35641 | Verifiable Visual Rewards (VVR) | 0 | Synthetic geometric scenes with deterministic verifiers; 10k tasks / 32 constraint types; GPT-Image-2.5 solves 21.4% of VVRBench-Challenge; RL lifts SD3.5-M from 2.8% → 28.3% [V] | Counting, relations | NEW — [src](https://huggingface.co/papers/2609.35641) |
| 2605.30038 | Alignment-Guided Score Matching (AGSM) | 2 | Fixes over-counting and repetition caused by the contrastive soft-token method SoftREPA; >35% higher GenEval counting accuracy [V] | **Unrequested extras (over-counting)** | NEW — [src](https://huggingface.co/papers/2605.30038) |
| 2609.02864 | RIG-Bench (Thinking in Pictures) | 2 | 2,000 reasoning-driven generation samples; "locally plausible but globally illogical" outputs [V] | Reasoning | OLD — [src](https://huggingface.co/papers/2609.02864) |
| 2609.00525 | GenScale (relative object scale) | 0 | 900 entries / 1,643 pairwise scale relations; SOTA generators do not reliably keep relative scale [V] | Object-level plausibility | NEW — [src](https://huggingface.co/papers/2609.00525) |
| 2609.02502 | VMetaphor-Bench (visual metaphor) | 0 | 1,500 metaphors; tests deliberate blending of concepts from different domains [V] | Prior-violating (creative) | NEW — [src](https://huggingface.co/papers/2609.02502) |
| 2605.20758 | Conflict-Aware Additive Guidance (g^car) | 0 | Composing several guidance constraints drifts off the data manifold, with error growing as gradients misalign; detects and resolves gradient conflicts [V] | **Conflicting constraints** (sampler level) | NEW — [src](https://huggingface.co/papers/2605.20758) |
| 2605.07078 | Test-Time Compositional Generalization via Concept Discovery | 0 | Product-of-experts teacher built from score-derived density modes for out-of-distribution queries [V] | Novel compositions | NEW — [src](https://huggingface.co/papers/2605.07078) |

**C. On arXiv but not on HF at all (HF API returned "Paper not found"), from the earlier report**
- 2605.14270 "Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers" — exists on arXiv [V-title] — [arXiv](https://arxiv.org/abs/2605.14270)
- 2610.03084 "NegT2IBench: When Negation Changes the Picture. A Polarity Benchmark for Text-to-Image Models" — [V-title] — [arXiv](https://arxiv.org/abs/2610.03084)
- 2610.25645 "Negation without Confusion: Decomposing Bias and Recomposing Style in Text Embeddings for Negation-aware Image Generation" — [V-title] — [arXiv](https://arxiv.org/abs/2610.25645)
- 2608.14976 "Benchmarking Frontier Text-to-Image Models on Image-Description Prompts" — [V-title] — [arXiv](https://arxiv.org/abs/2608.14976)
- Also not indexed on HF (titles not checked): 2605.29390, 2606.30262, 2609.02255, 2609.24228, 2610.11286. 2606.05949 is on HF with 0 upvotes ("Faithful, Enriched, and Precise: Benchmarking Natural-Science Illustration Generation…") — [src](https://huggingface.co/papers/2606.05949)

### Inferences
- [I] On HF the dedicated omission and negation work is almost invisible. The single most direct omission paper (2605.14270) and both 2026 negation papers (2610.03084, 2610.25645) were never submitted to HF. The featured items closest to these topics (DEPICT 1 up, TILT 2 up, BiDPO 0 up) got almost no votes. Topically this is open ground. In attention terms, a paper here will not ride an existing wave unless it is framed in terms the community upvotes (RL rewards, UMMs, agents).
- [I] HF does upvote object-level *mechanism* work when it covers modern MM-DiTs: 2607.19139 got 73 upvotes for showing template tokens causally maintain object identity. MM-DiT interpretability tied to object identity is the most "HF-friendly" way to frame an omission study.

### Gaps
- Upvotes are a snapshot on 2026-11-04; papers featured late October have had less time to collect votes.
- The arXiv API was rate-limited (HTTP 429), so I could not systematically sweep arXiv cs.CV for May–Nov 2026. Section C covers only IDs already known from the earlier report. Other arXiv-only papers on omission or negation may exist.
- HF "organization" is self-declared and often missing ("–"), so lab attribution is incomplete.

## Q2. Interest (1): object omission under long/dense prompts — what is HF surfacing?

### Takeaway
Nothing featured on HF is framed as "object omission under long prompts". Omission comes up in three indirect forms:
- per-requirement reward or verification: decompose the prompt into atomic checks (AlphaGRPO, Arena-T2I Hard, SCOPE, Auto-Rubric);
- "dense/long prompt" as a capability of foundation models, usually text rendering (Qwen-Image-2.0 with 1K-token instructions, MOSS-Image, Lens);
- controlled studies showing that difficulty is driven by the *number of grounded entities/rules*, not prompt length (Imag-Eval, mosaic).

### Cited Findings
- Qwen-Image-2.0 (featured 2026-05-12, 119 up, Qwen): uses Qwen3-VL as the condition encoder with an MM-DiT; supports instructions "up to 1K tokens" for text-rich content (slides, posters) [V] — [HF](https://huggingface.co/papers/2605.10730)
- Qwen-Image-2.0-RL (2026-06-29, 53 up, Qwen): GRPO-based RLHF plus on-policy distillation; chain-of-thought VLM reward models for alignment, aesthetics and portraits; hybrid CFG strategy [V] — [HF](https://huggingface.co/papers/2606.27608)
- MOSS-Image (2026-10-19, 90 up, OpenMOSS): 5B single-stream DiT; agentic prompt enhancement turns requests into detailed content and layout specs; new LongLongTextBench for long-form text rendering. "Dense" here means dense text and layout, not dense objects [V] — [HF](https://huggingface.co/papers/2610.23374)
- Lens (2026-05-25, 104 up, Microsoft): 3.8B T2I trained on Lens-800M, dense GPT-4.1 captions averaging about 109 words; about 19.3% of Z-Image's training compute [V] — [HF](https://huggingface.co/papers/2605.21573)
- Imag-Eval: "compositional difficulty is primarily governed by the number of grounded rules and their binding to instances, rather than by prompt length alone" [V] — [HF](https://huggingface.co/papers/2608.29210)
- mosaic study: scene complexity is the dominant factor in multi-object failure [V] — [HF](https://huggingface.co/papers/2605.00273)
- Arena-T2I Hard: about 30 constraints per prompt; best closed system 0.855; arena preference rankings fail to predict faithfulness [V] — [HF](https://huggingface.co/papers/2606.31711)
- SCOPE / Gen-Arena: entity-gated pass rate (EGIP = 0.60 for SCOPE) is an entity-first, omission-sensitive metric [V] — [HF](https://huggingface.co/papers/2605.08043)
- Post-Training Frontier T2I by Composing Preference and Rubric Rewards (2026-10-09, 30 up, Arena): rubric rewards check prompt faithfulness; RL-trained Flux2dev gains +69 Elo; post-trained Ideogram-4 reaches Elo 1223.5 (leaderboard snapshot of 2026-09-04); releases Arena-T2I-Training (1K) [V] — [HF](https://huggingface.co/papers/2610.02967)
- AutoRubric-T2I (2026-05-22, 15 up, Arena): learns explicit rubrics from preference pairs to guide VLM judges [V] — [HF](https://huggingface.co/papers/2605.17602)
- Qwen-Image-PE (2026-10-27, 23 up, Qwen): prompt enhancement learned via Caption-Based Reverse Distillation plus a Groupwise Design Reward; +7.81 on Qwen-Image-Bench [V] — [HF](https://huggingface.co/papers/2610.33472)
- Qwen-Image-Agent (2026-06-26, 53 up, Qwen): frames underspecified requests as a "Context Gap"; plan / reason / search / memory / feedback agent [V] — [HF](https://huggingface.co/papers/2606.26907)
- Earlier-report omission paper 2605.14270 ("Diagnosing and Correcting Concept Omission in MM-DiTs") is on arXiv but not on HF [V-title] — [arXiv](https://arxiv.org/abs/2605.14270)

### Inferences
- [I] Industry handles long prompts by (a) stronger LLM/VLM text encoders, (b) dense-caption pretraining, (c) prompt enhancement / rewriting agents and (d) checklist rewards in RL. None of these explains *which* entities drop out of the denoising process, or *why*. Mechanism-level omission diagnosis on 2026 MM-DiTs (Qwen-Image-2.0-class, FLUX.2) is still a gap, unaddressed by any highly upvoted paper.
- [I] Prompt enhancement (Qwen-Image-PE, MOSS PE, Boogu agentic rewriting) makes prompts *longer*. That may make omission worse, which is a possible angle.
- [I] The finding that difficulty scales with entity count rather than prompt length (Imag-Eval, mosaic) suggests an omission study should separate entity count from token length.

### Gaps
- No featured paper reports an omission rate as a function of entity count or prompt length for 2026 frontier models. DPG/TIIF numbers in tech reports are aggregate scores.
- I did not open the full PDFs (abstracts only), so per-category omission results inside the tech reports were not checked.

## Q3. Interest (2): conflicting / contradictory / counterfactual / prior-violating requirements and negation

### Takeaway
On HF the counterfactual side shows up mainly as *reasoning or world-knowledge* benchmarks for unified models (CF-World, Tailor-Bench "Impossible" mode, RIG-Bench, WISE-style "knowledge-informed" generation), and these get modest attention (11–30 upvotes). Negation in T2I has **no featured paper**. The only featured negation result is on the *evaluator* side: DEPICT raises negation accuracy of a VQA-based metric from 19% to 88%. Conflict among *multiple constraints* appears only at the sampler-guidance level (g^car, not featured).

### Cited Findings
- CF-World: all models degrade sharply from factual to counterfactual; the cause is attributed to world knowledge and visual appearance being coupled, with models defaulting to frequent co-occurrences; metrics PRR and RRR [V] — [HF](https://huggingface.co/papers/2606.24548)
- Tailor-Bench (30 up, UIUC): an "Impossible" mode introduces attribute-violating tools to probe constraint awareness in image and video world models [V] — [HF](https://huggingface.co/papers/2606.24256)
- RIG-Bench: a "reasoning–generation gap"; outputs are locally plausible but globally illogical [V] — [HF](https://huggingface.co/papers/2609.02864)
- DEPICT (1 up): fixed-YES decomposed metrics penalize faithful images in negation cases; answer agreement raises negation accuracy 19% → 88% [V] — [HF](https://huggingface.co/papers/2610.03617)
- SABER (19 up): text-only binding supervision for embedders also improves negation retrieval benchmarks [V] — [HF](https://huggingface.co/papers/2610.22602)
- g^car: composing guidance constraints causes off-manifold drift that grows with gradient misalignment; it detects and resolves the conflicts [V] — [HF](https://huggingface.co/papers/2605.20758)
- 2606.03715: DiT T2I models use only word identity and word order from the text encoder; structure is decoded by the image model [V] — [HF](https://huggingface.co/papers/2606.03715)
- NegT2IBench (2610.03084) and "Negation without Confusion" (2610.25645) exist on arXiv but are not on HF [V-title] — [arXiv](https://arxiv.org/abs/2610.03084), [arXiv](https://arxiv.org/abs/2610.25645)
- Unified-model "knowledge" work is well upvoted but is about knowledge *transfer*, not conflict: "Can Visual Generation Expand What UMMs Know?" (29 up, Z.ai) [V] — [HF](https://huggingface.co/papers/2610.21459); "What Generation Knows" (37 up) finds generation-side representations more sensitive to fine-grained attribute and spatial changes [V] — [HF](https://huggingface.co/papers/2610.23987)
- Google's SC-CMJP (26 up): a masked-diffusion sampler that detects and repairs *cross-modal contradictions* by re-masking tokens when evidence from the other modality conflicts [V] — [HF](https://huggingface.co/papers/2607.13188)

### Inferences
- [I] If 2606.03715 is right that composition is resolved inside the image model, then negation and conflict handling must also happen in the DiT, not the text encoder. That makes MM-DiT-internal analysis of how conflicting tokens are resolved (e.g., with 2610.06844's contextual-token reader, or 2607.19139's causal head interventions) a natural, under-explored method. No featured paper applies these tools to negation or counterfactual prompts.
- [I] The community frames prior violation as a *reasoning* problem to be solved by UMMs with thinking or RL. A direct representation-level account (how a T2I model internally represents "a cat that is not black" or "ice that is hot") is absent from HF.

### Gaps
- No featured paper measures how T2I models resolve two *mutually contradictory* explicit requirements in one prompt (as opposed to counterfactual-but-consistent worlds).
- Because of the arXiv rate limit I could not check for other 2026 negation T2I papers beyond the two from the earlier report.

## Q4. Secondary: unrequested object additions (precision), attribute binding, counting

### Takeaway
Counting and binding are mainly treated as targets for RL rewards or structural planning (VVR, ABACUS, IV-CoT, BiDPO). Unrequested additions show up only indirectly: AGSM's over-counting and repetition fix, plus decomposed-metric critiques. "Hallucination" in diffusion papers on HF means implausible samples, not extra objects.

### Cited Findings
- VVR: deterministic verifiers on synthetic scenes; SD3.5-M 2.8% → 28.3% on VVRBench; GPT-Image-2.5 solves 21.4% of the Challenge set [V] — [HF](https://huggingface.co/papers/2609.35641)
- AGSM: contrastive alignment over-penalizes negative pairs and causes over-counting and repetition; fixing this gives >35% higher GenEval counting accuracy [V] — [HF](https://huggingface.co/papers/2605.30038)
- ABACUS: cycle-consistent GRPO where the understanding branch self-critiques the counts in generated images [V] — [HF](https://huggingface.co/papers/2606.23835)
- IV-CoT: structural queries carry counts, relations, binding and layout [V] — [HF](https://huggingface.co/papers/2606.24849)
- mosaic: counting is uniquely hard to learn with little data [V] — [HF](https://huggingface.co/papers/2605.00273)
- Score-Control (VSM): "hallucination" = samples outside the data support, explained via score smoothness [V] — [HF](https://huggingface.co/papers/2606.00377)
- Read It Back / SpectraReward (57 up, ByteDance Seed): reward = image-conditioned log-likelihood of the original prompt under an MLLM, with no labels; Self-SpectraReward uses a UMM's own understanding branch; larger reward MLLMs are not always better [V] — [HF](https://huggingface.co/papers/2607.11886)

### Inferences
- [I] A prompt-recovery reward (SpectraReward) mostly rewards *recall*: it checks that what the prompt names is present. It is probably weak at penalizing unrequested extras. Precision-aware rewards are an open niche, not addressed by any featured paper.

### Gaps
- No featured paper benchmarks unrequested-object rate (precision) in T2I as its main topic.

## Q5. Trend synthesis: which directions get the most HF attention (May–Nov 2026)?

### Takeaway
By upvotes the order is roughly:
1. native **unified multimodal models** (SenseNova-U1/U1.5, Boogu, Lance) and big-lab **T2I tech reports** (Qwen-Image-2.0, MOSS-Image, Lens, Mage-Flow);
2. **post-training recipes**: GRPO/NFT-style RL for flow models and, newly prominent in 2026, **on-policy (self-)distillation**;
3. **reward models / verifiers** (thinking reward models, rubric rewards, MLLM-as-reward);
4. **agentic generate-verify-edit / tool-use / harness** systems;
5. DiT/MM-DiT internals.

Dedicated compositional or faithfulness benchmarks are many but individually get few votes.

### Cited Findings
Top image-generation upvotes (featured, May–Nov 2026), all [V]:
- Editing Images into Reusable Layers via Compositional Edit-Prior Distillation — 177 up, jingdong, 2026-10-23 — [HF](https://huggingface.co/papers/2610.24738)
- SenseNova-U1 — 152 up, 2026-05-13 — [HF](https://huggingface.co/papers/2605.12500); SenseNova-U1.5 — 138 up, 2026-09-11 (OLD) — [HF](https://huggingface.co/papers/2609.11929)
- UniEvo-VL — 133 up, Stanford NLP, 2026-10-01: self-distillation in which the UMM's own self-critique is privileged information for a teacher context; built on Qwen-image-2512 — [HF](https://huggingface.co/papers/2609.38721)
- Boogu-Image-0.1 — 124 up, 2026-07-16: argues that closed systems such as Nano-Banana-Pro and GPT-Image-2 win through "system-level integration", and that open models can catch up via stronger understanding, agentic prompt rewriting and agentic inference-time scaling — [HF](https://huggingface.co/papers/2607.13125)
- Qwen-Image-2.0 — 119 up; Lens — 104 up; MOSS-Image — 90 up; Mage-Flow — 76 up, Microsoft — [HF](https://huggingface.co/papers/2607.19064)
- DiffusionART — 83 up, KlingAI, 2026-10-12: diagnoses Flow-GRPO credit assignment and DiffusionNFT instability; aggregation-first reward tilting — [HF](https://huggingface.co/papers/2610.12572)
- On-policy distillation cluster: DanceOPD 82 up (ByteDance Seed) [HF](https://huggingface.co/papers/2606.27377); On-Policy Self-Distillation in Diffusion Models 68 up (ByteDance Seed) [HF](https://huggingface.co/papers/2608.24646); Flow-OPD 65 up [HF](https://huggingface.co/papers/2605.08063); Self-OPD 41 up [HF](https://huggingface.co/papers/2608.26872)
- Reward models: Z-Reward 61 up (Tongyi-MAI) [HF](https://huggingface.co/papers/2606.09076); Think Before You Score / TRM + PD-GRPO 58 up [HF](https://huggingface.co/papers/2609.37372); Read It Back 57 up [HF](https://huggingface.co/papers/2607.11886); Diffusion Reward Models 28 up (Tsinghua NLP) [HF](https://huggingface.co/papers/2609.33803)
- Agentic: ToolArtist 58 up [HF](https://huggingface.co/papers/2608.04436); Qwen-Image-Agent 53 up; Search Beyond What Can Be Taught 42 up (TIGER-Lab) [HF](https://huggingface.co/papers/2607.05382); GenClaw (code-driven agentic generation) 38 up (Tencent Hunyuan) [HF](https://huggingface.co/papers/2605.30248); GenRouter 27 up [HF](https://huggingface.co/papers/2608.16721); APT (agentic post-training) 19 up (MSR) [HF](https://huggingface.co/papers/2610.18097); survey "Agentic Visual Generation" 20 up (Fudan) [HF](https://huggingface.co/papers/2609.06758)
- Unified reasoning / reflection: Learning Native Reflection in Unified Models with Interleaved RL 41 up (MMLab@NTU) [HF](https://huggingface.co/papers/2609.35767); Latent Reasoning in Continuous Space for UMMs 39 up (KAIST) [HF](https://huggingface.co/papers/2610.42149); AlphaGRPO 34 up
- MM-DiT internals: Text Template Tokens as Semantic Registers 73 up; Rethinking Cross-Layer Information Routing in DiTs 112 up (RTP-LLM; architecture/residual, not semantics) [HF](https://huggingface.co/papers/2605.20708)

Theme counts over the 317 featured image-generation papers (keyword match on title and abstract, so noisy and overlapping) [V, own computation from HF API data]:

| Theme | n papers | sum of upvotes | median up |
|---|---|---|---|
| Reasoning / CoT / "think" | 78 | 2,107 | 15 |
| Unified multimodal (UMM) | 68 | 2,016 | 17 |
| Compositional / binding / counting / spatial | 58 | 1,473 | 13 |
| RL / GRPO / NFT / policy optimization | 55 | 1,571 | 20 |
| Agentic / tool-use / harness | 50 | 1,606 | 20 |
| Interpretability (DiT internals) | 48 | 1,125 | 18 |
| Reward model / verifier / rubric / judge | 41 | 1,203 | 20 |
| On-policy distillation (OPD/OPSD) | 16 | 656 | **27** |
| Text rendering | 15 | 946 | **32** |

The "benchmark" keyword matched 143 papers, but it appears in most abstracts, so it is not meaningful.

### Inferences
- [I] **The direction getting the most HF attention:** UMM + RL/OPD + MLLM-derived rewards. Here the model's own understanding branch acts as the verifier (Self-SpectraReward, ABACUS cycle-GRPO, UniEvo-VL, AlphaGRPO, native reflection). Faithfulness is improved by *closing the loop* rather than by diagnosing failure modes.
- [I] **Newly prominent in 2026 compared with 2025:** on-policy distillation for flow models (highest median upvotes among the method themes), and "harness" / agentic system framing. These fit the Boogu claim that closed systems win through system integration.
- [I] Test-time scaling for T2I images is a smaller theme on HF (Flash-BoN 9 up, PreviewDiff not featured). Streaming-video test-time scaling (Stream-T1, 106 up) got more votes than any image test-time-scaling paper.
- [I] For the user's direction: an omission or conflict study gets more visibility if it (a) works on 2026 MM-DiT/UMM backbones (Qwen-Image-2.x, FLUX.2, SenseNova-U1), (b) produces a reward or verifier signal usable in GRPO/OPD (for example a precision- and recall-aware, negation-correct checklist reward), or (c) offers causal MM-DiT mechanism findings in the style of 2607.19139.

### Gaps
- Upvote counts reflect HF community behavior (biased toward industry tech reports and Chinese labs' releases). They measure attention, not quality.
- Theme counts come from regex over abstracts and were not hand-labeled.
- HF week and month pages were not scraped as HTML. The daily API covers the same data, but HF's own "trending" ranking was not captured.
