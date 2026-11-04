# Benchmarks & Metrics for Text–Image Faithfulness in T2I (object omission, unrequested objects, binding, counting), 2022–2026

> Researcher notes. Sources fetched or searched in this session (Nov 2026) are marked **[verified]**. Items marked **[prior-knowledge]** come from well-known papers whose arXiv IDs and core facts I am confident of, but which I did not re-fetch this session. The report writer should treat their exact numbers as needing a quick check.

---

## Q1. Benchmarks: what failure types they cover, size, scoring

### Takeaway
The field has moved through three generations. (1) 2022–23 had small, human-rated prompt sets (DrawBench, PartiPrompts). (2) 2023–24 had automatic compositional benchmarks scored by detectors or VQA (GenEval, T2I-CompBench, TIFA, DSG, DPG-Bench, GenAI-Bench). (3) 2025–26 has large, multi-dimensional benchmarks scored by MLLM judges (OneIG, TIIF, UniGenBench++, T2I-CoReBench, GenEval 2, SpatialGenEval). Almost every benchmark measures **recall** of requested content: object presence, attributes, counts and relations. Essentially none scores **unrequested additions** as a first-class dimension.

### Cited Findings

**First generation (human-rated prompt suites)**
- **DrawBench** (Imagen paper, NeurIPS 2022; arXiv 2205.11487) has 200 prompts in 11 categories, including colors, counting, positional, conflicting and rare words. It is scored by human side-by-side preference. **[prior-knowledge]** — [Saharia et al. 2022](https://arxiv.org/abs/2205.11487)
- **PartiPrompts / P2** (Parti paper, 2022; arXiv 2206.10789) has more than 1,600 prompts across 12 categories and 11 "challenge" aspects. It is human-rated. **[prior-knowledge]** — [Yu et al. 2022](https://arxiv.org/abs/2206.10789)
- **HRS-Bench** (ICCV 2023; arXiv 2304.05390) covers 13 skills in 5 categories (accuracy, robustness, generalization, fairness, bias), including counting, spatial composition, attribute binding and emotion. It uses automatic detectors and metrics per skill. **[prior-knowledge]** — [Bakr et al. 2023](https://arxiv.org/abs/2304.05390)
- **HEIM** (Stanford CRFM, NeurIPS 2023 D&B; arXiv 2311.04287) is a holistic benchmark covering 12 aspects, including alignment, quality, aesthetics, originality, reasoning, knowledge, bias, toxicity, fairness, robustness, multilinguality and efficiency. It evaluates 26 models with both human and automatic metrics. **[prior-knowledge]** — [Lee et al. 2023](https://arxiv.org/abs/2311.04287)

**Second generation (automatic compositional benchmarks)**
- **GenEval** (NeurIPS 2023 D&B; arXiv 2310.11513) is the object-focused benchmark most directly relevant to omission and counting. It has 553 templated prompts in 6 tasks: single object, two object, counting, colors, position and attribute binding. Scoring uses a Mask2Former COCO detector (80 COCO classes) for presence, count and position (from box coordinates), plus CLIP to classify color on the cropped object. It originally reported about 83% agreement with humans. **[prior-knowledge]**; 553 prompts / 6 dims **[verified]** via the [UniGenBench++ Table I](https://arxiv.org/html/2510.18701v1) — [Ghosh et al. 2023](https://arxiv.org/abs/2310.11513)
  - Early agreement with humans was about 83% (91% on images where annotators were unanimous) for SD2.1, IF-XL and LAION retrieval. **[verified]** — [GenEval 2 paper](https://arxiv.org/html/2512.16853v1)
  - Note: GenEval checks that requested objects exist. It does **not** penalize extra detected objects, except implicitly in counting, where too many instances of the requested class fails. **[inference from the design; see also the position paper below]**
- **T2I-CompBench** (NeurIPS 2023 D&B; arXiv 2307.06350) has 6,000 prompts across color, shape and texture binding, 2D-spatial, non-spatial and complex. It scores with BLIP-VQA (attributes), UniDet (spatial), CLIPScore (non-spatial) and a "3-in-1" combination. **T2I-CompBench++** (IEEE TPAMI 2025) adds 3D-spatial and **numeracy**, for 8 sub-categories and 8,000 prompts, and adds GroundingDINO/depth-based and MLLM scoring. **[prior-knowledge]** — [Huang et al. 2023](https://arxiv.org/abs/2307.06350). UniGenBench++ lists T2I-CompBench(++) as 8 dims / 2,400 prompts (likely the test split) — [UniGenBench++](https://arxiv.org/html/2510.18701v1) **[verified]**. *Prompt counts conflict between sources (6k/8k total vs 2.4k test).*
- **TIFA** (ICCV 2023; arXiv 2303.11897) uses an LLM to generate QA pairs from the prompt, and a VQA model answers them on the image. TIFA v1.0 has about 4K prompts, about 25K questions and 12 question categories (object, counting, color, spatial, etc.). **[prior-knowledge]** — [Hu et al. 2023](https://arxiv.org/abs/2303.11897)
- **DSG — Davidsonian Scene Graph** (ICLR 2024; arXiv 2310.18235) generates atomic, unique, dependency-aware questions as a DAG, so that a child question such as "is the dog brown?" is skipped if the parent "is there a dog?" is answered no. This fixes TIFA's hallucinated or duplicated questions. DSG-1k has about 1,060 prompts. **[prior-knowledge]** — [Cho et al. 2024](https://arxiv.org/abs/2310.18235)
- **DPG-Bench** (from the ELLA paper; arXiv 2403.05135) has 1,065 dense, long prompts across 5 dimensions: global, entity, attribute, relation and other. It uses DSG-style questions answered by mPLUG-large. **[prior-knowledge]**; 1,065 prompts / 5 dims **[verified]** via the [UniGenBench++ Table I](https://arxiv.org/html/2510.18701v1)
- **GenAI-Bench** (CVPR 2024 SynData workshop / NeurIPS 2024 D&B; arXiv 2406.13743) has 1,600 prompts tagged with "basic" skills (object, attribute, scene, spatial or action relation) and "advanced" skills (counting, comparison, differentiation, **negation**, universality). It comes with a large set of human 1–5 alignment ratings and is paired with VQAScore. **[prior-knowledge]** — [Li et al. 2024](https://arxiv.org/abs/2406.13743)
- **Gecko** (Google DeepMind; arXiv 2404.16820, ICLR 2025) has about 2K skill-tagged prompts (Gecko2K) and more than 100K human annotations under 4 rating templates. It also introduces the Gecko QA-based metric. **[prior-knowledge]** — [Wiles et al. 2024](https://arxiv.org/abs/2404.16820). An alphaXiv listing reports Gecko at Spearman 0.58 vs VQAScore 0.54 — [alphaXiv Gecko](https://www.alphaxiv.org/benchmarks/google-deepmind/gecko) **[verified, secondary]**
- **ConceptMix** (NeurIPS 2024 D&B; arXiv 2408.14339) samples k visual concepts from 8 categories (objects, colors, numbers, shapes, sizes, textures, spatial, styles) and grades with GPT-4o-generated questions. Accuracy falls steeply as k grows. **[prior-knowledge]** — [Wu et al. 2024](https://arxiv.org/abs/2408.14339)
- **EvalMuse-40K** (arXiv 2412.18150) has 40K image–text pairs from about 4K prompts with fine-grained, element-level human annotations, i.e. which prompt element is missing or wrong. It proposes the FGA-BLIP2 and PN-VQA metrics. **[prior-knowledge]** — [Han et al. 2024](https://arxiv.org/abs/2412.18150)
- **T2I-FactualBench** (arXiv 2412.04300) evaluates factual and knowledge-intensive concept generation at three levels, from single concept to multi-concept composition, with VQA-based scoring. **[prior-knowledge]** — [arXiv 2412.04300](https://arxiv.org/abs/2412.04300)

**Third generation (2025–2026, MLLM-judged, multi-dimensional)**
- **WISE** (arXiv 2503.07265) has 1,000 knowledge and reasoning prompts and the WiScore metric. **[prior-knowledge]**; 1,000 prompts / 6 dims **[verified]** via [UniGenBench++](https://arxiv.org/html/2510.18701v1)
- **OneIG-Bench** (arXiv 2506.07977, June 2025, StepFun and collaborators) scores 5 dimensions: prompt-image alignment, text rendering, reasoning, stylization and diversity. It has EN and ZH splits. **[verified]** — [arXiv abs](https://arxiv.org/abs/2506.07977). *The prompt count was not on the abstract page (gap).*
- **TIIF-Bench** (arXiv 2506.02161) tests instruction following across compositional dimensions, with a short and a long version of each prompt. UniGenBench++ lists it as 9 primary dims / 5,000 prompts. **[verified]** — [search result](https://www.arxiv.org/pdf/2506.02161); [UniGenBench++ Table I](https://arxiv.org/html/2510.18701v1)
- **DetailMaster** (arXiv 2505.16915) uses long, compositionally dense prompts across 4 attribute dimensions. Performance degrades as prompt length grows. **[verified, search snippet]** — [arXiv 2505.16915](https://arxiv.org/html/2505.16915v2)
- **LongBench-T2I** (arXiv 2505.24787) has 500 complex prompts across 9 visual dimensions. **[prior-knowledge]** — [arXiv 2505.24787](https://arxiv.org/abs/2505.24787)
- **MMIG-Bench** (arXiv 2505.19415) has about 4.8K text prompts plus multi-view reference images for customized generation. It introduces an "Aspect Matching Score" for mid-level fine-grained alignment. **[prior-knowledge; ID verified via search]** — [arXiv 2505.19415](https://arxiv.org/pdf/2505.19415v1)
- **UniGenBench / UniGenBench++** (arXiv 2510.18701, Oct 2025; Shanghai AI Lab, SJTU, Fudan, Tencent Hunyuan) has 600 prompts × {EN, ZH} × {short, long}, 10 primary dims / 27 sub-dims (attribute incl. quantity, layout, relation, logical reasoning, grammar, text, etc.). **Scoring:** Gemini-2.5-Pro judges each test point (binary 0/1 plus rationale). An offline evaluator built on UnifiedReward-2.0-qwen-72b was trained on about 375K Gemini-labeled samples. **[verified]** — [UniGenBench++](https://arxiv.org/html/2510.18701v1). The leaderboard added Nano Banana Pro, FLUX.2-dev and Z-Image in Nov 2025, and GPT-4o-1.5, Seedream-4.5 and FLUX.2 variants in Feb 2026 — [GitHub](https://github.com/codegoat24/unigenbench) **[verified via search]**
- **T2I-CoReBench** ("Easier Painting Than Thinking", arXiv 2509.03516; latest version 28 Feb 2026) tests composition at high density, separately from reasoning. A 2026 paper citing the official leaderboard (Gemini 2.5 Flash judge) reports Nano Banana 80.6 vs GPT-Image-1 79.8 overall. **[verified, secondary]** — [arXiv 2509.03516](https://arxiv.org/abs/2509.03516v2)
- **GenEval 2** (arXiv 2512.16853, Dec 2025; Kamath, Chang, Krishna, Zettlemoyer, Hu, Ghazvininejad; Meta FAIR / UW / UCLA). **[verified]** — [arXiv html](https://arxiv.org/html/2512.16853v1)
  - It has 800 templated prompts, 100 at each "atomicity" level from 3 to 10. Atomicity is the number of objects + attributes + relations.
  - The vocabulary is 40 objects (20 COCO, 20 non-COCO; 20 animate, 20 inanimate), 18 attributes (colors, materials, patterns), 9 relations (3D prepositions such as "in front of" and "behind", and transitive verbs such as "chasing" and "jumping over"), and counts of 2–7.
  - The judge is **Soft-TIFA**: one templated question per atom, scored by Qwen3-VL's P("Yes"). The arithmetic mean gives atom-level accuracy and the geometric mean gives prompt-level accuracy.
  - Code is at github.com/facebookresearch/GenEval2.
- **SpatialGenEval** (arXiv 2601.20354, Jan 2026) has 15,400 text–image pairs with information-dense spatial prompts. **[verified, search snippet]** — [arXiv 2601.20354](https://arxiv.org/html/2601.20354)
- **Imag-Eval** (EMNLP 2026; listed as arXiv 2608.29210) is a "language-grounded framework for interpretable T2I instruction-following evaluation". It argues that difficulty should be controlled by compositional structure, not just prompt length. **[verified, search snippet only — ID/venue not independently confirmed]** — [HF papers](https://huggingface.co/papers/2608.29210.md)
- **Natural-science illustration benchmark** (arXiv 2606.05949, v2 June 2026) treats every element named in the prompt as required content. It is motivated by GPT-Image-2 and Nano Banana Pro. **[verified, snippet]** — [arXiv 2606.05949](https://arxiv.org/html/2606.05949)
- Other 2025–26 text and business benchmarks that appear in tech reports are **CVTG-2K**, **BizGenEval**, **IGenBench** and **LongText-Bench**. **[verified]** — [SenseNova-U1.5 report, arXiv 2609.11929](https://arxiv.org/html/2609.11929)

**Hallucination-specific T2I benchmarks**
- **I-HallA** (arXiv 2409.12784; AAAI 2025 per prior knowledge) is a VQA-based measure of *factual* image hallucination, for example historically or scientifically incorrect content. It covers 12 categories, including object and counting. It targets factuality, not unrequested objects. **[verified, search snippet]** — [I-HallA](https://fugumt.com/fugumt/paper_check/2409.12784v6_enmode)
- **Scene-graph QA hallucination agent** (Qin et al., arXiv 2412.05722, Dec 2024). **[verified]** — [arXiv html](https://arxiv.org/html/2412.05722v1)
  - Its taxonomy is attribute / **omission** / relation / **extraneous** objects.
  - The data is 12,000 images from 1,000 T2I-CompBench prompts, generated by SDv1-4, SDv2 and SDXL, with human scores.

### Inferences
- Coverage of omission (object recall) is near-universal. Binding and counting are covered by GenEval, T2I-CompBench++, GenAI-Bench, ConceptMix, DPG, UniGenBench++ and GenEval 2. Negation is explicitly covered mainly by GenAI-Bench (and partly by UniGenBench++ and logical-reasoning sub-dimensions).
- 2025–26 benchmarks shift the difficulty axis from "which skill" to **density**: atomicity in GenEval 2, concept count k in ConceptMix, long prompts in DPG, DetailMaster and LongBench-T2I, and composition density in T2I-CoReBench. Prompt-level all-correct accuracy collapses at high density.

### Gaps
- Exact prompt counts for HRS-Bench, OneIG-Bench and T2I-FactualBench were not confirmed this session.
- I could not confirm whether T2I-CoReBench or TIIF-Bench include any explicit "no extra objects" checks. Nothing found suggests they do.

---

## Q2. Metrics: CLIPScore → detectors → VQA → MLLM judges → preference models; and is there a metric for UNREQUESTED additions?

### Takeaway
Faithfulness metrics evolved from embedding similarity (CLIPScore), to detector-based checks (GenEval), to decompositional VQA (TIFA, DSG, Gecko, Soft-TIFA), to whole-prompt VQA probability (VQAScore), to MLLM-as-judge (UniGenBench++, T2I-CoReBench, OneIG). Every one of these is fundamentally a **recall** check over prompt-derived atoms. **No standard, widely adopted metric exists for unrequested extra objects (precision / hallucination).** The only explicit treatments are a 2024 scene-graph QA preprint, which found extraneous objects too subjective to automate reliably, and a 2025 position paper that formalizes object hallucination as a set difference but runs no experiments.

### Cited Findings

**Embedding-based**
- **CLIPScore** (EMNLP 2021; arXiv 2104.08718) is reference-free cosine similarity between CLIP image and text embeddings. **[prior-knowledge]** — [Hessel et al. 2021](https://arxiv.org/abs/2104.08718)
- CLIP-like VLMs behave like **bag-of-words** models: they are largely insensitive to word order, relations and attribute–object binding. This was shown by the ARO benchmark, "When and why vision-language models behave like bags-of-words" (ICLR 2023 oral; arXiv 2210.01936). **[prior-knowledge]** — [Yuksekgonul et al. 2023](https://arxiv.org/abs/2210.01936)
- In the scene-graph hallucination study, CLIPScore correlated with human hallucination scores at only Pearson 0.14 / Kendall 0.10 / Spearman 0.13. BLIP-Score was 0.12 / 0.06 / 0.09. **[verified]** — [Qin et al. 2412.05722](https://arxiv.org/html/2412.05722v1)

**Detector-based**
- GenEval uses Mask2Former and CLIP. Its vocabulary is limited to the 80 COCO classes, which motivated GenEval 2's move to non-COCO objects and a VQA judge. **[verified]** — [GenEval 2](https://arxiv.org/html/2512.16853v1)
- T2I-CompBench++ uses UniDet and GroundingDINO for spatial and numeracy. **[prior-knowledge]** — [T2I-CompBench](https://arxiv.org/abs/2307.06350)

**VQA / question-decomposition**
- TIFA uses LLM-generated QA and a VQA model to compute accuracy. DSG adds atomic, dependency-aware questions. **[prior-knowledge]** — [TIFA](https://arxiv.org/abs/2303.11897), [DSG](https://arxiv.org/abs/2310.18235)
- **VQAScore** (ECCV 2024; arXiv 2404.01291) is P("Yes" | image, "Does this figure show {text}?") from a VQA model (CLIP-FlanT5). Its authors report state-of-the-art results on DrawBench, EditBench, COCO-T2I and others, outperforming methods trained on human feedback or relying on proprietary models. **[verified, search snippet]** — [Lin et al. 2024](https://arxiv.org/pdf/2404.01291)
- **Soft-TIFA** (GenEval 2) uses per-atom soft probabilities. The geometric mean better approximates prompt-level correctness because any low-scoring atom drags it down. **[verified]** — [GenEval 2](https://arxiv.org/html/2512.16853v1)

**MLLM-as-judge**
- UniGenBench++ uses Gemini-2.5-Pro for binary per-test-point judgments plus a distilled offline 72B judge. T2I-CoReBench uses a Gemini 2.5 Flash evaluator. **[verified]** — [UniGenBench++](https://arxiv.org/html/2510.18701v1), [T2I-CoReBench](https://arxiv.org/abs/2509.03516v2)
- EvalAlign reports that using MLLMs directly for T2I evaluation "does not yield optimal results" without supervised fine-tuning on human-annotated generated images. **[verified, snippet]** — [EvalAlign arXiv 2406.16562](https://arxiv.org/html/2406.16562v1)

**Human-preference reward models.** These are used as metrics, but they measure overall preference, not object-level faithfulness. All entries below are **[prior-knowledge]**.
- **ImageReward** (NeurIPS 2023; arXiv 2304.05977) is trained on 137K expert comparisons. — [Xu et al.](https://arxiv.org/abs/2304.05977)
- **PickScore** / Pick-a-Pic (NeurIPS 2023; arXiv 2305.01569) is trained on crowd preferences from a web app. — [Kirstain et al.](https://arxiv.org/abs/2305.01569)
- **HPSv2** (arXiv 2306.09341) is trained on HPD v2, about 798K choices. — [Wu et al.](https://arxiv.org/abs/2306.09341)
- **HPSv3** (ICCV 2025; arXiv 2508.03789) is trained on HPDv3, about 1.08M text–image pairs and 1.17M comparisons. — [arXiv 2508.03789](https://arxiv.org/abs/2508.03789)
- **MPS** (Multi-dimensional Preference Score, CVPR 2024; arXiv 2405.14705) scores aesthetics, semantic alignment, detail quality and overall separately. — [Zhang et al.](https://arxiv.org/abs/2405.14705)

**Unrequested additions (precision / hallucination)**
- **Formal definition:** "Hallucination as an Upper Bound: A New Perspective on T2I Evaluation" (Kasaei & Rohban, Sharif Univ.; arXiv 2509.21257, Sept 2025, a position paper; a papernotes site lists it under NeurIPS 2025, likely a workshop). **[verified]** — [arXiv html](https://arxiv.org/html/2509.21257v1); [papernotes](https://en.papernotes.org/NeurIPS2025/hallucination/hallucination_as_an_upper_bound_a_new_perspective_on_text-to-image_evaluation/)
  - Object hallucination is defined as follows: prompt objects O = {o1…on}; if the image contains a non-empty set O′ with O′ ∩ O = ∅, then O′ is hallucination.
  - The paper contrasts this with "object neglect", i.e. omission.
  - It also defines attribute hallucination (e.g. "a doctor" → male in a white coat) and relation hallucination (e.g. "a man and a dog" → leash-walking).
  - Its key framing: "alignment metrics establish only a *lower bound*, measuring whether requested elements are present", whereas "hallucination evaluation sets an *upper bound*, capturing what the model introduces beyond the prompt."
  - On plausible context: additions "may be plausible" but "often distort the focus or semantics of the prompt", and they "reflect model assumptions about scene completion." Relation defaults "are not errors per se, but overlearned associations that compromise prompt neutrality."
  - It runs **no experiments** and proposes no benchmark numbers.
- **Empirical treatment:** Qin et al. (arXiv 2412.05722). **[verified]** — [arXiv html](https://arxiv.org/html/2412.05722v1)
  - They state that "even human raters have not reached a consensus" on extraneous objects, so automated evaluation of this type is hard.
  - Their human rubric gives an image with unmentioned objects **6/7**, treating it as a tolerable error.
  - Multiple instances of a requested class are not counted as extraneous.
  - F1 for detecting extraneous objects was only 0.33–0.45.
  - Their rule-based GraphQA (GPT-4) scorer reached Pearson 0.26 with human scores, tied with T2I-CompBench on Spearman (0.27).
- **Captioning analogue:** CHAIR and OpenCHAIR (arXiv 2312.03631) measure object hallucination as the hallucinated fraction of mentioned objects in *captions*. OpenCHAIR uses SDXL-generated images, but the metric is not used for T2I precision. **[verified, snippet]** — [OpenCHAIR](https://arxiv.org/html/2312.03631v3)
- **Recall-only by design:** GenEval 2's Soft-TIFA, TIFA, DSG and VQAScore only ask about prompt-derived atoms, so extra content goes unscored. **[verified for Soft-TIFA design]** — [GenEval 2](https://arxiv.org/html/2512.16853v1)

### Inferences
- **Answer to "is there a standard metric for unrequested additions?" — No.** The closest candidates are:
  - (a) open-vocabulary detection or captioning of the image, followed by a set difference against prompt entities (CHAIR-style precision). This is proposed conceptually but not standardized or validated for T2I.
  - (b) human rubrics, which treat extra objects as minor.
- **Background and contextual objects:** in practice every benchmark treats plausible context (a table under a cup, a rider on a horse) as **neutral, i.e. not penalized**. Detector and VQA metrics never ask about it. Human rubrics (Qin et al.) give at most a small penalty. Only the 2025 position paper argues it should count as hallucination. Any precision metric would need an explicit policy separating "scene-completion priors" (plausible, usually tolerated) from "semantic contamination" (an added object that changes meaning or conflicts with counts or negation).
- Counting and negation prompts are where "extra" objects are *implicitly* penalized: an extra instance fails a count check, and "no X" fails if X appears. These are the only places existing benchmarks measure precision at all.

### Gaps
- I found no peer-reviewed, large-scale T2I benchmark with a validated *extra-object precision* score as of Nov 2026. Some may exist after my search horizon or under different terminology, e.g. "over-generation", "content leakage" or "prompt neutrality".
- I did not verify I-HallA's prompt and QA counts.

---

## Q3. Reliability: metric–human agreement, MLLM judge hallucination, GenEval saturation/drift, reward hacking

### Takeaway
Automatic metrics agree with humans reasonably well at the *atom* level (AUROC around 91–94 for VQA-style judges on GenEval 2). They are weak on holistic hallucination judgments, with correlations around 0.1–0.3. They also **drift**: a static judge increasingly mis-scores newer models. GenEval, the most-cited compositional benchmark, is saturated under human evaluation (≥94.8% for SOTA models) while its automatic scorer *under*-reports them by 11–18 points. Optimizing directly on detector-based rewards (Flow-GRPO on GenEval) yields near-ceiling scores, with documented reward-hacking side effects.

### Cited Findings

**GenEval saturation and "benchmark drift"** — all from the GenEval 2 paper. **[verified]** — [arXiv 2512.16853](https://arxiv.org/html/2512.16853v1)
- The human study covered 8 T2I models, 553 GenEval prompts and 23,226 annotations, with 3 annotators per image.
- Gemini 2.5 Flash Image scored 75.4% on automatic GenEval but **93.1%** with humans, a 17.7-point gap. With prompt rewriting it scored 82.1% automatic vs **96.7%** human.
- The average automatic–human deviation for SOTA models is **11.0 points** without rewriting.
- All four SOTA models score ≥94.8% with humans under rewriting (Qwen-Image 94.8%), so GenEval is "effectively saturated".
- Benchmark drift is defined as the decline in alignment between a static benchmark's automatic judge and humans as model output distributions shift.
- VQAScore's human alignment falls for models released after its VQA backbone. Soft-TIFA_GM drops less, but the authors caution that drift may still occur.

**Atom-level human alignment on GenEval 2** (AUROC; original / rewritten / both prompts). **[verified]** — [GenEval 2](https://arxiv.org/html/2512.16853v1)

| Judge | Original | Rewritten | Both |
|---|---|---|---|
| TIFA | 91.2 | 91.2 | 91.6 |
| VQAScore | 93.5 | 90.5 | 92.4 |
| Soft-TIFA_AM | 92.8 | 92.9 | 93.0 |
| Soft-TIFA_GM | 94.2 | 94.4 | 94.5 |
| GPT-4o-based VQAScore | 91.2 | — | — |

**Judge errors on GenEval 2.** **[verified]** — [GenEval 2](https://arxiv.org/html/2512.16853v1)
- Soft-TIFA *overestimates* prompt-level accuracy for some models: Gemini Soft-TIFA_GM 44.6 vs human 31.0.
- On the verb-relation skill it is *stricter* than humans (31.4 vs 65.0).
- The paper has internal inconsistencies between text and Table 5. For example, Gemini counting is 69.2 in the text vs 70.1 in the table.

**Hallucination-level human agreement is low.** Correlation with human hallucination scores (Pearson):

| Scorer | Pearson |
|---|---|
| Rule-based GraphQA (GPT-4) | 0.26 |
| T2I-CompBench | 0.24 |
| LLM-Score | 0.19 |
| CLIPScore | 0.14 |
| BLIP-Score | 0.12 |

**[verified]** — [Qin et al. 2412.05722](https://arxiv.org/html/2412.05722v1)

**Gecko**
- The Gecko study showed that metric rankings change with prompt set and human-rating template. This is why it collected annotations under 4 templates. **[prior-knowledge]** — [Wiles et al. 2404.16820](https://arxiv.org/abs/2404.16820)
- A secondary listing gives Gecko Spearman 0.58 vs VQAScore 0.54. **[verified, secondary]** — [alphaXiv](https://www.alphaxiv.org/benchmarks/google-deepmind/gecko)

**MLLM judges**
- Off-the-shelf MLLMs are poorly calibrated on generated images without fine-tuning. **[verified, snippet]** — [EvalAlign](https://arxiv.org/html/2406.16562v1)
- In MLLM-as-a-Judge (2024, general multimodal tasks, not T2I), human agreement with GPT-4V was about 70%. **[verified, snippet]** — [alphaXiv 2402.04788](https://alphaxiv.org/overview/2402.04788v3)
- A 2025 study (arXiv 2509.12750) finds frontier MLLMs and fine-tuned reward models perform similarly as overall judges, but are less consistent on specific attributes. **[verified, secondary review]** — [Moonlight review](https://www.themoonlight.io/review/what-makes-a-good-generated-image-investigating-human-and-multimodal-llm-image-preference-alignment)
- GenExam validates its MLLM judge against 5 experts on 250 images (MAE plus correlation). This is the emerging validation practice. **[verified, snippet]** — [GenExam 2509.14232](https://arxiv.org/pdf/2509.14232)

**Reward hacking**
- **Flow-GRPO** (arXiv 2505.05470) raised SD3.5-Medium GenEval accuracy from **63% to 95%** using GenEval's detector-based rule as an RL reward. Its authors claim minimal hacking thanks to a KL constraint. **[verified, snippet]** — [Flow-GRPO](https://arxiv.org/pdf/2505.05470)
- Critiques:
  - **Adv-GRPO** says rule-based rewards (OCR, GenEval) are "prone to reward hacking". **[verified, snippet]** — [arXiv 2511.20256](https://arxiv.org/html/2511.20256)
  - **DenseGRPO** reports that a GenEval reward improves counting while image quality may decline. **[verified, snippet]** — [DenseGRPO 2601.20218](https://arxiv.org/pdf/2601.20218)
  - **GDRO** states Flow-GRPO "suffered from alignment and quality degradation, indicating that reward hacking truly occurs". **[verified, snippet]** — [GDRO 2601.02036](https://arxiv.org/pdf/2601.02036)
  - **GRPO-Guard** (CVPR 2026) describes implicit over-optimization in which proxy scores stay high while gold scores drop. It reports a GenEval gold score of 0.89 vs Flow-GRPO's 0.84. **[verified, snippet]** — [CVPR 2026 supp.](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_GRPO-Guard_Mitigating_Implicit_CVPR_2026_supplemental.pdf)

### Inferences
- GenEval numbers above about 0.90 in 2025–26 tech reports are no longer informative. Their two failure modes are opposite:
  - *under*-crediting SOTA models through detector drift, i.e. judge false negatives on non-COCO-looking renders;
  - *over*-crediting RL-tuned models that learn to satisfy the detector.
- Judges built on VQA or MLLMs inherit the hallucination tendencies of the judging MLLM. Because the evaluation is recall-oriented ("is there an X?"), the dominant judge error is likely a false "yes" on near-miss objects or counts. The Gemini 44.6 vs 31.0 over-estimate fits this. The verb-relation result shows judges can also be over-strict.
- A practical rule emerging from GenEval 2: prefer a geometric-mean or all-atoms-correct prompt-level score, report atom-level and prompt-level scores separately, and re-validate the judge against fresh human labels for each new model generation.

### Gaps
- I found no systematic study of how MLLM-judge *hallucination* specifically produces false positives for object presence in T2I evaluation, separate from generic agreement numbers.
- I did not verify the exact human-agreement numbers for GenAI-Bench, Gecko or DPG-Bench this session.

---

## Q4. Leaderboard state (2025–2026) and lowest-scoring failure categories

### Takeaway
On legacy benchmarks top models cluster at the ceiling: GenEval about 0.84–0.92 automatic (≥95% human), DPG-Bench about 85–88.6, OneIG-EN overall about 0.54–0.58. The remaining differentiation is in **counting, position/spatial, attribute binding, transitive-verb relations, high-density (many-atom) prompts, reasoning and negation**. On GenEval 2, even the best models get only about 32–36% of prompts fully right, and only 2–4% at atomicity 10.

### Cited Findings

**GenEval (automatic)** — single / two / counting / colors / position / attr-binding / overall. Reported in the SenseNova-U1.5 report (arXiv 2609.11929, Sept 2026). **[verified]** — [arXiv 2609.11929](https://arxiv.org/html/2609.11929)

| Model | Single | Two | Count | Colors | Position | Attr. | Overall |
|---|---|---|---|---|---|---|---|
| SenseNova-U1.5 | 1.00 | 0.97 | 0.93 | 0.92 | 0.92 | 0.81 | **0.92** |
| Qwen-Image (20B) | 0.99 | 0.92 | 0.89 | 0.88 | 0.76 | 0.77 | 0.87 |
| GPT-Image-1 | 0.99 | 0.92 | 0.85 | 0.92 | 0.75 | **0.61** | 0.84 |
| Seedream 4.0 | 0.99 | 0.92 | **0.72** | 0.91 | 0.76 | 0.74 | 0.84 |
| Nano-Banana-2.0 | 1.00 | 0.96 | **0.71** | 0.84 | 0.86 | 0.65 | 0.83 |

- The weakest GenEval sub-tasks for frontier closed models are **attribute binding (0.61–0.74)**, **counting (0.71–0.85)** and **position (0.75–0.86)**. Single and two object are saturated (≥0.92).
- The same Qwen-Image GenEval row also appears in the GEMS paper (arXiv 2603.28088). **[verified, snippet]** — [GEMS](https://arxiv.org/pdf/2603.28088)
- Flow-GRPO-tuned SD3.5-M reaches 0.95 on GenEval. **[verified]** — [Flow-GRPO](https://arxiv.org/pdf/2505.05470)

**DPG-Bench** — global / entity / attribute / relation / other / overall. **[verified]** — [arXiv 2609.11929](https://arxiv.org/html/2609.11929)

| Model | Global | Entity | Attr. | Relation | Other | Overall |
|---|---|---|---|---|---|---|
| Seedream 4.5 | — | — | — | — | — | 88.63 |
| Qwen-Image | 91.32 | 91.56 | 92.02 | 94.31 | 92.73 | 88.32 |
| Z-Image (6B) | 93.39 | 91.22 | 93.16 | 92.22 | 91.52 | 88.14 |
| SenseNova-U1.5 | 86.54 | 93.61 | 91.52 | 93.33 | 92.13 | 88.11 |
| Nano-Banana-Pro | 84.19 | 92.47 | 89.58 | 94.74 | 81.12 | 86.59 |
| BAGEL (7B) | 88.94 | 90.37 | 91.29 | 90.82 | 88.67 | 85.07 |

- The spread among top models is about 2 points, which points to saturation.

**OneIG-Bench EN** — alignment / text / reasoning / style / diversity / overall. **[verified]** — [arXiv 2609.11929](https://arxiv.org/html/2609.11929)

| Model | Align. | Text | Reas. | Style | Div. | Overall |
|---|---|---|---|---|---|---|
| Nano-Banana-Pro | 0.889 | 0.962 | 0.333 | 0.483 | 0.247 | **0.583** |
| SenseNova-U1.5 | 0.883 | 0.988 | 0.329 | 0.402 | 0.159 | 0.552 |
| Qwen-Image | 0.882 | 0.891 | 0.306 | 0.418 | 0.197 | 0.539 |
| FLUX.1-dev | 0.786 | 0.523 | 0.253 | 0.368 | 0.238 | 0.434 |
| BAGEL | 0.769 | 0.244 | 0.173 | 0.367 | 0.251 | 0.361 |

- Alignment is near the ceiling (about 0.88). **Reasoning (≤0.33)** and diversity are the weak dimensions.
- OneIG-ZH overall: Nano-Banana-Pro 0.569, Qwen-Image 0.548, SenseNova-U1.5 0.536, BAGEL 0.370.

**WISE (knowledge/reasoning) overall.** **[verified]** — [arXiv 2609.11929](https://arxiv.org/html/2609.11929)

| Model | Overall |
|---|---|
| Nano-Banana-Pro | 0.87 |
| SenseNova-U1.5 (CoT) | 0.81 |
| GPT-Image-1 | 0.80 |
| Seedream 4.0 | 0.78 |
| Qwen-Image | 0.63 |
| BAGEL (no CoT) | 0.49 |

**UniGenBench++ overall (Gemini-2.5-Pro judge).** **[verified]** — [UniGenBench++](https://arxiv.org/html/2510.18701v1)
- EN-short: GPT-4o 92.77, Imagen-4.0-Ultra 91.54, Nano Banana 87.45, Seedream-4.0 87.35, Qwen-Image 78.81 (best open model), Hunyuan-Image-2.1 74.64, FLUX.1-dev 61.30.
- EN-long: GPT-4o 92.63, Imagen-4-Ultra 90.95, Seedream-4.0 89.77, Qwen-Image 83.94.
- **Logical reasoning** is the weakest dimension: Qwen-Image 53.64, HiDream 26.73, SDXL 9.55. Grammar is a weak point for open models (Qwen-Image 60.29).
- Quantity and counting per-model values were not extractable.

**T2I-CoReBench.** Nano Banana 80.6 vs GPT-Image-1 79.8 overall. **[verified, secondary]** — [arXiv 2509.03516](https://arxiv.org/abs/2509.03516v2)

**GenEval 2 (human-judged, with rewriting)** — atom-level / prompt-level. **[verified]** — [GenEval 2](https://arxiv.org/html/2512.16853v1)

| Model | Atom-level | Prompt-level |
|---|---|---|
| Qwen-Image | 85.3% | 35.8% |
| Gemini 2.5 Flash Image | 84.4% | 32.3% |
| FLUX.1-dev | 80.4% | 29.5% |
| BAGEL+CoT | 76.8% | 22.3% |
| SD3.5-L | 69.7% | 19.0% |
| SD3-M | 69.3% | 15.8% |
| SDXL (no rewrite) | 45.3% | 4.3% |
| SD2.1 (no rewrite) | 33.7% | 2.5% |

- At atomicity 10, Gemini reaches 4% and Qwen-Image 2%.
- The hardest skills are **counting, spatial position and transitive-verb relations**. Qwen-Image is weakest on relations.

### Inferences
- **Lowest categories, ranked across benchmarks:**
  1. dense or multi-object composition (prompt-level, high-atomicity);
  2. counting, especially beyond about 4 and in multi-object scenes;
  3. spatial position and 3D relations;
  4. action and verb relations;
  5. attribute binding with several objects;
  6. reasoning, logical constraints and negation (UniGenBench++ logical reasoning, GenAI-Bench negation).
- Object omission in single- or two-object prompts is essentially solved. Omission persists in long and dense prompts, as shown by the DPG, DetailMaster, LongBench-T2I and GenEval 2 atomicity curves.
- Because no benchmark measures unrequested additions, **leaderboards say nothing about whether frontier models over-generate content**. This is an open measurement gap, distinct from saturation.

### Gaps
- I did not obtain official, current numbers for GPT-Image-2, Nano Banana Pro, Seedream 4.5/5 or FLUX.2 on GenEval 2, T2I-CompBench++ or TIIF-Bench. Those leaderboards (HF/GitHub) should be checked directly.
- GenEval numbers reprinted across tech reports can differ by evaluation setting, e.g. with or without prompt rewriting. The reprinted values above come from a single report (SenseNova-U1.5) and may differ from the original model reports.
- The T2I-CompBench++ per-category leaderboard (numeracy, 3D-spatial) for 2025–26 models was not retrieved.
