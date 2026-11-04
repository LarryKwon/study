# Conflicting, Contradictory, Counterfactual & Prior-Violating Requirements in T2I Prompts (2023–2026)

> Scope note: Claims marked **[verified]** were checked against the paper page/HTML in this session. Claims marked **[prior-knowledge]** come from the researcher's background knowledge of well-known papers; the arXiv link is given, but numbers were not re-checked in this session. Treat those as "verify before quoting." Newest work (mid/late 2026) is flagged **[NEW 2026]**.

---

## Q1. Explicitly contradictory / conflicting constraints within a prompt: which concept wins?

### Takeaway
A small but growing body of work (mostly 2025–2026) shows that when concepts in a prompt conflict, either explicitly or through learned associations, the model does not blend them evenly. One concept dominates, usually the one with stronger or more visually homogeneous training priors, or the one that controls early layout and color. The suppressed concept gets **dropped** (omission) or **overwritten** (e.g., "midday" becomes night). Conflicts are mostly decided in the **first denoising steps** and cannot be undone later.

### Cited Findings
- **"Image Generation from Contextually-Contradictory Prompts"** (Huberman, Patashnik, Dahary, Mokady, Cohen-Or; arXiv 2506.01929, v1 Jun 2025, v2 Mar 2026; no venue listed). It defines *contextual contradiction*: "one concept implicitly negates another due to entangled associations learned during training." **[verified]** — [arXiv](https://arxiv.org/abs/2506.01929)
  - Examples: "A butterfly is in a bee's hive" (the butterfly→flower association overrides the hive); "A woman writing with a dart" (the "writing" token dominates the hand/tool region and the dart is dropped); "A howling wolf and a flying bat at midday" (early steps impose night and a moon, contradicting "midday"); "A dragon blowing water" (fire entanglement). **[verified]** — [HTML](https://arxiv.org/html/2506.01929)
  - Mechanism: denoising runs coarse-to-fine. Layout and dominant colors form at about steps 0–2, shape and pose at 3–6, identity and material at 7–10, fine details at 11+. Coarse decisions "cannot be revised in later stages, even if they conflict with the prompt." **[verified]** — [HTML](https://arxiv.org/html/2506.01929)
  - Method **SAP** (stage-aware prompt decomposition): an LLM detects the contradiction and writes a *sequence of proxy prompts*, one per denoising stage (e.g., "dog"/"bird" sets a daytime layout, then wolf/bat identities are swapped in). **[verified]**
  - Benchmarks: Whoops! (500 prompts), Whoops-Hard (100), and **ContraBench** (40 GPT-generated, human-filtered contradictory prompts). GPT-4o alignment scores (20–100 scale): SD3.0 82.63 / 55.73 / 57.5; FLUX 78.85 / 44.3 / 57.16; R2F 83.50 / 57.06 / 59.16; **SAP (SD3.0) 85.87 / 64.40 / 65.33**; SAP (FLUX) 85.10 / 62.13 / 66.16. In the user study (61 users, 1,464 comparisons), SAP beat FLUX 81%, SD3 70%, and R2F 75%. Ablation on Whoops-Hard: a static single prompt scores 44.3 and "no reasoning" scores 46.46, against 62.13 for the full method. **[verified]** — [HTML](https://arxiv.org/html/2506.01929)
- **"Dominant vs. Dominated: Concept-Level Generative Collapse in Diffusion Models"** (Jeong & Lee, Yonsei; arXiv 2512.20666, v2 Jul 2026, no venue) **[NEW 2026]**. In two-concept prompts, one concept "dominates the output while others are suppressed." Example: "Neuschwanstein Castle coaster": the castle always appears and the coaster rarely does. **[verified]** — [arXiv HTML](https://arxiv.org/html/2512.20666)
  - **DominanceBench**: 300 two-concept LAION-derived prompts that pair low-diversity concepts (artists, landmarks, characters) with everyday objects. The DvD score is computed from VLM yes/no questions. **[verified]**
  - Cause: concepts trained on **visually homogeneous images** (low diversity) dominate more. Controlled fine-tuning shows a monotonic trend (shown qualitatively with box plots). **[verified]**
  - Attention: the dominant token gets the highest cross-attention in **84.33%** (253/300) of prompts on SD 1.4 and 54.0% on SD 2.1. The peak is in low-resolution UNet blocks at the **first denoising step**, and the dominated token's attention collapses by about t=50→40. **[verified]**
  - Dominance is **distributed across heads**. Single-head ablation fixes 77.6% of *memorization* prompts but only 45.3% (SD1.4) and 12.0% (SD2.1) of DvD prompts, and multi-head ablation makes it worse. No validated fix is proposed; the authors suggest data-side remedies. **[verified]**
- **CoMat** (arXiv 2404.03653) splits misalignment into *concept ignorance* (a concept is dropped) and *concept mismapping* (attributes go to the wrong object), and reports that mismapping is most common among attributes. **[verified via search snippet]** — [arXiv](https://arxiv.org/html/2404.03653v2)
- **Attend-and-Excite** (Chefer et al., SIGGRAPH 2023, arXiv 2301.13826) named "catastrophic neglect": in multi-subject prompts, one subject is dropped. Its fix strengthens the cross-attention of neglected tokens at test time. This is the earliest framing of "concept competition → omission." **[prior-knowledge]** — [arXiv](https://arxiv.org/abs/2301.13826)
- **SPDiffusion** (arXiv 2409.01327) targets "attribute confusion" in multi-concept prompts. **[verified via snippet]** — [arXiv](https://arxiv.org/html/2409.01327v1)
- Long, detailed prompts can introduce "ambiguities or conflicting instructions, making it challenging for the model to determine which elements to prioritize" (TextInVision, arXiv 2503.13730). **[verified via snippet]** — [arXiv](https://arxiv.org/html/2503.13730)

### Inferences
- Across these papers, "which constraint wins" depends on (a) the strength and homogeneity of the training prior, (b) which token takes cross-attention at the **first step**, and (c) whether the attribute is coarse (color, lighting, layout: decided early, hard to override) or fine (identity, texture: changeable later). This gives a testable hypothesis: in a long prompt, conflicting *global* attributes such as time of day, color palette, or style are resolved by early-step attention, while conflicting *local* details are resolved later.
- Omission and contradiction are linked: a dropped concept is often the *losing* side of an implicit conflict. Measuring omission without measuring the conflict may misattribute the cause.

### Gaps
- I found no benchmark of **explicit, logically unsatisfiable** prompts (e.g., "a red apple that is entirely blue", "a cat both left and right of the dog") that records *which* side the model picks, or whether it blends or drops. ContraBench (40 prompts) covers *contextual* (prior-induced), not logical, contradictions.
- I found no systematic study of **conflicting styles** ("photorealistic watercolor"), **position bias** (first- vs. last-mentioned attribute wins), or **conflicting spatial constraints**. A third-party summary claimed models favor the first-mentioned attribute, but I could not confirm that in the paper, so it is unverified.
- I found no work linking *conceptual blending* (creative fusion) to *conflict resolution* in a measured way.

---

## Q2. Counterfactual / prior-violating compositions and role reversal ("a horse riding an astronaut")

### Takeaway
Rare or reversed compositions reliably collapse to the frequent version. The gap between frequent and reversed prompts is often more than 40 VQAScore points, and scaling (DALL-E 3, SD3.5) does not close it. 2026 benchmarks such as CF-World formalize "prior resistance" and show steep drops from factual to counterfactual settings for all models.

### Cited Findings
- "A horse riding an astronaut" came from Gary Marcus's 2022 critique of Imagen as an informal probe, not a benchmark. Adding "on back of" improved success, which suggests the failure is in language-to-image binding rather than rendering ability. — [Marcus on AI](https://garymarcus.substack.com/p/horse-rides-astronaut)
- **"Role Bias in Diffusion Models: Diagnosing and Mitigating through Intermediate Decomposition"** (Malakouti & Kovashka, Pitt; arXiv 2503.10037; no venue listed). **RoleBench** covers 10 action verbs (chasing, riding, feeding…), each as a frequent/reversed pair ("cat chasing mouse" vs. "mouse chasing cat"), 420 prompts in total. **[verified]** — [arXiv HTML](https://arxiv.org/html/2503.10037)
  - **"Role collapse"**: on the rare prompt, the model produces nearly the same image as on the frequent one. All models tested (SDXL, SD3, SD3.5, AuraFlow2, DALL-E 3) show it, with rare–frequent gaps often **>40 VQAScore points**. Larger models do not fix it. **[verified]**
  - Compositional methods (IterComp, RPG, SLD, **R2F**, RRNet) reduce the bias "only modestly, if at all." **[verified]**
  - Cause: distributional asymmetry in training data, not model capacity. Intermediate compositions ("mouse chasing boy") show no bias. **[verified]**
  - **ReBind**: LLM decomposition into active/passive intermediates, synthetic data, and a SDXL LoRA. The β bias drops from 23.1 to 7.9, and humans prefer it over R2F 74% and over DALL-E 3 73%. **[verified]**
- **CF-World — "Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning"** (Lei, …, Hongsheng Li; arXiv 2606.24548, Jun/Jul 2026) **[NEW 2026]**. It has three levels: factual → *explicit* counterfactual (direct visual instruction) → *implicit* counterfactual (deduce from altered world rules). Metrics are **Prior Resistance Rate (PRR)** and **Reasoning Retention Rate (RRR)**, scored by the VLM judge CF-Eval. "All evaluated models show sharp degradation from factual to counterfactual settings," which the authors attribute to world knowledge and appearance being encoded as "tightly coupled patterns" and to reliance on frequent co-occurrence. (Per-model numbers were not on the abstract page.) **[verified abstract]** — [arXiv](https://arxiv.org/abs/2606.24548)
- **R2F — "Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance"** (Park et al.; **ICLR 2025 spotlight**; arXiv 2410.22376). It is training-free: an LLM plans a schedule that starts sampling with *frequent* related concepts and switches to the *rare* target. It introduces **RareBench**. Gains over SD3.0/FLUX are up to **28.1 %p** in T2I alignment. The theory and empirics argue that exposing frequent related concepts during sampling improves rare composition. **[verified]** — [arXiv](https://arxiv.org/abs/2410.22376)
- **Whoops!** (Bitton-Guetta et al., ICCV 2023, arXiv 2303.07274) contains 500 commonsense-defying image/caption pairs and is reused as a T2I prompt set (e.g., by SAP above). **[prior-knowledge for origin; usage verified via SAP]** — [arXiv](https://arxiv.org/abs/2303.07274)
- **T2I-CompBench** (Huang et al., NeurIPS 2023, arXiv 2307.06350) has non-spatial relation and attribute-binding categories, including some "counter-intuitive" combinations. **[prior-knowledge]** — [arXiv](https://arxiv.org/abs/2307.06350)
- **R2I-Bench** (arXiv 2505.23493) has 3,068 reasoning-driven prompts in 7 categories, including commonsense, logical, causal, and **concept mixing**. **[verified via snippet]** — [arXiv](https://arxiv.org/html/2505.23493)

### Inferences
- Role reversal is a *prior conflict* that produces **substitution** (the frequent version is drawn) rather than omission. For an omission/hallucination study it is a different failure class: every entity is present, but the relation is hallucinated.
- R2F and ReBind share one principle: route generation through *frequent* intermediates the model can render, then steer toward the rare target. SAP makes this stage-aware. This is the dominant 2024–2026 mitigation paradigm for prior conflicts.

### Gaps
- No public per-model CF-World numbers were retrieved, so the full paper should be checked.
- I found no study of counterfactual or role-reversed relations embedded **inside long, multi-entity prompts**. All the benchmarks above use short, single-relation prompts.
- I found no "Winoground-T2I"-style minimal-pair benchmark with large coverage for generation (RoleBench has only 10 relation pairs, and the authors call it small). I found no reliable source for a "Winoground-T2I" benchmark under that name.

---

## Q3. Negation ("no", "without", "not"), a direct source of unrequested object insertion

### Takeaway
Negation is still badly handled in 2026. Saying "without X" in the positive prompt often *adds* X, because the negated noun still gets attention and because X is strongly associated with the scene ("a bakery without bread"). The best recent benchmark (NegT2IBench, Oct 2026) finds about 25% overall correctness and drops of up to 43.8 points when a statement is negated. Negation-aware text encoders barely help, while embedding-level debiasing and negative guidance help more.

### Cited Findings
- **NegT2IBench: "When Negation Changes the Picture. A Polarity Benchmark for Text-to-Image Models"** (Elfatairy, Bravo, Bader, Akata et al.; arXiv 2610.03084, Oct 2026) **[NEW 2026]**. It has 4,800 prompts over 79 COCO classes. It negates *attributes and relations* ("a non-red cup", "is not above") and requires every named object to appear, so a model cannot "pass by omission." Scoring is detector/segmentation/geometry-based. **[verified]** — [arXiv HTML](https://arxiv.org/html/2610.03084)
  - 9 of 11 models score lower on a single negated statement than on a positive one, with drops up to **43.8 points**. Examples (positive 1:0 vs. negated 0:1): FLUX.2-klein-9B **72.8 vs 29.0**; Z-Image-Turbo 63.7 vs 31.4; Qwen-Image 67.5 vs 35.2; FLUX.2-dev 76.3 vs 42.6; SD3.5-L 57.9 vs 47.5; Janus-Pro-7B 55.2 vs 54.8 (the most polarity-robust). **[verified]**
  - Color has the largest penalty: **85.0% asserted vs 31.3% denied**. No model exceeds **8.4%** at load 4 (four statements). **41.5%** of failed statements are "negated violations," meaning the image shows the forbidden attribute or relation. About 158k of 211k generations are wrong (≈25% correct). **[verified]**
  - **NegationCLIP** (negation-aware encoder) raises SD1.5 from only 14.5% to 14.9%. **[verified]**
  - Prior negation sets it cites: NEG-TTOI (2,000 wanted/unwanted object pairs, GPT-4o judge), LMD negation (100 "without" prompts), NegGenBench (200 pairs), and NegationCLIP (107 prompts). **[verified]**
- **"Negation without Confusion" (DeRO)** (Park, Singh, Min, Yu, Cho; arXiv 2610.25645, Oct 2026) **[NEW 2026]**. For "a bakery without bread" the bread appears because "bias-aligned semantics dominate." *Strongly coupled* pairs (bakery/bread) resist removal, while weakly coupled ones (room/bread) are removed reliably. The method is training-free SVD decomposition of text embeddings into bias vs. semantic subspaces. Mean TIFA on 480 object-negation prompts: FLUX.1 **0.661 → 0.842** (vs. NAG 0.826); SD3 0.512 → 0.610; SD1.5 0.460 → 0.595; SDXL 0.476 → 0.518. **[verified]** — [arXiv HTML](https://arxiv.org/html/2610.25645)
- **"Know 'No' Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP"** (Park et al., ICCV 2025) builds NegationCLIP and tests it on the LMD negation benchmark with SD/SDXL. Its supplement notes that LMD "only checks object removal, not negated actions or attributes." **[verified via snippet]** — [ICCV supp](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Park_Know_No_Better_ICCV_2025_supplemental.pdf)
- **TNG-CLIP** (arXiv 2505.18434) introduces the **NEG-TTOI** negation T2I benchmark and criticizes the earlier 107-prompt set as using only "naive negation patterns." **[verified via snippet]** — [arXiv](https://arxiv.org/pdf/2505.18434)
- **LMD (LLM-grounded Diffusion)** (Lian et al., arXiv 2305.13655) has a negation template, "A realistic photo of a scene without [object]". An LLM layout planner converts negation into *absence* of a box, so it fixes negation by moving it out of the text encoder. **[verified via snippet]** — [arXiv](https://arxiv.org/pdf/2305.13655)
- **"Understanding the Impact of Negative Prompts: When and How Do They Take Effect?"** (Ban et al., **ECCV 2024**, arXiv 2406.02965). Negative prompts show a **delayed effect**: they act only after the positive prompt has already rendered the concept. They delete by **neutralization**, cancelling the concept against the positive prompt in latent space. **[verified via search snippet]** — [arXiv](https://arxiv.org/abs/2406.02965); [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12484.pdf)
- **"Dynamic Negative Guidance of Diffusion Models"** (ICLR 2025) is a follow-up on adaptive, state-dependent negative guidance. (I could not retrieve the abstract; OpenReview was blocked.) — [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/5647763d4245b23e6a1cb0a8947b38c9-Paper-Conference.pdf)
- **VSF (Value Sign Flip)** (arXiv 2508.10931) handles negative guidance for *few-step* models. It notes that "even when a text encoder understands negation, the diffusion model might still fail to avoid certain items because of their strong association." **[verified via snippet]** — [arXiv](https://arxiv.org/pdf/2508.10931)
- Other negative-guidance methods: **Dynamic VLM-Guided Negative Prompting (VL-DNP)** (arXiv 2510.26052; also a baseline in SAP), **Adaptive Negative Sampling** (Desai et al., ICCV 2025), and **Orthogonal Negative Guidance** with DCS-Bench on FLUX (arXiv 2605.29390). — [VL-DNP](https://arxiv.org/pdf/2510.26052); [ICCV'25](https://openaccess.thecvf.com/content/ICCV2025/papers/Desai_Guiding_Diffusion_Models_with_Adaptive_Negative_Sampling_Without_External_Resources_ICCV_2025_paper.pdf); [ONG](https://arxiv.org/html/2605.29390v1)
- **T2I-CoReBench** (ICLR 2026) reports the *opposite* pattern for its NEG (exclusion) items: models score higher on NEG than POS checklist items in several dimensions. A summary notes it is "easier to exclude an item than to correctly place and attribute it." This conflicts with NegT2IBench, likely because CoReBench allows success by omission while NegT2IBench requires objects to be present. — [project](https://t2i-corebench.github.io/); [arXiv](https://arxiv.org/abs/2509.03516)
- **NegBench** ("Vision-Language Models Do Not Understand Negation", Alhamoud et al., CVPR 2025, arXiv 2501.09425) is a **CLIP/VLM retrieval** benchmark, not a generation one. It is relevant because CLIP-style encoders and VLM judges inherit the same negation blindness. **[prior-knowledge]** — [arXiv](https://arxiv.org/abs/2501.09425). Related: NegT2IBench reports that VLM judges lose up to **18.8 κ points** when negations are phrased as negations, so *evaluators* are also negation-blind. **[verified]**

### Inferences
- For an omission/hallucination agenda, negation is the cleanest controlled source of **unrequested object insertion**: the negated noun is a strong attention target and often co-occurs with the scene. DeRO's strong- vs. weak-coupling result suggests negation failure scales with co-occurrence strength, a measurable variable.
- Evaluation must block "success by omission" (NegT2IBench's design) and must not rely on negation-blind VLM judges. Otherwise results flip, as the CoReBench vs. NegT2IBench disagreement shows.

### Gaps
- I found no benchmark of negation **inside long prompts** (e.g., a 200-token scene description with 2–3 "no X" clauses), apart from NegT2IBench's load ≤4 statements.
- CoCoNo was not found in this session's sources, so its details are unverified.
- Few per-model negation results exist for closed models (GPT-Image, Nano Banana, Seedream). UniGenBench++ (arXiv 2510.18701) has a negation dimension, but no negation-specific numbers were retrieved. — [arXiv](https://arxiv.org/html/2510.18701)

---

## Q4. Commonsense, physical, and reasoning conflicts; concept association / co-occurrence leakage

### Takeaway
When the literal prompt implies a state that contradicts the visual default (a lightbulb *without* electricity is unlit), models render the default. Commonsense-T2I found DALL-E 3 at only 48.9%, and GPT prompt enrichment did not close the gap. 2025–2026 reasoning benchmarks (WISE, T2I-ReasonBench, R2I-Bench, T2I-CoReBench) show reasoning lags composition, and that "think-before-draw" helps only partly. Separately, *semantic leakage* (one entity's attributes or identity bleeding into another, or rendered text hijacking the scene) is a measurable source of unrequested content.

### Cited Findings
- **Commonsense-T2I Challenge** (Fu, He, Lu, Wang, Roth; **COLM 2024**; arXiv 2406.07546). Adversarial prompt pairs share action words but differ minimally and need opposite visuals (e.g., a lightbulb with vs. without electricity). Scores: **DALL-E 3 48.92%**, SDXL 24.92%. **GPT-enriched prompts do not solve it.** **[verified]** — [arXiv](https://arxiv.org/abs/2406.07546)
- **WISE** (arXiv 2503.07265): 1,000 prompts over 25 subdomains (cultural commonsense, spatio-temporal, natural science). T2I-R1 reaches **0.54** WISE overall, +4% over FLUX.1-dev. — [T2I-R1 arXiv](https://arxiv.org/pdf/2505.00703)
- **T2I-ReasonBench** (arXiv 2508.17472; ACL Findings 2026): 800 prompts covering Idiom, Textual Image Design, Entity Reasoning, and Scientific Reasoning. Its conclusion: models "primarily operate through direct semantic-to-visual mapping rather than incorporating structured reasoning." — [arXiv](https://arxiv.org/pdf/2508.17472); [ACL Anthology](https://aclanthology.org/2026.findings-acl.433/)
- **T2I-CoReBench — "Easier Painting Than Thinking"** (Kling/Kuaishou; **ICLR 2026**; arXiv 2509.03516): 1,080 prompts, about 13.5k checklist questions, 12 dimensions, 38 models. "Composition capability still remains limited in complex high-density scenarios, while the reasoning capability lags even further behind." BAGEL with "Think" improves reasoning from **34.1 to 41.9** but still trails diffusion models. — [arXiv](https://arxiv.org/html/2509.03516v3); [GitHub](https://github.com/KwaiVGI/T2I-CoReBench)
- **R2I-Bench** (arXiv 2505.23493): 3,068 instances in 7 reasoning categories (commonsense, logical, causal, concept mixing, …). — [arXiv](https://arxiv.org/html/2505.23493)
- **PhyBench** (arXiv 2406.11802) evaluates physical commonsense in T2I. **[prior-knowledge]** — [arXiv](https://arxiv.org/abs/2406.11802)
- **"Text-to-Image Diffusion Models Cannot Count, and Prompt Refinement Cannot Help"** (arXiv 2503.06884) shows LLM rewriting has limits for hard numeric constraints. — [arXiv](https://arxiv.org/pdf/2503.06884)
- **Semantic leakage**:
  - **T2LSC-Bench** (arXiv 2609.02255) **[NEW 2026]** defines a *Semantic Leakage Rate (SLR)*, and its stress tests show "substantial semantic leakage." — [arXiv](https://arxiv.org/html/2609.02255)
  - **"When Scene Text Hijacks the Scene"** (arXiv 2610.11286) **[NEW 2026]**: words that are rendered as text in the image leak into scene semantics. Leakage is high for entity presence, attribute binding, object interaction, scene setting, and global appearance, and lower or model-dependent for counting and spatial relations. — [arXiv](https://arxiv.org/html/2610.11286)
  - **Token Merging (ToMe)** for semantic binding (NeurIPS 2024, arXiv 2411.07132), **SynGen / Linguistic Binding in Diffusion Models** (NeurIPS 2023, arXiv 2306.08877), and **Bounded Attention "Be Yourself"** (ECCV 2024, arXiv 2403.16990) all address attribute and identity leakage between subjects. **[ToMe verified via snippet; SynGen and Bounded Attention prior-knowledge]** — [ToMe](https://arxiv.org/pdf/2411.07132); [SynGen](https://arxiv.org/abs/2306.08877); [Bounded Attention](https://arxiv.org/abs/2403.16990)
  - **DetailMaster** (arXiv 2505.16915; ICLR 2026 *withdrawn* submission on OpenReview): 4,116 long prompts averaging about 285 tokens. State-of-the-art models reach only about **50%** on character attributes, locations, and relationships. The degradation "is not merely due to character omission": text encoders miss syntactic dependencies, and diffusion models suffer **attribute leakage**. — [arXiv](https://arxiv.org/html/2505.16915v3); [OpenReview](https://openreview.net/forum?id=AZ0VQouDmR)
- **IMPLICIT-Bench** (arXiv 2609.24228): 5,493 neutral prompts across 11 categories measure what priors fill in when the prompt is silent, which is the *underspecified* end of the spectrum. — [arXiv](https://arxiv.org/abs/2609.24228v1)

### Inferences
- Commonsense benchmarks and counterfactual benchmarks (CF-World) are two sides of one axis. Commonsense-T2I asks whether the model *applies* a prior the prompt implies, and CF-World asks whether it can *override* one. A model that over-relies on co-occurrence passes neither consistently.
- Leakage plus negation plus co-occurrence priors together explain much of the **unrequested object addition** in long prompts. Omission and conflict (Q1) explain much of the **dropping**.

### Gaps
- Leakage benchmarks measure leakage *between named entities*. I found little on **co-occurrence hallucination**, where the model adds an *unnamed* object strongly associated with a named one (e.g., "a bat" plus a ball).
- No source located in this session handles the ambiguous "a bat" (animal vs. baseball) case as a benchmark. TIED (Q6) covers ambiguity generally.

---

## Q5. Mechanistic explanations

### Takeaway
There are three converging explanations. (1) **Text-encoder entanglement**: CLIP and T5 embeddings carry co-occurrence bias and ignore negation and word order. (2) **Early-step cross-attention competition**: the dominant token wins attention at step 1 in low-resolution blocks and fixes the layout. (3) **Training-data frequency and visual homogeneity**: the frequent or homogeneous concept wins. The evidence suggests fixes at only one level (a better encoder alone, or head ablation) are insufficient.

### Cited Findings
- Entanglement in embeddings: DeRO separates "bias" and "semantic" subspaces with SVD, and negation failure tracks coupling strength. — [arXiv 2610.25645](https://arxiv.org/html/2610.25645)
- Encoder-only fixes are weak: NegationCLIP moves SD1.5 from 14.5% to 14.9% on NegT2IBench. — [arXiv 2610.03084](https://arxiv.org/html/2610.03084). VSF reports diffusion models still fail when the encoder "understands" negation. — [arXiv 2508.10931](https://arxiv.org/pdf/2508.10931)
- Early-step attention: in DominanceBench, the dominant token holds peak attention in 84.33% of prompts (SD1.4) at the first step, and the effect is distributed across heads. — [arXiv 2512.20666](https://arxiv.org/html/2512.20666). In SAP, coarse layout and color decisions are fixed in the first ~2 steps. — [arXiv 2506.01929](https://arxiv.org/html/2506.01929)
- Negative prompts act late (delayed effect) and by neutralization. — [ECCV 2024 / arXiv 2406.02965](https://arxiv.org/abs/2406.02965)
- Data: role collapse comes from distributional asymmetry, not capacity (RoleBench). — [arXiv 2503.10037](https://arxiv.org/html/2503.10037). Dominance comes from low visual diversity of concept training images. — [arXiv 2512.20666](https://arxiv.org/html/2512.20666). CF-World cites coupling of world knowledge with appearance. — [arXiv 2606.24548](https://arxiv.org/abs/2606.24548)
- CLIP-style encoders do not distinguish word order (the horse/astronaut discussion), but this point is from informal community discussion. — [HN thread](https://news.ycombinator.com/item?id=42730538) (low-quality source; treat as anecdotal)

### Inferences
- Because conflicts are decided at step 1, interventions limited to late steps (standard negative prompts) or to the text embedding alone tend to underperform. Interventions that **schedule prompts across timesteps** (R2F, SAP) or **retrain on counter-prior data** (ReBind) do better. This is consistent across the verified numbers above.

### Gaps
- Mechanistic studies are almost all on **UNet SD1.x/2.x**. DiT and MMDiT models (SD3, FLUX) and unified autoregressive MLLMs (Janus, BAGEL, GPT-Image) have little mechanistic analysis of conflict resolution. Janus-Pro's polarity robustness in NegT2IBench is an unexplained hint.

---

## Q6. Methods that help, and how models *should* behave when requirements cannot all be satisfied

### Takeaway
The mitigation families are: (a) LLM reasoning or rewriting before generation (R2F, SAP, LMD, RePrompt, T2I-R1, GoT, unified-MLLM "think" modes); (b) attention, embedding, and guidance interventions (Attend-and-Excite, ToMe, DeRO, negative guidance); (c) counter-prior data and fine-tuning (ReBind). Interactive clarification for ambiguous prompts exists (TIED, ACL 2023; proactive belief-graph agents, 2024). However, **no work was found on prioritization or an instruction hierarchy for genuinely unsatisfiable image prompts**.

### Cited Findings
- **LLM planning or reasoning before drawing**
  - R2F (ICLR 2025): up to +28.1 %p on RareBench. — [arXiv](https://arxiv.org/abs/2410.22376)
  - SAP (2025/26): +8–18 points over base on Whoops-Hard and ContraBench, beating R2F. — [arXiv HTML](https://arxiv.org/html/2506.01929)
  - LMD (2023): an LLM converts the prompt to a layout, which handles negation as a missing box. — [arXiv](https://arxiv.org/pdf/2305.13655)
  - **T2I-R1** (arXiv 2505.00703): semantic- and token-level CoT trained with RL; WISE 0.54, +4% over FLUX.1-dev. — [arXiv](https://arxiv.org/pdf/2505.00703)
  - **RePrompt** (arXiv 2505.17540): RL-trained reasoning reprompter, evaluated on GenEval and T2I-CompBench. — [arXiv](https://arxiv.org/html/2505.17540)
  - **GoT: Generation Chain-of-Thought** (arXiv 2503.10639). **[prior-knowledge]** — [arXiv](https://arxiv.org/abs/2503.10639)
  - **Think-Then-Generate** (arXiv 2601.10332) and **StruVis** (arXiv 2603.06032) are 2026 reasoning-aware T2I methods. One 2026 paper reports SFT+GRPO models reaching WISE 0.79 and T2I-ReasonBench 68.3 (attribution to a specific paper is unclear from the snippet). — [Think-Then-Generate](https://arxiv.org/pdf/2601.10332); [StruVis](https://arxiv.org/pdf/2603.06032)
  - Limits: GPT-enriched prompts do not solve Commonsense-T2I. — [arXiv 2406.07546](https://arxiv.org/abs/2406.07546). Prompt refinement does not fix counting. — [arXiv 2503.06884](https://arxiv.org/pdf/2503.06884). BAGEL-Think gains +7.8 on CoReBench reasoning but still lags. — [CoReBench](https://arxiv.org/html/2509.03516v3)
- **Attention, embedding, and guidance**: Attend-and-Excite (neglect), ToMe and SynGen (binding), Bounded Attention (leakage), DeRO (negation, FLUX TIFA 0.661 → 0.842), negative guidance (DNG, VSF, VL-DNP, ONG). Citations are in Q1–Q4. **Composable Diffusion** (Liu et al., ECCV 2022, arXiv 2206.01714) supports AND/NOT concept composition, a precursor to multi-condition and negative guidance. **[prior-knowledge]** — [arXiv](https://arxiv.org/abs/2206.01714)
- **Data and fine-tuning**: ReBind (role bias β 23.1 → 7.9). — [arXiv](https://arxiv.org/html/2503.10037)
- **Clarification and ambiguity**
  - **TIED — "Resolving Ambiguities in Text-to-Image Generative Models"** (Mehrabi et al., **ACL 2023**): a benchmark of ambiguous prompts plus a framework that generates clarifying questions and folds the answer back into the prompt. — [ACL Anthology](https://aclanthology.org/2023.acl-long.804.pdf)
  - **"Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty"** (arXiv 2412.06771): a belief graph over uncertain prompt elements; the agent asks questions ranked by uncertainty × importance. — [arXiv](https://arxiv.org/pdf/2412.06771)
  - "Enhancing Intent Understanding for Ambiguous Prompts through Human-Machine Co-Adaptation" covers multi-turn disambiguation. — [ResearchGate](https://www.researchgate.net/publication/388422783_Enhancing_Intent_Understanding_for_Ambiguous_Prompts_through_Human-Machine_Co-Adaptation)
  - Anecdotal: an AIMultiple test with impossible configurations (e.g., analog and digital clocks showing conflicting times) found no model fully complied. This is non-academic and low rigor. — [AIMultiple](https://research.aimultiple.com/text-to-image-generators/)

### Inferences
- Reasoning-before-generation is the most active 2025–2026 direction. It helps most where the conflict can be *resolved by re-expression* (SAP proxies, R2F frequent surrogates) and least where it requires overriding deep priors (Commonsense-T2I, CF-World implicit level, counting).
- Clarification work targets **ambiguity** (underspecification), not **contradiction** (overspecification). The two need different behavior: ask, versus detect-then-prioritize or report.

### Gaps (main research opportunities, especially for long prompts with omissions and conflicts)
1. **No benchmark of genuinely unsatisfiable or contradictory long prompts.** Existing sets are short (ContraBench has 40 prompts; RoleBench has 10 relation pairs) or test prior conflicts, not logical ones. DetailMaster and LongBench-T2I test long prompts but not conflict. I found no benchmark tailored to conflicting or over-specified prompts.
2. **No metric for conflict-resolution behavior.** Nothing records whether the model *chose A*, *chose B*, *blended*, *dropped both*, or *hallucinated a third option*, and none ties this to position in the prompt (first vs. last mention), token distance, or prior strength.
3. **No instruction hierarchy or prioritization study for image generation.** LLM research has system/user priority and "clarify vs. answer" policies (e.g., the VQA work "Clarify or Answer", arXiv 2601.16400). No T2I analogue was found that defines which constraint *should* win (explicit over implicit, later over earlier, specific over generic) or tests models against it. — [Clarify-or-Answer](https://arxiv.org/html/2601.16400v1)
4. **Omission vs. conflict disentanglement.** DetailMaster notes that long-prompt failure is "not merely omission," and SAP shows omission is often the losing side of a conflict. No study has causally separated "dropped because of capacity or length" from "dropped because of conflict with another constraint or prior."
5. **Negation and leakage in long prompts.** NegT2IBench stops at load 4, and accuracy is already ≤8.4% there. Negation embedded in 200+ token prompts, a likely major source of unrequested object insertion, is unstudied.
6. **Mechanistic analysis on DiT, MMDiT, and unified AR models** for conflicts (most of it is on SD1.x UNet).
7. **Evaluator reliability.** VLM judges lose up to 18.8 κ on negations (NegT2IBench). VLM-judged conflict and negation benchmarks (CF-Eval, GPT-4o scoring in SAP and R2F, NEG-TTOI) may be systematically biased.
8. **Contradictory findings to reconcile.** On negation, T2I-CoReBench finds NEG easier than POS, while NegT2IBench finds it much harder. The likely cause is the success-by-omission design difference, which no paper has tested directly.
