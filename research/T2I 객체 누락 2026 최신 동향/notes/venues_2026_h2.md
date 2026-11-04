# H2 2026 Venues & In-Review Submissions on T2I Object Faithfulness (Jul–Nov 2026)

Scope note: searched as of 2026-11-04. Main source = OpenReview API v2 full-text search over the `ICLR.cc/2027/Conference` group (it returned title, abstract, keywords, and venueid = Submission / Desk_Rejected_Submission). The single-note endpoint was challenge-gated, so all abstracts were pulled through search. Other sources: arXiv abstract pages, the Paper Digest "NeurIPS 2026 Papers & Highlights" page, ACL Anthology (EMNLP 2026), and web search. The arXiv export API returned HTTP 429 (rate-limited), and the NeurIPS/ECCV virtual-site lists loaded through JavaScript and showed nothing to WebFetch.

Verification labels:
- **[V-abs]** means I read the claim in the primary abstract or listing (OpenReview, arXiv, or ACL Anthology).
- **[V-list]** means the venue is confirmed only through a listing (Paper Digest or a lab news page).
- **[U]** means unverified.

All ICLR 2027 items are anonymous submissions under review. They are not peer-reviewed, and their numbers are self-reported.

Already covered in the earlier report (by arXiv ID): 2606.24548 (CF-World), 2610.03084 (NegT2IBench), 2610.25645 (DeRO), 2606.30262 (One-and-Only), 2608.14976, and 2609.02864. Below, these appear only where a new venue status has emerged.

---

## Q1. Which NeurIPS 2026 / ECCV 2026 / EMNLP 2026 papers address omission, long/dense prompts, conflict/counterfactual/negation, extra objects, counting/binding?

### Takeaway
Accepted H2-2026 venue papers that directly target *object omission* or *extra objects* are scarce. What I could confirm is mostly compositional and binding training (ORCA, NeurIPS 2026), a counterfactual prior-resistance benchmark (CF-World, NeurIPS 2026), negation in text embeddings (DeRO, EMNLP 2026 main), and prior-override for canonical objects (One-and-Only, ECCV 2026). The omission, extra-object, and conflict work is concentrated in ICLR 2027 submissions (see Q2).

### Cited Findings
**NeurIPS 2026 (main; notifications ~Sept 2026)**
- **ORCA: Hunting Compositional Failures in Text-to-Image Diffusion.** arXiv 2610.09841; arXiv comment says "Accepted at NeurIPS 2026" [V-abs]. *Finding:* the authors argue that binding, spatial, and counting failures come from *misalignment*, not missing information: T5 keeps the compositional structure, but the denoising objective never rewards aligning it with visual structure. *Method:* a single auxiliary loss aligns the DiT latent with a low-rank target from a frozen visual encoder, with the basis guided by the T5−CLIP residual. *Numbers:* DiT-L/2 reaches FID 16.65 and GenEval 0.291 at 200K steps, beating the strongest 400K-step baseline at about half the cost, with zero inference cost. The largest gains are on binding, spatial, and multi-object prompts. *Interests:* binding, counting (secondary), and omission on multi-object prompts (indirect). — [arXiv 2610.09841](https://arxiv.org/abs/2610.09841)
- **Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning (CF-World).** arXiv 2606.24548 (already in the earlier report). **New:** it appears in Paper Digest's NeurIPS 2026 list [V-list]. Its metrics are the *Prior Resistance Rate* (PRR: whether a model overcomes entrenched real-world priors) and the *Reasoning Retention Rate* (RRR: whether a model keeps counterfactual generation without explicit visual cues). All models degrade sharply from factual to counterfactual settings. *Interest:* primary (2), prior-violating requirements. — [Paper Digest NeurIPS 2026](https://www.paperdigest.org/2026/10/neurips-2026-papers-highlights/); [HF papers 2606.24548](https://huggingface.co/papers/2606.24548)
- **Evaluating Depth and Breadth in Test-Time Scaling for Compositional Visual Generation** (Jaiswal, Prabhudesai, …, Fragkiadaki, Pathak). Listed as NeurIPS 2026 [V-list]. It systematically compares sampler-depth scaling, parallel sampling, step-by-step construction, iterative refinement, and hybrid breadth–depth for compositional T2I across the accuracy, latency, and compute trade-offs. arXiv ID not found. *Interest:* secondary (compositional; relevant to verifier-based omission fixes). — [Paper Digest NeurIPS 2026](https://www.paperdigest.org/2026/10/neurips-2026-papers-highlights/)
- **Read It Back: Pretrained MLLMs Are Zero-shot Reward Models for Text-to-Image Generation (SpectraReward).** Listed as NeurIPS 2026 [V-list]. It is a training-free reward that turns pretrained MLLMs into reward models for image-generation RL. Its object-level behaviour is unverified [U]. *Interest:* faithfulness reward. — [Paper Digest NeurIPS 2026](https://www.paperdigest.org/2026/10/neurips-2026-papers-highlights/)
- **Track name:** the NeurIPS 2026 Datasets & Benchmarks track was renamed the **Evaluations & Datasets (E&D) Track**, and the organizers explicitly welcome "analyses of why existing benchmarks fail". I found no confirmed T2I-faithfulness E&D paper. — [NeurIPS blog](https://blog.neurips.cc/2026/03/23/introducing-the-evaluations-datasets-track-at-neurips-2026/)

**ECCV 2026**
- **Intermediate Text Representation Guided Text-to-Image Generation for Enhancing One-and-Only Alignment.** arXiv 2606.30262, ECCV 2026 per its arXiv comment (already in the earlier report by ID). It targets models that default to strong visual priors for "one-and-only" objects (landmarks, celestial bodies, artworks). It reportedly evaluates on OAO-AttackBench, WHOOPS, and Gecko. *Interest:* primary (2), prior override. — [arXiv 2606.30262](https://arxiv.org/abs/2606.30262)
- **SCALE: Semantic-Calibrated Guidance Enhancement for Prompt-Faithful Diffusion.** The title appears in the ECCV 2026 schedule according to a search snippet, but I could not find any arXiv or abstract [U]. — [ECCV 2026 schedule](https://eccv.ecva.net/virtual/2026/calendar)

**EMNLP 2026**
- **Negation Without Confusion: Decomposing Bias and Recomposing Style in Text Embeddings for Negation-Aware Image Generation (DeRO).** Park, Singh, Min, Yu, Cho, EMNLP 2026 main, 2026.emnlp-main.2219 [V-abs]. It is the same work as arXiv 2610.25645 (earlier report); **new** is the venue confirmation. *Method:* training-free, text-embedding-only; auxiliary prompts build semantic and bias subspaces separated by bidirectional SVD, then style is restored. It works with both U-Net and DiT. *Interest:* primary (2), negation. — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.2219/); [SNU lab news](https://pi.snu.ac.kr/news/news/papers/2026/08/21/emnlp-accept.html)

**Other H2-2026 arXiv (no venue stated)**
- **LogicVal: A Logic-Formalized Benchmark for Constraint Satisfaction in T2I Generation.** arXiv 2610.24051, 16 Oct 2026, preprint with no venue [V-abs]. It specifies objects, attributes, relations, and constraints with quantifiers, counting, and implication. Prompts can be *unsatisfiable*, and models are told to output an empty white scene in that case. *Numbers:* 4 T2I models, 9 constraint classes. The best model scores 98% on single object-attribute constraints but 53% on complex conditional multi-object constraints, and drops from 84% (single constraint) to 20% (multiple simultaneous constraints). It **detects only 23% of unsatisfiable prompts**. The evaluator reaches ROC-AUC 0.88–0.98 against humans on CLEVR. *Interests:* primary (1) and (2): constraint load and contradiction detection. — [arXiv 2610.24051](https://arxiv.org/abs/2610.24051)
- **QC-T2I-Bench (Scalable Question-Centric T2I Evaluation).** arXiv 2608.24112, 25 Aug 2026, no venue [V-abs]. It uses atomic questions on Davidsonian Scene Graphs with hierarchy-constrained aggregation (downstream questions are skipped when a prerequisite fails). *Number:* joint completion drops from **80.7% (2 capabilities) to 37.2% (≥7 capabilities)**. *Interest:* primary (1), density. — [arXiv 2608.24112](https://arxiv.org/abs/2608.24112)
- **CoBind: Stage-Aware Compositional Binding for Training-Free T2I.** arXiv 2607.16307, 14 Jul 2026, no venue [V-abs]. It uses an entity/attribute/relation graph, sets the layout first, then applies contrastive cross-entity binding, and fades guidance in later steps. It reports gains on T2I-CompBench++ and GenEval, but the abstract gives no numbers. *Interest:* binding. — [arXiv 2607.16307](https://arxiv.org/abs/2607.16307)

### Inferences
- At accepted H2 venues, conflict and counterfactual work is framed as **prior resistance** (CF-World PRR, One-and-Only, DeRO's "bias subspace"). This matches the earlier report's framing that the prior wins. No accepted paper yet models *how* the winning requirement is chosen.
- LogicVal (23% unsatisfiable detection) and QC-T2I-Bench (80.7%→37.2%) give fresh quantitative anchors for "density and contradiction break faithfulness". Neither is peer-reviewed yet.

### Gaps
- I could not enumerate the full NeurIPS 2026 accepted list (~9,000 papers). The Paper Digest page holds about 730K characters; I grepped all of it for T2I titles, but it carries only titles and one-line highlights, so papers whose titles lack T2I keywords may be missed.
- The ECCV 2026 accepted list and the ACM MM 2026, BMVC 2026, WACV 2027, and AAAI 2027 lists could not be read (JS-rendered pages or not yet public). No confirmed omission or extra-object paper was found there.
- The arXiv API was rate-limited, so I could not run a systematic Sept–Nov 2026 arXiv sweep.

---

## Q2. Which ICLR 2027 submissions (OpenReview) touch these topics?

### Takeaway
The ICLR 2027 pool is rich and lines up closely with the user's two primary interests. Several submissions move from "is the object present?" toward **mechanisms of omission in MMDiT joint attention** (Filling the Slots, Images Talk Back), a systematic account of **unrequested extra objects** (Implicit Co-occurring Objects), and a new failure class: **over-realization**, where a mentioned-but-not-requested target gets rendered (negation, belief, figurative, occlusion). There is also a cluster of **region- and requirement-level rewards** (R3, HARP, DeltaPO, VoTEx).

### Cited Findings (all "ICLR 2027 submission", [V-abs] from the OpenReview abstract; forum link = openreview.net/forum?id=ID)

**Interest (1): omission, density, position**
- **Filling the Slots: Repeated Mentions Get Their Attributes Before Their Instances in MMDiT T2I Models** (ohAQYCjV15). Writing a count as repeated mentions ("a bicycle, a bicycle and a bicycle") instead of a numeral **lowers counting accuracy by 44.3 pts (SD3.5-M) and 59.2 pts (FLUX.1-dev)**. Mentions lose their instance in two ways: an *omitted* mention's image-to-text attention decays on its own, while *fused* mentions keep attention but never get a separate region. Restoring early attention causally restores color but not the instance. The proposed EMAR fix raises counting by **7–15 pts** across six prompt structures and lowers omission and fusion. *Interests:* omission (1), counting, binding. — [OpenReview](https://openreview.net/forum?id=ohAQYCjV15)
- **Images Talk Back: Guiding MMDiTs by Suppressing Image-to-Text Feedback** (l1eFFyZcZg). In MMDiT joint attention, image-to-text (I2T) feedback **preserves whatever objects and attributes are already in the evolving image, including errors**. Transplanting I2T messages can rescue failures or break successes. The fix, Semantic Feedback Guidance (SFG), runs during the first 20% of steps at about 1.2× NFE. **FLUX GenEval rises from 0.6661 to 0.7479** with CFG+SFG. *Interest:* omission mechanism (1); early-commitment lock-in. — [OpenReview](https://openreview.net/forum?id=l1eFFyZcZg)
- **Position-Sensitive Concept Acquisition in T2I Models** (oyZo8XCw18). With identical supervision, fine-tuning gives markedly different control depending on whether the target field sits at the *beginning or end* of the prompt. The gap is linked to information propagation in the frozen *causal* text encoder (Qwen-Image case). Repeating the condition as a prefix helps. *Interest:* position effects (emerging); relevant to (1). — [OpenReview](https://openreview.net/forum?id=oyZo8XCw18)
- **Text-to-Image Models Need Less from Text Encoders Than You Think** (jxs69zwtxQ; also arXiv 2606.03715). A "bag of position-tagged words" embedding with no cross-token context matches full embeddings in quality and fidelity. DiT T2I models mainly use word meaning and word order; the image model does the compositional decoding. *Interest:* mechanism for (1) and (2): it suggests omission and conflict resolution happen in the denoiser, not the encoder. — [OpenReview](https://openreview.net/forum?id=jxs69zwtxQ); [arXiv PDF](https://arxiv.org/pdf/2606.03715)
- **Graph2Scene: Relation-Aware T2I with Graph-Guided Diffusion** (0hTSqCRDG3). A training-free scene-graph attention control method. **ConceptMix relationship accuracy goes from 0.59 to 0.84** over the strongest FLUX.1 regional-control baseline, and it reports an advantage on **LongBench-T2I** long, complex prompts. *Interest:* (1) dense prompts. — [OpenReview](https://openreview.net/forum?id=0hTSqCRDG3)
- **CroGen: Probing Spatial Control at Extreme Target Densities** (o2wbM8RqOQ). Names the phenomenon "**spatial constraint overload**": spatial correspondence degrades as the number of controlled instances grows. Includes a 71K crowd dataset. *Interest:* (1), density (layout-conditioned). — [OpenReview](https://openreview.net/forum?id=o2wbM8RqOQ)
- **JIG: JSON-to-Image Generation** (nau33qzRKX). Releases JIG-14M (13.7M images with structured annotations). Both JSON and prose renderings of structured annotations beat conventional dense captions, and captions now run "tens to thousands of tokens". *Interest:* (1), long and structured conditioning (data side). — [OpenReview](https://openreview.net/forum?id=nau33qzRKX)

**Interest (2): conflict, negation, counterfactual, prior**
- **When Chekhov's Gun Fires Anyway: Over-Realization in T2I Models** (erJ21xq8Tv). Defines **over-realization**: a target the prompt *mentions* appears in a way the prompt does not call for. It covers four situations: negation, inside a belief or imagination, the vehicle of a figurative comparison, and behind an occluding viewpoint. Benchmark: OverReal-Gen (698 prompts) and OverReal-Det (2,841 human-labeled images). **All 8 systems over-realize**, CLIPScore *rewards* the failure, and prompt expansion only partly helps. Inside the generator, the backbone attends to a mentioned target almost as much as to a requested one, and the "cue words" that should hold it back get *less* attention than ordinary words. *Interests:* (2) and extra objects (precision). — [OpenReview](https://openreview.net/forum?id=erJ21xq8Tv)
- **Presence Is Not Faithfulness: Figurative Vehicle Intrusion in T2I (VISTA)** (GzFM1pJ3PN). A figurative "vehicle" gets rendered as a literal object. Includes a multilingual benchmark, a role-aware V-Score QA metric, and VISTA-Guard (partial mitigation). *Interest:* (2), plus the critique of presence-based metrics. — [OpenReview](https://openreview.net/forum?id=GzFM1pJ3PN)
- **NegT2IBench** (nFPNhCF9Y3; arXiv 2610.03084, already in the earlier report). **New:** it is now an ICLR 2027 submission, and the abstract adds numbers: 4,800 prompts; polarity of 0–2 positive × 0–2 negated statements; **11 models, 211,200 images; 9 of 11 score lower on one negated statement than on one positive one**. The loss is largest for color and near zero for proximity. **41.5% of failed statements render exactly what was forbidden.** Its detector-based scoring agrees with humans as well as VLM judges up to 30× larger. — [OpenReview](https://openreview.net/forum?id=nFPNhCF9Y3)
- **Training-Free Concept Negation in T2I Synthesis** (G7rvcLSHoN). A context-aware null-space projection on cross-attention *value* vectors plus "ghost token augmentation", with keywords pointing to visual autoregressive (VAR) models. It ships a new negation benchmark of mixed positive and negated prompts and claims results competitive with Nano Banana Flash. *Interest:* (2). — [OpenReview](https://openreview.net/forum?id=G7rvcLSHoN)
- **Structured Counterfactual Contrast for Compositional T2I: A Pilot Study of Prompt Negation and Candidate Reranking** (W55MnCnNxH). Builds minimally perturbed counterfactuals (attribute swap, relation inversion, role swap) and uses a frozen-VLM margin. Finding: **textual negation of counterfactuals ("not X") makes the margin *worse***; explicit structured comparison works better than textual negation. Self-described as a pilot. *Interest:* (2). — [OpenReview](https://openreview.net/forum?id=W55MnCnNxH)
- **A2I-Bench: Measuring the Value of Clarification in Interactive T2I** (d83IpNMEqC). 527 ambiguous prompts, each with two intents and "visual contracts". Revealing intent **improves contract satisfaction by 0.158 when the user's intent conflicts with the generator's default interpretation, 0.092 more than when it is aligned**. *Interest:* (2), generator *default priors* vs. user requirements. — [OpenReview](https://openreview.net/forum?id=d83IpNMEqC)
- **SUBTEXT: Can Image Generators Infer What Captions Leave Implicit?** (4bCzOcjS2a). 140 minimal image pairs differing in one implied outcome (e.g., "melted"). Seven frozen generators used as diffusion classifiers are near chance on implicit captions, while humans and a frontier LLM are ≥93%. *Interest:* (2)-adjacent, implicit requirements. — [OpenReview](https://openreview.net/forum?id=4bCzOcjS2a)
- **BridgePrompt** (HiHorGs0RW). Addresses the "implicit visual meaning gap" by optimizing the prompter instruction with VLM feedback. **SD3.5-M on T2I-ReasonBench goes from 46.8 to 74.6.** — [OpenReview](https://openreview.net/forum?id=HiHorGs0RW)
- **Draw What LLMs Intend: Prompt Alignment Is Not Enough** (muVITDZ4bj). Introduces "Latent Intent Transfer": whether the generator realizes the upstream LLM's resolved commitments for underspecified slots. Includes DaLI-Bench and the DaLI method. — [OpenReview](https://openreview.net/forum?id=muVITDZ4bj)
- **Semantic Underspecification as a Safety Risk** (s5GIYSs8JN). Omitted context lets the prior fill in unsafe scenes. 100-prompt benchmark; RSC repair reaches a 92.0% risk mitigation rate. — [OpenReview](https://openreview.net/forum?id=s5GIYSs8JN)
- **From Knowledge Gaps to Better Generation (CGTR / KnowGapBench)** (5xrP2XqcYt). Audits 370M annotation records and builds 3,286 underrepresented concepts, scoring concept identity separately from compositional compliance. Knowledge score rises from 15.3 to 22.5. — [OpenReview](https://openreview.net/forum?id=5xrP2XqcYt)

**Secondary: unrequested extra objects (precision)**
- **Unspoken, Controllable, Unavoidable: Implicit Co-occurring Objects (ICO) is a Fundamental Property in T2I** (IET7dtNjT8). Across 7 models (2022–2026), 56,000 images, and 80 categories, **34–50% of images of a common object contain ≥1 unprompted object**, with co-occurrence patterns shared across models. Training-data co-occurrence is the dominant cause, the text encoder secondary, and sampling parameters set the magnitude. Isolation phrasing plus embedding steering cuts leakage to **0.19% at 94.2% fidelity**. A theoretical trade-off shows ICO cannot be suppressed freely without a fidelity loss. Detectors trained on images with ICO show up to an **18.8% recall gap**. *Interest:* extra objects (precision) — the most direct H2 paper on this. — [OpenReview](https://openreview.net/forum?id=IET7dtNjT8)

**Secondary: binding and counting**
- **Binding Crystallization (CrystalBind)** (VVnpx7b69l). Attribute assignment becomes *resistant to reversal before spatial refinement finishes*. "**Object presence and attribute ownership are distinct control problems.**" On T2I-CompBench SD1.5: **+21.3 pts** over unguided and **+3.4** over CONFORM, with 28% less wall-clock time. — [OpenReview](https://openreview.net/forum?id=VVnpx7b69l)
- **Tracing Compositional Instability in Diffusion Transformers** (Hzo1Iu93j0). In SD3.5 and FLUX.1-dev, the requested binding is supported *early* but becomes unstable across depth and time; test-time steering reduces failures. — [OpenReview](https://openreview.net/forum?id=Hzo1Iu93j0)
- **ATHENA: Test-Time Steering for Count Fidelity** (92abCkPqp8). A cardinality-contrastive steering signal plus intermediate count feedback. Gains reported, but the abstract gives no numbers. — [OpenReview](https://openreview.net/forum?id=92abCkPqp8)
- **Hierarchical Cross-modal Representation Alignment (HIP)** (1G6idnFdJf). GenEval macro accuracy improves by **+14.48% (SD3.5-M) and +8.24% (FLUX.1-dev)** relative. — [OpenReview](https://openreview.net/forum?id=1G6idnFdJf)
- **Parachuting Parrot** (gMuxnZwhXb). A 2.5M-image controlled dataset for compositional-generalization splits; trains an LDM from scratch. — [OpenReview](https://openreview.net/forum?id=gMuxnZwhXb)
- **ESG: Early Spatial Guidance** (s3WKa2Iw7u). Spatial relations in rectified-flow transformers handled through early joint-attention readout. — [OpenReview](https://openreview.net/forum?id=s3WKa2Iw7u)
- **Lightweight Velocity Correction for Few-Step T2I** (f3Apyvd5R5). A <20M-parameter corrector on frozen Z-Image-Turbo gives **+17.60 pp NetWin** at 26.9 ms extra latency. — [OpenReview](https://openreview.net/forum?id=f3Apyvd5R5)

**Faithfulness rewards and requirement-level credit**
- **Seeds Teach Each Other Through Prompt Tokens (VoTEx)** (Hpbag6AlrY). Different seeds satisfy different requirements, and whole-image rewards throw away the correct parts. VoTEx uses prompt-token states as cross-seed correspondences and distills them. It proves that covering K requirements takes O(log(K/δ)) samples, versus exponentially many for whole-image selection. **+2.2 to 8.9 pts on six compositional benchmarks** (SD3.5-M, FLUX.1-dev, Qwen-Image-2512), with no reward model. — [OpenReview](https://openreview.net/forum?id=Hpbag6AlrY)
- **R3: Rubric-Guided Region Reward** (vMTXdfzwrc). Treats rewarding as target–evidence matching per entity. **GranuBench preference accuracy: 90.8% vs 80.7%** for direct Gemini. Releases R3Traj-100K and a distilled 27B VLM. — [OpenReview](https://openreview.net/forum?id=vMTXdfzwrc)
- **HARP: Hierarchy-Aware Regional Preferences** (9ZkQGWswsP; **desk-rejected**). Requirement-specific regional credit. SD3.5-M **GenEval 0.70→0.84, DPG 83.82→88.61**. — [OpenReview](https://openreview.net/forum?id=9ZkQGWswsP)
- **DeltaPO: Counterfactually Verified Preference Optimization** (nIlXkefSiD; **desk-rejected**). Verifies each preference by repairing or injecting the violation in the image. SDXL **GenEval 0.53→0.64, DPG 73.38→79.97**. — [OpenReview](https://openreview.net/forum?id=nIlXkefSiD)
- **AutoRubric-T2I** (gOhgX9lVFA). Automatically selected and weighted rubrics; uses 256 preference pairs and beats fine-tuned reward models on MMRB2. — [OpenReview](https://openreview.net/forum?id=gOhgX9lVFA)
- **WeGenBench** (qV0OSZop2o). 4,000 bilingual prompts with capability tags that include negation and attribute binding. — [OpenReview](https://openreview.net/forum?id=qV0OSZop2o)
- **From Prompt Alignment to Intent Recovery (PAIR)** (w7kd3eGI6m). Observer-side evaluation: PAIR-based selection gets 9–17% higher recovery than CLIP/VQAScore. — [OpenReview](https://openreview.net/forum?id=w7kd3eGI6m)

### Inferences
- **Mechanism shift toward MMDiT joint attention:** Filling the Slots, Images Talk Back, Tracing Instability, and Binding Crystallization all find failures *locked in early* through image→text feedback or early commitment. This turns the earlier report's "first steps decide" claim into concrete MMDiT-specific causal levers: suppressing I2T feedback, restoring a mention's early attention, and choosing the guidance window.
- **The precision axis is getting a theory:** ICO (34–50% extra objects, with a provable fidelity trade-off) and OverReal (mentioned ≠ requested) together form the strongest H2-2026 basis for an extra-object or precision metric. OverReal also directly bridges interest (2): negation, belief, and figurative contexts are "conflicts" between mention and request.
- **Textual negation is weak as a control channel:** NegT2IBench (41.5% render the forbidden thing), Structured Counterfactual Contrast (negation lowers the margin), and OverReal (cue words get under-attended) converge on this.
- **Requirement-level credit assignment** (VoTEx, R3, HARP, DeltaPO) is the H2 analogue of "verifier-based optimization" for long prompts. None of them explicitly evaluates very long (>200-token) prompts, except Graph2Scene on LongBench-T2I.

### Gaps
- OpenReview search is term-based with a cap of 25 results per term. Some relevant submissions (e.g., on "object neglect" in titles phrased differently) may be missed; "object neglect" returned no T2I hits.
- I did not open the PDFs, so per-model omission rates and long-prompt (>200-token) results inside these submissions are unverified.
- arXiv IDs for most ICLR 2027 submissions are unknown (anonymous). Only jxs69zwtxQ (2606.03715) and NegT2IBench (2610.03084) were linked.

---

## Q3. Emerging directions in H2 2026 vs earlier

### Takeaway
Compared with H1 2026 (attention and verifier fixes, long-prompt decomposition, prior-wins diagnosis), H2 2026 adds four things:
- (a) **precision and over-generation as named failure classes**: ICO, over-realization, figurative intrusion;
- (b) **position, order, and phrasing effects**: position-sensitive acquisition, bag-of-position-tagged-words, and repeated-mention vs numeral;
- (c) **formal constraint and unsatisfiability evaluation**: LogicVal, polarity-controlled NegT2IBench, and the QC-T2I hierarchy;
- (d) **requirement-level reward and credit assignment**.

"Instruction priority" for image generation (which requirement should win) remains essentially unstudied.

### Cited Findings
- Precision and extra objects: 34–50% of single-object prompts yield ≥1 unprompted object; suppression trades off against fidelity. — [ICO, OpenReview IET7dtNjT8](https://openreview.net/forum?id=IET7dtNjT8)
- A mentioned-but-not-requested target is rendered by all 8 tested systems, and CLIPScore rewards it. — [OverReal, OpenReview erJ21xq8Tv](https://openreview.net/forum?id=erJ21xq8Tv)
- Position and phrasing effects:
  - Beginning vs end placement changes what fine-tuning teaches. — [oyZo8XCw18](https://openreview.net/forum?id=oyZo8XCw18)
  - Word order plus word meaning suffice for DiT conditioning. — [jxs69zwtxQ](https://openreview.net/forum?id=jxs69zwtxQ)
  - Repeated mentions lose 44–59 counting points relative to numerals. — [ohAQYCjV15](https://openreview.net/forum?id=ohAQYCjV15)
- Unsatisfiable or contradictory constraints are detected only 23% of the time by the best model. — [LogicVal arXiv 2610.24051](https://arxiv.org/abs/2610.24051)
- Generator default vs user intent conflict: clarification helps most when the intent conflicts with the generator default (+0.158). — [A2I-Bench d83IpNMEqC](https://openreview.net/forum?id=d83IpNMEqC)
- Prior resistance is a named metric (PRR) at NeurIPS 2026. — [Paper Digest NeurIPS 2026](https://www.paperdigest.org/2026/10/neurips-2026-papers-highlights/)
- Density effect: joint completion falls from 80.7% to 37.2% as capabilities per component rise from 2 to ≥7. — [QC-T2I-Bench arXiv 2608.24112](https://arxiv.org/abs/2608.24112). "Spatial constraint overload" with instance count. — [CroGen o2wbM8RqOQ](https://openreview.net/forum?id=o2wbM8RqOQ)
- Region- and requirement-level rewards: VoTEx proves O(log K) sample coverage vs exponential for whole-image selection. — [Hpbag6AlrY](https://openreview.net/forum?id=Hpbag6AlrY). R3 region reward reaches 90.8% preference accuracy. — [vMTXdfzwrc](https://openreview.net/forum?id=vMTXdfzwrc)

### Inferences
- **Open intersection (relevant to the user's direction):** no H2 paper jointly measures *omission (recall)* and *extra objects (precision)* as prompt density grows, nor how conflicting requirements trade off under density. ICO studies single-object prompts; OverReal studies mention vs request; LogicVal studies unsatisfiability but has only 4 models and CLEVR-style validation. A density × conflict × precision/recall benchmark with MMDiT mechanism analysis (I2T feedback, early mention attention) appears unoccupied.
- **Position effects in prompts** for *object omission* specifically (are later objects dropped more often?) remain untested. The H2 papers show position and order matter (fine-tuning acquisition, word-order reliance), but none measures omission rate against an object's position in a long prompt. This is a concrete, low-cost study opportunity.
- **"Instruction priority" for image generation**, analogous to LLM instruction hierarchy, has no dedicated paper. The closest are A2I-Bench (default vs intent), CF-World PRR, and LogicVal's "empty scene if contradictory" protocol.

### Gaps
- No reliable evidence was found on ACM MM 2026, BMVC 2026, WACV 2027, or AAAI 2027 T2I-faithfulness papers; their lists were not accessible or searchable within budget.
- I found no blog or X discussion that synthesizes these trends; the inferences above are mine, drawn from the abstracts.
