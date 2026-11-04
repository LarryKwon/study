# Mechanisms of Object Omission / Binding Leakage / Merging / Counting Errors in T2I Diffusion & Flow Models, and Training-Free (Inference-Time) Fixes (2023–2026)

Note on sourcing: claims marked (verified) were checked this session against the arXiv page, venue page, or a fetched abstract. Unmarked well-known papers are cited to their arXiv abs pages from prior knowledge of their content. Numbers are given only where verified. Secondary review sites (themoonlight, liner) are flagged where used.

---

## Q1. What mechanistic analyses exist for why T2I models drop, merge, or mis-bind objects?

### Takeaway
Failures have several separable causes. (1) The text encoder leaks information: CLIP is causal, so the first-mentioned object leaks into later tokens. CLIP behaves like a bag of words, and EOT/padding tokens carry entangled prompt-wide semantics. (2) In UNet models, cross-attention maps of different subjects overlap or one is weak during the early steps, when the layout gets fixed. (3) The initial noise and seed largely decide layout and count. (4) In MM-DiT (SD3, FLUX), joint attention is imbalanced (far more image tokens than text tokens), CLIP and T5 streams can disagree, and similar subjects become ambiguous. By 2026, linear-probe work showed omission can be read out of the concept-token keys in middle layers at intermediate timesteps.

### Cited Findings

**Text-encoder-side mechanisms**
- **Causal-encoder "information bias" (first-object dominance).** In CLIP's causal text encoder, information from earlier tokens bleeds into later token embeddings. For "a cat and a dog", the cat's information mixes into the dog's embedding, so two-cat images become more likely. The paper proposes training-free Text Embedding Balance Optimization (TEBOpt), which reports a 125.42% improvement in information balance for SD. It also proposes an automatic information-loss metric with 81% agreement with humans. — *A Cat Is A Cat (Not A Dog!): Unraveling Information Mix-ups in Text-to-Image Encoders through Causal Analysis and Embedding Optimization*, Chen et al., NeurIPS 2024, arXiv 2410.00321 (verified) — [arXiv](https://arxiv.org/abs/2410.00321); [NeurIPS page](https://neurips.cc/virtual/2024/poster/94705)
- **CLIP as a bag of words.** VLMs such as CLIP behave like bags of words and largely ignore word order and attribute-object relations, because contrastive retrieval training rarely needs compositional order. — *When and Why Vision-Language Models Behave like Bags-of-Words*, Yuksekgonul et al., ICLR 2023, arXiv 2210.01936 — [arXiv](https://arxiv.org/abs/2210.01936)
- A 2025 follow-up argues CLIP is bag-of-words cross-modally but not uni-modally: binding information exists in the text embedding but is lost in cross-modal alignment. — *CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally*, Koishigarina et al., arXiv 2502.03566 — [arXiv](https://arxiv.org/abs/2502.03566)
- **EOT / padding embeddings carry entangled semantics.** The [EOT] embedding is trained without word-level supervision and "tries to remember all important words", including adjectives and nouns. Padding embeddings inherit this, which the authors tie to color leakage and "object sticking". Magnet (training-free) applies positive and negative binding vectors to disentangle per-object embeddings. — *Magnet: We Never Know How Text-to-Image Diffusion Models Work, Until We Learn How Vision-Language Models Function*, Zhuang et al., NeurIPS 2024, arXiv 2409.19967 (verified) — [arXiv](https://arxiv.org/html/2409.19967)
- **Padding tokens.** Whether padding is used depends on the encoder's training objective. With frozen encoders (e.g., the CLIP and T5 used by SD and FLUX), images from prompt tokens only score nearly the same CLIP score as full prompts, so padding is largely ignored by the encoder. Padding can still matter inside the diffusion model, for example in MM-DiT where padding tokens join attention. When the encoder is trained jointly (LDM, LLaMA-UNet), padding gains semantic weight. — *Padding Tone: A Mechanistic Analysis of Padding Tokens in T2I Models*, Toker et al., NAACL 2025, arXiv 2501.06751 (verified) — [arXiv](https://arxiv.org/pdf/2501.06751); [ACL Anthology](https://aclanthology.org/2025.naacl-long.389.pdf)
- **Information flow across text tokens.** T5 is bidirectional, so representative tokens of a lexical item can sit at various positions. CLIP and Gemma are unidirectional and place them at the final token, and CLIP concentrates most semantics in its final/[CLS]-like token. — *Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models*, arXiv 2504.01137 (verified via snippet) — [arXiv](https://arxiv.org/pdf/2504.01137)
- **Diffusion Lens.** Decoding intermediate text-encoder layers through the diffusion model shows complex multi-object prompts are composed gradually and only in later encoder layers. Rare concepts also emerge later than common ones. — *Diffusion Lens: Interpreting Text Encoders in Text-to-Image Pipelines*, Toker et al., ACL 2024, arXiv 2403.05846 — [arXiv](https://arxiv.org/abs/2403.05846)
- **Correlated tokens cause omission.** When prompt tokens are statistically correlated, SD drops simple but important objects. Attention analysis shows "biased text representations mislead the visual denoising process". Same-meaning prompts with slightly different token order give different attention scores. The fix is a fine-tuning method, not training-free: it decorrelates self-attention maps inside the text module. — *Decoding Correlation-Induced Misalignment in the Stable Diffusion Workflow for Text-to-Image Generation*, Tong et al., ICCV 2025 (verified) — [ICCV page](https://iccv.thecvf.com/virtual/2025/poster/1705)
- **Spatial relations in DiTs.** A 2026 circuit analysis studies how DiTs generate spatial relations under T5 vs. random embeddings. It is adjacent to this topic: relations, not omission. — *Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers*, arXiv 2601.06338 — [arXiv](https://arxiv.org/pdf/2601.06338)

**Cross-attention / early-step layout mechanisms (UNet era)**
- **Catastrophic neglect.** Attend-and-Excite named "catastrophic neglect": one or more subjects in the prompt are not generated. It links neglect to weak maximum cross-attention for that subject token in early denoising steps. — *Attend-and-Excite*, Chefer et al., SIGGRAPH 2023 / ACM TOG, arXiv 2301.13826 — [project](https://yuval-alaluf.github.io/Attend-and-Excite/) (verified); [arXiv](https://arxiv.org/abs/2301.13826)
- **Layout is fixed early.** Cross-attention maps fix spatial layout in early timesteps. This is the basis for both attention-based editing and early-step-only guidance. — *Prompt-to-Prompt*, Hertz et al., ICLR 2023, arXiv 2208.01626 — [arXiv](https://arxiv.org/abs/2208.01626)
- **Attention overlap, not weak intensity, drives entity missing.** The analysis attributes entity missing mainly to large overlap between different entities' attention maps, rather than low attention intensity or excessive spread. Reducing overlap raised entity-generation success by more than 17% for 3-entity prompts versus about 7% for 2-entity prompts. Their example: A&E fails to generate one of "boat, cow, and cup". — *Attention Overlap Is Responsible for The Entity Missing Problem in Text-to-image Diffusion Models!*, arXiv 2410.20972 (verified) — [arXiv](https://arxiv.org/pdf/2410.20972)
- **Cross-attention information decays.** A-STAR observes that cross-attention maps of later or secondary tokens lose information across denoising steps, and that subject maps overlap. It proposes a segregation loss and a retention loss. — *A-STAR: Test-time Attention Segregation and Retention for Text-to-image Synthesis*, Agarwal et al., ICCV 2023, arXiv 2306.14544 — [arXiv](https://arxiv.org/abs/2306.14544)
- **Self-attention leakage, not only cross-attention.** Leakage also happens through self-attention: features of similar subjects attend to each other, which causes "semantic leakage" and subject blending even with correct cross-attention. — *Be Yourself: Bounded Attention for Multi-Subject T2I Generation*, Dahary et al., ECCV 2024, arXiv 2403.16990 — [arXiv](https://arxiv.org/abs/2403.16990)
- **Typology of binding failures.** SynGen sorts improper binding into semantic leak within the prompt, semantic leak out of the prompt, and attribute neglect. ToMe reuses this typology (verified via ToMe). — *Linguistic Binding in Diffusion Models* (SynGen), Rassin et al., NeurIPS 2023, arXiv 2306.08877 — [arXiv](https://arxiv.org/abs/2306.08877); [ToMe](https://arxiv.org/pdf/2411.07132)

**Initial noise / seed mechanisms**
- **Some initial-noise regions are reliably good.** Initial noise falls into "valid" and "invalid" regions for a given prompt. InitNO defines a cross-attention response score and a self-attention conflict score, then optimizes the initial noise toward the valid region. — *InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization*, Guo et al., CVPR 2024, arXiv 2404.04650 — [arXiv](https://arxiv.org/abs/2404.04650)
- **Reliable seeds.** Some noise patterns are more reliable for compositional prompts, especially counts and positions. Using mined reliable seeds instead of uniform random ones gives about +6% absolute accuracy for both SD and PixArt-α. Fine-tuning on self-generated images from those seeds gives a relative numerical-composition gain of 29.3% (SD) and 19.5% (PixArt-α). Later arXiv versions retitled the paper *All Seeds Are Not Equal*. — Li, Le, Xu, Salzmann, ICLR 2025, arXiv 2411.18810 (verified) — [ICLR](https://www.iclr.cc/virtual/2025/poster/30962); [arXiv](https://arxiv.org/html/2411.18810v5)
- **Seeds are highly identifiable.** A classifier separates images by seed with very high accuracy, and seeds systematically affect composition and style. "Golden" seeds can be mined. — *Good Seed Makes a Good Crop: Discovering Secret Seeds in Text-to-Image Diffusion Models*, Xu et al., WACV 2025, arXiv 2405.14828 — [arXiv](https://arxiv.org/abs/2405.14828)
- **Same seed, similar layout.** With a fixed seed, small changes to the count in the prompt produce similar layouts. CountGen relies on this to build training pairs (verified via project page). — [Make It Count project](https://make-it-count-paper.github.io)

**MM-DiT (SD3 / SD3.5 / FLUX) mechanisms**
- **Token-count imbalance.** MM-DiT joint attention suppresses cross-modal attention because visual tokens vastly outnumber text tokens, and attention weighting is not timestep-aware. TACA scales the visual→text logits (Q_vis K_txtᵀ) by a temperature greater than 1, with a timestep-dependent schedule. It has no learnable parameters but pairs with LoRA to suppress artifacts. It improves object appearance, attribute binding, and spatial relations on FLUX and SD3.5 (T2I-CompBench). — *Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers* (TACA), Lv et al., ICCV 2025, arXiv 2506.07986 (verified) — [arXiv](https://arxiv.org/pdf/2506.07986); [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Lv_Rethinking_Cross-Modal_Interaction_in_Multimodal_Diffusion_Transformers_ICCV_2025_paper.html)
- **Similar subjects in SD3 are neglected or merged.** For prompts with several semantically or visually similar subjects, SD3's success rate falls below 20%. The paper names three ambiguities. Inter-block ambiguity. Text-encoder ambiguity: cross-attention activations from the CLIP and T5 subject tokens are inconsistent. Semantic ambiguity. The training-free fix combines a block-alignment loss, a text-encoder alignment loss, an overlap loss, online overlap detection, and a "back-to-start" re-sampling strategy. It transfers to SD3.5 and FLUX. — *Enhancing MMDiT-Based Text-to-Image Models for Similar Subject Generation*, Wei, Chen, Zhou, Pan, arXiv 2411.18301 (verified) — [arXiv](https://arxiv.org/pdf/2411.18301)
- **Omission is linearly decodable (2026).** Linear probes on concept-token key vectors in FLUX.1-dev and SD3.5-Medium separate present from omitted concepts. Probe accuracy peaks at intermediate timesteps. At early steps, labels agree with the final image only 0.409 of the time, and late steps carry little signal. Accuracy also peaks in middle layers: early layers are near chance because text tokens have not yet absorbed image information. The best FLUX head reaches 91.0%, and 300 of 1,368 heads exceed 80%. In SD3.5, T5 tokens carry the omission signal and CLIP tokens do not. — *Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers* (OSI), Baek, Lew, Shin, Lee, Yoon, arXiv 2605.14270 (v3 31 Jul 2026, ICML template; acceptance not stated) (verified) — [arXiv](https://arxiv.org/html/2605.14270)
- **Block roles in MMDiT.** Semantics are set in earlier blocks and fine details in later ones. Spatial relations and color are handled early, while amount (count) is influenced by later blocks. Removing text conditioning from early blocks hurts most. Scaling text hidden states by 2× in selected blocks raises SD3.5 T2I-CompBench++ from 56.92% to 63.00% and GenEval from 66.42% to 71.63%. On FLUX, GenEval goes from 0.6538 to 0.6826, and on Qwen-Image from 0.8551 to 0.8777. Count gains are limited, which the authors attribute to MMDiT's difficulty with quantity. — *Unraveling MMDiT Blocks: Training-free Analysis and Enhancement of Text-conditioned Diffusion*, Li et al., arXiv 2601.02211 (Jan 2026) (verified) — [arXiv](https://arxiv.org/html/2601.02211v1)
- **Joint-attention outputs localize concepts.** In MM-DiT, attention-output projections give sharper concept saliency maps than raw cross-attention, which is a diagnostic tool for where each concept lives. — *ConceptAttention*, Helbling et al., ICML 2025, arXiv 2502.04320 — [arXiv](https://arxiv.org/abs/2502.04320)
- **Vital layers and RoPE roles in FLUX.** Stable Flow (arXiv 2411.14430) and FreeFlux (arXiv 2503.16153) identify layer-specific roles in FLUX's MM-DiT. These underpin block-selective interventions. — [FreeFlux](https://arxiv.org/html/2503.16153); [Stable Flow](https://arxiv.org/abs/2411.14430)

### Inferences
- The causal-encoder leak, EOT/padding entanglement, and bag-of-words effects mean some binding and omission errors are already present in the conditioning before denoising starts. Denoising-time attention guidance can only partly repair them, which is why embedding-level methods (TEBOpt, Magnet, ToMe) appeared in 2024.
- In MM-DiT, the 2025–2026 consensus is that the failure point moved from "weak cross-attention to one token" to "text tokens being swamped in joint attention" plus "dual-encoder (CLIP vs. T5) disagreement". Interventions therefore moved from optimizing latents to scaling or steering attention logits, keys, and hidden states.
- A recurring theme across UNet and DiT is a timestep window: layout and presence are decided early-to-mid. The OSI paper's finding that the signal peaks at intermediate steps refines the UNet-era "first few steps" heuristic.

### Gaps
- No direct mechanistic study found of whether T5-XXL (bidirectional) removes the first-object dominance that the CLIP causal encoder shows. OSI's SD3.5 result (only T5 tokens carry the signal) is suggestive but not a test of this.
- No quantitative study found linking padding/EOT tokens specifically to MM-DiT object omission. Padding Tone covers frozen-encoder text-side usage, not the joint-attention effects.

---

## Q2. What is the lineage of training-free methods, and which target omission, binding, or counting?

### Takeaway
The line runs from: (a) 2023 UNet cross-attention loss-guided latent updates (A&E, Divide & Bind, A-STAR, SynGen, Structured Diffusion), to (b) 2024 methods acting on initial noise, contrastive attention, logic losses, and text embeddings (InitNO, CONFORM, Predicated Diffusion, Attention Regulation, EBAMA, ToMe, Magnet, TEBOpt, Bounded Attention), to (c) 2024–2026 MM-DiT-native methods that rescale or steer joint attention or text hidden states (EnMMDiT, TACA, PLADIS, block-wise text scaling, OSI probe steering). In parallel, noise-search and reward-optimization methods (ReNO, inference-time scaling) became a general fix for all failure types.

### Cited Findings

**2023: UNet cross-attention guidance**
| Method | Venue / arXiv | Target | Mechanism |
|---|---|---|---|
| Structured Diffusion Guidance | ICLR 2023, 2212.05032 | Binding (also omission) | Parses the prompt into noun phrases, encodes each separately, and mixes them into cross-attention values. Training-free. — [arXiv](https://arxiv.org/pdf/2212.05032) |
| Attend-and-Excite | SIGGRAPH 2023, 2301.13826 | **Omission** (catastrophic neglect) | During early steps, updates the latent so the most neglected subject token reaches a high max cross-attention. — [project](https://yuval-alaluf.github.io/Attend-and-Excite/) |
| A-STAR | ICCV 2023, 2306.14544 | Omission and overlap | Segregation loss (less overlap between subject maps) and retention loss (keeps maps across steps). — [arXiv](https://arxiv.org/abs/2306.14544) |
| SynGen (Linguistic Binding) | NeurIPS 2023, 2306.08877 | **Binding** | Dependency parse, then a KL-based loss that pulls each modifier's attention map toward its noun's map and pushes it away from other nouns' maps. — [arXiv](https://arxiv.org/abs/2306.08877) |
| Divide & Bind | BMVC 2023, 2307.10864 | Omission and binding | Total-variation "attendance" loss for presence plus a JSD binding loss. — [arXiv](https://arxiv.org/abs/2307.10864) |
| Detector Guidance | 2023, 2306.02236 | Merging / mixing | Uses a detector on intermediate predictions to separate subjects. — [arXiv](https://arxiv.org/abs/2306.02236) |

**2024: noise, contrastive, and logic losses; embedding-level fixes**
| Method | Venue / arXiv | Target | Mechanism |
|---|---|---|---|
| Predicated Diffusion | CVPR 2024, 2311.16117 | Omission, binding, possession | Expresses prompt relations as predicate logic and turns them into a differentiable fuzzy-logic loss on attention. — [arXiv](https://arxiv.org/abs/2311.16117) |
| CONFORM | CVPR 2024, 2312.06059 | Omission and binding together | InfoNCE-style contrastive loss on attention maps: an object and its attributes are positives, other objects are negatives. — [arXiv](https://arxiv.org/abs/2312.06059) |
| Attention Regulation | 2024, 2403.06381 | Omission and binding | Constrained optimization of cross-attention at inference. Efficient because it edits attention, not the latent. — [arXiv](https://arxiv.org/abs/2403.06381) |
| Bounded Attention | ECCV 2024, 2403.16990 | Merging / leakage | Masks cross- and self-attention between subjects. — [arXiv](https://arxiv.org/abs/2403.16990) |
| EBAMA | 2024, 2404.07389 | Binding and omission | Object-conditioned energy-based attention-map alignment. — [arXiv](https://arxiv.org/abs/2404.07389) |
| InitNO | CVPR 2024, 2404.04650 | **Omission** (via noise) | Optimizes the initial noise using cross-attention response and self-attention conflict scores. — [arXiv](https://arxiv.org/abs/2404.04650) |
| Magnet | NeurIPS 2024, 2409.19967 | **Binding** (EOT leakage) | Adds positive and negative binding vectors to object embeddings. Training-free. — [arXiv](https://arxiv.org/html/2409.19967) (verified) |
| TEBOpt | NeurIPS 2024, 2410.00321 | **Omission and merging** from information bias | Optimizes text embeddings to balance object information. — [arXiv](https://arxiv.org/abs/2410.00321) (verified) |
| ToMe (Token Merging) | NeurIPS 2024, 2411.07132 | **Binding** | Merges an object's tokens and its attribute tokens into one composite token, then refines it in early steps with entropy and semantic-binding losses. — [arXiv](https://arxiv.org/pdf/2411.07132) (verified) |
| Self-Cross Diffusion Guidance | 2024, 2411.18936 | Subject mixing | Penalizes overlap between self-attention and cross-attention of different subjects. — [arXiv](https://arxiv.org/abs/2411.18936) |
| Attention-overlap reduction | 2024, 2410.20972 | **Omission** | Reduces overlap between entity attention maps: +17% entity success for 3 entities, +7% for 2. — [arXiv](https://arxiv.org/pdf/2410.20972) (verified) |

**Noise search and reward optimization (architecture-agnostic)**
- **ReNO.** Optimizes the initial noise against differentiable reward models for one-step models. It reports large GenEval gains, which cover counting, two-object, and color-attribute items. — *ReNO: Enhancing One-step T2I Models through Reward-based Noise Optimization*, Eyring et al., NeurIPS 2024, arXiv 2406.04312 — [arXiv](https://arxiv.org/abs/2406.04312)
- **Inference-time search over noise.** Spending more compute on searching noise candidates with verifiers improves FLUX and SiT outputs, including on GenEval. This is the 2025 generalization of "seed selection". — *Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps*, Ma et al., arXiv 2501.09732 — [arXiv](https://arxiv.org/abs/2501.09732)
- **Golden Noise.** Learns a noise prompt that maps random noise to better noise for alignment. This one uses a small trained network. — Zhou et al., arXiv 2411.09502 — [arXiv](https://arxiv.org/abs/2411.09502)

**2024–2026: MM-DiT / DiT-native training-free methods**
| Method | Venue / arXiv | Target | Mechanism / result |
|---|---|---|---|
| EnMMDiT (similar subjects) | arXiv 2411.18301 | **Omission and merging** of similar subjects | Block-alignment, text-encoder-alignment, and overlap losses, plus overlap detection with back-to-start re-sampling. Works on SD3, SD3.5, and FLUX. — [arXiv](https://arxiv.org/pdf/2411.18301) (verified) |
| Regional Prompting for FLUX | arXiv 2411.02395 | Composition with user-given regions | Regional attention masks in FLUX. Requires a layout. — [arXiv](https://arxiv.org/html/2411.02395v1) (verified) |
| TACA | ICCV 2025, 2506.07986 | Omission, binding, spatial | Temperature scaling of the visual→text logits plus a timestep schedule. The core is parameter-free, but the paper recommends LoRA to remove artifacts. — [arXiv](https://arxiv.org/pdf/2506.07986) (verified) |
| PLADIS | ICCV 2025 (Kim & Sim) | General alignment | Extrapolates sparse vs. dense attention. Used as a baseline in OSI. — cited in [OSI](https://arxiv.org/html/2605.14270) |
| Block-wise text hidden-state scaling | arXiv 2601.02211 (Jan 2026) | Binding (color, shape, texture), spatial | SD3.5 T2I-CompBench++ 56.92%→63.00%; GenEval 66.42%→71.63%. Count gains are limited. — [arXiv](https://arxiv.org/html/2601.02211v1) (verified) |
| OSI (Omission Signal Intervention) | arXiv 2605.14270 (2026) | **Omission** (also binding) | Adds a mean-shift "omission direction" to concept-token keys in the top-K heads during the first 15 of 30 steps (FLUX: K=300, α=5; SD3.5: K=100, α=7.5). The probes need labeled data, but the intervention itself is training-free. — [arXiv](https://arxiv.org/html/2605.14270) (verified) |

- **OSI headline numbers (verified).** GenEval multi-object accuracy, average over 2–6 objects:
  - FLUX: base 0.47, TACA 0.53, PLADIS 0.53, OSI 0.61.
  - SD3.5: base 0.55, TACA 0.58, OSI 0.64.

  On 100 FLUX failure cases, applying OSI to the omitted tokens raised accuracy from 0.00 to 0.70. T2I-CompBench attribute binding on FLUX (color/shape/texture) went from 0.79/0.50/0.64 to 0.80/0.58/0.70. — [arXiv](https://arxiv.org/html/2605.14270)
- **Agentic self-correction.** Marmot is a 2025 multi-agent, object-level self-correction approach that detects and fixes object errors after generation. It is a different paradigm (VLM in the loop), noted here for completeness. — *Marmot*, arXiv 2504.20054 — [arXiv](https://arxiv.org/pdf/2504.20054)
- **Training-based contrast.** TokenCompose (CVPR 2024, arXiv 2312.03626) adds segmentation-grounded token losses during fine-tuning. CoMat and GLIGEN are also training-based and are cited as contrasts in OSI. — [TokenCompose](https://arxiv.org/abs/2312.03626); [OSI related work](https://arxiv.org/html/2605.14270)

### Inferences
- **What targets what:**
  - Omission-first: A&E, A-STAR, InitNO, attention-overlap reduction, TEBOpt, EnMMDiT, OSI.
  - Binding-first: Structured Diffusion, SynGen, Magnet, ToMe.
  - Both jointly: Divide & Bind, CONFORM, Predicated Diffusion, Attention Regulation, EBAMA, TACA.
  - Merging / leakage-first: Bounded Attention, Self-Cross guidance, Detector Guidance.
  - Counting: see Q4.
- **Where the methods act** has moved over time:
  - 2023: the latent at step t, via gradients of attention losses.
  - 2024: the initial noise (InitNO, ReNO) or the text embeddings (ToMe, Magnet, TEBOpt).
  - 2025–2026: attention logits, keys, or hidden states inside MM-DiT (TACA, block scaling, OSI). These are cheaper (no backprop through the UNet) and fit FLUX's guidance-distilled, flow-matching sampling, where latent-gradient methods tuned for SD1.x DDIM do not transfer directly.

### Gaps
- No systematic head-to-head benchmark was found that re-implements the UNet-era methods (A&E, SynGen, CONFORM, ToMe) on FLUX or SD3 under one protocol. OSI only compares to TACA and PLADIS.
- PLADIS and Seg4Diff details (arXiv IDs, numbers) were not verified this session.

---

## Q3. How do these methods scale with the number of objects, and what are the known failure modes?

### Takeaway
All families degrade sharply as the object count rises. Base FLUX goes from 0.81 accuracy at 2 objects to 0.18 at 6 (GenEval-style). The best 2026 intervention (OSI) only lifts 6-object accuracy to 0.40. UNet-era attention-loss methods already fail at 3 entities, mainly because the subjects' attention maps overlap.

### Cited Findings
- **FLUX scaling with object count (OSI paper).**
  - 2 objects: base 0.81, TACA 0.89, PLADIS 0.87, OSI 0.92.
  - 6 objects: base 0.18, TACA 0.20, PLADIS 0.20, OSI 0.40.
  - TACA and PLADIS barely help at 6 objects.

  — [arXiv 2605.14270](https://arxiv.org/html/2605.14270) (verified)
- **Small gains on T2I-CompBench non-spatial.** FLUX scores 0.3069 base, 0.3078 TACA, 0.3083 OSI, so some benchmarks barely respond. OSI's stated limitations include occasional over-generation (extra objects) and no evaluation on quantities. — [arXiv 2605.14270](https://arxiv.org/html/2605.14270)
- **A&E at 3 entities.** A&E fails on 3-entity prompts such as "boat, cow, and cup". The problem is worse at 3 entities than at 2, and overlap reduction gives a larger gain at 3 entities (+17% vs. +7%). — [arXiv 2410.20972](https://arxiv.org/pdf/2410.20972) (verified)
- **A&E with nearby maxima.** A&E only acts on the neglected object and struggles when the subjects' maximum-attention regions are close together. — *Obtaining Favorable Layouts for Multiple Object Generation*, arXiv 2405.00791 (verified via snippet) — [arXiv](https://www.arxiv.org/pdf/2405.00791)
- **Similar subjects in SD3.** SD3's success on similar-subject prompts falls below 20%, and SD3.5 still neglects or mixes such subjects by default. — [arXiv 2411.18301](https://arxiv.org/pdf/2411.18301) (verified)
- **Over-scaling text tokens.** Scaling text hidden states too much hurts quality. Gains on Qwen-Image are smaller because its baseline is stronger, and count is hardly improved. — [arXiv 2601.02211](https://arxiv.org/html/2601.02211v1) (verified)
- **TACA artifacts.** Amplified cross-modal logits shift the output distribution and cause artifacts, so the paper recommends LoRA. The method is therefore not purely plug-and-play. — [arXiv 2506.07986](https://arxiv.org/pdf/2506.07986) (verified)
- **Multi-subject personalization.** Methods degrade substantially with four or more reference subjects. This is the personalization setting, not plain T2I, but it shows the same trend. — *MOSAIC*, arXiv 2509.01977 (verified via snippet) — [arXiv](https://arxiv.org/pdf/2509.01977)

### Inferences
- The known failure modes of training-free fixes:
  - Latent gradient steps push the latent off-manifold, causing artifacts and lower quality. This is also why they are restricted to early steps.
  - Most methods depend on parsing nouns and modifiers or on user-chosen token indices.
  - Losses defined for pairs of subjects grow quadratically with the number of subjects, and their maps compete for limited image area.
  - Extra compute: iterative refinement, and back-to-start re-sampling in EnMMDiT.
  - Over-generation and duplication when presence is over-excited (OSI notes this).
  - Hyperparameters (K, α, temperature, step window) are model-specific.
- The 2→6 object collapse in strong 2024–2025 models (FLUX at 0.18 for 6 objects) suggests a capacity or representation bottleneck that attention reweighting can only partly fix. Layout-conditioned or LLM-planned pipelines usually handle 5+ objects; those are adjacent work, outside training-free attention guidance.

### Gaps
- Found no benchmark that systematically reports omission rate versus object count (2→10) for UNet-era training-free methods (A&E, SynGen, CONFORM, ToMe, InitNO) under a common protocol.
- No computational-cost comparison across methods was verified.

---

## Q4. What counting-specific methods exist, and what have they found?

### Takeaway
Counting is the least solved compositional axis. Benchmarks show T2I models rarely exceed 50% exact-count accuracy, accuracy falls as the count grows, and prompt rewriting does not help. Methods divide into:
- Counter-network guidance: Counting Guidance, Iterative Object Count Optimization, CountDiffusion, YOLO-Count.
- Attention clustering: CountCluster.
- Instance-feature detection with layout repair: Make It Count / CountGen.
- Seed mining: Reliable Seeds.

### Cited Findings
- **T2ICountBench.** Prompts ask for 1–15 objects, and 5 human raters require exact counts. Overall accuracy rarely exceeds 50% and drops as the requested count rises. Prompt refinements (decomposed counting, grid layouts, positional hints) mostly make things worse. The benchmark covers around 15 models, including Imagen-3 and DALL·E 3. — *Text-to-Image Diffusion Models Cannot Count, and Prompt Refinement Cannot Help*, Cao et al., arXiv 2503.06884 (v2 Jan 2026) (verified; some figures via secondary review) — [arXiv](https://arxiv.org/abs/2503.06884v2)
- **Counting Guidance.** Estimates the object count at each timestep with a counting network (RCC) applied to the predicted clean image. It then updates the latent based on the count gap. — Kang et al., arXiv 2306.17567 (WACV 2025) — [arXiv](https://arxiv.org/abs/2306.17567); description verified via [CountCluster](https://arxiv.org/abs/2508.10710)
- **Iterative Object Count Optimization.** Optimizes a counting token embedding with a detection-based counting loss. The learned token can be reused across counts. — Zafar et al., arXiv 2408.11721 — [arXiv](https://arxiv.org/abs/2408.11721)
- **Make It Count / CountGen.**
  - Self-attention features at a particular layer and timestep act as instance-level features. Clustering them counts instances during denoising and detects over- and under-generation.
  - A trained "ReLayout" model predicts the shape and location of a missing object from the layout of existing ones. The method is therefore not fully training-free.
  - It uses the diffusion model's own prompt- and seed-dependent layout prior, with no external layout.
  - A third-party review reports about 26% accuracy for base models versus about 54% with CountGen. This was not verified in the paper.

  — Binyamin et al., CVPR 2025, arXiv 2406.10210 (verified) — [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Binyamin_Make_It_Count_Text-to-Image_Generation_with_an_Accurate_Number_of_CVPR_2025_paper.html); [project](https://make-it-count-paper.github.io)
- **CountDiffusion.** Training-free. Generates an intermediate denoising result, counts objects in it with a counting model, and corrects the quantity by changing the object's attention map with universal guidance. — Li et al. (HIT Shenzhen), arXiv 2505.04347 (May 2025) (verified) — [arXiv](https://arxiv.org/abs/2505.04347)
- **CountCluster.** Training-free and uses no external tools. Guides the object's cross-attention map to split into as many clusters as the requested count. — Lee, Lee, Lee, arXiv 2508.10710 (Aug 2025) (verified) — [arXiv](https://arxiv.org/abs/2508.10710)
- **YOLO-Count.** A fully differentiable open-vocabulary counter. It regresses a "cardinality map" that accounts for object size and spatial spread, trained with representation alignment and hybrid strong/weak supervision. It is designed to give gradient guidance for quantity control in T2I. The counter is trained; using it for guidance is inference-time. Numbers were not read. — Zeng et al., ICCV 2025, arXiv 2508.00728 (verified abstract) — [arXiv](https://arxiv.org/abs/2508.00728)
- **Reliable seeds for counting.**
  - Reliable seeds alone: about +6% absolute numerical accuracy for SD and PixArt-α.
  - Fine-tuning on self-generated data from those seeds: relative gains of 29.3% (SD) and 19.5% (PixArt-α).
  - Example: SD2.1 numerical accuracy 43.0% vs. 37.5% for random seeds (reported via secondary review).

  — [arXiv 2411.18810](https://arxiv.org/html/2411.18810v5) (verified)
- **MM-DiT text scaling and quantity.** In MMDiT, the "amount" attribute is influenced by later blocks, and text hidden-state scaling barely improves it. The authors note MMDiT's difficulty with quantity. — [arXiv 2601.02211](https://arxiv.org/html/2601.02211v1) (verified)

### Inferences
- Counting is coupled to the initial noise: seed-dependent layouts, reliable seeds, and CountGen's same-seed layout stability all point the same way. Count errors are therefore largely fixed early, and post-hoc attention reweighting has little leverage. Methods that work on counting either use an external counter or explicitly segment instances (self-attention clustering).
- Over-generation (too many instances) and under-generation are both common. Count methods must handle both directions, unlike omission methods, which only push presence up.

### Gaps
- Quantitative results of YOLO-Count, CountCluster, and CountDiffusion versus each other on a shared benchmark were not verified this session.
- No counting-specific training-free method for FLUX or SD3 was confirmed. Most counting work above evaluates on SD1.x, SD2.x, or SDXL. Whether YOLO-Count guidance was demonstrated on FLUX was not verified.

---

## Q5. Is there literature on unrequested object addition (hallucinated extras), and what mechanism does it propose?

### Takeaway
Yes, but it is sparse and mostly framed as evaluation or as spurious correlations (contextual priors). There is no dedicated mechanistic line comparable to the omission literature. Proposed mechanisms are: (1) co-occurrence and contextual priors from training data, where the model adds or substitutes the typical context; (2) mode interpolation, which produces structural extras such as additional fingers; (3) over-excitation by omission fixes, which causes duplicates.

### Cited Findings
- **Extraneous objects as an error class.** A scene-graph QA evaluation splits T2I hallucination into four error types: attribute errors, relational errors, object omissions, and extraneous objects. Humans often disagree on whether an extra object is acceptable, which makes this type hard to score automatically. — *Evaluating Hallucination in Text-to-Image Diffusion Models with Scene-Graph based Question-Answering Agent*, arXiv 2412.05722 (verified via snippet) — [arXiv](https://arxiv.org/html/2412.05722v1)
- **Contextual priors override the prompt.** For "a butterfly is in a bee's hive", models tend to produce a butterfly on a flower, because the learned prior associates butterflies with flowers. The authors frame this as spurious correlation and propose Stage-dependent Prompt decomposition (SAP), with evaluation on Whoops!. — *Image Generation from Contextually-Contradictory Prompts*, Huberman et al., CVPR 2026, arXiv 2506.01929 (verified via snippet) — [arXiv](https://arxiv.org/html/2506.01929v1); [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huberman_Image_Generation_from_Contextually-Contradictory_Prompts_CVPR_2026_paper.pdf)
- **Co-occurrence bias.** Co-occurrence bias is described as over-reliance on imbalanced training-data patterns, with examples of factual hallucination such as the Statue of Liberty's color. The proposed fix is retrieval of factual images. — arXiv 2407.10683 — [arXiv](https://arxiv.org/pdf/2407.10683)
- **Correlated tokens and omission.** The flip side of co-occurrence: correlated tokens cause omission in SD via biased text representations (Tong et al., ICCV 2025). — [ICCV](https://iccv.thecvf.com/virtual/2025/poster/1705)
- **Mode interpolation.** Diffusion models interpolate between nearby data modes and create samples outside the training support, such as extra fingers. High variance in the late denoising trajectory flags these samples, and filtering on it removed over 95% of hallucinations in synthetic experiments. — *Understanding Hallucinations in Diffusion Models through Mode Interpolation*, Aithal et al., NeurIPS 2024 (verified) — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/f29369d192b13184b65c6d2515474d78-Paper-Conference.pdf)
- **Over-generation from omission fixes.** OSI lists occasional over-generation as a limitation. Make It Count explicitly detects over-generation, meaning too many instances. — [OSI](https://arxiv.org/html/2605.14270); [Make It Count](https://make-it-count-paper.github.io)
- **"Semantic leak out of prompt".** SynGen's binding typology includes "semantic leak out of prompt": an attribute spills onto unmentioned background objects. This is a binding-level analogue of unrequested content. — [SynGen arXiv](https://arxiv.org/abs/2306.08877); typology confirmed via [ToMe](https://arxiv.org/pdf/2411.07132)
- **Related ideas from captioning and VLMs (not diffusion).** Co-occurrence normalization reduces caption hallucination (arXiv 2110.01705). A 2025 training-bias analysis finds VLM heads favor objects that are statistically associated with the image even when absent (arXiv 2508.04567). — [2110.01705](https://arxiv.org/html/2110.01705); [2508.04567](https://arxiv.org/pdf/2508.04567)
- **Diffusion-augmented retrieval.** Extra distracting objects in diffusion-generated query images are treated as a hallucination to suppress. — arXiv 2601.20391 — [arXiv](https://arxiv.org/html/2601.20391v1)

### Inferences
- Unrequested addition in T2I seems to have two main sources: dataset co-occurrence priors (scene context gets filled in) and side effects of presence-boosting guidance (duplicates and extra instances). Because extra objects are often "acceptable" context, benchmarks such as GenEval and T2I-CompBench largely do not penalize them. That likely explains why the topic is under-studied relative to omission.

### Gaps
- No paper was found that mechanistically traces T2I object addition to specific attention or text-encoder components (for example, whether EOT or padding tokens, or pooled CLIP embeddings, inject co-occurring objects). This looks like an open research gap.
- No benchmark was found that quantifies the extra-object rate across FLUX, SD3, and SDXL.
