# Training-based and System-level Approaches to Object Faithfulness in T2I (2023 → late 2026)

> Scope: omission (object neglect), unrequested additions (object hallucination), attribute binding, counting.
> Verification legend: items marked **[verified]** were checked against a primary source or arXiv page in this research session (Nov 2026). Items marked **[memory]** are well-known papers whose arXiv IDs/venues come from prior knowledge and were not re-fetched this session; their *existence and IDs* are high-confidence, but specific numbers marked [memory] should be spot-checked before final publication.

---

## Q1. LLM planning / layout-to-image: what did it achieve, and what is the omission-vs-naturalness trade-off?

### Takeaway
2023–2024 "plan-then-render" systems (LLM → boxes/regions → layout-conditioned diffusion) were the first reliable fix for omission and counting, because every requested object gets an explicit region; their cost is rigid/unnatural composition, dependence on layout-conditioned backbones, and no handling of unrequested background content. From 2025 the explicit-box idea survived mainly *inside* reasoning models (GoT, Spatial CoT, semantic CoT in T2I-R1) rather than as standalone pipelines.

### Cited Findings
- **GLIGEN** (CVPR 2023, arXiv 2301.07093): adds gated self-attention grounding layers to frozen SD so boxes+phrases condition generation; the base layout-to-image component reused by LMD and others — [arXiv](https://arxiv.org/abs/2301.07093) [memory]
- **LayoutGPT** (NeurIPS 2023, arXiv 2305.15393): GPT-3.5/4 in-context generates CSS-style layouts from prompts, then layout-to-image (GLIGEN); targets numerical and spatial correctness — [arXiv](https://arxiv.org/abs/2305.15393) [memory]
- **LMD: LLM-grounded Diffusion** (TMLR 2024, arXiv 2305.13655): two stages — LLM produces boxes + captions + background prompt, then training-free layout-grounded diffusion; evaluated on negation, numeracy, attribute binding, spatial relations, where vanilla SD fails most — [arXiv](https://arxiv.org/abs/2305.13655) [memory]
- **VPGen** ("Visual Programming for T2I Generation and Evaluation", NeurIPS 2023, arXiv 2305.15328): fine-tuned Vicuna decomposes T2I into object/count → layout → image steps; also introduced VPEval — [arXiv](https://arxiv.org/abs/2305.15328) [memory]
- **LayoutLLM-T2I** (ACM MM 2023, arXiv 2308.05095): LLM layout planning + relation-aware layout-to-image — [arXiv](https://arxiv.org/abs/2308.05095) [memory]
- **Ranni** (CVPR 2024, arXiv 2311.17002): LLM converts prompt into a "semantic panel" (boxes, colors, keypoints) as a middleware for diffusion; supports interactive editing of the panel — [arXiv](https://arxiv.org/abs/2311.17002) [memory]
- **RPG: Recaption, Plan and Generate** (ICML 2024, arXiv 2401.11708): MLLM recaptions into subprompts, plans complementary regional layout, and runs regional diffusion; reported beating DALL-E 3 and SDXL on T2I-CompBench multi-category composition — [arXiv](https://arxiv.org/abs/2401.11708) [memory]
- **MIGC** (CVPR 2024, arXiv 2402.05408) and **MIGC++** (TPAMI, arXiv 2407.02329): divide-and-conquer multi-instance shading; per-instance attention to fix attribute leakage and missing instances; COCO-MIG benchmark — [MIGC](https://arxiv.org/abs/2402.05408), [MIGC++](https://arxiv.org/abs/2407.02329) [memory]
- **InstanceDiffusion** (CVPR 2024, arXiv 2402.03290): per-instance conditioning with boxes, masks, points, scribbles — [arXiv](https://arxiv.org/abs/2402.03290) [memory]
- **GoT: Generation Chain-of-Thought** (arXiv 2503.10639, 2025): reasoning chain that explicitly writes object descriptions *with bounding-box coordinates* before generation (MLLM + diffusion), i.e., layout planning folded into a learned reasoning trace — [arXiv](https://arxiv.org/abs/2503.10639) [memory]
- **Spatial Chain-of-Thought** (arXiv 2602.11980, 2026): bridges understanding and generation models for spatial-reasoning generation — evidence that explicit spatial planning persists in 2026 in CoT form — [arXiv](https://arxiv.org/pdf/2602.11980) [verified: title/ID only]
- 2026 critique of multi-step planners: CLVR (arXiv 2605.14876) lists failure modes of existing multi-step methods as "ungrounded planning hallucinations, monolithic post-hoc reflection, long-context instabilities, and high latency" — [arXiv](https://arxiv.org/abs/2605.14876) [verified]

### Inferences
- Layout planning directly attacks omission (each object gets a slot) and counting (N boxes), but cannot prevent the backbone from filling background with unrequested objects, and boxes tend to produce "collage-like" scenes with poor inter-object interaction — the naturalness side of the trade-off. This explains the shift to implicit/learned planning (semantic CoT) and to RL on end-to-end models.
- Planning errors propagate: if the LLM plan itself omits/hallucinates an object, the renderer faithfully reproduces the error ("ungrounded planning hallucinations").

### Gaps
- No single source found in this session that quantifies omission rate vs. naturalness (e.g., FID/aesthetics) side-by-side for LMD/RPG/MIGC; papers report T2I-CompBench/COCO-MIG success rates but naturalness comparisons are mostly qualitative or user studies. Specific numbers for these 2023–24 papers were not re-verified.

---

## Q2. Reward / preference fine-tuning (RL): gains and reward hacking

### Takeaway
RL post-training moved from DDPO/DPOK/ReFL (2023) and Diffusion-DPO (2023–24) to online GRPO for flow-matching models in 2025 (Flow-GRPO, DanceGRPO, MixGRPO, TempFlow-GRPO, Pref-GRPO, DiffusionNFT) and, by 2026, to *joint* RL over reasoning text + image (UniGRPO, Think-Then-Generate). Using GenEval-style detector rewards, Flow-GRPO pushed SD3.5-M from 0.63 → 0.95 GenEval, essentially saturating the benchmark — but this exposes reward/benchmark hacking, KL/regularizer dependence, and the fact that GenEval rewards don't penalise extra unrequested objects.

### Cited Findings
**2023–2024 foundations**
- **DDPO** (ICLR 2024, arXiv 2305.13301): denoising as multi-step MDP, policy gradient; rewards incl. compressibility, aesthetics, and VLM (LLaVA)-based prompt-image alignment — [arXiv](https://arxiv.org/abs/2305.13301) [memory]
- **DPOK** (NeurIPS 2023, arXiv 2305.16381): online RL with KL regularisation using ImageReward — [arXiv](https://arxiv.org/abs/2305.16381) [memory]
- **ImageReward / ReFL** (NeurIPS 2023, arXiv 2304.05977): human-preference reward model (137k expert comparisons) + Reward Feedback Learning (direct backprop of reward at late denoising steps) — [arXiv](https://arxiv.org/abs/2304.05977) [memory]
- **Diffusion-DPO** (CVPR 2024, arXiv 2311.12908): DPO adapted to diffusion likelihood, trained SDXL on Pick-a-Pic (~851k pairs) — [arXiv](https://arxiv.org/abs/2311.12908) [memory]
- **D3PO** (CVPR 2024, arXiv 2311.13231): DPO-style fine-tuning from human feedback without a reward model — [arXiv](https://arxiv.org/abs/2311.13231) [memory]
- **SPO: Step-aware Preference Optimization** (arXiv 2406.04314): step-wise preference with a step-aware preference model — [arXiv](https://arxiv.org/abs/2406.04314) [memory]
- Generic RL alignment did not fix counting by itself: a 2024 Berkeley tech report states SD prompted with "N objects" fails to produce the right count "even after DDPO prompt alignment fine-tuning" — [Berkeley EECS-2024-19](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-19.html) [verified via search snippet]

**2025 GRPO wave for flow models**
- **Flow-GRPO** (arXiv 2505.05470; NeurIPS 2025): ODE→SDE conversion for stochastic rollouts + denoising reduction; GenEval-as-reward raises SD3.5-Medium from **63% → 95%** GenEval ("outperforming GPT-4o"), visual text rendering 59% → 92%; claims "minimal reward hacking"; KL penalty needed — "with an appropriate KL term, the method can reach the same high reward as the KL-free version while maintaining image quality, though with a longer training time" — [arXiv](https://arxiv.org/abs/2505.05470) [verified]
- RL-tuned SD3.5-M "generates nearly perfect object counts, spatial relations, and fine-grained attributes" (Flow-GRPO) — [arXiv html v5](https://arxiv.org/html/2505.05470v5) [verified]
- **DanceGRPO** (arXiv 2505.07818): unified GRPO across diffusion and rectified flow, T2I/T2V/I2V, multiple reward models (HPS, CLIP, VideoAlign, GenEval-style) — [arXiv](https://arxiv.org/abs/2505.07818) [memory]
- **MixGRPO** (arXiv 2507.21802): mixed SDE/ODE sampling with a sliding window to cut training cost — [arXiv](https://arxiv.org/abs/2507.21802) [memory]
- **TempFlow-GRPO** (arXiv 2508.04324): temporally-aware credit assignment (trajectory branching, noise-aware weighting) because uniform per-step credit is poor — [arXiv](https://arxiv.org/abs/2508.04324) [memory]
- **Pref-GRPO** (arXiv 2508.20751): diagnoses reward hacking as "illusory advantage" — pointwise RMs give near-identical scores within a group and group normalisation inflates tiny gaps; shows UnifiedReward scores rising monotonically while images "drift toward an unnaturally dark style"; fix = pairwise win-rate rewards; also introduces **UniGenBench** (600 prompts, 5 themes, 20 subthemes) — [arXiv](https://arxiv.org/abs/2508.20751), [project](https://codegoat24.github.io/UnifiedReward/Pref-GRPO) [verified]
- **DiffusionNFT** (arXiv 2509.16117): forward-process ("negative-aware") online RL, CFG-free; reported large GenEval gains with much higher efficiency than Flow-GRPO — [arXiv](https://arxiv.org/abs/2509.16117) [memory; exact numbers not re-verified]
- Detector rewards are hard to differentiate through: a 2026 paper notes non-differentiable rewards (OCR accuracy, object-detection metrics) are "prohibitively expensive to differentiate through" — [arXiv 2603.14128](https://arxiv.org/html/2603.14128) [verified via search snippet; paper identity (Centered Reward Distillation) per snippet]
- **D2D: Detector-to-Differentiable Critic for Improved Numeracy** (arXiv 2510.19278): converts non-differentiable detectors into differentiable critics for counting — [arXiv](https://arxiv.org/pdf/2510.19278) [verified: title/ID]
- **Synthetic Curriculum Reinforces Compositional T2I Generation** (arXiv 2511.18378): RL with a synthetic compositional curriculum — [arXiv](https://arxiv.org/pdf/2511.18378) [verified: title/ID]

**2026: joint RL over thinking + image**
- **UniGRPO** (arXiv 2603.23500, CUHK + ByteDance Seed, Mar 2026): treats Prompt→Thinking→Image as one MDP; GRPO for text + FlowGRPO for image; removes CFG in training and replaces latent KL with velocity-MSE penalty. Base BAGEL (SFT). GenEval: BAGEL 0.78 → SFT 0.83 → ReFL 0.85 → FlowGRPO 0.88 → **UniGRPO 0.90**. Category breakdown (Overall/Single/Two/Counting/Colors/Position/AttrBinding): BAGEL 0.78/0.98/0.96/0.78/0.84/0.52/0.58; FlowGRPO 0.88/0.99/0.98/0.93/0.94/0.60/0.86; UniGRPO 0.90/0.99/0.99/0.91/0.91/0.73/0.86 — [arXiv](https://arxiv.org/html/2603.23500) [verified]
- UniGRPO reward-hacking observations: without KL "validation reward rises and then drops, with oversaturated textures"; latent KL produced "grid-like artifacts by step 250"; UniFPO collapsed; sparse terminal reward gives poor credit assignment — "flawed reasoning can still yield good images"; thinking did not consistently improve GenEval for base BAGEL — [arXiv](https://arxiv.org/html/2603.23500) [verified]
- **Think-Then-Generate** (arXiv 2601.10332, ICML 2026): joint RL of LLM text encoder and diffusion decoder so the model reasons about intent before drawing; motivated by diffusion models taking prompts "too literally" — [arXiv](https://arxiv.org/abs/2601.10332v1), [ICML 2026](https://icml.cc/virtual/2026/poster/64073) [verified]

### Inferences
- Note in UniGRPO's breakdown: Single/Two-object (the omission-sensitive GenEval categories) are already ~0.96–0.99 at baseline; RL gains concentrate in Counting (+0.13–0.15), Attribute binding (+0.28) and Position — i.e., by 2025–26 *simple-prompt* omission is solved and remaining headroom is binding/counting/spatial.
- Training directly on GenEval-style rewards and then reporting GenEval is reward–benchmark overlap; Flow-GRPO's 0.95 should be read alongside GenEval 2 evidence (Q6) that the same capabilities collapse at higher atomicity.
- Reward hacking modes documented in primary sources are mostly *quality/style* collapse (dark style, oversaturation, grid artifacts) and diversity loss; I found no primary source quantifying "adds extra objects" as a hacking mode.

### Gaps
- No primary-source evidence found for RL models learning to *add extra objects* to satisfy detector/VQA rewards (plausible, since GenEval's two-object/colour tasks check presence of requested objects rather than absence of others, but not documented here).
- Exact DanceGRPO/MixGRPO/TempFlow-GRPO/DiffusionNFT numbers not re-verified this session.

---

## Q3. Data-centric approaches (recaptioning, compositional curation, hard negatives, self-generated loops)

### Takeaway
The single most influential data intervention is dense synthetic recaptioning (DALL-E 3), now universal in frontier models; on top of it, 2024–25 work added composition-aware preference data from model galleries (IterComp), contrastive hard-negative image pairs (EvoGen/ConPair), fine-grained atomic-question feedback (CompAlign), and self-generated verified-reasoning data engines (CLVR, 2026).

### Cited Findings
- **DALL-E 3: "Improving Image Generation with Better Captions"** (OpenAI, 2023): trains on highly descriptive synthetic captions from a bespoke captioner, showing prompt-following improves with descriptive captions — [PDF](https://cdn.openai.com/papers/dall-e-3.pdf) [memory]
- **IterComp** (ICLR 2025, arXiv 2410.07171): "iterative feedback learning framework … by aggregating composition-aware model preferences from multiple diffusion models" (model gallery → composition reward models → iterative ReFL); evaluated on T2I-CompBench; not a contrastive hard-negative method — [arXiv](https://arxiv.org/pdf/2410.07171), [ICLR](https://iclr.cc/virtual/2025/poster/30983) [verified]
- **EvoGen — "Progressive Compositionality in T2I Generative Models"** (arXiv 2410.16719): "multi-stage curriculum for contrastive learning of diffusion models to learn from hard negative images" using a dataset of contrastive image pairs (ConPair) — the closest match to "contrastive training for binding" — [arXiv](https://arxiv.org/abs/2410.16719v1) [verified]
- **CompAlign** (arXiv 2505.11178): benchmark of 900 complex multi-subject prompts combining numerical and 3D-spatial relations with attribute bindings; feedback via **CompQuest**, which decomposes prompts into atomic questions; used for alignment fine-tuning — [arXiv](https://arxiv.org/abs/2505.11178) [verified via search summary]
- **TokenCompose** (CVPR 2024, arXiv 2312.03626): token-level grounding loss using segmentation masks to align each noun token's cross-attention with its object region, targeting multi-category instance composition (objects missing) — [arXiv](https://arxiv.org/abs/2312.03626) [memory]
- **Make It Count / CountGen** (CVPR 2025, arXiv 2406.10210): counting via identifying object instances in features and layout correction — [arXiv](https://arxiv.org/abs/2406.10210) [memory; cited in D2D references per search]
- **CLVR** (arXiv 2605.14876, May 2026): "automated data engine" using step-level visual verification to synthesize reliable reasoning trajectories; plus Proxy Prompt RL and Δ-space weight merge to 4 NFEs; claims to beat open-source baselines and approach proprietary models (no numbers in abstract) — [arXiv](https://arxiv.org/abs/2605.14876) [verified]
- Root-cause claim from a 2025 analysis: compositional failures stem partly from "sparse explicit negation in training data" — [arXiv 2511.10136](https://arxiv.org/abs/2511.10136) [verified]

### Inferences
- Data work increasingly overlaps with RL: "self-generated data loops" (generate → verify with VLM/detector → keep/prefer) are the data side of the same verifier-driven paradigm.

### Gaps
- No quantitative ablation found this session isolating recaptioning's effect on *object omission* specifically.
- Numbers for IterComp/EvoGen/CompAlign on T2I-CompBench not re-verified.

---

## Q4. Reasoning-enhanced and unified MLLM generation; reflection; test-time scaling — do they reduce omission on long/complex prompts?

### Takeaway
Unified AR/AR+diffusion models (Janus-Pro, BAGEL, Show-o2, BLIP3-o, MetaQuery, OmniGen2) and large DiT models with MLLM encoders (Qwen-Image) plus proprietary GPT-Image/Nano Banana have essentially eliminated omission on short prompts (GenEval single/two-object ≈ 0.99). On dense prompts they still fail badly: on GenEval 2 the best *prompt-level* accuracy is 35.8% and drops to ~2–4% at 10 atoms; T2I-CoReBench (~20 elements/prompt) finds composition "still limited". Test-time scaling and reflection add +0.1–0.25 GenEval but at 16–32× compute.

### Cited Findings
**Unified / MLLM-based models (IDs)**
- Transfusion (arXiv 2408.11039), Show-o (2408.12528), Emu3 (2409.18869), Janus-Pro (2501.17811), MetaQuery (2504.06256), BAGEL (2505.14683), BLIP3-o (2505.09568), Show-o2 (2506.15564), OmniGen2 (2506.18871), Qwen-Image (2508.02324), HunyuanImage 3.0 (2509.23951), Emu3.5 (2510.26583) — [Transfusion](https://arxiv.org/abs/2408.11039), [Show-o](https://arxiv.org/abs/2408.12528), [Emu3](https://arxiv.org/abs/2409.18869), [Janus-Pro](https://arxiv.org/abs/2501.17811), [MetaQuery](https://arxiv.org/abs/2504.06256), [BAGEL](https://arxiv.org/abs/2505.14683), [BLIP3-o](https://arxiv.org/abs/2505.09568), [Show-o2](https://arxiv.org/abs/2506.15564), [OmniGen2](https://arxiv.org/abs/2506.18871), [Qwen-Image](https://arxiv.org/abs/2508.02324), [HunyuanImage 3.0](https://arxiv.org/abs/2509.23951), [Emu3.5](https://arxiv.org/abs/2510.26583) [memory for IDs]
- Reported GenEval scores: BAGEL overall **0.82** (sub-scores 0.99/0.94/0.81/0.88/0.64/0.63) per InternVL-U's comparison table — [InternVL-U arXiv 2603.09877](https://arxiv.org/pdf/2603.09877) [verified via search snippet]; BAGEL reproduced at 0.78 (0.79 with thinking) in UniGRPO's setup — [UniGRPO](https://arxiv.org/html/2603.23500) [verified]. Janus-Pro-7B GenEval 0.80 / DPG 84.19 — [arXiv](https://arxiv.org/abs/2501.17811) [memory]
- Qwen-Image DPG-Bench **88.32** (as reported in its report and reproduced in the GEMS table), but InternVL-U reports 86.1 for Qwen-Image under its own setup — evaluation-setup inconsistency — [GEMS arXiv 2603.28088](https://arxiv.org/pdf/2603.28088); [InternVL-U](https://arxiv.org/pdf/2603.09877) [verified via snippets]. Qwen-Image GenEval reported ~0.87 base / ~0.91 after RL — [Qwen-Image report](https://arxiv.org/abs/2508.02324) [memory; aggregator partially supports, not confirmed]
- **T2I-R1** (NeurIPS 2025, arXiv 2505.00703): semantic-level CoT (planning) + token-level CoT, trained with BiCoT-GRPO using an ensemble of rewards on Janus-Pro-7B; **+13% T2I-CompBench, +19% WISE**; largest gains in attribute binding (+19% avg); spatial 0.3378 (>5% above previous SOTA); authors attribute CompBench gains to semantic-CoT planning — [arXiv](https://arxiv.org/abs/2505.00703) [verified]
- **Image Generation CoT** ("Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step", arXiv 2501.13926): test-time verification (PARM reward model) + DPO on Show-o / AR generation — [arXiv](https://arxiv.org/abs/2501.13926) [memory]
- **Thinking-while-Generating** (arXiv 2511.16671): interleaves textual reasoning throughout visual generation — [arXiv](https://arxiv.org/pdf/2511.16671) [verified: title/ID]
- **Self-Reflective RL for diffusion-based image reasoning generation** (arXiv 2505.22407) — [arXiv](https://arxiv.org/pdf/2505.22407) [verified: title/ID]
- **EndoCoT** (arXiv 2603.12252, 2026): scaling endogenous chain-of-thought reasoning inside diffusion models — [arXiv](https://arxiv.org/pdf/2603.12252) [verified: title/ID]

**Reflection / test-time scaling**
- **Inference-time scaling for diffusion models beyond scaling denoising steps** (Ma et al., arXiv 2501.09732): search over initial noise with verifiers (CLIP/Aesthetic/VLM) — [arXiv](https://arxiv.org/abs/2501.09732) [memory]
- **ReflectionFlow** ("From Reflection to Perfection", arXiv 2504.16080): FLUX.1-dev GenEval **0.67 → 0.85 (noise-level scaling) → 0.87 (+prompt-level) → 0.91 (+reflection-level)** with 32 samples — [arXiv](https://arxiv.org/pdf/2504.16080) [verified]
- **Reflect-DiT** (ICCV 2025, arXiv 2503.12271): in-context reflection on SANA-1.0-1.6B, **+0.19 GenEval, 0.81 with 20 samples**, beating SANA-1.5-4.8B's 0.80 that used 2048 samples — [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Reflect-DiT_Inference-Time_Scaling_for_Text-to-Image_Diffusion_Transformers_via_In-Context_Reflection_ICCV_2025_paper.html) [verified]
- 2026 agentic/closed-loop: CLVR frames single-step T2I as hitting "diminishing returns from parameter scaling" and positions closed-loop verified reasoning as "general test-time scaling" — [arXiv 2605.14876](https://arxiv.org/abs/2605.14876) [verified]; GEMS "Agent-Native Multimodal Generation with Memory and Skills" (arXiv 2603.28088) — [arXiv](https://arxiv.org/pdf/2603.28088) [verified: title/ID]

**Do they fix complex prompts? (evidence from 2025–26 benchmarks)**
- **GenEval 2** (Kamath, Chang, Krishna, Zettlemoyer, Hu, Ghazvininejad; arXiv 2512.16853, Dec 2025), 800 prompts, atomicity 3–10, human-judged atom-level / prompt-level: SD3.5-large 67.8/14.3; Flux.1-dev 67.0/15.3 (80.4/29.5 with rewriting); **BAGEL+CoT 71.7/16.0** (76.8/22.3 rewritten); **Qwen-Image 82.0/26.8** (85.3/35.8); **Gemini 2.5 Flash Image 84.4/31.0** (84.4/32.3). Best prompt-level = **35.8%** — [arXiv](https://arxiv.org/html/2512.16853v1) [verified]
- GenEval 2 atomicity collapse: Gemini 58% (3 atoms) → **4%** (10 atoms); Qwen-Image 58% → **2%**; Flux.1-dev ≈0% at ≥7 atoms. Counting ≈70% for Gemini and Qwen-Image; Qwen-Image position ≈60%; attribute errors largely binding failures. Object-presence skill is high (Qwen-Image 99.1%, Gemini 99.0%) so omission of *individual* objects is not the dominant atom-level error — failures are about satisfying *all* constraints jointly — [arXiv](https://arxiv.org/html/2512.16853v1) [verified]
- Note: BAGEL's CoT ("thinking") gives only modest gains on GenEval 2 (BAGEL+CoT 16.0 prompt-level vs. Flux 15.3), consistent with UniGRPO's finding that thinking did not consistently help base BAGEL on GenEval — [GenEval 2](https://arxiv.org/html/2512.16853v1); [UniGRPO](https://arxiv.org/html/2603.23500) [verified]
- **T2I-CoReBench** ("Easier Painting than Thinking", ICLR 2026, arXiv 2509.03516): 1,080 prompts, ~13,500 checklist questions, ~20 visual elements per composition prompt; "composition capability still remains limited in high compositional scenarios, while the reasoning capability lags even further behind." Nano Banana 75.9 vs GPT-Image 72.6 overall (Gemini 2.5 Flash evaluator); another paper with Qwen3-VL evaluator lists Nano Banana 75.3, GPT-Image 73.1, **Nano Banana Pro 82.9** — scores depend on evaluator — [arXiv](https://arxiv.org/abs/2509.03516v2), [project](https://t2i-corebench.github.io/), [ICLR 2026](https://mlanthology.org/iclr/2026/li2026iclr-easier/) [verified via search; column mapping uncertain]
- **Thinking in Pictures** (arXiv 2609.02864, 2026): systematic benchmark for reasoning-driven image generation — [arXiv](https://arxiv.org/pdf/2609.02864) [verified: title/ID only]

### Inferences
- Unified/reasoning models substantially reduce *short-prompt* omission and improve binding (T2I-R1 +19% binding), but persistent failures on long prompts are joint-constraint satisfaction, counting (~70% ceiling), spatial/position (~60%), binding with multiple attributed objects, and reasoning-derived content.
- Prompt rewriting (an LLM "system-level" step) is itself one of the biggest levers: Flux.1-dev prompt-level 15.3 → 29.5 on GenEval 2 just from rewriting.
- Explicit CoT inside unified models helps when trained jointly with RL (UniGRPO, T2I-R1) more than when bolted on (BAGEL thinking).

### Gaps
- No public, controlled omission-rate measurement found for GPT-4o / GPT-Image / Nano Banana on long prompts; GenEval 2 and T2I-CoReBench report aggregate/category accuracy, not omission vs. hallucination separately.
- Show-o2, BLIP3-o, MetaQuery, OmniGen2 GenEval/DPG numbers not verified this session.

---

## Q5. Work optimising for NOT adding unrequested objects (precision) and conflicting constraints

### Takeaway
Precision (no unrequested objects) is almost entirely unaddressed by training methods: benchmarks/rewards measure recall of requested content. 2024–25 position/evaluation papers formally define "object hallucination" in T2I as added entities and note that even human raters disagree on extraneous objects; negation (the main "conflicting/negative constraint" case) remains a known weakness.

### Cited Findings
- **"Hallucination as an Upper Bound: A New Perspective on T2I Evaluation"** (Kasaei & Rohban, arXiv 2509.21257): defines T2I object hallucination as adding entities never mentioned in the prompt, "driven by learned priors rather than user intent", distinct from object neglect (omission); taxonomy of object/attribute/relation hallucination; argues current evaluations focus almost entirely on prompt alignment — [arXiv](https://arxiv.org/abs/2509.21257) [verified]
- **Scene-graph QA hallucination evaluation** (Qin et al., arXiv 2412.05722): QA over existence, relations, attributes; labels "extraneous objects" by humans but "due to the subtlety of these extraneous objects, even human raters have not reached a consensus, making it challenging to provide an automated evaluation" — [arXiv](https://arxiv.org/abs/2412.05722) [verified]
- Survey of hallucination benchmarks/detection across I2T and T2I (arXiv 2507.19024): free-form generation makes hallucination evaluation harder — [arXiv](https://arxiv.org/pdf/2507.19024v1) [verified via snippet]
- **"Right Looks, Wrong Reasons: Compositional Fidelity in T2I Generation"** (Vatsa, Bharati, Singh; arXiv 2511.10136, Nov 2025): organises failures around **negation, counting, spatial relations**; models that handle each alone "degrade sharply when combined"; causes: sparse explicit negation in data, attention ill-suited to discrete logic, metrics that reward plausibility over constraint satisfaction; argues scaling and current fixes won't close the gap — [arXiv](https://arxiv.org/abs/2511.10136) [verified]
- LMD (2023) explicitly evaluated **negation** prompts ("without X") as a category where layout planning helps — [arXiv 2305.13655](https://arxiv.org/abs/2305.13655) [memory]
- Think-Then-Generate (ICML 2026) targets models taking prompts "too literally", i.e., resolving intent/implicit constraints before drawing — [arXiv](https://arxiv.org/abs/2601.10332v1) [verified]

### Inferences
- Because GenEval/T2I-CompBench/VQA rewards score presence of requested items, RL on them provides no gradient against extra objects; a precision-aware reward (e.g., detector-based "unrequested object" penalty or CHAIR-style metric for T2I) is an open opportunity.
- "Conflicting constraints" (e.g., negation vs. strong co-occurrence priors) remain largely tested, not trained for.

### Gaps
- Found no 2025–26 training method whose primary objective is precision / suppression of unrequested objects, and no standardised automatic T2I-CHAIR-style metric with broad adoption.
- I-HallA (arXiv 2409.12784) concerns factual hallucination, not extra objects; not re-verified.

---

## Q6. Dominant 2025–2026 trend, quantitative gains, and open problems from surveys

### Takeaway
The frontier moved: (2023) LLM layout planning + training-free guidance → (2024) data recaptioning + preference fine-tuning (Diffusion-DPO, IterComp) → (2025) online GRPO for flow models with verifiable/VLM rewards + reasoning-enhanced unified models (T2I-R1, BAGEL, Qwen-Image) + test-time reflection → (2026) **joint RL over "think-then-generate" pipelines** (UniGRPO, Think-Then-Generate, EndoCoT) and **closed-loop verified reasoning / agentic test-time scaling** (CLVR, GEMS), together with a benchmark reset (GenEval 2, T2I-CoReBench) because GenEval is saturated.

### Cited Findings
- GenEval saturated: 2024–25 SOTA models score ≥94.8% (human-judged, with rewriting), Gemini 96.7%; automated GenEval scores drift from humans by up to 17.7% (Gemini 2.5 Flash Image 75.4 auto vs 93.1 human), ~11.0% average gap; authors propose GenEval 2 + Soft-TIFA judge — [arXiv 2512.16853](https://arxiv.org/html/2512.16853v1) [verified]
- Key gains table (reported):
  - Flow-GRPO: SD3.5-M GenEval 0.63 → 0.95 — [arXiv](https://arxiv.org/abs/2505.05470) [verified]
  - UniGRPO: BAGEL 0.78 → 0.90 GenEval; internal TA 0.681 → 0.838 — [arXiv](https://arxiv.org/html/2603.23500) [verified]
  - T2I-R1: +13% T2I-CompBench, +19% WISE over Janus-Pro-7B — [arXiv](https://arxiv.org/abs/2505.00703) [verified]
  - ReflectionFlow: FLUX.1-dev 0.67 → 0.91 GenEval (32 samples) — [arXiv](https://arxiv.org/pdf/2504.16080) [verified]
  - Reflect-DiT: SANA-1.6B +0.19 → 0.81 GenEval (20 samples) — [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Reflect-DiT_Inference-Time_Scaling_for_Text-to-Image_Diffusion_Transformers_via_In-Context_Reflection_ICCV_2025_paper.html) [verified]
  - Qwen-Image DPG-Bench 88.32 (86.1 under InternVL-U's setup) — [GEMS](https://arxiv.org/pdf/2603.28088); [InternVL-U](https://arxiv.org/pdf/2603.09877) [verified via snippets]
  - GenEval 2 best prompt-level 35.8% (Qwen-Image + rewriting) — [arXiv](https://arxiv.org/html/2512.16853v1) [verified]
- Open problems from 2025 surveys/analyses: combined negation+counting+spatial failures; data sparsity of negation; attention unsuited to discrete logic; metrics rewarding plausibility — [arXiv 2511.10136](https://arxiv.org/abs/2511.10136) [verified]; evaluation drift of VLM/detector judges — [GenEval 2](https://arxiv.org/html/2512.16853v1) [verified]; hallucination (extra objects) unmeasured — [arXiv 2509.21257](https://arxiv.org/abs/2509.21257) [verified]; reward hacking via illusory advantage — [Pref-GRPO](https://arxiv.org/abs/2508.20751) [verified]; credit assignment with sparse terminal rewards and single-round only — [UniGRPO](https://arxiv.org/html/2603.23500) [verified]; planning hallucination, latency and long-context instability of multi-step systems — [CLVR](https://arxiv.org/abs/2605.14876) [verified]
- General surveys: "Text to Image Generation and Editing: A Survey" (arXiv 2505.02527; 141 works 2021–24) — [papers.cool](https://papers.cool/arxiv/2505.02527) [verified via snippet]; "Survey on Quality Metrics for T2I Generation" (Hartwig et al., Ulm, 2025) separates compositional quality from image quality — [PDF](https://viscom.publications.uni-ulm.de/api/uploads/290/survey-on-quality-metrics.pdf) [verified via snippet]; "Reinforcement Learning for Large Model: A Survey" (arXiv 2508.08189) — [arXiv](https://arxiv.org/pdf/2508.08189) [verified: title/ID]

### Inferences
- Dominant 2025–26 trend = **verifier-driven optimisation**: the same MLLM/detector/VQA judges are used (a) as RL rewards (Flow-GRPO family, UniGRPO), (b) as test-time selectors/critics (ReflectionFlow, Reflect-DiT, CLVR), and (c) as data filters (CLVR data engine) — and increasingly applied to models that *reason before drawing* (unified AR+diffusion, LLM-encoder diffusion). The main risk is judge drift/hacking, hence the parallel push for harder, human-audited benchmarks (GenEval 2, T2I-CoReBench).
- Practical takeaway for object faithfulness: omission on simple prompts is solved; the open frontier is (i) joint satisfaction at high atomicity, (ii) counting beyond ~70%, (iii) spatial/binding, (iv) precision (unrequested objects) and negation, which no mainstream reward currently optimises.

### Gaps
- Late-2026 (Jul–Nov) papers were sparsely covered; e.g., "Scaling Properties of Text Conditioning in Visual Generation" (arXiv 2607.29679) and "Thinking in Pictures" (2609.02864) appeared in search but were not read.
- Could not find a dedicated 2025–26 survey specifically on *compositional* T2I or *T2I hallucination* beyond 2511.10136 (analysis/position) and 2507.19024 (cross-modal hallucination survey).
- GPT-4o / GPT-Image / Nano Banana training details are undisclosed; their improvement mechanisms (autoregressive + reasoning, RL) cannot be attributed from primary sources.
