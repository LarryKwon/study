# 대화 정리: 멀티모달 LLM Post-training 연구 조사 (2026-10-27)

관련 문서: `omni_modality_decoupled_rl_design.md` (실험 설계안, 같은 폴더)

## 0. 읽기 전 주의
- 최초 개요(§1)는 **모델 기억 기반**이라 인용 전 확인이 필요하다.
- 이후 조사(§3~§6)는 웹 검색 스니펫과 일부 논문 본문 요약(WebFetch) 기준이다. 수치는 원문 대조 전에는 확정으로 쓰지 말 것.
- 논문 본문 중 일부(CogniRoute Appendix, MoD-DPO Supp. E)는 읽지 못했다.
- "못 찾았다"는 "없다"가 아니다. 인용 그래프(cited by)는 직접 확인하지 못했다.

## 1. 출발점: 연구 개요 (기억 기반)
멀티모달 post-training의 7개 축:
1. Multimodal SFT (LLaVA, InternVL 계열 등)
2. 선호 최적화 (RLHF/DPO: RLHF-V, POVID, mDPO 등)
3. 환각 완화 (POPE, CHAIR, MMHal-Bench)
4. 추론 강화 RL (RLVR/GRPO: Vision-R1 계열 등)
5. Reward Modeling / Judge
6. 안전성 / 정렬
7. 모달리티 확장 (비디오, 오디오/Omni, GUI·agentic)

공통 과제: 시각 근거 부족, 데이터, 보상 설계, 일반화, 평가 오염, 효율.

## 2. 사용자가 선택한 5가지 연구 방향과 관심 3개
| # | 방향 | 요지 |
|---|---|---|
| 1 | Perception-aware 보상 | 정답뿐 아니라 "본 것"의 정확성도 보상 |
| 2 | 이미지 의존성 강제 | 이미지를 손상시켰을 때 답이 달라져야 함 |
| 3 | Verifier 없는 도메인 RL | rubric/생성형 RM으로 개방형 질문에 RL |
| 4 | On-policy 효율화 | 토큰 압축, rollout 재사용 |
| 5 | Omni 통합 정렬 | 모달리티 간 충돌·간섭·편향 처리 |

사용자는 **1, 2, 5번**을 선택해 심화했다. 심화 내용의 핵심:
- 1: 응답 구조화(`<perception>`), grounding/IoU 보상, caption 기반 검증, 보조 라벨 활용. 위험은 reward hacking과 가중치 민감성.
- 2: mDPO류 대조 학습, 이미지 교란 negative, 정보이득 보상 `log p(y|이미지) − log p(y|손상 이미지)`, VCD. 위험은 이미지가 불필요한 문항과 교란 난이도.
- 5: 충돌·간섭·편향 세 문제, 충돌 데이터 구축, 우선순위 정책, 간섭 방지(데이터 혼합, KL 정규화 등).

## 3. 학회별 흐름 (2025~2026)

### 3.1 확인된 논문
| 학회 | 논문 | 주제 |
|---|---|---|
| NeurIPS 2025 | Open Vision Reasoner | Qwen2.5-VL-7B 오픈소스 RL, 시각 인지 행동 분석 |
| NeurIPS 2025 | VideoChat-R1.5 | Visual Test-Time Scaling |
| ICLR 2026 | Perception-R1 | visual perception reward (RLVR이 인식을 잘 못 높인다는 지적) |
| ICLR 2026 | Vision-SR1 | 인식/추론 분해 + 자기 보상 |
| ICLR 2026 | Visual Jigsaw | self-supervised RLVR (순서 복원) |
| ICML 2026 | IC-VCO, Semantic Curriculum PO, IRIS | 환각 완화, 이미지 의존성(DPO 계열) |
| ICML 2026 | OmniVideo-R1 | 오디오-비주얼 RL, contrastive modality reward |
| CVPR 2026 | Perceval | perception 중심 process reward model |
| CVPR 2026 | PEARL | 검증된 시각 증거에 추론을 고정 |
| CVPR 2026 | Understanding the Role of Hallucination in RL Post-Training | 환각이 RL에 도움이 될 수 있다는 "Hallucination-as-Cue" |
| CVPR 2026 | MoD-DPO | Omni 환각의 모달리티 분리 선호 최적화 |
| CVPR 2026 | R-C2 | cycle-consistency 보상 |
| AAAI 2026 | RL-CoMM ("When Eyes and Ears Disagree") | 오디오-비주얼 혼동 RL |
| ACL 2026 | Omni-RewardBench | Omni 모델은 reward model로 부족 |
| EMNLP 2026 Findings | Seeing Through Conflicts | 텍스트/이미지 지시 충돌, 규칙 보상 RL |
| EMNLP 2025 | SoundMind | 오디오 논리 추론 RL |

NeurIPS 2026(12월 6~12일, 시드니)은 제목 수준만 확인(Self-Distilled RLVR, SAGE). ECCV 2026과 ICLR 2027 투고분은 확인하지 못했다.

### 3.2 흐름 해석
1. **GRPO 기반 규칙 보상 RL이 주류.** 변형(S-GRPO, Faithful GRPO, GMPO 등)이 증가.
2. **병목이 추론에서 인식으로 이동.** Perception-R1 → Perceval/PEARL로 정교화(process 단위 검증, 시각 증거 고정). **1번은 포화에 가까움.**
3. **RL과 환각의 관계가 새 질문.** 환각이 RL에 도움이 될 수 있다는 결과.
4. **2번(이미지 의존성)은 DPO 계열에서 진행** (mDPO → IC-VCO). RL 보상 쪽 직접 연구는 뚜렷하지 않았다.
5. **Omni(5번)가 2026 하반기에 가장 크게 이동.** 문제 정의가 구체화됨: 모달리티 간 간섭, 시각 shortcut으로 부풀려진 벤치마크, 충돌 시 prior 지배.
6. 부상 중인 축: thinking with images, agentic 멀티턴 RL, self-supervised RLVR, latent reasoning.

## 4. Omni 충돌 + RL 교집합 조사

### 4.1 직접 해당 논문
- **RL-CoMM** (2511.10059, AAAI 2026): Qwen2.5-Omni-3B. 외부 오디오 LLM을 참조로 한 Step-RR + Ans-CO. AV-ConfuseBench는 **73+100문항**으로 작고, 소리 없는 객체(muted) 정확도 8.2→27.4(Gemini 2.5 Pro는 68.5). 간섭 미측정.
- **CogniRoute** (2606.20970): Qwen3-Omni-30B-A3B, MoE 라우팅 + RL. 보상에 모달리티 인용 및 관계(none/support/conflict) 서술 검사. 한계: Gemini-3.1-Pro가 데이터 생성·판정·라벨링을 모두 수행(순환성), 판정기 검증 없음, 충돌은 합성이 아니라 라벨. MMAU 75.4→73.2 소폭 하락.
- **MoD-DPO** (2603.03192, CVPR 2026): Qwen2.5-Omni-7B, MiniCPM-O. DPO + 무관 모달리티 교란 시 불변 + 관련 모달리티 교란 시 민감 + 언어 prior 패널티. 확산 노이즈 500 step이 최적(10 step은 실패). AVHBench 매칭 54.69→69.68. 한계: 상보적 시청각 추론 미촉진, 하이퍼파라미터 민감. 본문에 τ 정의와 epoch 서술 불일치. **오프라인 DPO이며 RL 확장 논의 없음.**
- **OmniVideo-R1** (ICML 2026): contrastive modality reward. 충돌 명시 모델링 없음.
- **OmniNFT**: 생성 쪽, 보상별 advantage를 modality로 라우팅(샘플의 약 절반에서 보상 방향이 반대).

### 4.2 벤치마크/진단
AVID(11.2K 비디오, 오디오-비주얼 불일치), C³PO(Counterfactual Conflict), OMD-Bench(abstention), XModBench, Beyond Text-Dominance(삼중 충돌에서 시각 선호, 오디오 과소평가: MSR 21% 미만), CAD(training-free 충돌 인지 디코딩), Chain of Modality(충돌 시 어느 분기도 일관되게 우세하지 않음), Compositional Failure(정렬 단계 개입이 충돌 해소를 복구하지 못함).

### 4.3 공백 검증 결과 (인용 추적 포함)
| 공백 | 결과 |
|---|---|
| ① 합성 충돌 데이터로 RL | 유지. 오디오 counterfactual 교체 + GRPO 조합 못 찾음 |
| ② 신뢰 모달리티 arbitration 학습 | 유지(정답 정의가 어려워 위험 큼). CAD/Chain of Modality는 학습 없이 또는 분석만 |
| ③ RL 후 간섭 정량화 | 유지. 시각 RL→오디오 저하를 직접 본 연구 없음. 주변: Omni-R1(MIT)은 Qwen2.5-Omni에서 망각 발생, Nemotron 3 Nano Omni는 단계적 학습으로 완화, RL's Razor(ICLR 2026)는 RL이 덜 잊는다고 주장, RL Forgets!(2607.04364)는 반론 |
| ④ 불변/민감 정규화의 RL화 | 유지. MoD-DPO의 GRPO 버전 못 찾음 |

**정정 기록**: 중간에 "Omni RL + 충돌 교집합이 비어 있을 가능성"이라고 말했으나, 이후 조사에서 RL-CoMM, CogniRoute 등 **직접 해당 논문이 이미 있음**이 확인되어 철회했다. 현재 판단은 "교집합 자체는 차 있고, 위 ①~④의 세부 공백이 남아 있다"이다.

## 5. 합의된 방향과 설계안
- 선택: **Modality-Decoupled RL** (공백 ①+④를 묶고 ③을 평가에 포함). ②는 확장 옵션.
- 베이스: Qwen2.5-Omni-7B (RL-CoMM, MoD-DPO와 직접 비교 가능).
- 데이터: 영상 A 비디오 + 영상 B 오디오의 mismatched 샘플, 질문별 관련 모달리티 태그로 **판정 모델 없이 결정론적 검증**.
- 보상: `R_fmt + R_acc + λ_inv·R_inv + λ_sens·R_sens` (+선택적 언어 prior 패널티)
  - **옵션 A (KL 기반)**: 토큰별 분포 KL. 충실하지만 비용 큼, CoT 반영.
  - **옵션 B (정보이득 기반)**: 답 토큰 로그확률 차이. 저렴하고 해석 쉬움, CoT 미반영. **기본 시작점으로 권장**, 효과가 있으면 A로 상한과 비용을 비교(H4).
- 평가: 환각/충돌 벤치 + 일반 Omni 벤치 + 간섭 평가(오디오/비디오/텍스트/음성 인식) + 새 지표 MRI(Modality Reliance Index), 3 seeds 이상.
- 상세는 `omni_modality_decoupled_rl_design.md` 참고.

## 6. 남은 확인 사항 (미해결)
1. RL-CoMM, MoD-DPO 인용 논문의 직접 확인 (Semantic Scholar/Google Scholar의 cited by)
2. Qwen2.5-Omni 오디오-비디오 입력과 GRPO를 함께 지원하는 학습 스택
3. 외부 베이스라인(MoD-DPO, OmniDPO, RL-CoMM, OmniVideo-R1)의 공개 코드/체크포인트
4. 학습 소스(VALOR32K, AudioCaps)와 AVHBench의 데이터 중복 가능성
5. 열어보지 못한 논문: Omni-Perception Policy Optimization(2606.25325), NeurIPS 2026 채택 목록, ECCV 2026, ICLR 2027 투고분
6. 보상식(`kappa` 상한, `R_sens` 게이팅)은 제안 단계이며 어떤 논문에서도 검증되지 않았음
7. GPU 시간 추정은 프로파일링 후 확정

## 7. 대화 진행 순서 (참고)
1. 연구 개요 제시(기억 기반) → 2. 5가지 방향 쉬운 설명 → 3. 1·2·5번 심화 → 4. 논문 리스트 웹 조사 → 5. 주요 학회 흐름 조사 → 6. 2026년 하반기로 갱신(오늘이 2026-10-27이므로) → 7. Omni 충돌 + RL 심화 조사 → 8. 핵심 3편 비교(1단계) → 9. 인용 추적(2단계) → 10. 실험 설계안(3단계) → 11. 본 정리 문서

## 8. 주요 참고 링크
- RL-CoMM: https://arxiv.org/html/2511.10059 (코드: https://github.com/rikeilong/AudioVisual_Confusion)
- CogniRoute: https://arxiv.org/html/2606.20970
- MoD-DPO: https://arxiv.org/html/2603.03192
- mDPO: https://arxiv.org/pdf/2406.11839
- Perception-R1: https://openreview.net/forum?id=KttCXdjj4w
- Perceval: https://arxiv.org/pdf/2604.24583
- PEARL: https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Perceptual-Evidence_Anchored_Reinforced_Learning_for_Multimodal_Reasoning_CVPR_2026_paper.html
- Hallucination in RL Post-Training: https://arxiv.org/pdf/2604.03179
- OmniVideo-R1: https://arxiv.org/pdf/2602.05847
- OmniNFT: https://arxiv.org/html/2605.12480
- AVID: https://arxiv.org/pdf/2604.13593
- C³PO: https://arxiv.org/pdf/2608.05381
- OMD-Bench: https://arxiv.org/pdf/2603.27187
- CAD: https://arxiv.org/html/2609.04247v1
- Chain of Modality: https://arxiv.org/pdf/2604.14520
- Beyond Text-Dominance: https://arxiv.org/pdf/2604.16902
- Compositional Failure in Audio-Visual LLMs: https://arxiv.org/pdf/2608.27785
- Omni-R1 (audio): https://arxiv.org/html/2505.09439v1
- RL's Razor: https://proceedings.iclr.cc/paper_files/paper/2026/file/618c95f4557c15b253fb0e6f548ea0c0-Paper-Conference.pdf
- RL Forgets!: https://arxiv.org/html/2607.04364v2
- Nemotron 3 Nano Omni: https://arxiv.org/html/2604.24954v2
- Qwen3.8-Omni: https://arxiv.org/html/2609.25611v1
- Relax: https://arxiv.org/pdf/2604.11554
