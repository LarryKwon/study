# LLM 에이전트 페르소나 → 추천 시스템 전이: 연구 조사

- 작성일: 2026-10-28
- 조사 방식: 웹 검색 + 페이지 fetch. 대부분 초록·스니펫 수준이며 본문 정독 아님. 수치는 검증하지 않음.
- 표기: "(배경 지식)"은 검색이 아니라 기존 지식이므로 인용 전 확인 필요.

## 연구 방향 (사용자 의도)

LLM 에이전트로 다양한 페르소나를 생성하고, 그 preference 데이터를 추천 모델로 잘 전이시키는 방법.

1. 에이전트 preference 데이터 기반 추천 Foundation Model (LLM보다 경량, continual learning 가능)
2. Cold-start 해결
3. 기타 (sim-to-real, bias, RL 환경 등. 모두 LLM과 추천 시스템 결합 관점)

---

## 1. 기반: LLM 에이전트 유저 시뮬레이션

| 논문 | 요점 |
|---|---|
| [Agent4Rec](https://arxiv.org/pdf/2310.10108) (SIGIR 2024) | profile/memory/action 모듈. MovieLens, Steam, Amazon-Book 프로필 초기화. 실유저와의 정렬·편차 보고 |
| RecAgent (배경 지식) | 사회적 관계로 행동 다양성 확대 |
| [SimUSER](https://arxiv.org/pdf/2504.12722) | 이력에서 페르소나 추론, 이력과 가장 일관된 후보 선택(self-consistent). 시각 지각 모듈 |
| PUB | 행동 로그에서 Big Five 성향 추론 → 합성 상호작용 생성. Amazon 리뷰에서 통계적 충실도 보고 |
| [Task-aware profile generation](https://arxiv.org/pdf/2605.13497) (SIGIR 2026 #111) | 기존 연구가 memory/action에 치우쳤고 profile 생성은 소홀했다고 지적 |
| [Persona Fidelity Gap](https://icml.cc/virtual/2026/75683) (ICML 2026) | 약 268K Amazon 유저와 비교. 랭킹 정확도 0.72~0.85로 높지만 PRDC Coverage 4.6~9.4%. 그라운딩해도 선호 다양성 압축 |
| [Beyond Offline A/B Testing](https://arxiv.org/pdf/2604.09549) | context-aware 에이전트 시뮬레이션으로 추천 평가 |
| [PersonaGym형 합성 상호작용](https://www.themoonlight.io/de/review/synthetic-interaction-data-for-scalable-personalization-in-large-language-models) | persona bank로 합성 데이터 생성. LLM 개인화용 |

핵심 시사점: Persona Fidelity Gap. 합성 데이터로 학습하면 선호 다양성이 붕괴할 위험.

## 2. 방향 1: 경량·continual 추천 Foundation Model

직접 겨냥한 논문은 검색에서 못 찾음 (공백 가능성). 인접 연구:

- LLM → 경량 모델 증류
  - [DLLM2Rec](https://arxiv.org/pdf/2405.00338): 순위 증류 + 협업 임베딩 증류. 평균 47.97% 향상, 경우에 따라 LLM 추천기 능가
  - ALKDRec: active learning으로 LLM 질의 인스턴스 선택
  - [Li et al. 2409.14810](https://arxiv.org/abs/2409.14810v1): LLM 파인튜닝 후 경량 모델 증류, 실시간 서빙
  - [arXiv 2502.14305](https://arxiv.org/pdf/2502.14305): Llama-3 증류 + pruning. 증류가 SFT보다 일관되게 우수
- Continual 증류: SCoRD (2026 프리프린트). 비정상 스트림에서 LLM 리랭커를 경량 ID 기반 retriever로 증류. 저신뢰 시퀀스만 선택 증류. 가장 가까운 선행 연구
- (배경 지식) HSTU/Generative Recommenders, P5, TIGER(semantic ID), 추천 도메인 continual learning(replay, LoRA incremental update)

연구 포인트 후보: (1) 페르소나 조건화와 preference 데이터 형태(pairwise/listwise/rationale), (2) 합성 데이터의 다양성 붕괴 보정, (3) 합성 사전학습 → 실데이터 continual fine-tuning 파이프라인.

## 3. 방향 2: Cold-start

- [LLM-InS / ColdLLM](https://arxiv.org/pdf/2402.09176) (WSDM 2024): cold 아이템의 상호작용을 LLM으로 시뮬레이션해 warm화. coupled filter로 후보 유저를 수백 명으로 축소. 온라인 A/B Cold-GMV +16.90% (요약 글 출처). 명시적 페르소나 없음
- 공백(검색에서 못 찾음): 페르소나 기반 유저 cold-start (신규 유저 ↔ 페르소나 매칭, 시뮬레이션 이력으로 초기 임베딩)
- (배경 지식) LLM 인터뷰/대화형 elicitation, MeLU, Heater, CLCRec

## 4. Sim-to-real

추천 도메인 정면 연구는 확인 못 함. 일반 유저 시뮬레이션 쪽:

- [Mind the Sim2Real Gap](https://www.alphaxiv.org/abs/2603.11245v1): LLM 시뮬레이터 31종 vs 사람 451명, User-Sim Index. 지나치게 협조적이고 문체 균일. 모델 성능이 높다고 더 현실적이지 않음
- [RealUserSim](https://arxiv.org/html/2605.20204), [ParaStudent](https://arxiv.org/pdf/2507.12674), [Sim-to-Real Gap of FM Agents](https://arxiv.org/pdf/2606.07017), [Decision-Aware User Simulator (광고)](https://arxiv.org/pdf/2607.26893)
- [How Reliable is Your Simulator?](https://www.researchgate.net/publication/380546447_How_Reliable_is_Your_Simulator_Analysis_on_the_Limitations_of_Current_LLM-based_User_Simulators_for_Conversational_Recommendation) (CRS)
- NeurIPS 2026 [UserSim 워크숍](https://usersim-workshop.github.io/)

해석: 논의가 "시뮬레이터 현실성 평가"에 몰려 있고, "시뮬 데이터로 학습한 추천 모델이 실서비스에서 통하는가"는 열린 문제.

## 5. 강화학습 환경

- [LLM-Powered User Simulator for RecSys](https://ojs.aaai.org/index.php/AAAI/article/view/33456/35611) (AAAI 2025): 논리 모델 + 통계 모델 앙상블, 3개 모델 합의로 보상. 5개 도메인 검증
- [Lusifer](https://arxiv.org/pdf/2405.13362): 온라인 추천 RL용 LLM 피드백 환경
- [LLM as User Simulator: Training News Recommender without Real User Interactions](https://dl.acm.org/doi/10.1145/3726302.3730224) (SIGIR 2025): 실제 상호작용 없이 학습. 제안 방향과 가장 가까운 직접 선행
- [User Behavior Simulation with LLM-based Agents](https://dl.acm.org/doi/10.1145/3708985) (TOIS), [SR-Agent](https://arxiv.org/pdf/2607.17719)
- 비-LLM 선행: RecoGym, Virtual Taobao

## 6. Bias

- 시뮬레이터 자체: 인기도 편향(아이템 언급 균질화), 긍정 편향(제안 대부분 수락, 실유저는 약 절반 거절). 근거: [CRS 프롬프트 최적화 논문](https://arxiv.org/pdf/2607.00010), [NAACL 2024](https://aclanthology.org/2024.naacl-long.83.pdf)
- 완화: entropy 인지 프롬프트 최적화, APG4RecSim (둘 다 미검증 프리프린트)
- 추천 모델로의 전파: 직접 측정 논문 못 찾음. 닫힌 루프 재학습이 에이전트 기본 성향을 증폭하고 카탈로그를 좁힌다는 [arXiv 2604.26960](https://arxiv.org/html/2604.26960)
- 서베이: [Conversational User Simulation (EACL 2026)](https://aclanthology.org/2026.eacl-long.200.pdf)

## 7. 합성 preference → 추천 모델 전이

| 이슈 | 근거 |
|---|---|
| 합성 데이터를 그냥 섞으면 분포 이동으로 성능 저하 가능 | [SynAlign (MMD 정렬)](https://arxiv.org/pdf/2502.08661) |
| 합성 비율이 커질수록 격차 확대. 10%만 섞어도 test loss 상승 가능 | [arXiv 2506.19262](https://arxiv.org/pdf/2506.19262) |
| 합성 데이터만으로 학습한 순차 추천이 실데이터 학습을 Recall@K에서 능가 (자체 보고, 미검증) | [Principled Synthetic Data … Scaling Laws](https://arxiv.org/html/2602.07298v2) |
| 선호 구조가 보존되면 프롬프트 분포가 달라도 합성 pair가 DPO에 유효 | [DP preference 합성](https://arxiv.org/html/2605.30808) |

현실적 레시피 후보: 합성 분포를 실로그에 정렬(MMD 등), 합성 비율은 작게, 실데이터 holdout 검증. 캘리브레이션에 대한 직접 근거는 없음.

---

## 8. 2026 학회 프로그램 조사

### 조사 범위의 한계

| 학회 | 확인 범위 |
|---|---|
| SIGIR 2026 | 제3자 서베이 [gist](https://gist.github.com/rankun203/4f1a77c2705b2945417d0e1c581c4859/ed5a208b9b628e02bef68a0a809a9795de4f730a)에서 Generative & Sequential Rec 62편 확인. Recommender Systems 섹션 29편은 잘려서 못 읽음. 총 668편 중 일부. 공식 목록 아님 |
| WWW 2026 | [short paper](https://www2026.thewebconf.org/accepted/short-papers.html)와 workshop 일부. 76만 자 중 앞 10만 자만. full paper 미확인 |
| KDD 2026 | 공식 목록 못 찾음. 구두 발표 1편만 확인 |
| RecSys 2026 | 목록 못 찾음. Industry Track 1편만 확인 |
| ICLR/NeurIPS/ACL | 직접 대조 못 함 |

"X는 없다"고 단정하면 안 됨. 아래 공백은 확인 범위 안에서 안 보였다는 뜻.

### 확인된 2026 논문

유저 시뮬레이션 / 페르소나 / 합성 데이터
- SIGIR #111 Task-Aware User Profile Generation: 이력 기반 task-aware 프로필. 평가용
- SIGIR '26 [Verifiable User Simulation](https://arxiv.org/html/2606.14474v1) ([DOI](https://doi.org/10.1145/3805712.3808645)): 튜토리얼. Persona→Contract→Execution→Trace→Verification→Feedback→Refinement 7요소 감사 프레임워크. 논문이 직접 밝히듯 sim-to-real, 추천 모델 학습은 다루지 않음 (평가용)
- [AlignUSER](https://arxiv.org/pdf/2601.00930) (학회 미확인): world model 다음 상태 예측 + counterfactual 궤적 기반 페르소나 정렬. 평가용
- WWW '26 workshop [RecoWorld](https://arxiv.org/pdf/2509.10397): 에이전트형 추천 시뮬레이션 환경
- [RecRM-Bench](https://arxiv.org/pdf/2605.11874) (학회 미확인): 에이전트형 추천 reward modeling 벤치
- WWW '26 short "RL Based Hyper-Parameter Generation System Guided by LLM-Powered Virtual Users": LLM 가상 유저를 RL 피드백원으로. 제목만 확인
- SIGIR #561 IKEA (`Negative Data Mining for Contrastive Learning in Dense Retrieval at IKEA.com`): LLM-as-judge 합성 hard negative. 오프라인 개선이 온라인으로 이어지지 않았고 합성 질의 의도 범위가 훨씬 좁음. 합성 데이터 전이 실패 직접 사례
- SIGIR #81 Formalized Information Needs: 구조화된 정보 요구가 LLM 판단 positivity bias 감소

합성 데이터를 학습에 쓰는 쪽
- SIGIR #88 LGCD: 합성 pseudo-overlapping 유저로 cross-domain cold-start
- SIGIR #233 LLM-EDT: LLM 증강 + 2단계 학습, cross-domain 순차 추천
- SIGIR #85 ProMax: LLM 유저·아이템 프로필을 CF의 분포 조정 신호로

페르소나 / 서빙
- [RecSys 2026 Industry: LLM-Based User Personas for Recommendations at Scale](https://arxiv.org/abs/2606.12198) (Ed H. Chi 등): 대형 상용 비디오 플랫폼에서 자연어 관심 페르소나를 실시간 생성. 지식 증류, 비동기 추론, 입력 최적화로 10억 유저 규모. 오프라인/유저 스터디/라이브 A/B 개선 (구체 수치 없음)

Cold-start
- WWW '26 short [LLM Reasoning for Cold-Start Item Recommendation](https://arxiv.org/abs/2511.18261) (UT Austin·Netflix), COINS (Semantic ID cold 아이템 표현)
- SIGIR #122 GenRecEdit (모델 편집으로 cold 패턴 주입), #35 SG-URInit (training-free 유저 초기화), #199 Selective Knowledge Augmentation

Continual / Foundation model
- SIGIR #190 DACT (drift한 semantic ID만 증분 재양자화), #400 Semantic ID staleness (옛/새 ID 정렬 warm-start)
- SIGIR #541 SIGMA (Qwen3 기반 단일 생성 모델로 7개 과제), #146 UTGRec (범용 MLLM 토크나이저)
- LLM 임베딩 융합 계열: #232 FAERec, #374 SICSRec, #427 ACE, #121 SpecTran, #83 DCGL 등

Bias
- WWW '26 short "What Is Your AI Agent Buying?": 에이전트형 이커머스의 구매 편향
- SIGIR #603 Bridging Passive and Active: 피드백 루프 편향 완화

### 흐름 해석 (확인 범위 안에서)

1. 시뮬레이터는 "평가·감사" 쪽으로 성숙 중 (Verifiable User Simulation, AlignUSER, Task-aware profile, RecoWorld 모두 평가용).
2. 합성 데이터의 추천 모델 학습 활용은 증강형이 주류 (LGCD, LLM-EDT, ProMax). 에이전트 시뮬레이션 데이터로 모델 전체를 학습하는 형태는 소수.
3. 전이 실패 사례가 이미 보임 (IKEA).
4. 산업계는 "LLM 페르소나 + 증류 + 서빙"을 한 번에 풀고 있음 (RecSys Industry).
5. Continual은 LLM 추천이 아니라 Semantic ID drift 쪽에서 진행.

### 사용자 방향의 포지셔닝

- 에이전트 preference 데이터로 추천 모델을 학습해 전이하는 방법, 특히 분포 정렬과 sim-to-real 격차 보정을 정면으로 다룬 논문은 확인한 목록에서 안 보임.
- 시뮬레이터 신뢰도(평가 측)와 증강형 합성 데이터(학습 측)를 잇는 연구는 상대적으로 비어 보임.
- Semantic ID 기반 continual 연구는 "합성 사전학습 → 실데이터 continual"과 결합할 접점이 될 수 있음. 이는 판단이며 검증된 결과가 아님.

공백 후보 정리:
1. 합성 preference → 추천 모델 전이의 분포 정렬/캘리브레이션 (현재는 일반 NLP 합성데이터 기법뿐)
2. 편향이 추천 모델까지 전파되는 정도의 정량화와 완화
3. 페르소나 다양성 보존형 생성 + 그 데이터로 학습한 경량·continual 모델
4. 추천 도메인 sim-to-real 평가 프로토콜 (시뮬 성능 ↔ 실서비스 성능 상관)

## 우선 정독 논문

1. [LLM as User Simulator: Training News Recommender without Real User Interactions](https://dl.acm.org/doi/10.1145/3726302.3730224) — 직접 선행
2. [Persona Fidelity Gap](https://icml.cc/virtual/2026/75683) — 다양성 한계
3. [Principled Synthetic Data … Scaling Laws](https://arxiv.org/html/2602.07298v2) — 합성이 실데이터를 이길 수 있다는 근거 (검증 필요)
4. [LLM-Based User Personas for Recommendations at Scale](https://arxiv.org/abs/2606.12198) — 산업계 증류+서빙
5. SIGIR #561 IKEA — 합성 데이터 전이 실패 사례

## 남은 작업

- SIGIR Recommender Systems 섹션 (gist offset 200000 이후) 마저 읽기
- WWW full paper, KDD, RecSys 공식 목록 확인 (접근 가능한 URL 필요)
- 핵심 논문 본문 정독
