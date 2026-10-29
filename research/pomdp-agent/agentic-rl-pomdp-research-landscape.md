# Agentic RL × Tool Use × POMDP: 연구 지형 및 연구 기회 분석

- 작성일: 2026-10-27
- 조사 방법: 웹 검색 약 30회, 핵심 논문 원문/초록 확인 (Claude Code 조사 보조)
- 검증 수준 표기: **[원문]** 본문 확인 · **[초록]** arXiv 초록/abs 페이지 확인 · **[검색]** 검색 결과 요약 기준 · **[2차]** 블로그/리뷰 등 2차 자료 · **[배경]** 조사 보조의 배경지식(미검증)

---

## 0. 요약

1. 2026년 들어 tool 사용 LLM 에이전트를 **POMDP로 정식화**하는 것이 사실상 표준이 됐다. 단일 턴 RLHF/RLVR은 horizon 1의 퇴화 MDP로 대비된다.
2. 2026년 상반기부터 **"belief를 명시적으로 만들자"** 흐름이 빠르게 붐비고 있다: Agent-BRACE, ReBel, PABU, CGDP, Belief-State Engine, Graph of States(GoS).
3. 그러나 다음은 여전히 비어 있다.
   - belief RL은 대부분 **텍스트 게임·검색 QA**에 머물러 있고, 실제 tool 환경(SWE, 운영 장애 진단, API)에서는 드물다.
   - belief 보상은 **LLM judge, 자기일관성, 정답 likelihood**에 의존한다. 환경의 진짜 상태를 기준으로 한 검증 가능한 belief 보상은 보이지 않는다.
   - **belief→행동 변환**, **동적 환경에서의 재관측**, **비가역 행동 하의 탐색**은 문제 제기만 있고 RL 해법이 없다.
4. 가장 유망한 기회: **tool 샌드박스의 privileged 환경 상태를 이용한 state-anchored belief shaping** (로그 점수 기반 potential shaping)과 privileged critic의 결합.
5. 추천 로드맵: 기회 2(belief→행동 간극 진단, 1~2개월) → 기회 1(본 연구) → 기회 3·4(동적·비가역 환경) 확장.

---

## 1. 배경: 왜 POMDP인가

### 1.1 정식화

| 구분 | 단일 턴 LLM RL (RLHF/RLVR) | Agentic RL (tool 사용) |
|---|---|---|
| 정식화 | 퇴화 MDP: 상태 = 프롬프트, horizon = 1 | POMDP $(S, A, O, T, Z, R, \gamma)$ |
| 상태 | 완전 관측 | 숨은 상태: DB, 파일시스템, 웹, 외부 API, 사용자 의도 |
| 행동 | 응답 하나 | tool call, 발화, 코드 실행 |
| 관측 | 없음 | tool 응답: 부분적·노이즈·지연·낡을 수 있음 |
| 보상 | 응답 단위 | 대개 에피소드 종료 시점의 성공 여부 (희소) |

Landscape of Agentic RL 서베이가 이 대비를 프레임의 기준으로 쓴다 **[검색]**.

### 1.2 Tool 사용이 만드는 POMDP 특유의 난제

| 난제 | 내용 |
|---|---|
| belief 표현 | history 전체를 상태로 쓸지, 압축된 belief를 둘지 |
| 정보 획득 vs 활용 | 조회(read) 행동과 상태 변경(write) 행동이 섞여 있음. 언제 더 조사하고 언제 행동할지 |
| credit assignment | 최종 실패가 추론, tool 선택, 실행, 종합 중 어디서 왔는지 모호 |
| 장기 context | history가 길어지면 context가 넘치고 정책이 표류 |
| 환경 노이즈·비정상성 | tool 응답 오류, 타임아웃, 다른 행위자에 의한 상태 변화 |
| 비가역성 | 메시지 발송, 삭제, 결제처럼 되돌릴 수 없는 행동 |

### 1.3 History-as-state vs. 명시적 belief 논쟁

- **Belief-State Engine** **[초록]**: 전형적인 LLM 에이전트는 "숨은 상태에 대한 명시적 belief가 없는 history 조건부 정책"이며, LLM이 원본 history를 보지 않고 posterior만 볼 때에만 belief MDP 위의 건전한 Markov 정책이 된다고 주장.
- **PABU** **[원문]**: 약한 정보나 무관한 행동이 섞인 전체 history로 조건화하면 "representation collapse"가 생긴다고 주장. 단, 정식 정의나 직접 측정은 없음.
- **Agent-BRACE** **[원문]**: 요약 기반 belief는 분포를 점추정 하나로 붕괴시켜 불확실성을 버린다고 비판.

---

## 2. 연구 지형

### 2.1 서베이·프레임

| 연구 | 내용 | 확인 |
|---|---|---|
| The Landscape of Agentic RL for LLMs (2509.02547, v5까지 갱신) | 500편 이상 종합. 단일 턴 MDP vs 다중 턴 POMDP 대비, planning·tool use·memory·reasoning·self-improvement·perception 축 분류 | [검색] |
| A Practitioner's Guide to Multi-turn Agentic RL (2510.01132) | 환경·보상·정책 설계 실증 가이드 | [검색] |
| From Reasoning to Agentic: Credit Assignment (2604.09459) | 2024~2026 초 CA 방법 47개를 token/step/turn/multi-agent 단위로 분류. Awesome 리스트가 2026.09까지 갱신 | [검색] |
| Adaptation of Agentic AI (2512.16301) | post-training·memory·skills. 최종 오답의 원인 귀속 문제 지적 | [검색] |
| Agentic Search RL 서베이 (2510.16724) | RL 기반 검색 에이전트 종합. DeepRAG의 정답·검색비용 동시 최적화 언급 | [검색] |
| Agent harness 설계 서베이 (2606.20683), Memory 서베이 (2603.07670), Horizon Gap (2608.06663) | 하네스·메모리·장기 horizon 종합 | [검색] |

### 2.2 다중 턴 RL 인프라·알고리즘

| 연구 | 내용 | 확인 |
|---|---|---|
| AgentGym-RL (2509.08755) | 장기 horizon 다중 턴 RL을 POMDP로 정식화, N턴 후 outcome 보상 | [검색] |
| TSR (2602.11767) | POMDP 설정에서 trajectory search로 rollout 품질 개선 | [검색] |
| VerlTool (2509.01055), Tool-R1 (2509.12867), ToolBrain (2510.00023) | tool 사용 agentic RL 프레임워크 | [검색] |
| ToolPO / DeepAgent | 대규모 tool set에서 시뮬레이션 API와 token 단위 advantage 귀속으로 end-to-end RL | [2차] |
| Search-R1, ReTool, ToolRL, ToRL | tool-integrated RL과 GRPO 변형(turn 단위 credit, tool 출력 token masking) | [배경] |

### 2.3 Belief를 명시적으로 다루는 연구 (가장 직접적인 선행연구)

| 연구 | belief 표현 | 학습 | 환경 | 확인된 한계 | 확인 |
|---|---|---|---|---|---|
| **Graph of States** (2603.21250) | 기호적 인과 그래프 + 레벨 상태기계 | 없음 (추론 전용) | 의료 진단, 분산시스템 장애 진단 (tool 호출 있음) | 잘못된 행동 선택이 주된 오류, backtracking 실패 30.9%, 낮은 가치 조회로 예산 소진 | [원문] |
| **PABU** (2602.09138) | 텍스트 belief $[q, p_n, A_{att}, A_{avail}, O_{saved}]$ | SFT만 (오프라인) | AgentGym 8개 환경 | 온라인 탐색 없음(저자 future work), 평균 수치 집계 불일치 | [원문] |
| **Agent-BRACE** (2605.11436) | 슬롯별 자연어 주장 + 언어 확률 라벨(WEP) | SFT 후 PPO, belief 보상은 LLM judge | TextWorld, ALFWorld (tool 없음) | 슬롯이 과제 특화. "개방형 환경의 belief 차원 자동 발견은 future work" | [원문] |
| **ReBel** (2605.20061) | belief → think → action 구조 | 예측 belief와 관측 피드백 불일치를 자기지도 신호로 사용 | ALFWorld, WebShop | 일관성 ≠ 정확성 | [초록] |
| **T3 / Belief deviation** (2510.12264, ICLR 2026) | 진리 고정 퍼텐셜 $\Psi(b) = -\log b(s^*)$ | RL rollout 절단 (PPO/GRPO/GSPO에 plug-in) | 열거 가능한 가설 공간의 퍼즐 | 프록시가 과제 특화, 일부는 정답 필요 | [원문] |
| **CGDP** (2605.07042) | predicate 기반 belief + 프로그램식 소진 게이트 | 초록상 RL 미확인 | 검색 QA 3개 도메인 | 검색 과제 한정 | [초록] |
| **Belief-State Engine** (2609.10036) | LLM 외부의 Bayesian posterior | 없음 | Tiger POMDP, 공격 그래프 | POMDP 모델을 알아야 함 | [초록] |
| VAGEN (2510.16907, NeurIPS 2025) | StateEstimation·TransitionModeling 추론 | WorldModeling 보상 + Bi-Level GAE | VLM 시각 환경 | tool 사용 아님 | [검색] |
| LUMINA (2601.16649) | oracle이 상태를 자연어로 제공 | 진단용 개입 | 다중 턴 환경 | 학습된 belief 아님 | [검색] |
| Reward-Driven Workflows (2607.17038) | 관측 history + graph memory로 근사 | 휴리스틱 | – | 정식 RL 결과 아님 | [검색] |
| POMDP 기반 검증 프레임 (2606.17383) | belief·forecast·policy 검증 | 학습 아님 | 포트폴리오 관리 | – | [검색] |
| Hypothesis-driven Clinical RL (ICLR 2026) | 가설 에이전트 + 결정 에이전트 순환 | SFT + RL | 감별 진단 | 의료 한정 | [2차] |

세부 사항:

- **PABU**: Llama-3.1-8B, 평균 성공률 81.0%(ATLAS 65.4% 대비), 평균 9.5 step. ALFWorld ablation에서 full-history outcome SFT 48.5% → PABU 90.0%, 입력 token은 약 절반. **주의**: 표의 과제별 수치를 단순 평균하면 약 69.7%라 81.0%와 맞지 않으며, 집계 방식 설명이 없음.
- **Agent-BRACE**: belief 모델과 정책 모델을 분리하고, 정책은 전체 history를 보지 않음. belief 보상 = format gate × (상태 추적 + 상태 정확성 + 라벨 다양성 + 할인된 과제 성공)의 평균. Qwen2.5-3B에서 72.8%(최강 RL baseline 58.3%), Brier score가 에피소드 동안 0.40 → 0.28. **주의**: ablation 수치가 두 표에서 서로 맞지 않음(75.3% vs 60.8%).
- **T3**: belief deviation = LLM 갱신과 Bayesian 갱신 사이의 Ψ 기대 차이. 가설 공간이 k step 동안 의미 있게 줄지 않으면(belief-trap region) rollout을 절단해, 꼬리 부분이 앞선 행동의 credit을 오염시키지 않게 함. Qwen2.5-7B에서 18개 지표 중 14개 개선(예: GRPO on GuessNumbers +30.1).

### 2.4 정보 획득·비용 인식 tool 호출

| 연구 | 내용 | 확인 |
|---|---|---|
| IGPO (Information Gain-based Policy Optimization) | GRPO 기반. 각 turn에 정답에 대한 정보이득(정답 teacher-forcing likelihood 변화)을 보상 | [2차] |
| TIPS (2603.22293, ICLR 2026) | 추론+tool call 구간마다 정답 likelihood 상승분을 potential-based shaping으로 PPO에 추가. Qwen-2.5 7B에서 EM +11.8%, F1 +13.6% | [검색] |
| Calibrate-Then-Act (2602.16699) | 잠재 환경 상태에 대한 prior를 LLM에 제공해 탐색 비용과 조기 결정 위험을 저울질. RL 학습 시에도 이점 유지 | [검색] |
| Uncertainty-Aware Clarification (2606.03135) | 명확화 질문의 효용을 Bayesian belief 갱신으로 측정하는 information gain 보상 | [검색] |
| UQ for LLM Agents (2609.07395) | 에이전트 불확실성 정량화 분류 체계와 평가 프로토콜 | [검색] |
| ESearch-R1 (2512.18571) | 비용 인식 임베디드 검색 MLLM 에이전트 RL | [검색] |
| CS 224R 프로젝트 리포트 | outcome-only 보상은 거의 항상 tool 호출, 호출당 페널티가 과하면 tool 회피 | [검색] |

### 2.5 강건성·Sim-to-real

**When Simulation Lies / RobustBench-TC, ToolRL-DR** (2605.11928, NeurIPS 2026 D&B) **[초록]**

- tool 사용을 POMDP로 두고 교란이 들어오는 4개 구성요소를 정의: 관측, 행동 공간, 보상 관련 메타데이터, 전이 동역학.
- 실제 GitHub 이슈에 근거한 22개 교란 유형, 21개 모델(1.5B~32B, o4-mini) 평가.
- 관측 교란은 정확도 하락 5% 미만이지만, **보상 메타데이터 교란은 약 40%, 전이 교란은 약 30%** 하락. 스케일만으로는 해결되지 않음.
- ToolRL-DR: 정적으로 인코딩 가능한 3개 구성요소로 도메인 랜덤화 RL. **전이 교란은 학습에 넣을 수 없었음**에도 전이 격차의 약 27%를 줄임(재시도 정책이 전이된 것으로 추정).

### 2.6 Privileged information / Asymmetric critic

| 연구 | 내용 | 확인 |
|---|---|---|
| Le Critique (2608.16739) | Privileged Value Function(critic에 과제 관련 token 단위 신호 추가)과 TETHER. 추론 과제에서 GRPO와 경쟁적이거나 우위 | [검색] |
| SWEET-RL (2503.15478) | 학습 시점 추가 정보를 보는 turn 단위 advantage 모델, DPO로 사용. ColBench(협업 프로그래밍·디자인) | [검색] |
| Informed Asymmetric Actor-Critic (2509.26000, ICML 2026) | 전체 상태가 아닌 임의의 privileged signal로도 unbiased policy gradient. 전체 상태가 항상 최선은 아님 → **어떤 signal을 쓸지가 열린 문제** | [검색] |

→ multi-turn **tool 환경**에서의 privileged critic은 확인되지 않음.

### 2.7 Memory·Context 관리

| 연구 | 내용 | 확인 |
|---|---|---|
| MEM1 (2506.15841, ICLR 2026) | 매 turn 압축된 내부 상태만 유지하는 end-to-end RL. 16-목표 multi-hop QA에서 Qwen2.5-14B 대비 성능 3.5배, 메모리 3.7배 절감 | [검색] |
| Context folding (2510.11967) | 끝난 하위 trajectory를 live context 밖으로 접음 | [검색] |
| ACM (2607.23809), Mem-T (2601.23014), Agent-compatible context mgmt (2605.30785) | 장기 과제 context 관리, 메모리 에이전트 보상 밀집화 | [검색] |
| Agentic Memory (Yu et al., 2026) | 저장·검색·갱신·요약·폐기를 tool로 두고 step-wise GRPO | [검색] |
| Context Compression 리뷰 (preprints.org) | "미래 행동에 충분한 최단 표현"을 이상적 목표로 정의하지만 학습 가능한 목적함수는 아님 | [검색] |
| Agent-state 정책 (2409.15703), AIS 이론 | belief MDP를 넘는 agent-state 정책 이론. **LLM의 손실 있는 텍스트 요약에 대한 오차 한계는 없음** | [검색] |

### 2.8 World model·Tool 시뮬레이터

| 연구 | 내용 | 확인 |
|---|---|---|
| GTM (2512.04535) | 1.5B 범용 tool 시뮬레이터로 실제 API 대체 | [검색] |
| 8B 완전 시뮬레이션 환경 (2604.17739) | tool 학습 환경 전체를 8B 모델로 시뮬레이션 | [검색] |
| AWM (Agentic World Modeling 서베이 2604.22748에서 소개) | 실행 가능한 코드·DB 기반 환경, DB 상태로 RL 보상 | [검색] |
| MCP-Cosmos (2605.09131) | tool call 시퀀스를 시뮬레이션한 뒤 계획 수정 | [검색] |
| DyMo | 자신의 function call이 만들 상태를 예측하도록 학습, BFCL V2 개선 | [검색] |
| Agents Fail to Leverage World Model as Tool (2601.03905) | 에이전트가 시뮬레이터를 잘 부르지 않고 결과를 오용. "언제 시뮬레이션할지"가 열린 문제 | [검색] |
| Self-Evolving World Models (2606.30639), PlanBench-XL (2606.22388) | ALFWorld/ScienceWorld 계획, 1,665개 tool의 장기 계획 벤치마크 | [검색] |

### 2.9 Multi-agent·사용자·Dec-POMDP

| 연구 | 내용 | 확인 |
|---|---|---|
| τ²-Bench (2506.07982) | dual-control(사용자도 tool 호출)을 Dec-POMDP로 정식화 | [검색] |
| Trajectory2Task (ACL 2026) | 비정상적 잠재 사용자 의도를 갖는 POMDP, 모호·변화·다중 분기 의도 과제 합성 | [검색] |
| Divergence-Point Preference Learning (2606.23112) | 잠재 상태 = 사용자 목표 + DB 상태 + 상호작용 상태, 종료 보상 | [검색] |
| MUA-RL (2508.18669), τ-Rec (2606.10156) | 사용자 상호작용 RL, 숨은 제약을 대화로 추론하는 추천 | [검색] |
| RL for LLM-based MAS (2605.02801) | 에이전트 생성·제거를 허용하는 동적 Dec-POMDP | [검색] |
| MARFT (2504.16129), Multi² (2606.03698), PACT (2606.16215) | 다중 에이전트 RL fine-tuning, 계층적 다중 에이전트, multi-turn tool 에이전트의 privileged trace co-training(내용 미확인) | [검색] |

### 2.10 진단 연구: belief와 행동 사이의 병목

| 연구 | 발견 | 확인 |
|---|---|---|
| Why Do LLMs Struggle in Strategic Play? (2605.00226) | 내부 belief(probe)가 말로 보고한 belief보다 정확. 그러나 belief를 행동으로 바꾸는 것이 병목이며, 해독된 belief대로 최적 행동하면 약 95% 게임에서 보상 상승. 수정 방법은 제안 안 함 | [초록] |
| Agents Explore but Agents Ignore (2604.17609) | 환경에 정답을 심어두면 발견은 하지만(AppWorld 발견률 90% 이상) 사용하지 않음(상호작용 6.3% 이하). SFT 3가지 시도 모두 실패, **RL은 미시도** | [원문] |
| STALE (2605.06527) | 새 관측이 기존 belief를 무효화하는지 아는가 평가 | [검색] |
| STT-Arena (2605.18548) | 낡은 상태로 계속 실행하는 것이 주된 실패, 현재 상태 확인 없이 같은 tool 재호출 | [검색] |
| When Agent Context Goes Stale (2610.05281) | prompt 안의 tool 출력이 낡는 문제. 기존 freshness 연구는 메모리·검색에만 있음 | [검색] |
| 내부 표현 probe 연구 (2604.19775, 2609.09448 등) | trajectory hidden state로 성공/실패·신뢰도 예측 | [검색] |

### 2.11 안전·비가역 행동

| 연구 | 내용 | 확인 |
|---|---|---|
| SafeCommit (2608.04289) | 가역 변경은 staging 후 검사, 비가역 행동은 실행 전 인증 | [검색] |
| Cordon (2606.17573) | 가역 변경은 shadow state에서, 외부로 나가는 행동은 outbox에 보류 | [검색] |
| Parallax (2604.12986) | 파괴적 행동 전 스냅샷 | [검색] |
| Tree search·보상 설계 서베이 (2510.09988) | 많은 탐색 기법이 undo를 전제 → **비가역 환경의 탐색이 핵심 미해결 과제** | [검색] |

→ 모두 런타임 시스템이며 **RL 연구는 확인되지 않음**.

### 2.12 Bayes-adaptive·Meta-RL

| 연구 | 내용 | 확인 |
|---|---|---|
| BARL / Beyond Markovian (2505.20561, ICLR 2026) | MDP 가설에 대한 posterior를 유지하는 Bayes-adaptive RL, 추론 과제 | [검색] |
| LaMer / Meta-RL Induces Exploration (2512.16848, ICLR 2026) | 에피소드 간 meta-RL로 초반 탐색·후반 활용 학습 | [검색] |
| MAXS (ACL 2026 Findings) | 다중 tool 추론의 메타 적응 탐색(비 Bayesian) | [검색] |

→ "모르는 tool의 동작을 Bayes-adaptive하게 학습"하는 연구는 확인되지 않음.

---

## 3. Graph of States (GoS) 심층 분석

> 참고: 같은 약칭의 **Graph of Skills** (2604.05333, EMNLP 2026)는 대규모 skill 라이브러리의 의존성 그래프 검색 논문으로, 별개다. 후속작으로 SE-GoS(2609.08228), SkillDAG(2606.03056)가 있다.

**Graph of States: Solving Abductive Tasks with LLMs** (2603.21250) **[원문]**

### 3.1 문제 설정

- 귀추(abductive) 추론을 $\arg\max_H P(H \mid O_{surf}, O_{deep}, K)$로 정의.
  - $O_{surf}$: 처음 주어지는 표면 증상
  - $O_{deep}$: 비용이 드는 심층 증거, ReAct tool 호출로 획득
  - $K$: 도메인 지식(의료 가이드라인 등)
- **POMDP로 명시하지는 않음**. 전이 모델, 보상, 가치 함수 정의가 없고 구조만 POMDP와 유사.

### 3.2 Belief 표현 $B = (G, S)$

- **인과 그래프 G**: 노드 = 증상·증거·가설(가설마다 신뢰도). 엣지 = derive(증상→가설), refine(거친 가설→세밀한 가설), support/refute(증거→가설).
- **상태기계 S**: 현재 조사 중인 가설의 레벨. 사실상 깊이 카운터.
- **전이 규칙**:
  - Backtracking: 상위 레벨 조상이 반대 증거로 최상위에서 밀려나면 하위 노드를 모두 가지치기하고 그 레벨로 복귀.
  - Drill-down: 상위 두 가설의 신뢰도 차이가 δ 이상이고 지지 증거가 η개 이상이면 다음 레벨로 세분화(충분히 세밀하면 종료).
  - 그 외에는 증거를 더 수집.

### 3.3 구성과 결과

- 중앙 에이전트 + 전문가 에이전트(의료: 검사·병리·영상, 장애 진단: Linux·네트워크·DB 운영자). 전문가당 검색 3회, 최대 3회 반복. 백본은 GPT-5.1.
- **학습 없음**: 프롬프팅·그래프 구조·레벨 컨트롤러로 구성된 추론 전용 아키텍처.
- 의료(DiagnosisArena 변형): 사람 평가 Match 39.86% (최고 baseline 26.09%), 비용 $0.12/건 (baseline $0.73).
- 장애 진단(Lenovo 실서비스 150건, 비공개): LLM 평가 Match 70.67% (최고 baseline 34.00%).
- 인과 그래프나 상태기계를 제거하면 Match 12.32%로 급락.

### 3.4 한계 (연구 기회와 직결)

- 오류 분석: **잘못된 행동 선택**이 지배적, **backtracking 실패 30.90%**, 장애 진단에서 **낮은 가치의 조회로 예산을 소진해 조기 종료**.
- 저자는 원인을 LLM의 도메인 지식 부족으로 보고 RAG 결합을 제안.
- 평가 한계: 사람 평가자 1명, 백본 1개, ablation은 의료만, 장애 진단 데이터 비공개.

### 3.5 지형 속 위치

| | PABU·Agent-BRACE 등 | GoS |
|---|---|---|
| belief 표현 | 학습된 텍스트 belief | 기호적 인과 그래프 + 상태기계 |
| 학습 | SFT / RL | 없음 |
| 전이 통제 | 정책이 학습 | 손으로 설계한 규칙(δ, η) |
| 초점 | 환경 상호작용 | 가설-증거 추론 (tool로 증거 수집) |

**해석**: GoS의 실패 유형(행동 선택, backtracking, 조회 가치 판단)은 모두 **정책 학습으로 고칠 수 있는 문제**다. GoS 식 구조화 belief를 유지하면서 행동 선택을 RL로 학습하는 것이 자연스러운 다음 단계이며, 이번 조사에서 이 조합은 의료 감별 진단 RL(ICLR 2026) 외에는 찾지 못했다.

---

## 4. 종합: 지형 지도

### 4.1 Belief 표현 × 학습 방식 × 환경

| belief 표현 | 학습 없음 | SFT | RL (outcome) | RL (belief 보상) |
|---|---|---|---|---|
| 암묵적 history | ReAct 등 | 다수 | AgentGym-RL, TSR, Search-R1류 | – |
| 텍스트 요약/진행 | CGDP(검색) | PABU | MEM1(QA) | ReBel(자기일관성) |
| 구조화 + 불확실성 | – | – | – | Agent-BRACE(LLM judge, 텍스트 게임) |
| 기호 그래프 | **GoS**(진단, tool) | – | – | **(비어 있음)** |
| 외부 Bayesian | BSE(고전 POMDP) | – | – | – |
| 정답 likelihood | – | – | – | IGPO, TIPS(검색 QA) |
| 진리 고정 퍼텐셜 | – | – | T3(절단에 사용, 퍼즐) | **(보상으로는 비어 있음)** |

### 4.2 붐비는 영역 (진입 시 차별화 어려움)

- 일반 multi-turn credit assignment (서베이 기준 47개 이상, 계속 추가)
- 검색 에이전트의 정보이득 보상 (IGPO, TIPS)
- RL 기반 context 관리 (MEM1, context folding, ACM, Mem-T)
- tool 시뮬레이터·world model (GTM, MCP-Cosmos, DyMo)

### 4.3 비어 있는 영역

1. 실제 tool 환경에서 **검증 가능한 belief 보상**으로 학습하는 RL
2. **정답 문자열이 없는 행동형 과제**(고치기, 설정하기)의 정보이득 보상
3. multi-turn tool 환경의 **privileged critic**
4. **belief→행동 간극**을 RL로 메우기
5. **동적 환경**에서 언제 다시 관측할지 학습
6. **비가역 행동** 하의 탐색을 RL로
7. LLM context 압축의 **belief 충분성 이론**

---

## 5. 연구 기회

### 기회 1 (최우선 추천): State-anchored belief shaping

**근거**

- tool 샌드박스(DB, 파일시스템, 컨테이너)에서는 **학습 시점에 진짜 숨은 상태 $s^*$에 접근할 수 있다**.
- 기존 belief 보상은 각각 한계가 있다: LLM judge(Agent-BRACE), 자기일관성(ReBel, 정확성 보장 없음), 정답 likelihood(IGPO/TIPS, 정답 문자열 필요).
- T3는 $\Psi(b) = -\log b(s^*)$를 정의했지만 **rollout 절단에만** 썼고, 보상으로 쓰지 않았다.

**아이디어**

에이전트가 명시적 belief $b_t$(가설 집합 위의 확률)를 유지하게 하고, 로그 점수를 퍼텐셜로 하는 shaping 보상을 준다.

$$
r_t = r_{\text{outcome}} + \gamma\,\Phi(b_{t+1}) - \Phi(b_t), \qquad \Phi(b) = \log b(s^*)
$$

- **최적 정책 보존**: potential-based shaping이므로 최적 정책을 바꾸지 않는다 (Ng et al., 1999) **[배경]**.
- **해킹 방지**: 로그 점수는 strictly proper scoring rule이라, 근거 없이 확신만 높이면 기대 보상이 오히려 줄어든다. 단순 엔트로피 감소 보상과의 핵심 차이다.
- **정보 획득 tool call 보상이 자동으로 생긴다**: 로그 읽기, 테스트 실행이 $s^*$에 대한 belief를 개선하면 그만큼 보상을 받는다.

**선행연구와의 차별점**

| | 보상 기준 | 정답 문자열 필요 | 적용 범위 |
|---|---|---|---|
| IGPO / TIPS | 정답 likelihood | 필요 | QA |
| Agent-BRACE | LLM judge | 불필요 | 텍스트 게임 |
| ReBel | 자기일관성 | 불필요 | 정확성 보장 없음 |
| T3 | 진리 고정 퍼텐셜 | 과제에 따라 | 절단에만 사용 |
| **제안** | **privileged 환경 상태** | **불필요** | **행동형 과제 (고치기, 설정하기)** |

**자연스러운 확장**

- **Privileged critic**: 같은 $s^*$를 critic 입력으로도 사용. Le Critique, SWEET-RL은 추론·협업 과제만 다뤘다. Informed AAC가 지적한 "어떤 privileged signal이 좋은가"(전체 상태, diff, 정답 위치)를 tool 환경에서 실험적으로 답할 수 있다.
- **개방형 belief 스키마**: Agent-BRACE의 future work와 일치. GoS 식 가설 그래프를 정책이 직접 생성·갱신하고 RL로 학습.

**테스트베드 후보**

| 테스트베드 | 숨은 상태 $s^*$ | 장점 |
|---|---|---|
| SWE 디버깅 | gold patch의 결함 위치(파일/함수) | 공개 데이터, 정답 위치가 명확 |
| 운영 장애 진단 | 근본 원인 컴포넌트 | GoS를 추론 전용 baseline으로 직접 비교 가능 |
| τ²-bench | 사용자 목표 + DB 상태 | 시뮬레이터가 latent를 알고 있음, POMDP 정식화 선례 |

**Baseline**: outcome-only GRPO, GoS(추론 전용), Agent-BRACE, ReBel, IGPO/TIPS(적용 가능한 경우).

**위험 요소와 대응**

- **가설 집합 밖의 $s^*$**: $-\log b(s^*)$가 발산. "기타" 버킷, 계층적 가설(부분 일치 점수), 하한 clipping 필요.
- **$s^*$의 단위 정의**: 무엇을 숨은 상태로 볼지 과제마다 다름. 결함 위치처럼 이산적이고 검증 가능한 것부터 시작.
- **belief와 행동의 분리**: belief는 좋아지는데 행동이 안 따라올 수 있음(기회 2의 문제). 기회 2를 사전 실험으로 두는 이유.

### 기회 2: Belief→행동 간극을 RL로 메우기

**근거**

- Strategic play 논문: belief는 내부에 있는데 행동으로 이어지지 않음(최적 행동 시 95% 게임에서 개선).
- Agents explore but ignore: 발견하고도 사용하지 않음. SFT 3종 실패, RL 미시도. 저자는 RL도 SFT처럼 tool 출력 편향을 강화할 것이라 추측만 함.

**실험 설계**

- 평가: solution-injection 벤치마크 그대로 사용 (discovery@k vs. interaction@k vs. pass@k).
- 비교: (a) outcome-only GRPO, (b) Agent-BRACE 식 belief/policy 분리, (c) 관측 반영(observation uptake) 보상 추가.
- 질문: RL이 "무시" 현상을 악화시키는가 완화시키는가? belief를 명시하면 간극이 줄어드는가?

**장점**: 인프라 부담이 작고, 부정적 결과도 기여가 됨. 기회 1의 사전 실험 역할.

### 기회 3: 동적 환경의 재관측 정책

**근거**

- RobustBench-TC: 전이 교란이 가장 아프고(−30%), 정적 증강으로는 학습 불가.
- STALE, STT-Arena, Stale context 논문: 문제만 보였고 학습된 해법 없음. "언제 다시 볼지"에 대한 구체적 정책(주기, 쓰기 후 트리거, 행동 전 검증)이 없음.

**아이디어**

- POMDP 관점에서 시간이 지나면 belief 불확실성이 커지므로, 재관측은 정보 가치(VoI) 판단 문제다.
- belief 항목에 "신선도(age)"를 두고 재관측 시점을 RL로 학습.
- 환경: 에피소드 중 배경 프로세스나 다른 행위자가 상태를 바꾸는 시뮬레이터(AWM 같은 코드 기반 환경 확장).

**평가**: 인프라 비용이 크지만 경쟁이 적다.

### 기회 4: 비가역 행동 하의 탐색

**근거**: SafeCommit, Cordon, Parallax는 모두 런타임 시스템이고 RL이 없다. 탐색 서베이도 비가역 환경의 탐색을 핵심 미해결 과제로 꼽는다.

**아이디어**

- 불확실성이 높은 상태에서 비가역 행동을 하면 페널티를 주는 constrained RL (belief 엔트로피를 조건으로).
- 학습 시 Cordon 식 shadow state로 안전하게 탐색.
- 기회 1의 belief를 그대로 재사용 → 후속 연구로 적합.

### 기회 5: Belief 충분성 이론과 context 압축

**근거**

- AIS 이론의 오차 한계는 LLM의 손실 있는 텍스트 요약에 적용되지 않음.
- BSE(원본 history를 보면 Markov성 깨짐), Agent-BRACE(요약이 불확실성을 붕괴) 주장이 이론적으로 정리되지 않음.

**아이디어**

- 압축된 belief로 다음 관측·보상을 예측하는 보조 목적(predictive-state objective)으로 압축기 학습.
- "belief 충분성" 측정 지표 제안.

**평가**: 이론 기여가 크지만 실험만으로는 MEM1 계열과 차별화가 어려움.

### 우선순위 요약

| 기회 | 신규성 | 실현 난이도 | 영향 | 우선순위 |
|---|---|---|---|---|
| 1. State-anchored belief shaping | 높음 | 중 | 높음 | **1 (본 연구)** |
| 2. Belief→행동 간극 RL | 중~높음 | 낮음 | 중 | **사전 실험** |
| 3. 동적 환경 재관측 | 높음 | 높음 (인프라) | 높음 | 확장 |
| 4. 비가역 탐색 | 높음 | 중~높음 | 높음 (산업) | 확장 |
| 5. Belief 충분성 이론 | 중 | 중 (이론) | 중 | 선택 |

---

## 6. 추천 로드맵

| 단계 | 기간(예상) | 내용 | 산출물 |
|---|---|---|---|
| 1. 진단 | 1~2개월 | 기회 2: tool 환경(SWE 또는 τ²)에서 belief→행동 간극 재현, outcome RL vs belief 분리 비교 | 진단 결과, 워크숍 논문 가능 |
| 2. 본 연구 | 3~6개월 | 기회 1: 결함 위치 추정 또는 장애 진단에 state-anchored shaping + privileged critic | 본 학회 논문 |
| 3. 확장 | 이후 | 같은 belief 구조로 기회 3·4 (동적·비가역 환경) | 후속 논문, 환경/벤치마크 공개 |

착수 전 할 일: Semantic Scholar·OpenReview에서 "privileged belief reward", "proper scoring rule belief agent", "log score shaping LLM agent" 키워드로 신규성 재확인.

---

## 7. 한계와 검증 상태

- **원문 확인**: GoS, PABU, T3, Agent-BRACE, Agents Explore but Ignore.
- **초록 확인**: Belief-State Engine, RobustBench-TC, CGDP, ReBel, Strategic play.
- **검색 요약·2차 자료**: 그 외 전부. 특히 IGPO, ToolPO, Calibrate-Then-Act 세부는 2차 자료 기준.
- **수치 불일치**: PABU 평균 81.0%(단순 평균 약 69.7%), Agent-BRACE ablation(75.3% vs 60.8%). 인용 시 원문 재확인 필요.
- **신규성 판단**: "이번 검색에서 찾지 못했다"는 의미이며 부재의 보장이 아님.
- **시점**: 조사 보조의 학습 데이터는 2026년 6월까지이고, 이후 논문은 검색으로만 확인. 2026년 7~10월 논문 일부는 누락됐을 수 있음.
- **배경지식 표기 항목**(Search-R1, ReTool, ToolRL, ToRL, Ng et al. 1999)은 이번 조사에서 별도 확인하지 않음.

---

## 부록: 참고문헌

### 서베이·프레임
- [The Landscape of Agentic RL for LLMs: A Survey (2509.02547)](https://arxiv.org/html/2509.02547v5)
- [A Practitioner's Guide to Multi-turn Agentic RL (2510.01132)](https://arxiv.org/pdf/2510.01132)
- [From Reasoning to Agentic: Credit Assignment in RL for LLMs (2604.09459)](https://arxiv.org/html/2604.09459v1) · [Awesome 리스트](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- [Adaptation of Agentic AI (2512.16301)](https://arxiv.org/html/2512.16301v3)
- [RL-based Agentic Search 서베이 (2510.16724)](https://arxiv.org/pdf/2510.16724)
- [Agent System and Harness Design 서베이 (2606.20683)](https://arxiv.org/pdf/2606.20683)
- [Memory for Autonomous LLM Agents (2603.07670)](https://arxiv.org/html/2603.07670v1)
- [The Horizon Gap (2608.06663)](https://arxiv.org/pdf/2608.06663)
- [Agentic World Modeling 서베이 (2604.22748)](https://arxiv.org/pdf/2604.22748)
- [The Evolution of Tool Use in LLM Agents (2603.22862)](https://arxiv.org/pdf/2603.22862)

### 다중 턴 RL 인프라
- [AgentGym-RL (2509.08755)](https://arxiv.org/pdf/2509.08755)
- [TSR (2602.11767)](https://arxiv.org/pdf/2602.11767)
- [VerlTool (2509.01055)](https://arxiv.org/pdf/2509.01055)
- [Tool-R1 (2509.12867)](https://arxiv.org/pdf/2509.12867)
- [ToolBrain (2510.00023)](https://arxiv.org/pdf/2510.00023)
- [ToolPO / DeepAgent (2차 요약)](https://api.emergentmind.com/topics/toolpo)

### Belief 명시화
- [Graph of States (2603.21250)](https://arxiv.org/abs/2603.21250)
- [PABU (2602.09138)](https://arxiv.org/html/2602.09138v1)
- [Agent-BRACE (2605.11436)](https://arxiv.org/html/2605.11436)
- [ReBel: Rewarding Beliefs, Not Actions (2605.20061)](https://arxiv.org/abs/2605.20061)
- [Reducing Belief Deviation / T3 (2510.12264)](https://arxiv.org/html/2510.12264)
- [The Context Gathering Decision Process (2605.07042)](https://arxiv.org/abs/2605.07042)
- [Belief-State Engine (2609.10036)](https://arxiv.org/abs/2609.10036)
- [VAGEN (2510.16907)](https://arxiv.org/html/2510.16907v1)
- [LUMINA (2601.16649)](https://arxiv.org/pdf/2601.16649)
- [Reward-Driven LLM Agent Workflows (2607.17038)](https://arxiv.org/html/2607.17038v1)
- [POMDP 기반 Agentic AI 검증 (2606.17383)](https://arxiv.org/html/2606.17383)
- [Hypothesis-driven Clinical Decision Making RL (ICLR 2026 노트)](https://en.papernotes.org/ICLR2026/llm_agent/language_agents_for_hypothesis-driven_clinical_decision_making_with_reinforcemen/)
- [AgentTether (2607.06273)](https://arxiv.org/html/2607.06273)
- [Graph-based RL for Drift Diagnosis (2608.14109)](https://arxiv.org/pdf/2608.14109)
- [Belief-State RWKV (2604.09671)](https://arxiv.org/pdf/2604.09671)

### 정보 획득·비용 인식
- [TIPS (2603.22293)](https://arxiv.org/html/2603.22293)
- [IGPO (2차 요약)](https://lacuna.tiptreesystems.com/work/information-gain-based-policy-optimization-a-simple-and-effective-approach-for/wrk_9dc317dcb42668c4a5a3feccc36ec13a)
- [Calibrate-Then-Act (2602.16699)](https://arxiv.org/html/2602.16699v2)
- [Uncertainty-Aware Clarification with Information Gain (2606.03135)](https://arxiv.org/html/2606.03135v1)
- [Uncertainty Quantification for LLM Agents (2609.07395)](https://arxiv.org/pdf/2609.07395)
- [ESearch-R1 (2512.18571)](https://arxiv.org/html/2512.18571v1)
- [TRACE (2607.13988)](https://arxiv.org/pdf/2607.13988)
- [CS 224R 리포트](https://cs224r.stanford.edu/projects/pdfs/Vikram%20Garrett%20Srinivasan%20submission_416281263/cs-224r-report.pdf)

### 강건성·Privileged information
- [When Simulation Lies / RobustBench-TC (2605.11928)](https://arxiv.org/abs/2605.11928)
- [Le Critique (2608.16739)](https://arxiv.org/pdf/2608.16739)
- [SWEET-RL (2503.15478)](https://arxiv.org/pdf/2503.15478)
- [Informed Asymmetric Actor-Critic (2509.26000)](https://arxiv.org/html/2509.26000v3)

### Memory·Context
- [MEM1 (2506.15841)](https://arxiv.org/html/2506.15841v2)
- [ACM (2607.23809)](https://arxiv.org/pdf/2607.23809)
- [Mem-T (2601.23014)](https://arxiv.org/pdf/2601.23014)
- [Agent-Compatible Context Management (2605.30785)](https://arxiv.org/pdf/2605.30785)
- [Context Compression 리뷰 (preprints.org)](https://www.preprints.org/frontend/manuscript/098fda1d1490b8885d002521dbc08afa/download_pub)
- [Agent-state based policies in POMDPs (2409.15703)](https://arxiv.org/html/2409.15703)

### World model·Tool 시뮬레이터
- [GTM (2512.04535)](https://arxiv.org/html/2512.04535)
- [Environments Fully Simulated by an 8B LM (2604.17739)](https://arxiv.org/pdf/2604.17739)
- [MCP-Cosmos (2605.09131)](https://arxiv.org/pdf/2605.09131)
- [Current Agents Fail to Leverage World Model as Tool (2601.03905)](https://arxiv.org/html/2601.03905)
- [Self-Evolving World Models (2606.30639)](https://arxiv.org/html/2606.30639v1)
- [PlanBench-XL (2606.22388)](https://arxiv.org/html/2606.22388v1)

### Multi-agent·사용자
- [τ²-Bench (2506.07982)](https://arxiv.org/pdf/2506.07982)
- [Trajectory2Task (ACL 2026)](https://aclanthology.org/2026.acl-long.2037.pdf)
- [Divergence-Point Preference Learning (2606.23112)](https://arxiv.org/pdf/2606.23112)
- [MUA-RL (2508.18669)](https://arxiv.org/pdf/2508.18669)
- [τ-Rec (2606.10156)](https://arxiv.org/pdf/2606.10156)
- [RL for LLM-based Multi-Agent Systems (2605.02801)](https://arxiv.org/html/2605.02801v1)
- [MARFT (2504.16129)](https://arxiv.org/pdf/2504.16129)
- [Multi² (2606.03698)](https://arxiv.org/pdf/2606.03698)
- [PACT (2606.16215)](https://arxiv.org/pdf/2606.16215)

### 진단 연구
- [Why Do LLMs Struggle in Strategic Play? (2605.00226)](https://arxiv.org/abs/2605.00226)
- [Agents Explore but Agents Ignore (2604.17609)](https://arxiv.org/html/2604.17609v1)
- [STALE (2605.06527)](https://arxiv.org/html/2605.06527v1)
- [STT-Arena (2605.18548)](https://arxiv.org/pdf/2605.18548)
- [When Agent Context Goes Stale (2610.05281)](https://arxiv.org/html/2610.05281)
- [Conformal Interpretability of Temporal Concepts (2604.19775)](https://arxiv.org/html/2604.19775v2)
- [Do Agents Know When They Succeed? (2609.09448)](https://arxiv.org/pdf/2609.09448)
- [Leveraging Trajectory Graphs for Error Diagnosis (2607.27443)](https://arxiv.org/pdf/2607.27443)

### 안전·비가역
- [SafeCommit (2608.04289)](https://arxiv.org/html/2608.04289)
- [Cordon (2606.17573)](https://arxiv.org/html/2606.17573v1)
- [Parallax (2604.12986)](https://arxiv.org/html/2604.12986v1)
- [Tree Search & Reward Design 서베이 (2510.09988)](https://arxiv.org/pdf/2510.09988)

### Bayes-adaptive·Meta-RL
- [Beyond Markovian / BARL (2505.20561)](https://arxiv.org/pdf/2505.20561)
- [Meta-RL Induces Exploration in Language Agents (2512.16848)](https://arxiv.org/html/2512.16848v2)
- [MAXS (ACL 2026 Findings)](https://aclanthology.org/2026.findings-acl.670/)

### 혼동 주의: Graph of Skills
- [Graph-of-Skills (2604.05333)](https://arxiv.org/pdf/2604.05333) · [SE-GoS (2609.08228)](https://arxiv.org/pdf/2609.08228) · [SkillDAG (2606.03056)](https://huggingface.co/papers/2606.03056.md)
