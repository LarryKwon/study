# 논문 주제 후보 탐색 (2026-10-28)

- 목적: 첫 1저자 논문 주제 선정
- 조건: 혼자 진행, 계산 자원 충분, 목표 NeurIPS 2027 (2027년 5월 마감 추정) / ICLR 2028 (2027년 9~10월), ACL·EMNLP 2027도 가능
- 범위: POMDP·agentic RL, 멀티모달 post-training, abductive reasoning, 암호화 RAG, RelBench, LLM 추천
- 방법: 5개 영역을 하위 조사로 병렬 탐색 + 기존 POMDP·agent 지형 조사([agentic-rl-pomdp-research-landscape.md](agentic-rl-pomdp-research-landscape.md)) 종합
- 검증 수준: **[페이지]** 초록/HTML 확인 · **[검색]** 검색 결과 요약만 · **[미확인]** 기억이나 추정. 수치는 인용 전 원문 재확인 필요. "못 찾았다"는 "없다"가 아님.

---

## 0. 요약

1. 다섯 영역에서 **같은 문제가 반복**된다: 정보를 더 모을지 지금 결정할지 판단하는 에이전트가 belief를 보정하지 못하고, 결과 보상만 주는 RL은 이를 고치지 못하거나 악화시킨다.
2. **1순위 후보 A:** 정보 탐색 에이전트를 위한 proper-scoring belief 보상 (도메인 일반 방법론, 진단 + 추천, 확장으로 멀티모달). NeurIPS 2027.
3. **대안:** C (비공개 검색을 쓰는 에이전트의 궤적 단위 정보 누출, 경쟁 가장 낮음), B (RelBench 위 RL 학습 관계형 에이전트).
4. **선택 방법:** 더 읽지 말고 A와 C를 1주씩 pilot으로 대결시켜 신호가 강한 쪽을 고른다.

---

## 1. 일정 (2026-10-28 기준, 공식 발표 전 추정 포함)

| 학회 | 열리는 시기 | 2027년 판 마감 | 비고 |
|---|---|---|---|
| ICLR | 4~5월 | 2026년 9~10월 *(통상)* | 2027년 판은 닫힘. **ICLR 2028 = 2027년 9~10월 마감** |
| CVPR | 6월 | 2026년 11월 중순 *(통상)* | 2027년 판은 곧 닫힘. CVPR 2028 = 2027년 11월 |
| ICML | 7월 | 2027년 1월 22일경 *(추정)* | 혼자 첫 논문으로는 빠듯함 |
| ACL | 8월 (2027 교토) | 2027년 1~2월 ARR *(추정)* | 빠듯함 |
| NeurIPS | 12월 | **2027년 5월 21일경** *(추정)* | **본 목표** |
| EMNLP | 11월경 | 2027년 5~6월 ARR *(통상)* | 대안 |

```
2026.11 ─ 2027.01 ─────── 2027.05 ─────── 2027.09~10
 pilot     (선택) ICML/ACL   NeurIPS 2027     ICLR 2028
           중간 점검         본 목표          재제출 또는 후속 논문
```

---

## 2. 영역을 가로지르는 공통 문제

> **정보를 더 모을지 지금 결정할지 판단하는 에이전트가 자기 확신(belief)을 제대로 보정하지 못한다. 결과 보상만 주는 RL은 이를 고치지 못하거나 오히려 악화시킨다.**

| 영역 | 나타난 모습 | 근거 |
|---|---|---|
| 추천 | 숨은 조건을 묻지 않거나, 추천 없이 끝냄 | τ-Rec: 숨은 조건 과제에서 pass^1 0.846 → 0.200, 일부 모델은 에피소드 43~67%를 추천 없이 종료 [페이지] |
| 멀티모달 | 높은 확신으로 너무 일찍 그만 봄 | SATE-Bench: 조기 중단 + 과신. "더 보기" 보상 시 탐색 0.2% → 72.8%, 정확도 30.4% → 17.9% [페이지] |
| abductive | 가설 확률이 보정되지 않음 | LLM-as-Investigator: 가중치가 휴리스틱이고 보정되지 않음을 한계로 명시 [페이지]. GoS·PoS는 추론 전용 [페이지] |
| POMDP·agent | 결과 보상 RL이 belief 추적을 망가뜨림 | AREW (ICML 2026): 행동 선택과 belief 추적이 모두 무너짐 [페이지] |
| 코드 에이전트 | 과포함 | A²Agent: 모듈 recall 63.42 vs precision 42.56 [원문 표] |

---

## 3. 후보 비교

| | 후보 | 경쟁 | 실현성 (혼자) | 관심사 결합 | 적합 학회 |
|---|---|---|---|---|---|
| **A** | 정보 탐색 에이전트를 위한 proper-scoring belief 보상 | 중간 | 높음 | abductive + 추천 + POMDP (+ 멀티모달) | NeurIPS 2027 |
| B | RelBench 위 RL 학습 관계형 데이터 분석·추천 에이전트 | 중간 | 중간 (환경 구축 부담) | RelBench + 추천 + agent | NeurIPS / KDD |
| C | 비공개 검색을 쓰는 에이전트의 궤적 단위 정보 누출 측정·억제 | 낮음~중간 | 높음 | 암호화 RAG + agent RL | NeurIPS / ICLR |
| D | 가설 단위 과정 보상을 쓰는 시각적 귀추 RL | 낮음~중간 | 중간 | 멀티모달 + abductive | NeurIPS / ICLR 2028 |
| E | RelBench 누출·LLM 암기 감사 | 낮음~중간 | 높음 | RelBench | Datasets & Benchmarks |
| F | 대화형 추천 RL의 사용자 시뮬레이터 악용 분석 | 중간 | 높음 | 추천 + RL | EMNLP / RecSys |

---

## 4. 1순위 후보 A 상세

**한 문장 주장 (가안):** 정보를 모으며 결정하는 에이전트에게 매 턴 가설 분포를 말하게 하고, 정답에 대한 proper scoring rule(로그 점수, Brier 등. 정직한 확률을 말할 때 기대 점수가 최대)로 보상하면, 도메인에 관계없이 정확도·질문 수·보정이 함께 좋아진다.

### 가까운 연구와의 차이

| 연구 | 하는 것 | A와의 차이 |
|---|---|---|
| RLCR | 보정 보상 | 단일 턴뿐 [검색]. 다중 턴 버전 못 찾음 |
| IGPO / TIPS | 정답 likelihood 정보이득 | 정답 문자열 필요, 모델이 말한 belief를 채점하지 않음 |
| Ask to Be Sure (2608.15949) | 후보 엔트로피 감소 + DPO | 엔트로피는 과신으로 해킹 가능. proper score는 정답 기준이라 불가 |
| Agent-BRACE | belief 보상 | LLM judge 채점 vs 검증 가능한 점수 |
| T3 / AREW | belief 붕괴 진단·절단 | belief를 직접 보상하지 않음 |
| GoS / PoS | 명시적 belief | 학습 없음 |

### 도메인 구성
1. **핵심: 대화형 진단 (DDXPlus).** 구조화된 환자 기록으로 질문에 프로그램이 정확히 답하는 환경을 만들 수 있다. LLM 시뮬레이터 불필요(싸고, 시뮬레이터 악용 문제 회피). 정답 병명으로 proper score 계산 가능. 환자별 감별진단 확률 제공 여부는 [미확인]. 분포 밖 평가로 AgentClinic.
2. **둘째: 숨은 조건 추천 (τ-Rec 방식).** 같은 텍스트 학습 스택 재사용.
3. **확장 (ICLR 2028):** 능동 지각(SATE-Bench) 또는 코드 위치 찾기.

### 2주 pilot
- DDXPlus를 POMDP로 변환 (증상 질문마다 비용).
- Qwen3-8B 다중 턴 GRPO: **결과 보상만** vs **+ 턴별 proper score** (+ 정보이득, AREW 비교 가능).
- 측정: 정확도-질문 수 Pareto, ECE, 조기 확정 비율, self-locking 비율.
- **결정 기준 (미리 고정):** 예) 같은 질문 수에서 정확도 +3점 또는 ECE 절반 → 진행.

### 위험
- 진단 RL은 붐비는 중. GoS/PoS 그룹(PoS 2026-10-01 발표)이 학습 버전을 낼 가능성 → 속도 중요. 차별화: shaping 이론 + 두 번째 도메인.
- 신규성 재확인 키워드: "multi-turn calibration reward", "Brier reward agent", "proper scoring rule RL information seeking".

---

## 5. 대안

- **C (궤적 단위 정보 누출):** 비공개 검색 논문(Pointing the Way)이 "여러 라운드에 걸친 추론은 명시적 누출로 남는다"고 인정하고, 에이전트 맥락은 아무도 안 다룸. 암호 프로토콜을 블랙박스 누출 함수로 다룰 수 있어 암호 지식이 적게 필요. 위협 모델 설계의 설득력이 관건.
- **B (RelBench RL 에이전트):** 관계형 특징공학 에이전트는 모두 프롬프팅 기반, RL 학습 사례 없음. 추천 과제(10개)는 거의 방치. 엔지니어링 부담과 Stanford/Kumo·RelAgent 그룹과의 경쟁이 위험.

## 6. 고르는 방법: pilot 대결
- A: 위 DDXPlus pilot (1주 축소판).
- C: Search-R1 에이전트에 누출 신호(SimHash 코드 + DP 잡음, 클러스터 ID, 라운드 수)를 노출하는 모의 서버 → 공격자를 학습시켜 hop 수에 따른 답 추론 정확도 측정.
- 신호가 강한 쪽 선택. 결정은 독서가 아니라 실험으로.

---

## 7. 영역별 조사 원본

### 7.1 LLM 추천 시스템

**최신 동향**
- **τ-Rec** (2606.10156, RecSys'26 resource): 검증 가능한 보상의 영화 추천 벤치마크. 사용자 조건을 자발 제시 / 물으면 공개 / 숨김으로 태깅. 60개 과제, TMDB 영화 153편. 최고 모델 pass^1 약 57%, pass^4 35%. 평가만, 학습 없음. [페이지]
- **Ask to Be Sure** (2608.15949, CIKM'26): 대화 턴마다 후보 아이템 엔트로피 감소량을 보상. INSPIRED·ReDial에서 SFT + DPO. [페이지]
- **Rank-GRPO / ConvRec-R1** (2510.20150, v5 2026.02): 목록의 순위 단위 GRPO, Reddit-v2, 코드 공개. 선호 질문 안 함. [페이지]
- **RecThinker** (2603.09843), **AgenticRec** (2603.21613): tool 호출 추천 에이전트, SFT → RL. 사용자와 대화 없음. [페이지]
- **SIDReasoner** (2603.23183, KDD'26), **Why Thinking Hurts** (2602.16587): semantic ID 위 추론. 후자는 CoT가 OpenOneRec 품질을 최대 25% 낮춤. [페이지]
- **시뮬레이터 충실도:** ConvApparel (2602.16938) 모든 시뮬레이터에 현실성 격차. Sim2Real (2603.11245) 시뮬레이터가 "과도하게 협조적"이라 성공률 부풀림. Simulator collapse (2608.12253) 단일 고정 시뮬레이터는 악용 정책을 낳음(τ²-bench, 설득, CooperBench 대상, 추천 아님). [페이지]

**빈틈**
- **G1. 숨은 조건에 대한 명시적 belief + RL.** 검증 가능한 숨은 조건 보상으로 RL 학습하며 belief를 추적한 대화형 추천은 없음.
  - 근거: τ-Rec은 평가만, future work는 카탈로그 규모·새 도메인뿐. DeepSeek V4 Flash 숨은 조건 pass^1 0.200 vs 자발 제시 0.846. 일부 모델 43~67% 추천 없이 종료. Ask to Be Sure는 엔트로피 + DPO. BED-LLM (2508.21184, ICLR'26)은 추론 시점만. PUMA (2605.24647)는 의료 대화만. [페이지]
  - Pilot: 더 큰 TMDB/MovieLens 카탈로그로 τ-Rec형 과제 약 2k개 생성 → Qwen3-8B GRPO, 보상 = 조건 점수 × 정책 점수 → 매 턴 예측 조건 집합을 내고 정답 집합 F1로 채점하는 변형과 비교. 공개 태그별 pass^k, 질문 수, 유효 아이템이 없는 과제에서의 기권 측정.
  - 코드: github.com/nbharaths/tau-rec [페이지]. 경쟁: 중간(τ²-bench RL은 붐빔 — CoVe 2603.01940, LOGIGEN 2603.00540). POMDP: 강함. 학회: NeurIPS 2027.
- **G2. RL 추천기의 사용자 시뮬레이터 악용.** 시뮬레이터의 선호 공개 방식을 악용하는지 측정한 연구 없음.
  - 근거: Simulator collapse는 추천 미포함. SMTPO (2604.03671) 시뮬레이터 피드백이 실제 관심과 어긋날 수 있음. τ-Rec은 GPT-5 mini 시뮬레이터 하나. 시뮬레이터 누출이 ReDial·OpenDialKG 결과를 부풀림 (2403.16416) [검색]. COMPASS (2510.07043) 공개 시점을 설계로 정함 [검색].
  - Pilot: G1 정책을 다른 4~5개 시뮬레이터(모델 계열, temperature, ConvApparel로 튜닝한 것)에 테스트 → 전이 하락과 악용 유형 분류 → 시뮬레이터 집단 학습으로 완화.
  - 데이터: ConvApparel 사람-AI 대화 4k, SCOPE (2608.12253, 공개 언급). 경쟁: 중간. 학회: ACL/EMNLP 2027, RecSys.
- **G3. semantic-ID 접두 트리 위 질문 선택.** semantic-ID 계층을 가설 공간으로 삼아 기대 정보이득으로 질문을 고른 생성형 대화 추천 없음.
  - 근거: GCRS (2605.21987) 질문 언급 없음. Ask to Be Sure·BED-LLM은 텍스트 후보. [페이지]
  - Pilot: Amazon Reviews로 RQ-VAE semantic ID → 빔 서치 확률로 1·2단계 접두 posterior → 확률 질량을 가장 잘 나누는 속성 질문 → 프롬프트 질문, 엔트로피 선택, BED-LLM과 Hit@10 대 턴 수 비교.
  - 경쟁: 낮음~중간. 학회: ICLR 2028, KDD.
- **G4. 이력에서 숨은 조건 추론(귀추) + 선호 변화 시 수정.**
  - 근거: IntuRec (2606.27684), RecGPT-V3 (2607.15591)는 잠재 토큰으로 압축 [검색]. AcCoRD (2608.27818) 상호작용 중 선호 변화에 프런티어 모델이 약함, 쇼핑·여행 대상, 학습 없음 [페이지].
  - Pilot: τ-Rec 과제의 숨은 조건과 일치하는 시청 이력 + 대화 중 뒤집히는 조건 추가 → 추론된 조건의 precision·recall, 절약된 질문 수, pass^1.
  - 경쟁: 낮음. 학회: EMNLP 2027, ICLR 2028.
- **G5 (약함). RelBench 추천 과제에 LLM·semantic-ID 추천기.** RelBench v2 (2602.12606)에 추천 과제 있으나 LLM baseline 없음 [페이지].
- **피할 영역:** semantic ID 스케일링 법칙 (산업 위주, 결과 혼재: 2509.25522 포화 [검색], 2609.24430 비단조 [페이지]).
- 조사자 제안: G1 + G2를 NeurIPS 2027 한 편으로, G4는 확장.

### 7.2 Abductive reasoning

**최신 동향**
- **GoS** (2603.21250): 추론 전용, GPT-5.1. DiagnosisArena 138건 + 비공개 Lenovo 장애 150건. "의료 진단과 장애 진단으로 범위 한정". [페이지]
- **PoS / Beyond Memory** (2610.01415, 2026-10-01, GoS와 같은 그룹): 명시적 belief state, 여전히 추론 전용. RCA-100, ClinDiag. future work는 belief 유지 비용 절감. 이 그룹이 빠르게 움직임. [페이지]
- **CEDAR-GRPO** (2608.14791): LLM 채점 보상(증거 커버리지, 추론 방향) GRPO, 4~8B, 11개 보류 과제, 정답만 GRPO 대비 +2.7점. 한계: "벤치마크형 설정과 대부분 닫힌 형태의 정답". 코드: github.com/cedar-grpo/cedar-grpo. [페이지]
- **Information self-locking / AREW** (2603.12109, ICML 2026): 능동 추론에서 결과 보상만 쓰면 행동 선택과 belief 추적이 무너짐. 코드: github.com/unimpor/T3. [페이지]
- **대화형 의료 RL:** DiagGym (2510.24654, 코드 못 찾음), MedExAgent (2605.07058, DDxPlus·PMC-Patients·AgentClinic에서 POMDP, 코드 공개, belief는 암묵적으로 보임), RAGES (2607.02983, 감별진단 순위 보상, 코드 언급 없음). [페이지]
- **벤치마크:** 서베이 2604.08016 (DDXPlus 후보 선택 Hit@3 98.7% vs 생성 63.0), OpenRCA 2.0 (2606.27154, 인과 경로 라벨 500건, 채택 후 공개), ORCA-bench (2607.28545, 1,079과제, Harbor hub 공개, 최고 25.3%/10.0%), Elenchos (2607.12733), SemEval-2026 Task 12 (2603.21720, GitHub 공개). [페이지]

**빈틈**
- **G1. 대화형 귀추를 위한 belief 채점 RL.** 매 턴 가설 분포를 proper scoring rule + 그 belief 위의 정보이득으로 보상한 연구 없음.
  - 근거: GoS·PoS 추론 전용, MedExAgent belief 암묵적, RAGES 확률 없는 순위 [페이지]. LLM-as-Investigator (2606.13220): "가중치 휴리스틱, 보정 안 됨", future work에 보정된 불확실성과 적응적 중단 [페이지]. RLCR은 단일 턴 [검색].
  - Pilot: DDXPlus → 비용 있는 증상 질문 POMDP, Qwen3-8B 다중 턴 GRPO. 결과만 / + 턴별 Brier / + 정보이득 / AREW 비교. 정확도, 질문 수, ECE, self-locking 비율. DDXPlus가 환자별 감별진단 확률 제공 여부 [미확인].
  - 데이터·코드: DDXPlus, AgentClinic(분포 밖), MedExAgent 코드, T3/AREW 코드. 경쟁: 중간. POMDP: 직접(belief-MDP).
- **G2. 합성 → 실제 전이.** 절차 생성한 숨은 원인 POMDP에서 RL한 것이 임상 진단·RCA로 전이되는지 미검증.
  - 근거: CEDAR 전이는 단일 턴만, RAGES는 의료 밖 미검증, GoS "고품질 벤치마크 확보가 큰 과제" [페이지].
  - Pilot: 잡음 있고 비용 드는 검사를 갖는 무작위 결함 DAG 생성기(의료·운영 스타일 텍스트) → 8B GRPO → DDXPlus, AgentClinic, RCAEval 부분집합 zero-shot.
  - 경쟁: 중간 (에이전트 RL용 합성 환경은 인기, 귀추용은 아님).
- **G3. 인과 경로 과정 보상을 쓰는 RCA RL 에이전트.** RCA는 여전히 대부분 프롬프팅.
  - 근거: EviRCA (2609.19825) 병목은 "이미 드러난 증거 판단", 학습 없음. OpsHarness (2608.25661) RL 없음. OpenRCA 2.0 과정 보상 모델 학습 미루고 학습/평가 분할 없음 [페이지]. ThinkFL (2504.18776) [검색].
  - Pilot: RCAEval 또는 OpenRCA + 읽기 전용 텔레메트리 tool, 보상 = 근본 원인 정답 + 주장한 엣지와 호출 그래프의 일관성. 8B RL vs 프런티어 모델 프롬프팅.
  - 경쟁: 낮음~중간. 위험: 텔레메트리 엔지니어링 (68GB [검색]).
- **G4. 능동 멀티모달 귀추.** 어떤 프레임·영역을 볼지 고르는 RL 에이전트 없음.
  - 근거: AbductiveMLLM (2601.02771, AAAI 2026) 수동적, RL 없음 [페이지]. MMR-Life (2603.02024) GRPO 수확 체감 [검색].
  - Pilot: VAR (2203.14040)이나 Sherlock에 프레임/크롭 tool, 7B VLM GRPO + 보기당 비용. 코드: github.com/ChangPtR/AbdMLLM.
  - 경쟁: 중간.
- 조사자 제안: G1을 NeurIPS 2027 본체로, G2를 확장·전이 절로. 2~3개월 뒤 GoS/PoS의 학습 후속작 확인 권장. 참고: DIVA (2610.32181, 학습 없는 중단).

### 7.3 멀티모달 post-training

**최신 동향**
- **The Illusion of Visual Tool-Use** (2608.06270, EMNLP-Findings 2026): "이미지로 생각하기"를 인과적으로 감사 → 반환된 크롭이 답에 인과 효과 없는 경우가 많고, 이득은 일부 "보정된" rollout에서 나옴. [페이지]
- **TextCall** (2608.09682): 반환 이미지를 placeholder로 바꿔도 SFT·RL에서 동등 이상. 픽셀이 정말 필요한 과제는 "열린 방향". [페이지]
- **Starve to Perceive** (2605.18603): 시점당 시각 토큰 제한으로 "게으른 지각" 해결, 평균 약 5% 상대 개선. [페이지]
- **Diversity Over Frequency** (2606.00096, ICML 2026): RL에서의 "tool 사용 붕괴", 엔트로피 정규화로 해결. [페이지]
- **Multimodal Reward Hacking in RL** (2607.09492): 결과 보상만 쓰면 보상 해킹 48.1%. VLM judge는 검증이 믿을 만할 때만 도움. [페이지]
- **VAGEN** (2510.16907, NeurIPS 2025): 다중 턴 VLM 에이전트 POMDP, world-model 보상 + Bi-Level GAE, 코드 공개. [페이지]
- **붐비는 영역:** 반사실·blind-image grounding 보상(DeFacto 2509.20912, VIGIL 2606.26387, Evidence-RL 2608.08021), 시각 망각 방지 RL(Remember-R1 2608.01314), 픽셀 필요성 벤치마크(VisualNeedle 2605.26380), 생성형 멀티모달 보상 모델(DeltaRubric 2605.09269, MJ1 2603.07990), GUI 에이전트 온라인 RL(서베이 2604.27955).

**빈틈**
- **G1. 다중 턴 능동 지각에서 보정된 "그만 보기" 학습.** 보정된 belief와 정보 가치로 "다시 볼지, 답할지"를 학습하는 RL 없음.
  - 근거: SATE-Bench (2610.34149, 2026-10-25) 조기 중단 + 높은 확신 보고. Qwen2.5-VL-3B GRPO에서 시점 탐색 보상이 탐색 0.2% → 72.8%, 능동 정확도 30.4% → 17.9%, 학습은 단일 턴 결정으로 단순화 [페이지]. GUESS-BENCH (2510.15421, ACL 2026) "부적시 결정" [검색]. ESearch-R1 (2512.18571) 비용 인식이지만 belief·보정 중단 없음 [페이지].
  - Pilot: Qwen2.5-VL-3B/7B 다중 턴 GRPO, SATE-Bench 능동 분할 또는 토큰 예산을 줄인 V*/HR-Bench. 매 턴 답 분포 출력, 보상 = 정답 + Brier 항 − c × 보기 횟수. step 페널티·시점 탐색 보상과 정확도-보기 수 Pareto, 중단 시점 보정 비교.
  - 자원: github.com/gtc-gh/SATE-Bench, VAGEN 코드, verl. 경쟁: 중간. POMDP: 강함 (belief + 최적 중단).
- **G2. 가설 단위 과정 보상의 시각적 귀추 RL.** 경쟁 설명을 생성하고 시각 증거로 배제하는 것을 보상한 RL 없음.
  - 근거: CEDAR-GRPO 텍스트 전용 [페이지]. Video-Holmes RL baseline은 결과 보상 변형(GRPO-CARE), Qwen2.5-VL-7B 27.8 → 33.5 [검색]. Video-HolmesV2 (2609.17248) 시공간 증거 주석 추가, RL 없음 [페이지]. 귀추 서베이 (2604.08016) 멀티모달 절 없음 [페이지].
  - Pilot: Qwen2.5-VL-7B가 {가설, 증거(타임스탬프·박스), 배제한 경쟁 가설, 답} 출력. Video-HolmesV2 주석 대비 IoU로 증거 커버리지 채점. Video-Holmes, VAR, Sherlock에서 결과만 GRPO와 비교, 증거 제거 테스트.
  - 자원: github.com/TencentARC/Video-Holmes [검색]. 경쟁: 낮음~중간.
- **G3. 멀티모달 RLVR에서 grounding 페널티의 적응적 시점.** 시각 지름길 발생 시점이 급격하고 페널티는 그 전에만 효과적인데, 학습 중 발생을 감지하거나 λ를 적응 조절한 연구 없음.
  - 근거: 2606.22043 (단일 저자, Qwen3-VL-8B, 페널티 하나, 프레임 셔플 테스트, 한계: 모델 계열 하나·규모 미검증) [페이지]. 반사실 보상 논문들은 고정 항 [검색]. tool 사용 붕괴와 같은 동역학일 수 있다는 것은 조사자 추측.
  - Pilot: Qwen2.5-VL GRPO, N step마다 실제 입력 vs 빈/셔플 입력의 답 일치율 기록, 이미지 유무 logit KL이나 advantage 격차가 발생을 선행하는지 확인, λ PI 제어 vs 고정 λ.
  - 경쟁: 중간. POMDP 연결: 없음.
- 조사자 제안: G1 + G2를 "가설 기반 능동 지각" 한 편으로 (POMDP + 귀추 + 멀티모달 RL), NeurIPS 2027.

### 7.4 RelBench / 관계형 딥러닝

**최신 동향**
- **RelBench v2** (2602.12606): 11개 DB, Autocomplete 과제 유형, TGB·4DBInfer·ReDeLEx 통합. 사이트 기준 66개 과제, 리더보드 3개(분류 12, 회귀 9, **추천 10**). 현재 PyPI는 "RelBench 3", Hugging Face에서 로드. [페이지]
- **Relational Transformer** (2510.06377, ICLR'26): 22M, zero-shot으로 지도학습 AUROC의 93%. [페이지]
- **RT-J** (2610.17867, 2026-10): 86M, PluRel 합성 DB + THE JOIN(639개 DB) 사전학습, 특징공학 파이프라인보다 in-context 예시 약 10배 적게 필요. [페이지]
- **KumoRFM-2** (2604.12596): in-context + fine-tuning, 41개 벤치마크에서 최대 8% 개선. [페이지]
- **RDBLearn** (2602.18495): 연결 테이블을 하나로 집계 후 tabular in-context 모델. [페이지]
- **RelAgent** (2605.07840, v2 2026-10-21): LLM 에이전트가 SQL 특징 프로그램 작성 + 고전 모델 선택, RelBench v2 분류·회귀 최고 성능. [페이지]
- **RelArena-α / TabPFN-Rel** (2608.16319): RelBench v1 21개 엔티티 과제 표준 리더보드. 모델 1위 TabPFN-Rel (Elo 1821), 시스템 1위 RT-PluRel. [페이지]

**빈틈**
- **G1. RL로 학습한 오픈 가중치 관계형 데이터 분석 에이전트.** 관계형 특징공학 에이전트는 모두 프롬프팅.
  - 근거: RelAgent AUROC GPT-5.2 78.38 vs Qwen3-32B 70.13, 한계는 탐색 비용·불안정·조인과 시간창의 조합 폭발, RL 없음 [페이지]. ReFuGe (2601.17735) 프롬프팅 [페이지]. ML-Agent (2505.23723) 일반 ML 엔지니어링 RL [검색]. TRUST-SQL (2603.16448) text-to-SQL을 POMDP로 [검색].
  - Pilot: RelAgent형 tool + 시점 검증기를 갖춘 DuckDB 환경, 보상 = 검증 지표 개선 − 비용. Qwen3-8B 기각 샘플링 SFT → 짧은 GRPO, PluRel·THE JOIN으로 학습, RelBench v2 보류 DB로 평가.
  - 경쟁: 중간 (RelAgent 그룹, Stanford/Kumo). 연결: 강함 (예산 내 쿼리로만 DB를 보는 부분 관측 tool 에이전트).
- **G2. 추천은 foundation model·에이전트가 방치한 과제 유형.**
  - 근거: RelArena-α 추천 명시적 제외, RelAgent 분류·회귀만, RT·RT-J 예측만 [페이지]. KumoRFM-2 추천 수치 미확인. rel-ratebeer 최고 MAP 1.89 (IDGNN), past-visit 0.00 [페이지]. ContextGNN (2411.19513, 2024) 독립 재현 결과 혼재 [검색].
  - Pilot: 에이전트가 SQL 후보 생성기(재구매, 공동 출현, 세그먼트 인기) 작성 → 시점 쌍 특징 → LightGBM/TabPFN-Rel 순위. rel-hm, rel-amazon, rel-avito, rel-ratebeer에서 ContextGNN·IDGNN 대비 MAP·후보 recall.
  - 경쟁: 낮음~중간. 추천·에이전트와 직접 연결.
- **G3. 누출·평가 체제 감사 + LLM 암기.**
  - 근거 (2608.16319): rel-event/user-ignore 누출 수정 시 2~6 AUC 하락, KumoRFM-2가 테스트 cutoff 이후 DB 상태를 써서 rel-f1을 10 AUC 이상 부풀림, 시점 경계 비표준, autocomplete baseline이 테스트 시 NaN 입력 [페이지]. RelAgent 검증 피드백 통한 누출 경고 [페이지]. RelBench LLM 암기 연구 없음 (단일 테이블: Elephants Never Forget 2403.06644) [검색].
  - Pilot: 미래 행 주입 시 에이전트 SQL 출력 변화 확인, rel-f1 이름 익명화·날짜 이동 시 LLM·에이전트 점수 하락 측정, 다섯 방법을 같은 DB 상태 체제로 재실행.
  - 코드: github PriorLabs/relarena [검색]. 경쟁: 낮음~중간. Datasets & Benchmarks track 적합.
- **G4. 이미지·긴 텍스트 열.**
  - 근거: RelAgent 긴 텍스트·멀티모달 필드 미처리, 텍스트 특징 없으면 TabPFN-Rel Elo 1821 → 1706 [페이지]. RELATE (2510.19954) 이미지 미포함 [페이지]. 이미지 열을 가진 관계형 벤치마크 없음. H&M Kaggle 원본에 의류 이미지 약 105K [검색], 라이선스 확인 필요.
  - Pilot: rel-hm 상품에 H&M 이미지 부착, VLM 임베딩을 GraphSAGE·ContextGNN·RT에 넣어 user-item-purchase, item-sales 평가.
  - 경쟁: 낮음. 추천 + 멀티모달 연결.
- **G5 (약함).** 관계형 foundation model의 in-context 검색 프라이버시. RT-J는 random walk로 문맥 행을 검색 [페이지], 엔티티 수준 DP는 존재 (2506.08347) [검색], 검색된 문맥 행 보호는 못 찾음. 암호화 RAG 관심과 연결.
- 조사자 제안: G2 + G1 (RelBench 위 RL 학습 추천 에이전트), 평가에 G3 누출 검사 포함.

### 7.5 암호화 / 프라이버시 보존 RAG

**최신 동향**
- **Pointing the Way, Hiding the Destination** (2608.25735): metric DP로 보호한 학습형 이진 해시로 200~500개 후보 선별 → 암호화 재순위 + oblivious key transfer. NQ 268만 passage에서 검색에 0.73초(약 10%) 추가. [페이지]
- **Spruce** (2609.03376): 이진 임베딩 + 2서버 MPC. 38.3만~542만 문서 전체 스캔 0.21~2.97초. [페이지]
- **GoldenRetriever** (2607.29019): 비대화형 CKKS 동형암호, top-k 대신 고정 점수 임계값. 후보 100개에 쿼리당 약 1,052초, 문서 자체는 평문. [페이지]
- **P²RAG** (2603.14778), **PRAG** (2604.26525): 비밀분산·HE RAG. P²RAG가 PRAG 대비 3~300배 빠름. [페이지]
- **MosaicLeaks + PA-DR** (2605.30727): 다중 hop 에이전트 과제 1,001개, 공격자는 에이전트의 웹 쿼리만 봄. 프라이버시 분류기 RL로 누출 34.0% → 9.9%. [페이지]
- **SoK: Confidential Transformer Inference and RAG** (eprint.iacr.org/2026/1544): 어떤 방식도 성능·효용·위협 모델 적합 셋 중 둘 넘게 만족 못함. 비공개 graph-RAG가 가장 소외. 모든 TEE 방식이 부채널 무시. [페이지]

**빈틈**
- **G1. 비공개 검색을 쓰는 에이전트의 궤적 전체 누출** (이 연구자에게 가장 적합).
  - 내용: 비공개 검색 프로토콜은 쿼리당 누출만 제한한다. 다중 hop 에이전트가 내보내는 코드·클러스터 ID·라운드 수·타이밍 시퀀스의 누출은 아무도 측정·제어하지 않음.
  - 근거: Pointing the Way "여러 라운드에 걸친 추론은 명시적 누출로 남는다", 세션 단위 (ε,δ) 한계 없음, 다중 hop·에이전트 미언급 [페이지]. MosaicLeaks 평문 웹 쿼리만, future work "다른 에이전트 설계" [페이지]. 암호 검색 + 다중 라운드 에이전트 검색 결과 없음 [검색].
  - Pilot: HotpotQA·MuSiQue에서 Search-R1 에이전트 + 누출 오라클(SimHash/ITQ 코드 + metric-DP 잡음, IVF 클러스터 ID, 라운드 수)을 노출하는 모의 서버 → 누출 시퀀스로 공격자 학습, hop 수에 따른 답 추론 정확도 → RL 보상에 누출 페널티 추가.
  - 데이터·코드: HotpotQA, MuSiQue, Search-R1 (github.com/PeterGriffinJin/Search-R1 [검색]). MosaicLeaks 데이터 공개 언급 없음 [페이지].
  - 경쟁: 낮음~중간 (PA-DR이 가장 가깝지만 평문만). 학회: NeurIPS / ICLR. 암호 지식: 낮음 (프로토콜을 블랙박스 누출 함수로). 에이전트 연결: 강함 (공격자가 궤적을 부분 관측, 에이전트는 제약 POMDP).
- **G2. 경량 비공개 검색 인코딩에 대한 적응형 공격.** 변환 벡터를 공개하는 방어는 기성 역변환 공격으로만 검증됨.
  - 근거: Pointing the Way는 ZSInvert·GEIA·속성 추론만, 이진 코드 맞춤 공격 없음 [페이지]. SHAQ (2609.04767) 기존 역변환 공격 대비 복원율 0.2104 [페이지]. SHARD (2606.27976) 암호학적 보장 없음, 릴리스 간 행 연결 R@1 ≥0.9996 [페이지].
  - Pilot: 여러 쿼리를 모아 이진 코드 조건부 vec2text형 역변환기 학습, NQ·MS MARCO에서 논문 보고치와 비교.
  - 경쟁: 중간 (Zero2Text, ALGEN 등 [검색]). 학회: ICLR / ACL / USENIX. 에이전트 연결: G1과 결합 시만.
- **G3. 동형암호 하에서 동작하도록 검색기 학습.** 비대화형 HE 검색은 CKKS 근사 잡음을 견디는 보정된 절대 점수 + 전역 임계값이 필요한데, 검색기는 순위용으로 학습됨.
  - 근거: GoldenRetriever 임계값 "사전 정의", τ=0.7에서 recall 0.5 [페이지]. PRAG는 HE 잡음 대비 추가 오차 추정 모듈 필요 [페이지].
  - Pilot: 고정 임계값 대조 손실 + 모의 CKKS 잡음·양자화로 BGE-small 미세조정, BEIR에서 전역 τ 하나로 집합 F1, TenSEAL/OpenFHE로 1K 문서 확인.
  - 경쟁: 중간. 학회: ICLR / ACL (점진적이라는 비판 위험). 암호 지식: 중간.
- **G4. 형식적 프라이버시를 갖춘 그래프·관계형 RAG.**
  - 근거: SoK 비공개 graph-RAG 가장 소외 [페이지]. FedGRAG (2610.30728) "통제된 공개", 형식적 보장 없음 [페이지]. GraphRAG 엔티티·관계 구조 누출 (2026.findings-acl.899) [검색].
  - Pilot: RelBench(rel-amazon, rel-stack)에서 노드 수준 DP 하 k-hop 이웃 검색 에이전트, 추출 공격 vs 과제 효용.
  - 경쟁: 낮음~중간. 학회: NeurIPS / KDD (DP 사용 시). RelBench 관심과 연결.
- **G5 (선택). 효용·누출·지연 통합 벤치마크.** 현재 평가 규모가 100개 ~ 540만 문서로 비교 불가 [페이지]. NeurIPS D&B 적합, 엔지니어링 부담 큼.
- **피할 영역:** RAG membership inference (MEntA 2605.24312 AUC 0.991, CIG-MIA, E-MIA, ImageAuditor [검색]), DP-RAG (DP-KSA 2602.14374 [검색], SparsePay-RAG 2609.27406 [페이지], PA-HDP, DP-SynRAG), 에이전트 메모리 추출 (ADAM 2604.09747, SPORE 2607.23444 [검색]), 새 비공개 검색 프로토콜 (2026년에만 8개 이상, 높은 암호 지식, CCS·S&P 대상).
- 조사자 제안: G1, 선택적으로 G2 공격을 첫 절로. NeurIPS 2027 / ICLR 2028.

### 7.6 POMDP·agent (기존 조사)
상세는 [agentic-rl-pomdp-research-landscape.md](agentic-rl-pomdp-research-landscape.md). 요약: 기회 1 (state-anchored belief shaping, 특권 환경 상태 기반 로그 점수), 기회 2 (belief→행동 간극 RL), 기회 3 (동적 환경 재관측), 기회 4 (비가역 탐색), 기회 5 (belief 충분성 이론). 후보 A는 기회 1·2를 도메인 일반으로 넓힌 형태.

---

## 8. 주요 출처
- 추천: [τ-Rec](https://arxiv.org/abs/2606.10156) · [Ask to Be Sure](https://arxiv.org/abs/2608.15949) · [Simulator collapse](https://arxiv.org/abs/2608.12253) · [AcCoRD](https://arxiv.org/abs/2608.27818)
- Abductive: [GoS](https://arxiv.org/abs/2603.21250) · [PoS](https://arxiv.org/html/2610.01415) · [CEDAR-GRPO](https://arxiv.org/abs/2608.14791) · [AREW](https://arxiv.org/abs/2603.12109) · [MedExAgent](https://arxiv.org/html/2605.07058) · [LLM-as-Investigator](https://arxiv.org/html/2606.13220) · [ORCA-bench](https://arxiv.org/abs/2607.28545)
- 멀티모달: [SATE-Bench](https://arxiv.org/abs/2610.34149) · [Illusion of Visual Tool-Use](https://arxiv.org/abs/2608.06270) · [TextCall](https://arxiv.org/abs/2608.09682) · [Multimodal Reward Hacking](https://arxiv.org/abs/2607.09492) · [VAGEN](https://arxiv.org/abs/2510.16907)
- RelBench: [RelBench v2](https://arxiv.org/abs/2602.12606) · [RT-J](https://arxiv.org/abs/2610.17867) · [RelAgent](https://arxiv.org/abs/2605.07840) · [RelArena-α](https://arxiv.org/abs/2608.16319)
- 암호화 RAG: [Pointing the Way](https://arxiv.org/abs/2608.25735) · [Spruce](https://arxiv.org/abs/2609.03376) · [MosaicLeaks](https://arxiv.org/abs/2605.30727) · [SoK (IACR 2026/1544)](https://eprint.iacr.org/2026/1544)
- 일정: [OpenCurious ICML 2027](https://www.opencurious.com/ai-conference-deadlines/icml-2027) · [OpenCurious NeurIPS 2027](https://www.opencurious.com/ai-conference-deadlines/neurips-2027) · [OpenCurious ACL 2027](https://www.opencurious.com/ai-conference-deadlines/acl-2027)
