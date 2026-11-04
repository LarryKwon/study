# 핵심 논문 정리 색인

T2I 생성에서 객체 누락·요청 외 객체·상충 요구를 다룬 핵심 논문 16편을 정리했다. 각 폴더의 `정리.md`는 다음 다섯 항목으로 되어 있다.

1. 미리 알아야 할 기본 개념
2. 무엇을 문제로 보았나
3. 방법
4. 실험 결과
5. Challenge를 걸 부분

본문을 확보한 논문은 같은 폴더에 `paper.pdf`와 텍스트 추출본 `paper.txt`가 있다. 작성 기준일은 2026-11-04다.

**본문** 열 표기
- ✅ 본문을 읽었다.
- ⚠️ 본문을 확보하지 못해 초록 정보로만 정리했다.
- ❌ 초록도 확보하지 못했다.

## 1. 위치 효과 (프롬프트 안의 순서와 표현)

| # | 논문 | 상태 | 본문 | 핵심 |
|---|---|---|---|---|
| 01 | [Position-Sensitive Concept Acquisition](1_위치효과/01_Position-Sensitive-Concept-Acquisition/정리.md) | ICLR 2027 투고 | ❌ | 조건이 프롬프트 앞에 있는지 뒤에 있는지에 따라 미세조정으로 얻는 제어력이 달라진다. 이 내용은 이전 조사의 초록 요약에서 가져왔다. |
| 02 | [Filling the Slots](1_위치효과/02_Filling-the-Slots/정리.md) | ICLR 2027 투고 | ⚠️ | 개수를 숫자 대신 반복 언급으로 쓰면 개수 정확도가 SD3.5-M에서 −44.3점, FLUX에서 −59.2점 떨어진다. 언급이 인스턴스를 잃는 방식은 누락과 융합 두 가지다. |
| 03 | [CLIP Is Shortsighted (DeBias-CLIP)](1_위치효과/03_DeBias-CLIP/정리.md) | CVPR 2026 (Highlight로 전달받음, PDF에는 표기 없음) | ✅ | long-caption CLIP은 학습 캡션의 첫 요약 문장에 의존한다. 생성 쪽 검증은 약하다. |
| 04 | [T2I Models Need Less from Text Encoders](1_위치효과/04_T2I-Need-Less-from-Text-Encoders/정리.md) | arXiv 2606.03715 (ICLR 2027 투고로 전달받음) | ✅ | 단어 의미와 절대 위치만 남긴 임베딩으로도 충분히 생성된다. 다만 FLUX.2의 두 객체 항목은 92.7→58.9로 떨어지고, 긴 프롬프트에서는 이 주장이 약해진다. |
| 16 | [TIE](1_위치효과/16_TIE/정리.md) | arXiv 2610.32407 | ✅ | 뒤쪽 속성–객체 쌍의 속성 누락률은 33.2%로 앞쪽의 13.2%보다 높다. 원인 신호는 CLIP 토큰 엔트로피다. 객체 자체의 위치별 누락은 측정하지 않았다. |

## 2. 충족 불가·모순·반사실

| # | 논문 | 상태 | 본문 | 핵심 |
|---|---|---|---|---|
| 05 | [LogicVal](2_충족불가_모순/05_LogicVal/정리.md) | arXiv 2610.24051 | ✅ | 제약을 1개→6개로 늘리면 Gemini의 충족률이 0.85→0.20으로 떨어진다. 모순 프롬프트에 기권한 모델은 Gemini(23%)뿐이다. |
| 06 | [SAP: Contextually-Contradictory Prompts](2_충족불가_모순/06_SAP-Contradictory-Prompts/정리.md) | arXiv 2506.01929 (CVPR 2026으로 전달받음) | ✅ | 디노이징 단계별로 LLM이 만든 proxy prompt로 바꿔 넣는다. 레이아웃에 자리가 없는 객체는 사라진다. |
| 07 | [CF-World](2_충족불가_모순/07_CF-World/정리.md) | arXiv 2606.24548 (NeurIPS 2026으로 전달받음) | ✅ | 결과를 명시해 줘도 사전분포로 되돌아간다(prior lock-in). 단계마다 다른 채점 프롬프트를 쓴다는 문제가 있다. |
| 08 | [GenOlympiad](2_충족불가_모순/08_GenOlympiad/정리.md) | arXiv 2610.26850 | ✅ | 최고인 Nano Banana Pro도 strict 23.9점이다. 판정자가 Gemini라 편향 가능성이 있다. 난이도는 양이 아니라 과정에서 만들었다. |
| 09 | [NegT2IBench](2_충족불가_모순/09_NegT2IBench/정리.md) | arXiv 2610.03084 (ICLR 2027 투고로 전달받음) | ✅ | 부정 문장이 들어가면 평균 22.7%p 떨어진다. 실패한 부정 문장의 41.5%는 금지한 대상을 그렸다. |
| 10 | [DominanceBench](2_충족불가_모순/10_DominanceBench/정리.md) | arXiv 2512.20666 | ✅ | 학습 이미지가 시각적으로 균질한 개념이 이긴다. 첫 스텝의 저해상도 어텐션이 승부를 가르며, 근거는 SD1.4/2.1 분석이다. |

## 3. 요청 외 객체 (정밀도)

| # | 논문 | 상태 | 본문 | 핵심 |
|---|---|---|---|---|
| 11 | [ICO](3_요청외객체/11_ICO-Implicit-Co-occurring-Objects/정리.md) | ICLR 2027 투고 | ⚠️ | 객체 하나만 요청해도 34–50%의 이미지에 요청 외 객체가 들어간다. 억제와 충실도 사이에 상충 관계가 있다는 이론을 제시한다. |
| 12 | [OverReal](3_요청외객체/12_OverReal/정리.md) | ICLR 2027 투고 | ⚠️ | 언급만 하고 요청하지 않은 대상(부정·상상·비유·가려짐)도 그려진다. CLIPScore는 이 실패를 오히려 보상한다. |
| 13 | [VISTA](3_요청외객체/13_VISTA/정리.md) | ICLR 2027 투고, arXiv 2609.32188 | ✅ | 비유의 매개체가 실제 물체로 그려진다. TIFA와 V-Score의 상관은 ρ=−0.04에 그친다. |
| 14 | [AGSM](3_요청외객체/14_AGSM-Alignment-Guided-Score-Matching/정리.md) | ICML 2026 | ✅ | SoftREPA는 Two object 점수를 올리지만 Counting을 0.56→0.29로 망가뜨린다. 재현율을 얻고 정밀도를 잃는 교환이다. "+35%"는 SoftREPA 대비 수치다. |

## 4. 초기 고착 메커니즘

| # | 논문 | 상태 | 본문 | 핵심 |
|---|---|---|---|---|
| 15 | [Images Talk Back](4_초기고착_메커니즘/15_Images-Talk-Back/정리.md) | ICLR 2027 투고, arXiv 2610.13941 | ✅ | MM-DiT에서 이미지가 텍스트 쪽으로 보내는 피드백이 초기 오류를 고착시킨다. 이 피드백을 억제하는 SFG로 FLUX의 GenEval이 0.666→0.748로 오른다. |

## 추천 읽기 순서

두 관심사(긴 프롬프트 누락, 상충 요구)를 기준으로 고른 순서다.

1. **15 Images Talk Back → 10 DominanceBench → 06 SAP**: 생성 초반에 무엇이 이기고 고착되는지를 본다.
2. **16 TIE → 04 Text Encoders → 03 DeBias-CLIP**: 위치 효과의 인코더 쪽 증거를 본다.
3. **05 LogicVal → 09 NegT2IBench → 08 GenOlympiad → 07 CF-World**: 상충 요구를 어떻게 평가하는지와 각 평가의 약점을 본다.
4. **13 VISTA → 14 AGSM** (11, 12는 본문을 확보한 뒤): 요청 외 객체를 어떻게 정의하고 측정하는지 본다.

## 여러 논문에 공통으로 나타나는 점

- **승부는 생성 초반에 난다.** SAP, DominanceBench, Images Talk Back, Filling the Slots 모두 처음 몇 스텝(전체의 약 20%)에서 레이아웃과 개념 간 승부가 정해진다고 본다.
- **위치 효과를 직접 잰 논문이 없다.** 긴 프롬프트에서 객체 단위 누락을 위치별로 측정한 논문은 여전히 없다. TIE는 짧은 2쌍 프롬프트의 *속성* 누락만 쟀다. TIE가 인용한 "A Cat Is A Cat"(arXiv 2410.00321)은 다음에 읽을 후보다.
- **해소 방식을 분류하지 않는다.** 상충 벤치마크는 모두 점수 하나만 낸다. 모델이 한쪽을 택했는지, 섞었는지, 둘 다 빼먹었는지, 제3의 것을 그렸는지는 분류하지 않는다. 대상은 LogicVal, CF-World, NegT2IBench, GenOlympiad다.
- **정밀도를 보상으로 쓴 연구가 없다.** 정밀도 지표는 생기기 시작했지만(ICO, OverReal, VISTA) 이를 보상에 넣은 연구는 없다. AGSM은 재현율을 올리려는 정렬 기법이 정밀도를 깎는다는 사례다.
- **판정자 편향이 반복된다.** GenOlympiad(Gemini 판정자에 Gemini 1위), VISTA(GPT 판정자), CF-World(단계별로 다른 채점 프롬프트), NegT2IBench(VLM 판정자의 부정 이해 문제)에서 같은 문제가 보인다.

## 본문 미확보 논문 채우기

01, 02, 11, 12는 OpenReview가 자동 접근을 막아 본문을 받지 못했다. 이 네 편은 브라우저에서 OpenReview에 들어가 PDF를 받은 뒤, 각 폴더에 `paper.pdf`로 넣으면 다시 정리할 수 있다.

| # | OpenReview |
|---|---|
| 01 | https://openreview.net/forum?id=oyZo8XCw18 |
| 02 | https://openreview.net/forum?id=ohAQYCjV15 |
| 11 | https://openreview.net/forum?id=IET7dtNjT8 |
| 12 | https://openreview.net/forum?id=erJ21xq8Tv |
