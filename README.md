# 🌫️ AERIS: 재비산먼지 예측 기반 지능형 청소차 운영 시스템

AERIS is a project developed for the [Engineering Industry Competition](https://engcontest.kenca.or.kr/).

This repository contains the full pipeline for PM10 prediction, road dust extension, and routing optimization, aiming to support cleaner and smarter urban mobility.

---

## 개요

**동적 PM10 지도 생성 시스템**을 구축하고, 이를 두 가지 핵심 최적화 문제에 활용:

### 1️ 도로 청소 차량 경로 최적화

→ 흡입 차량의 경로를 최적화하여 도시 전체 오염을 최소화

### 2️ 저노출 경로 최적화

→ 보행자에게 오염 노출을 최소화하는 경로 제공

---

## Motivation

도시 대기오염은 단순한 예측 문제가 아닌

**의사결정 문제**이이다:

* 청소 차량은 어디서 운행해야 하는가?
* 사람들은 노출을 줄이기 위해 어떻게 이동해야 하는가?

이 질문에 답하기 위해서는:
→ **동적·고해상도 오염 지도**가 필요하하다.

---

## Methodology

### 1. ST-GNN (시공간 그래프 신경망)

* 오염물질이 지점 간에 어떻게 확산되는지 모델링
* 바람 방향을 고려한 방향성 그래프 활용
* GRU를 통한 시간적 동역학 포착

---

### 2. HiddenExtension (Wind-IDW)

* 지역별 오염 발생 요인 포착
* 포함 요소:

  * 도로 밀도
  * 토지 이용
  * 생활인구
  * 도시 구조

---

### 3. 하이브리드 모델

PM = f(지리적 특성, ST-GNN 표현)

→ **이동 역학 + 지역 발생**을 결합

---

### 4. 동적 PM 지도

시간에 따라 변화하는 **도시 전역 오염 지도**를 생성합니다.

---

## 최적화 태스크

### 🚗 (1) 흡입 차량 경로 최적화

목표:

* 도시 전체 PM 수준 감소

접근 방식:

* 고오염 구역 식별
* 청소 효율을 극대화하는 차량 경로 최적화

---

### 🚶 (2) 저노출 경로 최적화

목표:

* PM 노출을 최소화하는 이동 경로 탐색

접근 방식:

* 동적 PM 지도 활용
* 누적 오염이 가장 낮은 경로 탐색

---

## 주요 특징

* 물리적 해석 가능한 엣지 기반 ST-GNN
* 동적 오염 지도 생성
* 두 가지 실제 적용 최적화:

  * 차량 경로 최적화
  * 보행자 경로 최적화

---

## 📊 데이터

* 대기질 측정 관측소 (서울)
* 기상 데이터 (풍향, 기온)
* 지리적 특성 (도로 밀도, 토지 이용)

---

## 핵심 인사이트

본 프로젝트는 대기오염 모델링을 다음과 같이 재정의합니다:

> **단순 예측이 아닌, 도시 최적화를 위한 의사결정 지원 시스템**

---

## 📁 파일 구조

```
AERIS_Project/
│
├── 0_Data_Preprocess/                  # 데이터 수집 및 전처리
│   ├── Data_Preprocess_for_LUR/        # LUR 피처 전처리 (도로, 토지, 인구 등)
│   ├── Data_Preprocess_for_Pollutants/ # 대기오염 측정 데이터 전처리
│   ├── Data_Preprocess_for_Satellite/  # 위성 데이터 전처리
│   └── Data_Preprocess_from_NAS/       # NAS 원본 데이터 로딩
│
├── 1_ST-GNN_Modeling/                  # ST-GNN 모델 학습 및 평가
│   ├── checkpoints/                    # 학습된 모델 가중치
│   ├── graphs/                         # 그래프 구조 파일
│   ├── plots/                          # 학습 결과 시각화
│   └── Visualization_Graph/            # 그래프 구조 시각화
│
├── 2_HiddenExtension/                  # ST-GNN hidden → 격자 PM10 확장 실험
│   ├── HiddenExtension_V1/
│   ├── HiddenExtension_V2/
│   ├── HiddenExtension_V3/
│   ├── HiddenExtension_V4/
│   └── HiddenExtension_V5/
│
├── 3_RoadExtension/                    # 도로 재비산먼지 연계 실험
│   ├── RoadExtension_V1/
│   ├── RoadExtension_V2/
│   └── RoadExtension_V3/
│
├── 4_Routing_Optimization/             # 경로 최적화 알고리즘
│   ├── clutsp_multiview_topk/
│   ├── cvrp_topk_pomo/
│   ├── metaheuristic_routing/
│   ├── op_comparison/
│   ├── op_topk/
│   ├── policy_routing_simulation/
│   ├── topk_tsp_pomo/
│   └── tsp_topk_pomo/
│
├── Appendix/                           # 부록: 실험 상세 분석
│   ├── Appendix_C2/
│   ├── Appendix_C3/
│   ├── Appendix_C4/
│   ├── Appendix_D1/
│   ├── Appendix_D2/
│   ├── Appendix_D3/
│   └── Appendix_D4/
│
└── Visualization/                      # 최종 결과 시각화
    └── outputs/
```

---

## Hidden Extension 버전 히스토리

ST-GNN hidden vector를 city-wide grid로 확장하는 실험 시리즈.

| 버전 | 폴더 | 핵심 방법 | best direct MAE | vs ST-GNN baseline |
|------|------|----------|----------------|-------------------|
| ST-GNN | `1_ST-GNN_Modeling/checkpoints/` | Station forecasting | **2.6144** | - |
| V1 | `2_HiddenExtension/HiddenExtension_V1/` | Cross-attention + 6D LUR (공유 compressor) | 2.6659 | ↓악화 |
| V2 | `2_HiddenExtension/HiddenExtension_V2/` | Cross-attention + 9D LUR (독립 compressor) | 2.6105 | ↑0.0039 개선 |
| V3 | `2_HiddenExtension/HiddenExtension_V3/` | Wind-aware IDW + Random Forest | (진행 중) | 목표 < 2.55 |
| V4 | `2_HiddenExtension/HiddenExtension_V4/` | (진행 중) | - | - |
| V5 | `2_HiddenExtension/HiddenExtension_V5/` | (진행 중) | - | - |

각 버전의 상세 실험 설계, 결과, 교훈은 해당 폴더의 `README.md` 참고.

---

## 📚 Appendix

### 목차

- **A.** 데이터 설명 및 전처리
- **B.** ST-GNN 모델 상세
- **C.** ST-GNN 실험 상세
  - C.1 데이터 분할 전략
  - C.2 입력 시퀀스 및 그래프 구성 → [`Appendix/Appendix_C2/README.md`](Appendix/Appendix_C2/README.md)
  - C.3 피처 시나리오 → [`Appendix/Appendix_C3/README.md`](Appendix/Appendix_C3/README.md)
  - C.4 실험 결과 및 분석 → [`Appendix/Appendix_C4/README.md`](Appendix/Appendix_C4/README.md)
    - C.4.1 전체 실험 설정
    - C.4.2 전체 성능 비교
    - C.4.3 그래프 구성 방식의 영향
    - C.4.4 피처 시나리오의 영향
    - C.4.5 윈도우 크기의 영향
    - C.4.6 최종 모델 선정
- **D.** Attention 시각화 및 해석
  - D.1 공간 Attention 시각화 → [`Appendix/Appendix_D1/README.md`](Appendix/Appendix_D1/README.md)
    - D.1.1 노드 수준 Attention 집계
    - D.1.2 헤드별 Attention 특성
  - D.2 풍향과 Attention의 관계 → [`Appendix/Appendix_D2/README.md`](Appendix/Appendix_D2/README.md)
  - D.3 고농도 PM 이벤트 사례 분석 → [`Appendix/Appendix_D3/README.md`](Appendix/Appendix_D3/README.md)
  - D.4 해석 및 한계 → [`Appendix/Appendix_D4/README.md`](Appendix/Appendix_D4/README.md)
- **E.** 격자 단위 PM10 확장
  - E.1 격자 단위 확장의 필요성
  - E.2 IDW 기반 보간 방식
  - E.3 격자 feature 구성
  - E.4 격자 확장 실험 1–5 설명
  - E.5 최종 격자 PM10 지도 생성 방식
  - → 각 실험 상세는 [`2_HiddenExtension/`](2_HiddenExtension/) 하위 버전별 `README.md` 참고
- **G.** 도로 재비산먼지 확장 실험
  - G.1 도로 재비산먼지 데이터 설명
  - G.2 ST-GNN 예측값과 도로 미세먼지 연결 방식
  - G.3 RoadExtension 구조
  - G.4 현재 실험 결과
  - → 각 실험 상세는 [`3_RoadExtension/`](3_RoadExtension/) 하위 버전별 `README.md` 참고

---


## 👤 Author

Yonsei University
Industrial Engineering @duddnd1113 @dddlmss & Urban Planning and Engineering @tak & Quantitative Risk Management @sonyein
---
