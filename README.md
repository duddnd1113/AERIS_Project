# 🌫️ AERIS: 재비산먼지 예측 기반 지능형 청소차 운영 시스템

AERIS is a project developed for the [Engineering Industry Competition](https://engcontest.kenca.or.kr/).

**Git Repository** for Model Implementation, Experiments, and Analysis

---

## Project Overview

<p align="center">
  <img src="poster.png" width="850">
</p>

---

## Key Insight

This project reframes air pollution modeling as:

> **A decision-support system for urban optimization, not just prediction**


---

## 📊 Data

* Air quality monitoring stations (Seoul)
* Meteorological data (wind, temperature)
* Geographic features (road density, land use)


---



## 📁 File Structure

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
Industrial Engineering @duddnd1113 @dddlmss & 

Urban Planning and Engineering @tak & 

Quantitative Risk Management @sonyein

---
