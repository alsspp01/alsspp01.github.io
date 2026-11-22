---
title: "Chemi.lol"
title_en: "Chemi.lol"
type: page
aliases:
  - /portfolio/chemi-lol/
description: "LoL 듀오 데이터로 두 플레이어의 궁합을 설명할 수 있는지 실험한 프로젝트"
description_en: "Testing whether League of Legends duo data can explain how well two players play together"
---

<div class="lang-ko">

**기간** · 2024.12–2025.05  
**역할** · 프로젝트 발의 · 팀장 · 기획 · 통계 분석 · 예측 모델 실험  
**팀** · 5명  
**상태** · 조회 기능 출시(팀) · 예측 기능은 실험에서 중단

[분석 저장소](https://github.com/league-of-legend-project/Analysis)

## 요약

- **문제** · 두 플레이어가 함께할 때의 승률과 궁합을 데이터로 보여줄 수 있는가
- **한 일** · 서비스 기획, 플레이어 통계 분석, 변수 설계, 예측 모델 실험
- **결과** · 과거 기록 기반 듀오 승률 조회 기능 출시(팀). 예측 기능은 데이터 한계를 근거로 출시하지 않고 종료 제안 → 팀 합의

| 구분 | 내용 |
|---|---|
| 출시 | 함께한 경기 기록 기반 듀오 승률 조회 |
| 실험 | 개인 지표와 포지션 조합으로 듀오 승률 예측 (공개하지 않음) |
| 미구현 | 플레이 성향 유형 LoLBTI (네 가지 축과 지표까지 설계) |

## 역할 분담

- **나** · 기획, 통계 분석, 변수 설계, 모델 실험, 제품화 여부 판단
- **팀원** · 데이터 수집, EDA와 MongoDB 관리, 웹 구현, AWS 배포

## 과정

### 개인 기록을 듀오 데이터로 바꾸기

- 개인 승률을 포함해, 함께한 기록이 적은 듀오도 예측할 수 있게 함
- KDA, 킬 관여율, 시야 점수는 두 사람의 평균과 차이로 표현
- 포지션 조합은 입력 순서와 무관하게 처리 (`Lane_combo`)
- 네 경기 이상 함께한 듀오만 사용 → 후보 2,737개, 학습 데이터 5,145개

### 모델보다 데이터가 문제였음

Random Forest, Ridge, SVR, XGBoost, MLP, 앙상블을 비교했습니다.  
5,145개를 80:20으로 나눈 대표 결과는 아래와 같습니다. 예측 대상은 듀오 승률(%)입니다.

| 실험 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge | 9.94 | 12.60 | 0.203 |
| Ridge · XGBoost · MLP 평균 앙상블 | 9.92 | 12.59 | 0.204 |

모델을 바꿔도 R²는 0.2 근처에 머물렀습니다.

- 한 플레이어에게서 안정적으로 얻을 수 있는 기록은 최근 약 100경기
- 듀오 데이터가 바텀 조합에 몰려 있고, 조합별 표본 차이가 큼
- 실력이 좋은 사람과 함께하면 승률이 오르는 효과와 "궁합"을 분리하기 어려움

### 출시하지 않기로 한 판단

예측값을 화면에 띄우는 건 가능했습니다.  
하지만 근거가 약한 숫자를 "두 사람의 궁합"이라고 보여줄 수는 없다고 판단해, 팀에 종료를 제안했고 합의했습니다.

## 배운 점

- 모델을 고르기 전에, 예측하려는 개념을 데이터가 담을 수 있는지부터 확인
- 만들 수 있는 기능이라도 근거가 부족하면 내보내지 않는 것도 제품 판단

</div>

<div class="lang-en" style="display:none">

**Period** · Dec 2024–May 2025  
**Role** · Initiator · Team Lead · Planning · Statistical Analysis · Prediction Experiments  
**Team** · 5 members  
**Status** · Lookup feature shipped (team) · Prediction stopped at the experiment stage

[Analysis repository](https://github.com/league-of-legend-project/Analysis)

## Summary

- **Question** · Can data show how well two players do together?
- **What I did** · Product planning, player statistics, feature design, model experiments
- **Result** · The team shipped a duo win-rate lookup based on shared matches. I proposed not releasing the prediction feature because the data could not support it, and the team agreed to end the project

| Stage | Detail |
|---|---|
| Shipped | Duo win-rate lookup from shared match history |
| Experimental | Predicting duo win rate from individual stats and roles (never released) |
| Not built | LoLBTI play-style profile (four axes and metrics designed) |

## Who did what

- **Me** · Planning, statistics, feature design, model experiments, the call on whether to productize
- **Teammates** · Data collection, EDA and MongoDB, the web app, AWS deployment

## Process

### Turning individual records into duo data

- Included individual win rates so duos with few shared games could still be scored
- Expressed KDA, kill participation, and vision score as the pair's average and difference
- Made role combinations order-invariant (`Lane_combo`)
- Kept duos with four or more shared games → 2,737 candidates, 5,145 training rows

### The data, not the model, was the limit

I compared Random Forest, Ridge, SVR, XGBoost, an MLP, and ensembles.  
Representative results on an 80:20 split of the 5,145 rows (target: duo win rate %):

| Experiment | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge | 9.94 | 12.60 | 0.203 |
| Ridge · XGBoost · MLP mean ensemble | 9.92 | 12.59 | 0.204 |

R² stayed around 0.2 whatever the model.

- Only about the last 100 matches per player were reliably available
- Duo data clustered in bot-lane pairs, with large gaps between combinations
- The lift from playing with a stronger player could not be separated from "chemistry"

### Deciding not to ship it

Showing a number on screen was easy.  
Calling a weakly supported number "your compatibility" was not, so I proposed stopping and the team agreed.

## What I learned

- Check whether the data can carry the concept before choosing a model
- Not shipping a feature you can build is also a product decision

</div>

---

[← Projects](/projects/) · [Portfolio](/portfolio/)
