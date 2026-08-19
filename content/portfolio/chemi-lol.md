---
title: "Chemi.lol"
title_en: "Chemi.lol"
type: page
description: "LoL 듀오 데이터를 관계 중심 feature와 회귀 모델로 탐색한 데이터 제품 프로젝트."
description_en: "A data-product experiment modeling LoL duo relationships through engineered features and regression."
---

<div class="lang-ko">

[Analysis Repository →](https://github.com/league-of-legend-project/Analysis)

## The Question

두 사람이 같이 플레이했을 때의 결과를 **둘의 관계를 나타내는 데이터**로 어디까지 설명할 수 있는지 탐색했습니다.

## Data Before Model

분석 코드는 MongoDB의 `lol_data_hub`에서 summoner, match, timeline 데이터를 나누어 불러옵니다.

```text
Player / Match / Timeline
        ↓
Duo Detection
        ↓
Feature Engineering
        ↓
Regression Experiments
        ↓
Score Experiments
```

## Turning Two Players into Features

두 플레이어 수치를 옆에 붙이는 대신 관계를 표현하는 파생 변수를 만들었습니다.

- `KDA_diff`
- `KP_diff`
- `GPM_diff`
- `Vision_diff`
- `WinRate_diff`
- 순서에 영향을 받지 않는 `Lane_combo`

## Model Experiments

Random Forest, Ridge, SVR, XGBoost를 비교하고 MLP 및 단순 ensemble도 실험했습니다.

한 실험에서 Ridge는 MAE 약 **9.94**, RMSE **12.60**, R² **0.203**이었고,  
Ridge + XGBoost + MLP 평균 ensemble은 MAE 약 **9.92**, RMSE **12.59**, R² **0.204**였습니다.

숫자보다 중요했던 결과는 **복잡도를 높여도 설명력이 크게 올라가지 않았다는 점**이었습니다.

## Model Is Not the Product

R² 약 0.2는 “더 복잡한 모델이면 궁합을 맞힐 수 있다”보다  
**현재 지표만으로는 듀오 결과의 상당 부분을 설명하기 어렵다**는 신호에 가까웠습니다.

그래서 다음 질문이 더 중요해졌습니다.

> **우리가 ‘궁합’이라고 부르는 것은 무엇인가?**  
> **현재 데이터에 빠진 것은 무엇인가?**  
> **사용자에게 어느 수준의 확신으로 결과를 보여줘야 하는가?**

[Analysis Repository →](https://github.com/league-of-legend-project/Analysis)  
[← Portfolio](/portfolio/)

</div>

<div class="lang-en" style="display:none">

[Analysis Repository →](https://github.com/league-of-legend-project/Analysis)

## The Question

I explored how much of two players' outcomes together could be explained through **data describing the relationship between them**.

## Data Before Model

The analysis code works around a MongoDB `lol_data_hub`, separating summoner, match, and timeline data.

```text
Player / Match / Timeline
        ↓
Duo Detection
        ↓
Feature Engineering
        ↓
Regression Experiments
        ↓
Score Experiments
```

## Turning Two Players into Features

Instead of only placing two players' raw stats side by side, I created relationship-level features:

- `KDA_diff`
- `KP_diff`
- `GPM_diff`
- `Vision_diff`
- `WinRate_diff`
- order-invariant `Lane_combo`

## Model Experiments

I compared Random Forest, Ridge, SVR, and XGBoost, then experimented with an MLP and simple ensembles.

In one experiment, Ridge produced roughly MAE **9.94**, RMSE **12.60**, and R² **0.203**.  
A Ridge + XGBoost + MLP average produced roughly MAE **9.92**, RMSE **12.59**, and R² **0.204**.

The more useful finding was that **extra model complexity did not meaningfully improve explanatory power**.

## Model Is Not the Product

An R² around 0.2 suggested something more useful than “we need a bigger model”:

**the available features explained only part of what we were calling compatibility.**

That shifted the questions toward:

> **What do we really mean by compatibility?**  
> **What is missing from the data?**  
> **How confident should the product sound when showing a result?**

[Analysis Repository →](https://github.com/league-of-legend-project/Analysis)  
[← Portfolio](/portfolio/)

</div>
