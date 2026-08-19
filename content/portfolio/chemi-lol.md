---
title: "Chemi.lol"
title_en: "Chemi.lol"
type: page
description: "LoL 듀오 궁합을 데이터로 설명하려다 정확도 한계와 데이터 부족에 부딪혀 중단한 ML 프로젝트."
description_en: "An ML project that explored LoL duo compatibility, hit an accuracy ceiling, and was stopped after the data proved insufficient."
---

<div class="lang-ko">

**Type** · Data / ML Side Project  
**Status** · Discontinued  
**Focus** · Data Pipeline · Feature Engineering · Regression · Model Evaluation

[🔗 Analysis Repository](https://github.com/league-of-legend-project/Analysis)

---

## The Question

League of Legends에는 KDA, 골드, 시야, 승률처럼 수많은 지표가 있습니다.

하지만 두 사람이 같이 플레이할 때 궁금한 것은  
각자의 기록보다 **둘이 함께했을 때 어떤 결과를 만드는가**에 더 가깝습니다.

Chemi.lol은 이 관계를 데이터로 설명해보려는 프로젝트였습니다.

> **“두 플레이어의 플레이 특성만으로, 이 둘의 궁합을 어느 정도 예측할 수 있을까?”**

---

## Data Before Model

분석 코드는 MongoDB의 `lol_data_hub`를 기준으로  
summoner, match, timeline 데이터를 나누어 불러오는 구조로 만들었습니다.

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

처음에는 모델을 고르는 문제가 핵심이라고 생각했지만,  
실제로는 **어떤 듀오를 학습 데이터로 확보할 수 있는가**가 훨씬 중요한 문제였습니다.

---

## Turning Two Players into Features

두 플레이어의 수치를 단순히 나란히 놓기보다  
둘 사이의 차이와 조합을 표현하는 파생 변수를 만들었습니다.

- `KDA_diff`
- `KP_diff`
- `GPM_diff`
- `Vision_diff`
- `WinRate_diff`
- 순서에 영향을 받지 않는 `Lane_combo`

Lane 조합은 Player 1 / Player 2의 순서가 바뀌어도 같은 조합으로 취급하도록 정리했습니다.

제가 표현하고 싶었던 것은 **개별 플레이어의 실력**보다  
**두 사람 사이의 차이와 조합**이었습니다.

---

## Model Experiments

Random Forest, Ridge, SVR, XGBoost를 비교했고  
MLP와 단순 ensemble도 실험했습니다.

한 실험에서 Ridge는:

- MAE ≈ **9.94**
- RMSE ≈ **12.60**
- R² ≈ **0.203**

Ridge + XGBoost + MLP 평균 ensemble은:

- MAE ≈ **9.92**
- RMSE ≈ **12.59**
- R² ≈ **0.204**

였습니다.

모델을 바꾸고, scaling과 feature selection을 바꾸고,  
ensemble을 추가해도 성능은 비슷한 구간에서 머물렀습니다.

---

## The Accuracy Ceiling

처음에는 이 정체를 **모델의 문제**라고 생각했습니다.

그래서 다른 회귀 모델을 비교하고,  
scaling과 feature selection을 바꾸고,  
앙상블과 MLP까지 실험했습니다.

하지만 실험을 반복할수록 질문이 달라졌습니다.

> **“모델이 부족한 걸까?”**

에서

> **“애초에 이 관계를 학습할 데이터가 충분한 걸까?”**

로 바뀌었습니다.

제가 가장 큰 원인으로 본 것은  
**듀오 궁합을 학습할 만큼 충분하고 다양한 paired data를 확보하지 못한 것**이었습니다.

같은 두 사람이 함께 플레이한 데이터는 제한적이었고,  
‘궁합’이라는 개념 자체도 단순한 개인 지표의 차이만으로 안정적으로 설명하기 어려웠습니다.

이 부분은 실험으로 완전히 증명한 원인이라기보다,  
**성능이 반복해서 정체된 결과와 데이터 수집 과정에서 내린 가장 유력한 가설**입니다.

---

## Why I Stopped

프로젝트를 계속하려면 선택지는 있었습니다.

- 더 복잡한 모델을 계속 시도하기
- feature를 더 많이 만들기
- 점수를 보정해 서비스 형태를 먼저 완성하기

하지만 어느 순간부터는  
그 방식들이 문제를 해결하기보다 **낮은 설명력을 감추는 방향**이 될 수 있다고 생각했습니다.

정확도가 충분하지 않은 상태에서  
“궁합 점수”를 자신 있게 보여주는 것은 제품적으로도 마음에 들지 않았습니다.

그래서 Chemi.lol은 중단했습니다.

> **모델을 완성하지 못해서가 아니라,  
> 현재 데이터로는 약속하고 싶었던 수준의 결과를 만들기 어렵다고 판단했기 때문입니다.**

---

## What I Learned

이 프로젝트에서 가장 크게 배운 것은  
더 좋은 알고리즘을 찾는 방법이 아니었습니다.

### 1. 모델 성능의 ceiling은 데이터 문제일 수 있다

모델을 계속 바꾸기 전에  
target을 설명할 수 있는 데이터가 충분히 존재하는지 먼저 봐야 했습니다.

### 2. 만들 수 있는 점수와 보여줘도 되는 점수는 다르다

코드는 어떤 숫자든 출력할 수 있습니다.

하지만 그 숫자가 사용자의 의사결정에 영향을 준다면  
**얼마나 믿어도 되는 숫자인지**가 더 중요합니다.

### 3. 중단도 제품 결정이다

프로젝트를 끝까지 배포하는 것만이 성공적인 결론은 아니었습니다.

현재 조건에서 더 진행하는 것이 합리적인지 판단하고,  
아니라면 멈추는 것도 설계의 일부였습니다.

---

## If I Started Again

다시 시작한다면 모델보다 데이터 수집 설계부터 바꿀 것 같습니다.

단순히 개인 전적을 많이 모으는 것이 아니라,

- 동일 듀오의 반복 플레이
- 조합별 충분한 sample size
- 시간에 따른 실력 변화
- 포지션과 챔피언 조합
- 함께한 횟수와 관계의 안정성

처럼 **궁합이라는 target을 설명할 수 있는 paired data를 먼저 확보**하는 쪽에 더 많은 시간을 쓸 것입니다.

Chemi.lol은 서비스로 완성되지 않았습니다.

대신 저에게는  
**“더 만들 것인가?”만큼 “여기서 멈출 것인가?”도 데이터로 판단해야 한다**는 걸 배운 프로젝트로 남았습니다.

[🔗 관련 글: 정확도가 더 오르지 않을 때](/posts/2026-08-19-chemi-lol-accuracy-ceiling/)  
[🔗 Analysis Repository](https://github.com/league-of-legend-project/Analysis)  
[🔗 Portfolio](/portfolio/)

</div>

<div class="lang-en" style="display:none">

**Type** · Data / ML Side Project  
**Status** · Discontinued  
**Focus** · Data Pipeline · Feature Engineering · Regression · Model Evaluation

[🔗 Analysis Repository](https://github.com/league-of-legend-project/Analysis)

---

## The Question

League of Legends gives you plenty of numbers: KDA, gold, vision, win rate, and more.

But when two people queue together, the interesting question is less about their individual records  
and more about **what happens when those two players are paired.**

Chemi.lol started with a simple question:

> **“Can we predict how well two players fit together from their play data?”**

---

## Data Before Model

The analysis code was built around a MongoDB `lol_data_hub`,  
with summoner, match, and timeline data handled separately.

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

I initially treated model selection as the central problem.

In practice, **which duo relationships we could actually observe often mattered more than which regressor we chose.**

---

## Turning Two Players into Features

Instead of simply placing two players' raw statistics side by side,  
I created features describing the difference and combination between them.

- `KDA_diff`
- `KP_diff`
- `GPM_diff`
- `Vision_diff`
- `WinRate_diff`
- order-invariant `Lane_combo`

Lane combinations were normalized so swapping Player 1 and Player 2 did not create a different category.

The goal was to represent **the relationship between two players**, not just the quality of each player independently.

---

## Model Experiments

I compared Random Forest, Ridge, SVR, and XGBoost,  
then also experimented with an MLP and simple ensembles.

In one experiment, Ridge produced roughly:

- MAE **9.94**
- RMSE **12.60**
- R² **0.203**

A Ridge + XGBoost + MLP average produced roughly:

- MAE **9.92**
- RMSE **12.59**
- R² **0.204**

Changing models, scaling, feature selection, and ensemble structure repeatedly led back to a similar performance range.

---

## The Accuracy Ceiling

At first, I treated the plateau as a **model problem**.

So I compared more regressors,  
changed scaling and feature-selection approaches,  
and added ensembles and an MLP.

After enough repetitions, the question changed from:

> **“Which model are we missing?”**

to:

> **“Do we actually have enough data to learn this relationship?”**

My strongest hypothesis was that we did not.

We were not able to secure enough diverse paired data to represent duo compatibility reliably,  
and “compatibility” itself was probably more complicated than the differences between a few individual performance metrics.

That is an inference rather than a formally isolated causal result.

It is the explanation I found most convincing after seeing the same performance ceiling across repeated experiments and dealing with the limitations of the available duo data.

---

## Why I Stopped

There were still ways to keep the project moving:

- try increasingly complicated models
- create more derived features
- calibrate the output and finish the service around the existing score

But at some point, those options risked **hiding weak explanatory power instead of solving it**.

I was not comfortable presenting a confident “compatibility score” when the model itself was not reliable enough to support that confidence.

So I stopped Chemi.lol.

> **Not because the code could not produce a score,  
> but because the data could not support the level of promise I wanted the score to make.**

---

## What I Learned

The most valuable lesson was not how to find another algorithm.

### 1. A model ceiling may actually be a data ceiling

Before spending more time on model complexity,  
I should ask whether the target is adequately represented in the dataset at all.

### 2. A score you can generate is not automatically a score you should show

Software can always output a number.

If that number influences a user's judgment,  
**how much trust it deserves** matters more.

### 3. Stopping is also a product decision

Shipping is not the only valid ending to a project.

Deciding that the current evidence is not strong enough to justify further productization can be the more responsible decision.

---

## If I Started Again

I would begin with data-collection design, not model selection.

Instead of simply collecting more individual match histories, I would prioritize paired data that better represents compatibility:

- repeated games by the same duo
- sufficient samples across different pairings
- player skill changes over time
- role and champion combinations
- number of games played together
- stability of the relationship across sessions

Chemi.lol never became the service we originally imagined.

For me, it became something else:

**a project that taught me that “Should we keep building?” is also a question the data has to answer.**

[🔗 Related Post: When Accuracy Stops Improving](/posts/2026-08-19-chemi-lol-accuracy-ceiling/)  
[🔗 Analysis Repository](https://github.com/league-of-legend-project/Analysis)  
[🔗 Portfolio](/portfolio/)

</div>
