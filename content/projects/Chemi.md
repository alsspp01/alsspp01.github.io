---
title: "🧪 Chemi.lol"
title_en: "🧪 Chemi.lol"
type: page
description: "LoL 듀오 궁합 예측을 시도했지만 데이터와 정확도 한계로 중단한 ML 프로젝트."
description_en: "An ML experiment in LoL duo compatibility that was discontinued after hitting data and accuracy limits."
---

<div class="lang-ko">

프로젝트는 여러 회귀 모델과 feature engineering을 거쳤지만  
성능이 일정 수준에서 정체되었고, 결국 중단했습니다.

[🔗 Portfolio Case Study](/portfolio/chemi-lol/)  
[🔗 Analysis Repository](https://github.com/league-of-legend-project/Analysis)

---

## Pipeline

```text
Player / Match / Timeline
        ↓
MongoDB
        ↓
Duo Detection
        ↓
Feature Engineering
        ↓
Regression Experiments
        ↓
Accuracy Ceiling
        ↓
Discontinued
```

---

## Features

- KDA difference
- Kill Participation difference
- Gold Per Minute difference
- Vision difference
- Win Rate difference
- order-invariant Lane combination

---

## Models

- Random Forest
- Ridge
- SVR
- XGBoost
- MLP
- simple ensembles

한 실험에서 Ridge와 ensemble 모두 R²가 약 **0.20** 수준에 머물렀고,  
모델과 전처리 방식을 바꿔도 의미 있는 개선이 이어지지 않았습니다.

---

## Why It Stopped

제가 가장 큰 원인으로 본 것은  
**궁합을 학습할 만큼 충분하고 다양한 듀오 데이터를 확보하지 못한 것**입니다.

이는 단일 실험으로 원인을 확정한 결과라기보다  
반복된 성능 정체와 데이터 수집 과정에서 내린 가장 유력한 가설입니다.

낮은 설명력을 가진 점수를 제품에서 자신 있게 보여주는 것은 적절하지 않다고 판단해 프로젝트를 중단했습니다.

> **“점수를 만들 수 있는가?”보다 “그 점수를 믿어도 되는가?”가 더 중요했습니다.**

자세한 실험 과정과 중단 판단은 Portfolio Case Study에 정리했습니다.

[🔗 Portfolio Case Study](/portfolio/chemi-lol/)

</div>

<div class="lang-en" style="display:none">

After multiple regression and feature-engineering experiments, performance plateaued and the project was discontinued.

[🔗 Portfolio Case Study](/portfolio/chemi-lol/)  
[🔗 Analysis Repository](https://github.com/league-of-legend-project/Analysis)

---

## Pipeline

```text
Player / Match / Timeline
        ↓
MongoDB
        ↓
Duo Detection
        ↓
Feature Engineering
        ↓
Regression Experiments
        ↓
Accuracy Ceiling
        ↓
Discontinued
```

---

## Features

- KDA difference
- Kill Participation difference
- Gold Per Minute difference
- Vision difference
- Win Rate difference
- order-invariant Lane combination

---

## Models

- Random Forest
- Ridge
- SVR
- XGBoost
- MLP
- simple ensembles

In one set of experiments, both Ridge and the ensemble stayed around an R² of **0.20**,  
and changing models and preprocessing did not lead to sustained improvement.

---

## Why It Stopped

My strongest hypothesis was that  
**we did not have enough diverse duo data to learn compatibility reliably.**

That is not a formally isolated causal result; it is the explanation I found most convincing after repeated performance plateaus and the limitations of the available paired data.

I did not want to present a confident compatibility score when the model did not support that confidence, so I stopped the project.

> **“Can we produce a score?” mattered less than “Does the score deserve to be trusted?”**

The full experiment history and stopping decision are documented in the Portfolio case study.

[🔗 Portfolio Case Study](/portfolio/chemi-lol/)

</div>
