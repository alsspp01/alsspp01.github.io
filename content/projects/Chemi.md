---
title: "🧪 Chemi.lol"
title_en: "🧪 Chemi.lol"
type: page
description: "LoL 듀오 데이터를 활용한 관계 기반 feature engineering 및 회귀 모델 실험."
description_en: "Relationship-focused feature engineering and regression experiments using LoL duo data."
---

<div class="lang-ko">

[🔗 Portfolio Case Study](/portfolio/chemi-lol/)  
[🔗 Analysis Repository](https://github.com/league-of-legend-project/Analysis)

## Pipeline

```text
Player / Match / Timeline → MongoDB → Duo Detection → Feature Engineering → Regression → Score
```

## Features
- KDA / KP / GPM / Vision / Win Rate difference
- order-invariant Lane combination

## Models
- Random Forest
- Ridge
- SVR
- XGBoost
- MLP
- simple ensembles

모델 복잡도를 높여도 R²가 크게 개선되지 않았고, **현재 feature만으로 궁합을 설명하는 데 한계가 있다**는 점도 중요한 결과였습니다.

[🔗 Portfolio Case Study](/portfolio/chemi-lol/)

</div>

<div class="lang-en" style="display:none">

[🔗 Portfolio Case Study](/portfolio/chemi-lol/)  
[🔗 Analysis Repository](https://github.com/league-of-legend-project/Analysis)

## Pipeline

```text
Player / Match / Timeline → MongoDB → Duo Detection → Feature Engineering → Regression → Score
```

## Features
- KDA / KP / GPM / Vision / Win Rate differences
- order-invariant Lane combinations

## Models
- Random Forest
- Ridge
- SVR
- XGBoost
- MLP
- simple ensembles

Increasing model complexity did not produce a large improvement in R², which made **the limits of the available features** an important part of the result.

[🔗 Portfolio Case Study](/portfolio/chemi-lol/)

</div>
