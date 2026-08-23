---
title: "🧪 Chemi.lol"
title_en: "🧪 Chemi.lol"
type: page
aliases:
  - /portfolio/chemi-lol/
description: "LoL 듀오 데이터를 바탕으로 실제 · 예측 승률을 제공한 웹 서비스와 모델 검증 기록."
description_en: "A web service for observed and predicted League of Legends duo win rates, and a case study in testing the limits of the underlying data."
---

<div class="lang-ko">

<a id="top"></a>

**기간** · 2024.12–2025.05  
**역할** · 팀장 · 서비스 기획 · AI 개발  
**팀** · 5명  
**주요 작업** · API 기반 데이터 수집 · 변수 설계 · 회귀 모델 실험 · 제품 의사결정

[**Case Study ↓**](#case-study) · [**분석 저장소 ↗**](https://github.com/league-of-legend-project/Analysis)

---

<a id="case-study"></a>
## Case Study

### 친구와 듀오 랭크를 하다가 시작한 서비스

친구와 League of Legends 듀오 랭크를 하던 중, 사람 사이의 궁합을 보는 것처럼 게임에서도 두 플레이어의 궁합을 확인할 수 있다면 재미있겠다는 생각이 들었습니다. 여기서 두 플레이어의 전적을 분석해 함께 플레이했을 때의 승률을 보여주는 웹 서비스 Chemi.lol을 기획했습니다.

함께 플레이한 기록이 충분한 듀오에게는 실제 승률을 보여주고, 그렇지 않은 경우에는 각자의 플레이 지표와 포지션 조합을 바탕으로 예상 승률을 제공하는 방식이었습니다. 플레이 성향을 유형으로 보여주는 LoLBTI도 별도의 결과 콘텐츠로 기획했지만, 궁합 예측 모델의 입력값으로 사용하지는 않았으며 실제 서비스에서는 준비 중인 기능으로 남았습니다.

### 내가 맡은 일

5명으로 구성된 팀에서 팀장을 맡아 서비스의 방향을 정하고 AI 개발을 담당했습니다. Riot API를 활용한 수집 코드를 작성하고, 두 플레이어의 관계를 모델이 학습할 수 있도록 변수를 설계했으며 여러 회귀 모델을 비교했습니다.

EDA는 백엔드 팀원이 주도했고 MongoDB도 해당 팀원이 관리했습니다. 웹 구현과 AWS 배포는 백엔드 · 프런트엔드 팀원이 맡았습니다. 저는 분석 결과를 바탕으로 어떤 변수를 실험하고 모델의 결과를 제품에서 어떻게 다룰지 판단했습니다. 서비스는 도메인을 구매해 실제로 배포하는 단계까지 진행했습니다.

### 두 사람의 관계를 데이터로 표현하기

개인 승률은 두 사람이 자주 함께 플레이하지 않았더라도 예상 승률을 계산할 수 있도록 포함했습니다. KDA, 킬 관여율, 시야 점수 등의 지표는 각 플레이어의 값뿐 아니라 두 사람의 평균과 차이도 함께 살펴봤습니다. GPM은 유효성을 확인하기 위한 실험 변수로 사용했습니다.

포지션 조합은 두 입력의 순서와 무관하도록 처리했습니다. 예를 들어 A가 서포터이고 B가 원거리 딜러인 경우와 그 반대 순서로 입력된 경우는 같은 조합입니다. 입력 순서가 별개의 특성처럼 학습되거나 데이터 분할에 영향을 주지 않도록 `Lane_combo`를 순서가 없는 조합으로 만들었습니다.

표본이 지나치게 적은 듀오의 일시적인 결과를 줄이기 위해 네 경기 이상 함께한 듀오를 최소 기준으로 정했습니다. 수집 과정에서 이 기준을 충족한 듀오 후보 2,737개를 확보했고, 포지션 등 조건을 반영해 최종 5,145개의 학습 데이터를 구성했습니다.

### 모델을 바꿔도 해결되지 않았던 문제

Random Forest, Ridge, SVR, XGBoost를 비교하고 MLP와 앙상블도 실험했습니다. 공개 분석 저장소에 남은 80:20 학습 · 테스트 분할 실험의 대표 결과는 다음과 같습니다.

| 실험 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge | 9.94 | 12.60 | 0.203 |
| Ridge · XGBoost · MLP 평균 앙상블 | 9.92 | 12.59 | 0.204 |

모델을 복잡하게 만들었지만 개선 폭은 거의 없었습니다. 스케일링과 변수 선택 방식을 바꾼 실험에서도 결과는 비슷했습니다. 처음에는 더 적합한 모델을 찾지 못한 문제라고 생각했지만, 반복할수록 모델보다 데이터가 가진 한계가 더 크다고 판단했습니다.

League of Legends의 결과에는 챔피언과 숙련도, 포지션, 팀 조합과 상대 조합 등 많은 요소가 영향을 줍니다. 반면 한 플레이어에게서 안정적으로 얻을 수 있는 최근 경기 기록은 약 100경기 수준이었습니다. 듀오 데이터도 바텀 포지션 조합에 집중되어 있었고, 조합별 표본 수의 격차가 컸습니다. 챔피언이 100종 이상인 환경에서는 비주류 챔피언을 사용하는 플레이어의 지표가 적은 표본 때문에 크게 흔들릴 가능성도 있었습니다.

따라서 이 결과만으로 두 사람의 고유한 궁합을 안정적으로 설명하기 어려웠습니다. 개인 실력이 높은 플레이어와 함께할수록 팀 전체의 승률이 올라가는 현상도 존재하므로, 예측 승률이 순수한 관계의 효과만을 나타낸다고 말할 수도 없었습니다.

### 출시 이후 중단을 제안한 이유

Chemi.lol은 AWS와 구매한 도메인을 통해 실제 서비스로 배포했습니다. 그러나 서비스를 만들 수 있다는 사실과 결과를 신뢰할 수 있다는 것은 다른 문제였습니다. 예측값을 보여주는 기능은 구현할 수 있었지만, 데이터가 충분하지 않은 상태에서 그 값을 두 사람의 궁합이라고 단정하고 싶지는 않았습니다.

AI 개발을 담당한 팀장으로서 데이터의 편향과 모델의 설명력 한계를 팀에 공유하고 프로젝트 중단을 제안했습니다. 팀원들과 논의한 끝에 중단에 합의했습니다.

이 프로젝트를 통해 모델 선택보다 먼저 예측하려는 개념에 맞는 데이터를 확보할 수 있는지 검토해야 한다는 점을 배웠습니다. 또한 배포까지 마친 결과물이라도 근거가 충분하지 않다면 멈추는 것이 제품을 책임지는 결정일 수 있다는 것을 경험했습니다.

🔗 [Analysis Repository](https://github.com/league-of-legend-project/Analysis)

[↑ Top](#top)

</div>

<div class="lang-en" style="display:none">

<a id="top-en"></a>

**Period** · Dec 2024–May 2025  
**Role** · Team Lead · Product Planning · AI Development  
**Team** · 5 members  
**Scope** · API-based data collection · Feature design · Regression experiments · Product decisions

[**Case Study ↓**](#case-study-en) · [**Analysis Repository ↗**](https://github.com/league-of-legend-project/Analysis)

---

<a id="case-study-en"></a>
## Case Study

### A service inspired by playing duo queue with a friend

Chemi.lol began with a casual thought while I was playing League of Legends duo queue with a friend: people enjoy checking their compatibility with each other, so they might also enjoy seeing how well they play together in a game. I turned that idea into a web service that analyzed two players' match histories and presented their win rate as a duo.

When enough shared matches were available, the service displayed the duo's observed win rate. Otherwise, it could provide a predicted win rate based on the players' individual statistics and role combination. We also planned LoLBTI as a separate result that described play styles. It was not an input to the compatibility model and remained a planned feature rather than a completed part of the service.

### My role

I led the five-person team, set the product direction, and handled the AI development. I wrote Riot API data-collection code, designed features to represent the relationship between two players, and compared several regression models.

A backend teammate led the exploratory data analysis and managed MongoDB. The backend and frontend teammates implemented the web application and deployed it on AWS. My responsibility was to use the analysis to decide which variables to test and how the model's output should be treated in the product. The team purchased a domain and brought the service into production.

### Representing a pair of players

I included each player's individual win rate so that the service could estimate an outcome even when the two users had rarely played together. For statistics such as KDA, kill participation, and vision score, I examined both the average and the difference between the two players. GPM was included as an experimental feature.

Role combinations were made order-invariant. A support and an AD carry should represent the same pairing regardless of which player was entered first. I therefore encoded `Lane_combo` as an unordered combination so that an arbitrary input order would not become a learned feature or affect how otherwise equivalent records were separated.

To reduce the impact of one-off results from extremely small samples, we required a duo to have played at least four matches together. The collection pipeline identified 2,737 duo candidates that met this threshold, from which we constructed 5,145 training records with role and other conditions applied.

### A problem that model changes did not solve

I compared Random Forest, Ridge, SVR, and XGBoost, then tested an MLP and several ensembles. Representative results from an 80:20 train–test split in the public analysis repository were:

| Experiment | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge | 9.94 | 12.60 | 0.203 |
| Ridge · XGBoost · MLP mean ensemble | 9.92 | 12.59 | 0.204 |

The additional complexity produced almost no improvement. Experiments with scaling and feature-selection methods stayed in a similar range. I initially treated the plateau as a model-selection problem, but repeated experiments pointed to a more fundamental limitation in the data.

League of Legends outcomes depend on many interacting factors, including champion choice and mastery, player roles, team composition, and the opposing lineup. In contrast, the usable recent history for an individual player was limited to roughly 100 matches. Duo observations were also concentrated in bottom-lane pairings, with large differences in sample size across role and champion combinations. In a game with more than 100 champions, metrics for players using less popular champions could become especially unstable.

The available data therefore could not reliably isolate compatibility between two particular people. Playing with a stronger individual also tends to raise a five-player team's chance of winning, so a predicted win rate could not be presented as a pure measure of interpersonal synergy.

### Why I proposed stopping after deployment

Chemi.lol reached production through AWS and a purchased domain. Shipping the service, however, did not make every result trustworthy. We could generate and display a prediction, but I was not comfortable presenting it as a confident measure of compatibility when the supporting data was insufficient.

As the team lead responsible for the model, I explained the dataset bias and the model's limited explanatory power to the team and proposed that we stop the project. We discussed the evidence and agreed to discontinue it.

This project taught me to ask whether the right data can be collected before optimizing a model. It also showed me that stopping a deployed product can be the responsible decision when its central claim is not supported well enough.

🔗 [Analysis Repository](https://github.com/league-of-legend-project/Analysis)

[↑ Top](#top-en)

</div>
