---
title: "🧪 Chemi.lol"
title_en: "🧪 Chemi.lol"
type: page
aliases:
  - /portfolio/chemi-lol/
description: "LoL 듀오 승률 조회 웹사이트의 초기 출시와, 출시하지 않은 예측 기능 실험을 구분해 정리한 사례."
description_en: "A League of Legends duo win-rate lookup site the team launched, and a separate prediction experiment that was never released."
---

<div class="lang-ko">

<a id="top"></a>

**기간** · 2024.12–2025.05  
**역할** · 프로젝트 발의 · 팀장 · 서비스 기획 · 통계 분석 · 예측 모델 실험  
**팀** · 5명  
**상태** · 조회 기능 출시(팀) · 예측 기능은 실험 단계에서 중단

[**Case Study ↓**](#case-study) · [**분석 저장소 ↗**](https://github.com/league-of-legend-project/Analysis)

---

<a id="case-study"></a>
## Case Study

| 구분 | 내용 |
|---|---|
| 목표 | 두 플레이어의 듀오 예상 승률과 궁합을 보여주는 웹사이트 |
| 출시 | 과거 함께한 경기 기록 기반의 듀오 승률 조회 기능(초기 버전, 팀 전체의 결과) |
| 실험 | 개인 지표와 포지션 조합을 이용한 예상 승률 예측 모델(실제 서비스에는 공개하지 않음) |
| 미구현 | 플레이 성향 유형 콘텐츠 LoLBTI(기획 단계) |

### 친구와 듀오 랭크를 하다가 시작한 프로젝트

친구와 League of Legends 듀오 랭크를 하던 중, 사람 사이의 궁합을 보는 것처럼 게임에서도 두 플레이어의 궁합을 확인할 수 있다면 재미있겠다는 생각이 들었습니다. 이 아이디어를 팀에 제안해 5명으로 프로젝트를 시작했고, 팀장을 맡았습니다.

목표는 함께 플레이한 기록이 충분한 듀오에게는 실제 승률을, 그렇지 않은 듀오에게는 각자의 플레이 지표와 포지션 조합을 바탕으로 예상 승률을 보여주는 것이었습니다. 팀은 이 중 과거 기록 기반의 듀오 승률 조회 기능을 초기 버전으로 출시했고, 예상 승률 예측은 실험 단계에 머물렀습니다. LoLBTI는 궁합 예측 모델의 입력값이 아닌 별도 콘텐츠로 기획했지만 구현하지 않았습니다.

### 내가 맡은 일과 팀원이 맡은 일

저는 서비스 기획, 플레이어 통계 분석, 변수 설계(Feature Engineering), 예측 모델 실험, 그리고 데이터가 제품화를 뒷받침하는지에 대한 판단을 맡았습니다.

데이터 수집은 다른 팀원이 주로 담당했습니다. EDA와 MongoDB 관리는 백엔드 팀원이 주도했고, 웹 구현과 AWS 배포, 도메인 연결은 백엔드 · 프런트엔드 팀원이 맡았습니다. 조회 기능 출시는 이렇게 팀 전체가 함께 만든 결과입니다.

### 두 사람의 관계를 데이터로 표현하기

개인 승률은 두 사람이 자주 함께 플레이하지 않았더라도 예상 승률을 계산할 수 있도록 포함했습니다. KDA, 킬 관여율, 시야 점수 등의 지표는 각 플레이어의 값뿐 아니라 두 사람의 평균과 차이도 함께 살펴봤습니다. GPM은 유효성을 확인하기 위한 실험 변수로 사용했습니다.

포지션 조합은 두 입력의 순서와 무관하도록 처리했습니다. 예를 들어 A가 서포터이고 B가 원거리 딜러인 경우와 그 반대 순서로 입력된 경우는 같은 조합입니다. 입력 순서가 별개의 특성처럼 학습되거나 데이터 분할에 영향을 주지 않도록 `Lane_combo`를 순서가 없는 조합으로 만들었습니다.

표본이 지나치게 적은 듀오의 일시적인 결과를 줄이기 위해 네 경기 이상 함께한 듀오를 최소 기준으로 정했습니다. 팀이 수집한 데이터에서 이 기준을 충족한 듀오 후보는 2,737개였고, 포지션 등 조건을 반영해 최종 5,145개의 학습 데이터를 구성했습니다.

### 모델을 바꿔도 해결되지 않았던 문제

Random Forest, Ridge, SVR, XGBoost를 비교하고 MLP와 앙상블도 실험했습니다. 아래는 공개 분석 저장소에 남은 실험 중 학습 데이터 5,145개를 80:20으로 학습 · 테스트 분할한 대표 결과입니다. 예측 대상은 듀오 승률(%)이며, 이 모델은 실제 서비스에 배포하지 않았습니다.

| 실험 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge | 9.94 | 12.60 | 0.203 |
| Ridge · XGBoost · MLP 평균 앙상블 | 9.92 | 12.59 | 0.204 |

모델을 복잡하게 만들어도 개선 폭은 거의 없었습니다. 스케일링과 변수 선택 방식을 바꾼 실험에서도 결과는 비슷했습니다. 처음에는 더 적합한 모델을 찾지 못한 문제라고 생각했지만, 반복할수록 모델보다 데이터가 가진 한계가 더 크다고 판단했습니다.

League of Legends의 결과에는 챔피언과 숙련도, 포지션, 팀 조합과 상대 조합 등 많은 요소가 영향을 줍니다. 반면 한 플레이어에게서 안정적으로 얻을 수 있는 최근 경기 기록은 약 100경기 수준이었습니다. 듀오 데이터도 바텀 포지션 조합에 집중되어 있었고, 조합별 표본 수의 격차가 컸습니다. 챔피언이 100종 이상인 환경에서는 비주류 챔피언을 사용하는 플레이어의 지표가 적은 표본 때문에 크게 흔들릴 가능성도 있었습니다.

따라서 이 결과만으로 두 사람의 고유한 궁합을 안정적으로 설명하기 어려웠습니다. 개인 실력이 높은 플레이어와 함께할수록 팀 전체의 승률이 올라가는 현상도 존재하므로, 예측 승률이 순수한 관계의 효과만을 나타낸다고 말할 수도 없었습니다.

### 조회 기능 출시 이후 중단을 제안한 이유

조회 기능은 출시했지만, 예측 기능을 서비스에 추가할 수 있다는 것과 그 결과를 신뢰할 수 있다는 것은 다른 문제였습니다. 예측값을 화면에 보여주는 것은 가능했지만, 데이터가 충분하지 않은 상태에서 그 값을 두 사람의 궁합이라고 단정하고 싶지는 않았습니다.

분석과 모델 실험을 맡은 팀장으로서 데이터의 편향과 모델의 낮은 설명력을 팀에 공유하고 프로젝트 종료를 제안했습니다. 팀원들과 논의한 끝에 종료에 합의했습니다.

이 프로젝트를 통해 모델 선택보다 먼저 예측하려는 개념에 맞는 데이터를 확보할 수 있는지 검토해야 한다는 점을 배웠습니다. 또한 구현할 수 있는 기능이라도 근거가 충분하지 않다면 출시하지 않는 것이 제품을 책임지는 결정일 수 있다는 것을 경험했습니다.

🔗 [Analysis Repository](https://github.com/league-of-legend-project/Analysis)

[↑ Top](#top)

</div>

<div class="lang-en" style="display:none">

<a id="top-en"></a>

**Period** · Dec 2024–May 2025  
**Role** · Initiator · Team Lead · Product Planning · Statistical Analysis · Prediction Experiments  
**Team** · 5 members  
**Status** · Lookup feature shipped (team) · Prediction feature stopped at the experimental stage

[**Case Study ↓**](#case-study-en) · [**Analysis Repository ↗**](https://github.com/league-of-legend-project/Analysis)

---

<a id="case-study-en"></a>
## Case Study

| | |
|---|---|
| Goal | A website showing predicted duo win rates and compatibility between two players |
| Shipped | Initial version: duo win-rate lookup based on shared match history (team result) |
| Experimental | A model predicting duo win rates from individual stats and role combination (never released) |
| Not built | LoLBTI, a play-style profile (planning only) |

### A project inspired by playing duo queue with a friend

The idea came to me while playing League of Legends duo queue with a friend: people enjoy checking their compatibility with each other, so they might also enjoy seeing how well they play together in a game. I proposed it to a group, started a five-person project, and led the team.

The goal was to show the observed win rate for duos with enough shared matches and a predicted win rate, based on individual stats and role combination, for everyone else. The team launched the first part, a win-rate lookup built on past shared matches. The prediction never moved beyond experiments. LoLBTI was planned as separate content, not as a model input, and was not built.

### What I did and what the team did

I focused on product planning, player statistics analysis, feature engineering, prediction experiments, and assessing whether the data justified putting a model into the product.

Another teammate led most of the data collection. A backend teammate led the exploratory data analysis and managed MongoDB, and the backend and frontend teammates built the web application, deployed it on AWS, and connected the domain. The lookup launch was a team result.

### Representing a pair of players

I included each player's individual win rate so that a prediction would still be possible when the two users had rarely played together. For statistics such as KDA, kill participation, and vision score, I examined both the average and the difference between the two players. GPM was included as an experimental feature.

Role combinations were made order-invariant. A support and an AD carry should represent the same pairing regardless of which player was entered first. I therefore encoded `Lane_combo` as an unordered combination so that an arbitrary input order would not become a learned feature or affect how otherwise equivalent records were separated.

To reduce the impact of one-off results from very small samples, we required a duo to have played at least four matches together. The team's collected data contained 2,737 duo candidates that met this threshold, from which we built 5,145 training records with role and other conditions applied.

### A problem that model changes did not solve

I compared Random Forest, Ridge, SVR, and XGBoost, then tested an MLP and several ensembles. The table shows representative results from the public analysis repository, using an 80:20 train–test split of the 5,145 records. The target was duo win rate (%), and none of these models were deployed.

| Experiment | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge | 9.94 | 12.60 | 0.203 |
| Ridge · XGBoost · MLP mean ensemble | 9.92 | 12.59 | 0.204 |

The additional complexity produced almost no improvement. Experiments with scaling and feature selection stayed in a similar range. I first treated the plateau as a model-selection problem, but repeated experiments pointed to a more fundamental limitation in the data.

League of Legends outcomes depend on many interacting factors, including champion choice and mastery, player roles, team composition, and the opposing lineup. In contrast, the usable recent history for an individual player was limited to roughly 100 matches. Duo observations were also concentrated in bottom-lane pairings, with large differences in sample size across role and champion combinations. In a game with more than 100 champions, metrics for players using less popular champions could become especially unstable.

The data therefore could not reliably isolate compatibility between two particular people. Playing with a stronger individual also tends to raise a five-player team's chance of winning, so a predicted win rate could not be presented as a pure measure of interpersonal synergy.

### Why I proposed stopping after the lookup launch

The lookup feature was live, but being able to add a prediction was not the same as being able to trust it. We could have displayed a number, but I was not comfortable presenting it as a measure of compatibility when the data did not support it.

As the team lead responsible for the analysis and model experiments, I shared the dataset bias and the model's low explanatory power with the team and proposed ending the project. After discussing the evidence, the team agreed.

This project taught me to check whether the right data can be collected before optimizing a model. It also showed me that not releasing a feature you are able to build can be the responsible product decision when its central claim is not well supported.

🔗 [Analysis Repository](https://github.com/league-of-legend-project/Analysis)

[↑ Top](#top-en)

</div>
