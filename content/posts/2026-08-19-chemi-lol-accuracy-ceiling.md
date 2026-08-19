---
title: "🧪 정확도가 더 오르지 않을 때"
title_en: "🧪 When Accuracy Stops Improving"
date: 2026-08-19
description: "Chemi.lol에서 모델을 계속 바꾸다가 결국 데이터의 한계와 프로젝트 중단을 선택한 과정."
description_en: "What Chemi.lol taught me about recognizing a data ceiling and deciding when not to keep optimizing the model."
type: "post"
tags: ["Chemi.lol", "Machine Learning", "Data", "Product"]
---

<div class="lang-ko">

머신러닝 프로젝트를 시작하면 자연스럽게 이런 흐름을 기대하게 됩니다.

> 모델을 만든다.  
> 성능을 본다.  
> 부족하면 개선한다.  
> 더 좋은 모델을 찾는다.

Chemi.lol도 그렇게 시작했습니다.

League of Legends 듀오 데이터를 바탕으로  
두 플레이어가 얼마나 잘 맞는지를 예측해보고 싶었습니다.

처음에는 성능이 부족하면 **모델을 더 잘 만들면 된다**고 생각했습니다.

프로젝트를 멈출 때쯤에는 생각이 꽤 달라져 있었습니다.

---

## 궁합을 숫자로 만들기

두 플레이어를 각각 보는 것만으로는 부족했습니다.

그래서 KDA, KP, GPM, Vision, Win Rate 같은 지표에서  
둘 사이의 차이를 만들고,

Lane도 Player 1 / Player 2의 순서에 영향을 받지 않는 조합으로 바꿨습니다.

즉,

> **“A는 얼마나 잘하는가?”**

보다

> **“A와 B는 어떻게 다른가?”**

를 feature로 표현하려고 했습니다.

그다음 여러 회귀 모델을 비교했습니다.

- Random Forest
- Ridge
- SVR
- XGBoost
- MLP
- ensemble

scaling과 feature selection도 바꿔봤습니다.

---

## 그런데 계속 비슷한 곳으로 돌아왔다

한 실험에서 Ridge의 R²는 약 **0.203**,  
Ridge + XGBoost + MLP 평균 ensemble은 약 **0.204**였습니다.

물론 하나의 숫자만 보고 프로젝트를 판단한 것은 아닙니다.

문제는 여러 실험을 해도  
**성능이 계속 비슷한 구간으로 돌아왔다는 점**이었습니다.

처음에는 이런 생각을 했습니다.

> **“아직 좋은 모델을 못 찾은 건가?”**

그래서 모델을 바꿨습니다.

그다음에는 전처리를 바꿨습니다.

feature selection도 해보고,  
앙상블도 해봤습니다.

그런데 어느 순간 질문을 바꿔야 했습니다.

> **“모델을 더 바꾸는 게 맞나?”**

---

## 모델의 ceiling인가, 데이터의 ceiling인가

제가 결국 가장 큰 원인이라고 생각한 것은  
**궁합을 학습하기에 충분한 데이터를 확보하지 못했다는 것**입니다.

여기서 중요한 건 “게임 데이터가 적었다”는 의미와 조금 다릅니다.

개별 플레이어의 경기 데이터는 모을 수 있습니다.

하지만 제가 필요했던 것은  
**특정 두 사람이 함께했을 때의 관계를 학습할 수 있는 paired data**였습니다.

같은 듀오가 충분히 반복해서 플레이한 데이터,  
다양한 조합에서 비교할 수 있는 데이터가 있어야  
“이 두 사람이 함께했을 때”라는 target을 안정적으로 학습할 수 있습니다.

그런 데이터는 충분히 확보하기 어려웠습니다.

그리고 궁합이라는 개념 자체도  
KDA 차이 몇 개로 완전히 설명될 만큼 단순하지 않았습니다.

챔피언 조합, 포지션, 플레이 경험, 서로 함께한 횟수, 당시 실력 변화처럼  
빠져 있는 변수가 많을 수 있습니다.

다만 이것은 제가 **원인을 실험적으로 완전히 분리해 증명했다는 의미는 아닙니다.**

반복된 성능 정체와 데이터 수집 과정에서  
가장 가능성이 높다고 판단한 설명입니다.

---

## 그래도 점수는 만들 수 있었다

여기서 프로젝트를 계속하는 선택도 가능했습니다.

예측값을 보정하고,  
0~100 범위의 궁합 점수로 가공하고,  
UI를 붙이면 서비스처럼 보이게 만들 수 있습니다.

기술적으로 **숫자를 출력하는 것** 자체는 어렵지 않았습니다.

그런데 어느 순간 이 질문이 더 중요해졌습니다.

> **“이 점수를 사용자에게 얼마나 믿으라고 말할 수 있지?”**

R²가 낮고,  
모델을 바꿔도 성능이 크게 나아지지 않고,  
target을 충분히 설명할 데이터도 부족하다고 생각하는데,

화면에서

> **“두 분의 궁합은 87점입니다.”**

라고 자신 있게 보여주는 건 마음에 들지 않았습니다.

---

## 그래서 중단했다

Chemi.lol은 결국 중단했습니다.

처음에는 프로젝트 중단을  
완성하지 못한 결과처럼 생각하기도 했습니다.

지금은 조금 다르게 봅니다.

데이터가 충분하지 않은데도  
모델을 계속 복잡하게 만들고 서비스를 완성하는 것이  
항상 더 좋은 선택은 아닙니다.

특히 숫자가 사람에게 **객관적인 결과처럼 보이는 제품**이라면 더 그렇습니다.

그래서 이 프로젝트의 마지막 결정은  
새 모델을 하나 더 만드는 것이 아니라

> **“현재 조건에서는 여기까지가 이 데이터가 말할 수 있는 범위다.”**

라고 인정하는 것이었습니다.

---

## 다시 시작한다면

다시 Chemi.lol을 만든다면  
모델 notebook부터 열지는 않을 것 같습니다.

먼저 **데이터를 어떻게 확보할지**부터 설계할 겁니다.

예를 들면:

- 동일 듀오의 반복 플레이 횟수
- 듀오별 최소 sample size
- 포지션 조합
- 챔피언 조합
- 함께 플레이한 기간
- 시간에 따른 개인 실력 변화
- 첫 듀오와 장기 듀오의 구분

같은 조건을 먼저 정의하고,

**정말 궁합을 학습할 수 있는 dataset인지** 확인한 뒤 모델을 고를 것 같습니다.

---

## 프로젝트를 끝내면서 남은 것

Chemi.lol은 배포된 서비스로 남지는 않았습니다.

대신 꽤 선명한 질문 하나가 남았습니다.

> **모델이 더 좋아질 수 있는가?**

만 물으면  
계속 다음 실험을 할 수 있습니다.

하지만 제품을 만든다면 한 가지를 더 물어야 합니다.

> **이 데이터로 이 문제를 풀 수 있다는 전제부터 맞는가?**

Chemi.lol에서 제가 가장 늦게 배운 질문이고,  
다음 데이터 프로젝트에서는 가장 먼저 확인하고 싶은 질문입니다.

[🔗 Chemi.lol Case Study](/portfolio/chemi-lol/)  
[🔗 Analysis Repository](https://github.com/league-of-legend-project/Analysis)

</div>

<div class="lang-en" style="display:none">

Machine-learning projects often come with an intuitive loop:

> build a model,  
> measure it,  
> improve it,  
> find a better model.

Chemi.lol started that way.

I wanted to use League of Legends duo data to estimate how well two players fit together.

At the beginning, I assumed weak performance meant **the model needed to get better**.

By the time I stopped the project, I thought about the problem very differently.

---

## Turning compatibility into numbers

Looking at each player independently was not enough.

So I created differences between metrics such as KDA, KP, GPM, Vision, and Win Rate,  
and normalized lane combinations so Player 1 / Player 2 ordering did not create a different category.

The features were trying to represent:

> **“How good is Player A?”**

less, and:

> **“How are Player A and Player B different or complementary?”**

more.

Then I compared several regression approaches:

- Random Forest
- Ridge
- SVR
- XGBoost
- MLP
- ensembles

I also tried different scaling and feature-selection approaches.

---

## And the experiments kept returning to the same range

In one experiment, Ridge reached an R² of roughly **0.203**.

A simple Ridge + XGBoost + MLP average reached roughly **0.204**.

I did not stop because of one metric from one run.

The problem was that **many different experiments kept returning to a similar performance range.**

My first reaction was:

> **“Maybe I just haven't found the right model yet.”**

So I changed the model.

Then the preprocessing.

Then feature selection.

Then the ensemble.

Eventually, the question itself needed to change:

> **“Is another model actually the thing we are missing?”**

---

## A model ceiling, or a data ceiling?

My strongest hypothesis became that  
**we did not have enough data of the right kind to learn compatibility reliably.**

That is different from simply saying “there was not enough game data.”

Individual match histories were available.

What the target really needed was **paired data about the same two players playing together**, with enough repetition and enough variety across different pairings.

That kind of dataset was much harder to secure.

And compatibility was probably more complicated than a few differences in individual performance statistics.

Champion combinations, roles, time spent playing together, changing player skill, and many other variables could matter.

This is not a causal explanation I formally isolated and proved.

It is the explanation I found most convincing after repeated performance plateaus and the practical limitations of collecting paired duo data.

---

## We could still produce a score

The project did not become technically impossible.

We could calibrate the prediction,  
map it onto a 0–100 scale,  
and build a UI around it.

Producing **a number** was not the difficult part.

A different question became more important:

> **“How much should we ask the user to trust this number?”**

If the explanatory power was weak,  
model changes were not moving it much,  
and I already doubted whether the dataset represented the target well enough,

then confidently displaying:

> **“Your compatibility score is 87.”**

did not feel like a good product decision.

---

## So I stopped

Chemi.lol was eventually discontinued.

At first, stopping felt a little like failing to finish.

I do not see it that way now.

When the data is not strong enough,  
continuing to increase model complexity and polishing the service is not automatically the better choice.

That matters even more when a numerical output can look objective to the user.

The final decision in Chemi.lol was not another model.

It was accepting:

> **“This is about as much as the current data can responsibly tell us.”**

---

## If I started again

If I rebuilt Chemi.lol, I would not begin with the model notebook.

I would begin with **data-collection design**.

For example:

- repeated games by the same duo
- a minimum sample size per pairing
- role combinations
- champion combinations
- how long the pair has played together
- player skill changes over time
- separating new duos from established duos

I would want to answer:

**“Do we actually have a dataset capable of representing compatibility?”**

before spending much time selecting a model.

---

## What remained after the project ended

Chemi.lol did not survive as a deployed service.

But it left me with a clearer question.

If I only ask:

> **Can the model get better?**

there is always another experiment to try.

For a product, I also need to ask:

> **Was the assumption that this data can solve this problem valid in the first place?**

It was the question I learned latest in Chemi.lol.

It is probably the question I would ask first on the next data project.

[🔗 Chemi.lol Case Study](/portfolio/chemi-lol/)  
[🔗 Analysis Repository](https://github.com/league-of-legend-project/Analysis)

</div>
