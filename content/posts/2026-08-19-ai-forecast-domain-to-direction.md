---
title: "기상학을 모르는 PM이 요구사항을 정리하기까지"
title_en: "Learning the Domain Before Writing Requirements"
date: 2026-08-19
description: "모르는 분야의 과제에서 기능 목록보다 업무 흐름을 먼저 정리한 이유"
description_en: "Why I mapped the workflow before writing a feature list in an unfamiliar domain"
type: "post"
tags: ["PM", "Planning", "Domain", "AI"]
---

<div class="lang-ko">

AI 예보지원 과제에 PM으로 들어갔을 때 기상학을 전혀 몰랐습니다.  
첫 직장이었고 따로 온보딩도 없었습니다.  
개발은 약 3개월째 멈춰 있었고, 고객사와 개발 기관의 대화는 계속 어긋났습니다.

## 누가 틀린 게 아니었음

양쪽 이야기를 계속 들어 보니, 고객사는 예보 업무 순서로, 개발 기관은 기술이 동작하는 순서로 말하고 있었습니다.  
같은 단어를 쓰면서 다른 과정을 떠올리고 있었습니다.

## 기능 이름보다 먼저 확인한 것

국립기상과학원 연구진과 담당자에게 업무 흐름을 설명 듣고, 요구사항마다 아래를 먼저 봤습니다.

- 이 정보는 예보 과정의 어느 시점에 필요한가
- 지금 사람이 직접 비교하거나 판단하는 부분은 어디인가
- AI가 그 판단을 대신하는가, 판단할 재료를 정리해 주는가
- 각 기관의 결과는 앞뒤 단계와 어떻게 이어지는가

세 번째 질문이 예측 모델의 목표를 회귀에서 분류로 바꾸자는 제안으로 이어졌습니다.

## 요구사항을 빨리 쓰는 게 빠른 건 아니었음

모호할수록 기능 목록부터 쓰고 싶어집니다.  
하지만 사용자의 일을 모르고 쓴 목록은 개발이 시작된 뒤 다시 고치게 됩니다.

새 과제를 시작하면 안개 낀 게임 맵을 떠올립니다.  
보이지 않는 곳에 목적지만 찍기보다, 돌아다니며 지형을 밝혀야 다음 판단이 쉬워집니다.  
이 과제에서는 업무를 가장 잘 아는 사람들에게 배우고 용어를 익히는 일이 그 맵을 밝히는 일이었습니다.

관련 사례: [Alpha Weather](/projects/ai-forecast-support/)

</div>

<div class="lang-en" style="display:none">

When I joined an AI forecasting project as PM, I knew nothing about meteorology.  
It was my first job, with no formal onboarding.  
Development had been stalled for about three months, and the client and developers kept talking past each other.

## Nobody was wrong

Listening to both sides, I found the client described things in the order of forecasting work, and the developers in the order the technology runs.  
They used the same words and pictured different processes.

## What I checked before feature names

After NIMS researchers and staff walked me through the workflow, I asked of every requirement:

- At what point in forecasting is this information needed?
- Where does a person compare or decide today?
- Does the AI make that decision, or prepare what is needed to make it?
- How does each organization's output connect to the steps around it?

The third question led to the proposal to reframe the prediction model from regression to classification.

## Writing requirements fast was not faster

The vaguer things are, the more tempting it is to start with a feature list.  
A list written without knowing the user's work gets rewritten once development starts.

At the start of a project I picture a map under fog of war.  
Rather than dropping a waypoint somewhere unseen, I walk around until the terrain shows.  
Here, learning from the people who knew the work best, and learning their vocabulary, was how the map cleared.

Related case study: [Alpha Weather](/projects/ai-forecast-support/)

</div>
