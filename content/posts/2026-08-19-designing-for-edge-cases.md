---
title: "만들기 전에 한 번 플레이해보는 기획"
title_en: "Playing a Design Before It Exists"
date: 2026-08-19
description: "Arcanum Nights를 만들며 정상 흐름 밖의 플레이어 행동을 미리 검토한 방법"
description_en: "How I think through player behavior outside the happy path, from Arcanum Nights"
type: "post"
tags: ["Planning", "UX", "Game Design", "Edge Case"]
---

<div class="lang-ko">

기획서는 보통 기능이 잘 돌아가는 순서로 적힙니다.  
하지만 플레이어는 그 순서대로 움직이지 않습니다.  
그래서 기능을 정할 때, 아직 없는 게임을 머릿속으로 먼저 한 번 플레이해 봅니다.

## 개발자가 물어볼 상태를 먼저 떠올리기

- 입력을 연속으로 하면?
- 애니메이션이 끝나기 전에 다른 입력이 들어오면?
- 두 조건이 동시에 충족되면 무엇이 먼저?
- 처음 하는 사람도 내가 예상한 대로 이해할까?

전부 기획서에 적지는 않습니다.  
다만 한 번 생각해 두면, 질문이 왔을 때 기능의 목적을 기준으로 바로 답할 수 있습니다.  
줄이고 싶은 건 질문 자체가 아니라, 질문마다 처음부터 다시 고민하느라 작업이 멈추는 상황입니다.

## 예 1 · 이동 중 클릭

이동 애니메이션이 끝나야 다음 장소를 고를 수 있었고, 플레이어들은 그 사이 클릭을 반복했습니다.  
애니메이션을 없애면 어디로 갔는지 알기 어려웠습니다.

→ 애니메이션을 줄이고, 이동 중 들어온 첫 입력을 기억했다가 이어서 이동

## 예 2 · 미리 알려준 조건

히든 엔딩 조건(힌트를 적게 쓰기)을 튜토리얼에서 알려줬더니, 플레이어들이 힌트를 참다가 지쳐서 그만뒀습니다.

→ 조건을 처음엔 숨김. 힌트로 규칙을 익힌 뒤, 더 잘 풀고 싶어서 스스로 힌트를 줄임

정보를 빠짐없이 주는 게 항상 친절하지는 않았습니다.

## 다 지원할 필요는 없음

예외를 생각하는 목적은 기능을 늘리는 게 아닙니다.  
필요한 경우는 처리 방법을 정하고, 필요 없는 경우는 왜 지원하지 않는지 팀이 알고 있으면 됩니다.  
우연히 동작하게 내버려 두지 않는 것이 핵심입니다.

관련 사례: [Arcanum Nights](/projects/arcananights/)

</div>

<div class="lang-en" style="display:none">

Design documents usually describe features in the order they work.  
Players do not move in that order.  
So before settling a feature, I play the not-yet-built game in my head.

## Think of the states a developer will ask about

- What if input comes in rapidly?
- What if a new input arrives before an animation ends?
- If two conditions fire at once, which wins?
- Will a first-time player read it the way I expect?

Not all of this goes into the document.  
But having thought it through, I can answer questions from the feature's purpose right away.  
What I want to avoid is not questions, but work stopping while each one is re-decided from scratch.

## Example 1 · Clicking while moving

Players could pick the next spot only after the move animation ended, so they kept clicking.  
Removing the animation made it hard to tell where they had gone.

→ Shorter animation, and the first click during a move is queued and played right after

## Example 2 · A condition told too early

The tutorial explained the hidden-ending condition (use few hints), and players held off on hints until they gave up.

→ Hid the condition at first. Players learned with hints, then cut back on their own to solve more cleanly

Giving every piece of information up front was not always kind.

## No need to support everything

Thinking about edge cases is not about adding features.  
Decide how to handle the cases that matter, and make sure the team knows why the rest are not supported.  
The point is not to leave anything working by accident.

Related case study: [Arcanum Nights](/projects/arcananights/)

</div>
