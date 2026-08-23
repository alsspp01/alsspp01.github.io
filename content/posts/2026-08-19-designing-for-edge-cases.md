---
title: "🔬 만들기 전에 한 번 플레이해보는 기획"
title_en: "🔬 Testing a Design Before It Exists"
date: 2026-08-19
description: "Arcanum Nights를 만들며 입력, 힌트와 플레이어 행동을 미리 검토하고 수정한 경험."
description_en: "How I reviewed input, hints, and player behavior while designing Arcanum Nights."
type: "post"
tags: ["Planning", "UX", "Game Design", "Edge Case"]
---

<div class="lang-ko">

기획서를 작성할 때는 보통 기능이 정상적으로 작동하는 순서부터 적습니다. 버튼을 누르면 이동하고, 조건을 만족하면 다음 단계로 넘어가는 식입니다. 문서에서는 깔끔하지만 실제 플레이어는 그 순서대로만 움직이지 않습니다.

같은 버튼을 여러 번 누르기도 하고, 설명을 예상과 다르게 받아들이기도 하고, 기획자가 중요하지 않다고 생각한 조건에 집착하기도 합니다. 그래서 저는 기능을 정리할 때 아직 만들어지지 않은 게임을 머릿속으로 먼저 플레이해보는 편입니다.

## 개발자가 물어볼 상태를 먼저 생각합니다

기능 하나를 정하면 자연스럽게 다음 상태가 생깁니다.

- 입력을 연속으로 하면 어떻게 되는가
- 애니메이션이 끝나기 전에 다른 입력이 들어오면 어떻게 처리하는가
- 두 조건이 동시에 충족되면 무엇을 우선하는가
- 값이 없거나 저장 도중 상태가 바뀌면 어떻게 되는가
- 처음 플레이하는 사람도 내가 예상한 방식으로 이해하는가

이 경우를 모두 기획서에 적는 것은 아닙니다. 구현할 필요가 없는 예외도 있고, 개발 과정에서 정하는 편이 나은 것도 있습니다. 그래도 한 번 생각해두면 개발자가 질문했을 때 기능의 목적을 기준으로 답할 수 있습니다.

개발자의 질문이 나온다는 사실 자체는 기획서가 잘못됐다는 뜻이 아닙니다. 기획자는 플레이 흐름으로 본 기능을 개발자는 상태와 조건으로 다시 보기 때문에 질문은 생길 수밖에 없습니다. 제가 줄이고 싶은 것은 질문이 아니라, 그때마다 처음부터 목적을 다시 고민하느라 작업이 멈추는 상황입니다.

## 이동 입력을 기다리게 하지 않기

Arcanum Nights는 장소를 클릭해 이동하는 게임입니다. 초기 빌드에서는 이동 애니메이션이 끝나야 다음 장소를 선택할 수 있었습니다. 애니메이션이 조금 길다 보니 플레이어들은 이동 중에도 클릭을 반복했습니다.

퍼즐을 풀려면 맵을 오가며 구조를 확인해야 하는데, 이동할 때마다 입력이 막히면 생각의 흐름도 자주 끊겼습니다. 그렇다고 애니메이션을 완전히 없애면 플레이어가 어느 방향으로 이동했는지 파악하기 어려웠습니다.

그래서 애니메이션 길이를 줄이고, 이동 중 다음 장소가 입력되면 애니메이션이 끝난 직후 가장 먼저 선택한 장소로 이어서 이동하도록 정했습니다. 플레이어는 입력이 무시됐다고 느끼지 않으면서도 현재 이동을 확인할 수 있었습니다. 테스트에서도 이동의 답답함이 줄고 게임을 더 오래 플레이하는 모습을 확인했습니다.

여기서 검토한 것은 "클릭하면 이동한다"는 정상 흐름보다, 이동이 끝나기 전에 다시 클릭하는 플레이어를 어떻게 처리할지였습니다.

## 설명을 많이 할수록 잘 플레이하는 것은 아니었습니다

히든 엔딩을 보기 위해서는 힌트를 적게 사용해야 했습니다. 처음에는 이 조건을 튜토리얼에서 알려주는 편이 공정하다고 생각했습니다.

그런데 플레이 테스트에서는 예상과 다른 행동이 나왔습니다. 플레이어들이 처음부터 조건을 달성하려고 힌트를 사용하지 않은 채 한 스테이지에 오래 머물렀고, 이미 지친 뒤에도 포기하지 않다가 게임을 중단했습니다. 조건을 미리 알려준 것이 도전을 돕기보다 퍼즐에 익숙해질 기회를 막은 셈이었습니다.

그래서 히든 엔딩 조건을 처음부터 설명하지 않도록 바꿨습니다. 플레이어들은 필요할 때 힌트를 사용하며 기믹을 익혔고, 익숙해진 뒤에는 더 완벽하게 풀고 싶어서 스스로 힌트 사용을 줄였습니다.

정보를 빠짐없이 제공하는 것이 항상 친절한 것은 아니었습니다. 플레이어가 지금 집중해야 할 것보다 나중에 고려할 조건을 먼저 보여주면, 오히려 선택할 것이 늘고 실패했다는 느낌만 강해질 수 있었습니다.

## 모든 경우를 지원할 필요는 없습니다

예외를 생각하는 목적은 가능한 기능을 모두 추가하는 것이 아닙니다. 경우에 따라서는 지원하지 않기로 정하는 것이 더 단순하고 이해하기 쉬운 결과를 만듭니다.

중요한 것은 정상 흐름 밖의 상황이 생겼을 때 우연히 동작하도록 방치하지 않는 것입니다. 사용자에게 필요한 경우라면 처리 방법을 정하고, 필요하지 않다면 왜 지원하지 않는지 팀이 알고 있어야 합니다.

제가 기획할 때 미리 여러 상황을 생각하는 이유도 문서를 길게 만들기 위해서가 아닙니다. 개발과 아트가 작업을 시작한 뒤 처음부터 방향을 다시 정하는 일을 줄이고, 플레이어에게는 고민한 흔적이 복잡한 옵션이 아니라 자연스러운 경험으로 남게 하고 싶기 때문입니다.

[🔗 Arcanum Nights Case Study](/projects/arcananights/#case-study)

</div>

<div class="lang-en" style="display:none">

A design document usually begins with the intended sequence: the player presses a button, the character moves, and meeting a condition opens the next step. It looks orderly on paper, but players rarely follow only that sequence.

They click the same button repeatedly, interpret an explanation differently from what I expected, or become fixated on a condition that I assumed would remain secondary. When I define a feature, I therefore try to play through it in my head before the game exists.

## Thinking through the states developers will ask about

One feature quickly creates several states:

- What happens after repeated input?
- What if another input arrives before the animation ends?
- Which condition wins when two become true at once?
- What happens when a value is missing or the state changes before saving?
- Will a first-time player understand the interaction in the way I expect?

Not every case belongs in the design document. Some do not need to be supported, while others are better decided during implementation. Thinking through them once, however, lets me answer a developer's question from the purpose of the feature instead of reconsidering that purpose from the beginning.

A question from a developer does not mean the design has failed. Designers tend to see a play sequence, while developers reconstruct it as states and conditions. Questions are inevitable. What I want to reduce is the time the team loses when the intended behavior has never been considered at all.

## Letting the next movement wait instead of the player

Players move through Arcanum Nights by clicking locations. In an early build, they had to wait for the current movement animation to finish before selecting the next location. The animation was long enough that players repeatedly clicked while the character was still moving.

Solving the puzzles requires moving around the map to understand its structure. Blocking input during every movement kept interrupting that thought process. Removing the animation entirely was not a good answer either, because the player still needed to understand where the character had moved.

I shortened the animation and changed the behavior so that an input received during movement would be queued. As soon as the current animation ended, the character moved to the first location the player had selected. The input no longer felt ignored, while the current movement remained readable. During testing, players showed less frustration with navigation and stayed with the game longer.

The important question was not simply whether clicking moved the character. It was what should happen when the player clicked again before that movement had finished.

## More explanation did not produce better play

One hidden-ending condition rewarded players for using fewer hints. Initially, I thought stating that condition in the tutorial was the fairest approach.

Playtests produced the opposite behavior from what I expected. Players tried to meet the condition on their first attempt, refused to use hints, and remained stuck in a single stage even after they were tired. Some stopped playing altogether. Explaining the condition early had prevented them from using the support they needed to learn the puzzle.

I removed the condition from the initial explanation. Players then used hints when necessary, learned how the mechanic worked, and later chose to limit their own hint use when they wanted a more complete solution.

Providing every piece of information is not always helpful. Showing a later objective before the player understands the immediate task can increase the number of things they feel responsible for and make ordinary learning feel like failure.

## Not every case needs support

Thinking through exceptions does not mean adding support for every imaginable behavior. In some cases, deciding not to support something produces a simpler and more understandable result.

What matters is that behavior outside the intended flow is not left to accident. If the case matters to the player, the team should decide how it works. If it does not, the team should still understand why it is being left unsupported.

I consider these cases before implementation not to make the document longer, but to reduce the number of times development and art must stop and redefine the direction. For the player, that preparation should appear as a natural experience rather than a long list of visible options.

[🔗 Arcanum Nights Case Study](/projects/arcananights/#case-study-en)

</div>
