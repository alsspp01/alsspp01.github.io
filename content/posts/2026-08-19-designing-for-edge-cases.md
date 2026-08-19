---
title: "🔬 Edge Case를 수집하는 기획"
title_en: "🔬 Designing for Edge Cases"
date: 2026-08-19
description: "개발자가 질문한 뒤 생각하는 대신, 구현 전에 머릿속으로 한 번 굴려보는 기획 습관."
description_en: "A planning habit: mentally running the system before implementation questions arrive."
type: "post"
tags: ["Planning", "UX", "Game Design", "Edge Case"]
---

<div class="lang-ko">

기획서를 쓸 때 가장 쉬운 흐름은 정상적인 흐름입니다.

사용자가 버튼을 누르고, 다음 화면으로 가고, 필요한 값을 입력하고, 성공합니다.

문제는 실제 사용자가 정상적인 흐름만 밟지 않는다는 것입니다.

그래서 저는 기획을 하면 머릿속에서 한 번 **실제로 굴려보는 편**입니다.

---

## Happy Path 다음에 나오는 질문들

기능 하나를 생각하면 이런 질문이 따라옵니다.

- 사용자가 중간에 취소하면?
- 같은 버튼을 연속으로 누르면?
- 값이 없으면?
- 두 조건이 동시에 만족하면?
- 우선순위가 충돌하면?
- 저장 전 상태가 바뀌면?
- 처음 보는 사람은 내가 예상한 버튼을 정말 누를까?

이 질문을 전부 문서에 적는 것은 아닙니다.

하지만 적어도 한 번 생각해보려고 합니다.

---

## 개발자의 질문은 버그가 아니다

예전에는 개발자가 기획에 대해 질문하면 문서가 부족했다는 느낌을 받기도 했습니다.

지금은 조금 다르게 생각합니다.

질문은 당연히 생깁니다.

구현하는 사람은 기획자가 화면으로 본 것을 상태와 조건으로 다시 보기 때문입니다.

다만 제가 좋아하는 상태는

> **“이 경우에는 어떻게 해요?”**

라는 질문을 받았을 때

> **“그건 생각 안 해봤는데요.”**

보다

> **“그 경우에는 이렇게 처리하면 됩니다.”**

라고 말할 수 있는 경우가 더 많은 상태입니다.

---

## 기획서보다 먼저 존재하는 시뮬레이션

저에게 꼼꼼함은 문서가 길거나 체크리스트가 많다는 뜻과는 조금 다릅니다.

오히려 **아직 만들어지지 않은 것을 머릿속에서 한 번 써보는 것**에 가깝습니다.

사용자가 어떻게 움직일지,  
개발자는 어떤 상태를 만들어야 할지,  
아트는 어떤 리소스를 추가해야 할지,  
데이터가 어디에서 생기고 어디로 갈지.

한 사람의 관점만으로 굴리면 놓치는 것이 많아서  
가능하면 다른 직군의 질문도 같이 상상합니다.

---

## 작은 예: 데모에서 안 보이면 없는 걸까?

게임 데모를 준비할 때 특정 스테이지를 노출하지 않도록 설정한 빌드를 본 적이 있습니다.

화면에서는 잘 숨겨져 있었습니다.

하지만 프로젝트 안에는 원본 스토리 데이터가 그대로 포함되어 있었습니다.

정상적인 플레이 흐름만 보면 아무 문제도 없습니다.

하지만 질문을 조금 바꾸면 달라집니다.

> **“안 보이는 콘텐츠가 빌드 안에도 없어야 하는가?”**

데모라는 배포 목적을 생각하면 저는 분리하는 편이 더 맞다고 판단했고, 데모용 스토리 데이터를 따로 정리했습니다.

---

## Edge Case는 다 막는 것이 아니다

모든 예외를 완벽하게 지원해야 한다는 뜻은 아닙니다.

오히려 예외를 생각하다 보면

> **이 경우를 지원할 필요가 있나?**

라는 질문도 나옵니다.

예외를 모두 구현하면 제품이 더 복잡해지고 오히려 사용하기 어려워질 수 있습니다.

그래서 edge case를 찾는 목적은 기능을 늘리는 것이 아니라  
**어떤 경우를 지원하고, 어떤 경우는 의도적으로 지원하지 않을지 결정하는 것**에 가깝습니다.

---

## Think Deep, Ship Simple

예전에는 더 많이 지원할수록 좋은 기획이라고 생각했습니다.

지금은 뒤에서 충분히 많이 생각하되 앞에서는 단순하게 보이는 쪽을 더 좋아합니다.

edge case를 많이 생각한 결과가 옵션 20개짜리 화면일 필요는 없습니다.

오히려 잘 정리된 결과는 사용자가 그런 경우를 의식하지 않아도 자연스럽게 넘어갈 수 있습니다.

**많이 생각하되, 많이 보여주지는 않는 것.**

요즘 제가 기획할 때 자주 돌아오는 기준입니다.

[🔗 Arcanum Nights Case Study](/portfolio/arcanum-nights/)

</div>

<div class="lang-en" style="display:none">

The easiest flow to document is the happy path.

The user clicks a button, moves to the next screen, enters the expected value, and succeeds.

Real users are less cooperative.

So when I plan a feature, I tend to **run it once in my head before it exists.**

---

## The questions after the happy path

A single feature usually creates a small tree of questions:

- What if the user cancels halfway through?
- What if they press the same button repeatedly?
- What if a value is missing?
- What if two conditions become true at the same time?
- What if priorities conflict?
- What if the state changes before saving?
- Will a first-time user actually click the button I expect?

Not every answer belongs in the document.

I still want to think through the branch at least once.

---

## A developer's question is not a bug

I used to feel that a question from a developer meant the planning document was incomplete.

I see it differently now.

Questions are normal.

Developers look at the same idea through states, conditions, and implementation constraints.

What I do prefer is reaching a point where:

> **“What happens in this case?”**

more often gets:

> **“Then we handle it this way.”**

instead of:

> **“I hadn't thought about that.”**

---

## The simulation that exists before the document

For me, being meticulous is not the same thing as writing longer documents.

It is closer to **using something in my head before it has been built.**

How will the user move?  
What state does development need to represent?  
What new asset will art need?  
Where does the data come from, and where does it go?

A system simulated from only one discipline's point of view will miss a lot, so I also try to imagine the questions other disciplines are likely to ask.

---

## A small example: if the demo cannot show it, is it gone?

While preparing a game demo, I saw a build where certain stages were correctly hidden.

From the player's perspective, it looked fine.

But the original story data was still bundled.

Nothing was wrong on the happy path.

The question changed when I asked:

> **“If the demo should not expose this content, should the data be in the build at all?”**

Given the purpose of the demo, I preferred separating demo-specific story data instead.

---

## Edge cases are not a checklist of everything you must support

Thinking about edge cases does not mean implementing all of them.

Sometimes the most useful question is:

> **Do we need to support this case at all?**

Supporting every imaginable case can make the product harder to use.

So the point of finding edge cases is not to maximize features.

It is to **decide intentionally which cases the product supports and which ones it does not.**

---

## Think Deep, Ship Simple

I used to think broader support automatically meant better planning.

Now I prefer doing more of the complexity work behind the scenes so the final experience can remain simple.

A design that considered twenty edge cases does not need twenty visible options.

The better outcome may be that the user never has to think about those cases at all.

**Think a lot. Show less.**

That is one of the rules I keep returning to.

[🔗 Arcanum Nights Case Study](/portfolio/arcanum-nights/)

</div>
