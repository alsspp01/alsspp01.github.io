---
title: "🌦️ 낯선 도메인을 기술 방향으로 바꾸기"
title_en: "🌦️ From an Unfamiliar Domain to Technical Direction"
date: 2026-08-19
description: "기상학을 처음 접한 PM이 기능 목록보다 먼저 사용자 업무를 따라간 이유."
description_en: "Why I followed the user's workflow before writing a feature list in an unfamiliar meteorological domain."
type: "post"
tags: ["PM", "Planning", "Domain", "AI"]
---

<div class="lang-ko">

프로젝트 초반에 요구사항이 흐리면 마음이 급해집니다.

일정은 움직이고 있고, 개발자는 무엇을 만들지 알아야 하고, 회의에서는 다음 액션이 필요합니다.  
이럴 때 가장 빠르게 보이는 방법은 기능 목록을 만드는 것입니다.

그런데 기상 예보지원 프로젝트에서는 기능부터 적기 시작하면 오히려 더 늦어질 것 같았습니다.

왜냐하면 저는 **기상학을 몰랐기 때문입니다.**

---

## 모르는 상태에서 요구사항을 쓰면 생기는 일

처음 보는 도메인에서는 익숙한 단어도 의미가 다릅니다.

같은 "데이터"라는 말도 누가, 언제, 어떤 판단을 위해 보는 데이터인지에 따라 중요도가 달라집니다.

그래서 먼저 기상 용어를 공부했습니다.  
그다음에는 예보관이 기존 프로그램을 어떻게 사용하는지 봤습니다.

어떤 정보를 먼저 보고, 어디에서 다른 화면으로 넘어가고, 어떤 시점에 사람이 판단을 하는지 따라갔습니다.

제가 찾고 싶었던 건 기능명이 아니라 **업무의 문법**이었습니다.

---

## “무엇을 만들까?”보다 먼저 본 질문

프로젝트에서 반복해서 확인한 질문은 대체로 이랬습니다.

> **이 정보는 왜 필요한가?**

> **예보관은 어느 시점에 이걸 보는가?**

> **현재 사람이 직접 판단하는 부분은 어디인가?**

> **AI가 들어오면 그 판단을 대신하는가, 보조하는가?**

이 질문에 답할 수 있게 되자 그제야 기능이 기능처럼 보이기 시작했습니다.

그전에는 버튼과 화면 이름일 뿐이었던 것이  
업무 흐름 안에서 목적을 가지기 시작했습니다.

---

## 개발자에게 기능만 넘기지 않기

제가 프로젝트를 많이 이해해야 한다고 생각한 이유 중 하나는  
개발자가 모든 도메인 맥락을 다시 공부하게 만들고 싶지 않았기 때문입니다.

> “이 화면이 필요합니다.”

와

> “예보관이 이 판단을 내리는 시점에 이 정보를 비교해야 해서 이 화면이 필요합니다.”

는 구현 중 판단할 수 있는 범위가 다릅니다.

두 번째 설명을 알고 있으면 애매한 상황이 생겨도 원래 목적을 기준으로 다시 판단할 수 있습니다.

프로젝트가 진행되면서 개발자들이 방향을 확인할 일이 있을 때 제게 의견을 묻는 경우도 자연스럽게 생겼습니다.

저는 그게 PM이 모든 기술적 답을 안다는 의미라고 생각하지 않습니다.

오히려 **왜 만드는지에 대한 맥락이 한 곳에서 끊기지 않고 이어지고 있었다는 신호**에 더 가깝습니다.

---

## 빠르게 만드는 것과 빨리 이해하는 것은 다르다

이 경험 이후 요구사항이 흐릴 때 바로 표와 기능 목록부터 만드는 습관이 줄었습니다.

대신 먼저 확인합니다.

- 사용자에게 지금 무슨 일이 일어나고 있는가
- 기존 방식은 왜 이렇게 되어 있는가
- 어떤 판단이 가장 어렵거나 반복적인가
- 기술이 들어갔을 때 무엇이 실제로 달라져야 하는가

이 과정은 겉으로 보면 구현보다 느려 보입니다.

하지만 도메인을 잘못 이해한 채 만든 요구사항을 개발 중간에 다시 뜯어고치는 것보다는 훨씬 빠를 때가 많았습니다.

---

## Fog of War

저는 새로운 프로젝트에 들어갈 때 머릿속에 **Fog of War가 깔려 있다**고 생각하는 편입니다.

안 보이는 상태에서 목표 지점만 찍고 달리는 것보다  
먼저 맵을 조금씩 밝히는 쪽을 좋아합니다.

사용자를 보고, 용어를 배우고, 기존 시스템의 이유를 따라가다 보면 어느 순간 전체 지형이 보이기 시작합니다.

그때부터 질문이 바뀝니다.

> **“그래서, 어떻게 만드는 게 제일 좋을까?”**

저에게 기획은 보통 그 질문부터 시작됩니다.

[🔗 Portfolio Case Study](/portfolio/ai-forecast-support/)

</div>

<div class="lang-en" style="display:none">

When requirements are blurry at the beginning of a project, there is pressure to move fast.

The schedule is already moving, developers need something concrete to build, and every meeting is expected to end with a next action.  
The most obvious shortcut is a feature list.

On an AI forecasting-support project, however, starting with features felt like the slower option.

Because I **did not understand meteorology yet**.

---

## Writing requirements before understanding the domain

In an unfamiliar domain, even familiar words can mean something different.

“Data” is not just data. Its importance depends on who looks at it, when, and for what decision.

So I started by learning the terminology.  
Then I followed how forecasters used their existing software.

What did they look at first? Where did they switch views? At what point did human judgment enter the workflow?

I was not trying to memorize feature names.

I was trying to learn **the grammar of the work**.

---

## The questions before “What should we build?”

I kept returning to a small set of questions:

> **Why is this information needed?**

> **At what point does the forecaster look at it?**

> **Where does the current workflow depend on human judgment?**

> **If AI enters the process, is it replacing that judgment or supporting it?**

Once I could answer those questions, features started to look like features rather than labels on a screen.

---

## Do not hand developers a feature with the context removed

One reason I wanted a strong understanding of the overall project was that I did not want every developer to rediscover the domain from scratch.

There is a real difference between:

> “We need this screen.”

and:

> “Forecasters need to compare this information at this point in the decision process, so this screen exists to support that comparison.”

The second explanation gives implementation decisions a reference point.

As the project progressed, developers naturally began asking me to check direction when an implementation felt ambiguous.

I do not interpret that as “the PM has every technical answer.”

I see it as a sign that **the reason behind the work had not been disconnected from the implementation**.

---

## Building fast and understanding fast are not the same thing

Since that project, I have become less eager to start with a spreadsheet full of features when requirements are unclear.

I first ask:

- What is happening to the user today?
- Why does the current workflow look this way?
- Which judgment is difficult or repetitive?
- What should actually change when technology enters the process?

This can look slower than implementation.

It is often much faster than discovering halfway through development that the requirements were built on the wrong mental model.

---

## Fog of War

I tend to imagine a new project as a map covered in **Fog of War**.

Rather than placing a waypoint on an invisible map and running toward it, I prefer to reveal the terrain first.

Watch the user. Learn the language. Trace why the current system exists.

Eventually the map starts to open up.

Then the question changes.

> **“So, what is the best way to build this?”**

That is usually where planning starts for me.

[🔗 Portfolio Case Study](/portfolio/ai-forecast-support/)

</div>
