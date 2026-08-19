---
title: "🔗 사람 사이에도 Interface가 있다"
title_en: "🔗 People Have Interfaces, Too"
date: 2026-08-19
description: "10명 규모 원격 팀을 운영하며 협업 툴을 기능이 아니라 인터페이스로 보기 시작한 과정."
description_en: "What running a remote team taught me about treating collaboration tools as interfaces between people."
type: "post"
tags: ["D3F!B", "Collaboration", "Planning", "Automation"]
---

<div class="lang-ko">

팀이 작을 때는 대부분의 일이 기억으로 굴러갑니다.

> “그 파일 어디 있었지?”

> “저번 회의에서 뭐로 결정했더라?”

> “이건 개발팀 채널에 올리면 되나?”

다 아는 사람들이라면 누군가 답해줍니다.

문제는 사람이 늘어날 때입니다.

기획, 개발, 아트로 나뉜 10명 정도의 원격 팀에서는  
**사람의 기억이 협업 시스템이 되는 방식이 금방 한계에 옵니다.**

---

## 툴을 많이 쓰는 것과 구조가 있는 것은 다르다

D3F!B에서도 Discord, Notion, Google Drive를 사용합니다.

중요했던 것은 어떤 툴을 쓰느냐보다 **각 툴이 무엇을 책임지는지 겹치지 않게 만드는 것**이었습니다.

### Discord
지금 일어나는 대화.

- 빠른 질문
- 회의
- 직군별 대화
- 알림
- 당장 필요한 결정

### Notion
나중에도 이유를 알아야 하는 것.

- 기획 문서
- 회의 기록
- 프로젝트 구조
- 다시 찾아볼 지식

### Google Drive
실제 파일 자산.

- 이미지
- 제작 리소스
- 공유할 큰 파일

모든 걸 한 도구에 넣는 것보다 이 역할 구분이 훨씬 중요했습니다.

---

## 권한도 UX다

권한은 보안 설정처럼 보이지만 실제로는 사람의 행동을 크게 바꿉니다.

너무 열려 있으면 관련 없는 정보가 계속 들어오고,  
너무 잠겨 있으면 작은 일을 할 때마다

> “이거 권한 좀 주세요.”

라고 말해야 합니다.

그래서 **“누가 보면 안 되는가?”뿐 아니라 “이 사람이 일할 때 무엇까지 자연스럽게 보여야 하는가?”**도 함께 봤습니다.

잘 만든 권한은 존재감이 별로 없습니다.

필요한 것은 보이고, 필요하지 않은 것은 방해하지 않습니다.

---

## 반복되는 기억은 자동화하기

회의 내용을 정리해서 올리고, 개발일지가 생기면 알리고, 정해진 시간에 메시지를 보내는 일은 한 번은 쉽습니다.

문제는 **매번 누군가가 기억해야 한다는 것**입니다.

그래서 하나씩 자동화했습니다.

- Discord 회의 → AI 요약 → Notion
- Notion 개발일지 → Discord 알림
- 예약 메시지
- Git 관련 알림

목적은 거창한 생산성 향상이 아니라  
**사람이 중요한 판단보다 “기억해야 하는 잡일”에 신경을 덜 쓰게 하는 것**이었습니다.

---

## “그럼 내가 다 하면 되겠네?”

예전에는 제가 할 수 있는 일이 늘어나는 게 좋았습니다.

그러다 아주 자연스럽게 이런 결론에 도착했습니다.

> **“그럼 내가 다 하면 되겠네?”**

효율적으로 들립니다.

그리고 전혀 확장되지 않습니다.

제가 모든 걸 알고, 확인하고, 전달해야 한다면 팀이 커질수록 제가 bottleneck이 됩니다.

---

## 요즘 더 좋아하는 구조

지금은 제가 직접 해결하는 것보다 다음 질문에 더 관심이 있습니다.

> **사람들이 나를 기다리지 않고 움직일 수 있는가?**

필요한 문서를 찾을 수 있는지, 어디에서 질문해야 하는지 아는지, 자기 역할에 필요한 정보가 보이는지.

이런 것들이 잘 맞아 있을 때 사람 사이에도 interface가 있다는 생각을 합니다.

좋은 interface가 사용자를 매번 교육하지 않듯,  
좋은 협업 구조도 계속 “여기서 이걸 해야 해”라고 설명하지 않아도 됩니다.

그냥 다음 행동이 자연스럽게 보입니다.

[🔗 Portfolio Case Study](/portfolio/d3fib-workspace/)

</div>

<div class="lang-en" style="display:none">

When a team is small, memory can carry a surprising amount of infrastructure.

> “Where was that file again?”

> “What did we decide in the last meeting?”

> “Which channel should I post this in?”

If everyone knows everyone, somebody usually answers.

That stops scaling once a remote team grows across planning, development, and art.

At around ten people, **“someone will remember” is no longer a collaboration system.**

---

## Using many tools is not the same as having a structure

D3F!B uses Discord, Notion, and Google Drive.

The important part was not choosing a fancy tool.  
It was making sure **each tool had a clear responsibility instead of all three becoming different versions of the same messy folder.**

### Discord
What is happening now.

- quick questions
- meetings
- discipline-specific conversations
- notifications
- immediate decisions

### Notion
What still needs to make sense later.

- planning documents
- meeting records
- project structure
- knowledge worth finding again

### Google Drive
Actual production assets.

- images
- source resources
- larger shared files

That division mattered more than the individual products.

---

## Permissions are UX

Permissions look like security settings, but they also shape how people move through a workspace.

Open everything too widely and irrelevant information becomes noise.  
Lock everything down too aggressively and even a small task requires:

> “Can you give me access to this?”

So I stopped asking only **“Who should not see this?”**

I also asked **“What should this person naturally be able to see while doing their job?”**

Good permissions mostly disappear.

The right things are visible, and the wrong things do not get in the way.

---

## Automate the things people should not have to remember

Write up a meeting. Notify everyone when a devlog appears. Send a message at a specific time.

Each task is easy once.

The problem is that **someone has to remember it every time.**

So I automated them one by one:

- Discord meeting → AI summary → Notion
- Notion devlog → Discord notification
- scheduled messages
- Git-related notifications

The point was not dramatic “productivity optimization.”

It was to spend less attention on remembering chores and more on decisions that actually need a person.

---

## “Then I can just do everything myself.”

As I learned more development, planning, and design, I enjoyed being able to cover more ground.

That led naturally to:

> **“Then I can just do everything myself.”**

It sounds efficient.

It scales terribly.

If I have to know everything, approve everything, and relay everything, the team gets slower as soon as I get busy.

I become the bottleneck.

---

## The structure I prefer now

These days I care more about a different question:

> **Can people move without waiting for me?**

Can they find the document? Do they know where to ask? Can they see the information their role needs?

When those things line up, I start thinking of collaboration as an interface between people.

A good product interface does not repeatedly explain the next step.

It simply makes the next action easy to see.

I think good team infrastructure should feel similar.

[🔗 Portfolio Case Study](/portfolio/d3fib-workspace/)

</div>
