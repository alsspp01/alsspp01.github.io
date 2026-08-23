---
title: "🌦️ AI Forecast Support"
title_en: "🌦️ AI Forecast Support"
type: page
aliases:
  - /portfolio/ai-forecast-support/
description: "낯선 기상 도메인을 이해하고 기술 요구사항과 프로젝트 방향으로 구조화한 경험."
description_en: "How I learned an unfamiliar meteorological domain and translated it into technical direction."
---

<div class="lang-ko">

<a id="top"></a>

**기간** · 2025.09–2026.02  
**역할** · PM / 기획 / 협업 조율  
**소속** · Ineeji(주관기관)  
**고객사** · 국립기상과학원  
**협업** · 대학 연구실 2곳

[**Case Study ↓**](#case-study) · [**공식 보고서 ↗**](https://dl.nanet.go.kr/detail/MONO12026000015571)

---

<a id="case-study"></a>
## Case Study

### 무엇을 만들어야 하는지부터 찾아야 했습니다

이 프로젝트의 목표는 **예보관의 초기 판단을 돕는 AI 에이전트**를 만드는 것이었습니다.

여러 언어 모델이 내놓은 예측 결과를 예보관이 일일이 확인하는 대신, 시기와 기상 환경에 맞는 결과를 찾고 예보에 활용할 수 있도록 기상 용어로 설명해주는 서비스였습니다.

연구 목표는 있었지만 구체적인 구현 범위는 정해져 있지 않았습니다. 국립기상과학원은 예보 업무에서 필요한 것이 무엇인지는 알고 있었지만, 이를 어떤 기술로 구현해야 하는지는 개발 기관과 함께 찾아야 했습니다. 반대로 컨소시엄의 각 기관은 기술적으로 맡은 연구에 집중하고 있었지만, 그 결과가 어떻게 연결되어 하나의 에이전트가 되어야 하는지는 공유되지 않은 상태였습니다.

저는 PM으로서 그 사이를 연결했습니다.

### 예보관의 일을 먼저 배웠습니다

기상학도, 기존 예보 프로그램도 처음 접하는 분야였습니다. 별도의 온보딩이 없어 문서만으로는 실제 사용 맥락을 알기 어려웠습니다.

직접 국립기상과학원에 방문해 프로그램의 작동 방식과 기상예보 과정을 배웠습니다. 예보관이 어떤 순서로 정보를 보고, 어느 지점에서 판단하며, AI의 설명이 언제 필요한지 확인했습니다.

> **예보관이 원하는 것은 무엇인가?**  
> **각 연구기관의 결과는 그 과정에서 어떤 역할을 하는가?**  
> **서로 떨어진 연구 결과를 어떻게 하나의 서비스로 연결할 것인가?**

이 질문을 기준으로 고객사의 요구와 각 기관의 연구를 다시 살펴봤습니다.

### 흩어진 작업을 하나의 흐름으로 정리했습니다

각 기관이 진행하던 작업의 의미와 연결 지점을 파악하고, 이들이 하나의 에이전트로 동작하기 위한 전체 흐름을 그렸습니다. 보안과 계약상 구체적인 모델명과 구조는 공개할 수 없지만, 각 연구 결과가 어디에서 입력되고 다음 단계에 무엇을 전달해야 하는지를 정리했습니다.

개발자들을 모아 화이트보드에 흐름을 그려가며 프로젝트의 목적과 각 작업의 관계를 설명했습니다. 이후 개발자들은 구현 방향을 확인할 때 저를 찾아왔고, 기관 간 협력이 필요한 지점도 구체적으로 요청할 수 있게 됐습니다.

약 3개월 동안 정체되어 있던 작업은 방향을 맞춘 뒤 1~2개월 안에 마무리됐습니다.

### 기상과 기술 사이에서 통역했습니다

갈등의 원인은 어느 한쪽이 틀렸기 때문이 아니었습니다.

국립기상과학원은 실제 기상예보 과정을 기준으로 이야기했고, 개발 기관은 기술 개발 과정을 기준으로 이야기했습니다. 같은 목표를 두고도 서로 다른 전제에서 출발하다 보니 요구사항이 제대로 전달되지 않았고, 불신도 쌓이고 있었습니다.

저는 국립기상과학원과 실시간으로 요구사항을 확인하고, 그 의미를 연구진과 개발자가 판단할 수 있는 기술적 맥락으로 바꿔 전달했습니다. 반대로 기술적인 제약과 개발 방식을 고객사가 이해할 수 있는 언어로 설명했습니다.

대화의 기준이 맞춰지자 고객사는 개발 방향을 확인할 수 있었고, 개발자들은 기상학을 모두 알지 못해도 자신이 무엇을 만들어야 하는지 이해할 수 있게 됐습니다.

### 프로젝트가 보이도록 운영했습니다

전체 방향을 맞추는 일과 함께 네 기관의 실제 진행 상황도 관리했습니다.

- 기관별 작업과 협력이 필요한 지점 확인
- 통합 마일스톤과 WBS 정리
- 주간 · 월간 회의 아젠다와 후속 작업 관리
- 신규 개발자에게 프로젝트 목적과 업무 흐름 설명
- 일정, 이슈, 의사결정과 산출물 추적
- 기관별 산출물 취합과 최종 제출본 관리

마감 시점에는 필요한 산출물 목록을 만들고 각 기관의 자료를 취합해 제출했습니다. 1차년도의 기존 프로그램 개선, 에이전트 설계, 구성 모델 연구가 하나의 결과로 이어지도록 끝까지 작업 상태와 맥락을 맞췄습니다.

### 공식 보고서 검수

주관기관 PM으로서 1차년도 최종보고서의 작성과 제출 과정을 관리했습니다.

- 회사 담당 내용 검토
- 기술 설명과 실제 구현 내용 대조
- 기관별 작성 내용 취합
- 목차와 전체 흐름 정리
- 문장, 형식, 수치, 그림, 표와 인용 확인
- 최종 제출본 검수

보고서 『AI기반 예보지원 기술개발 1』은 2025년 기상청 국립기상과학원에서 발행됐으며 국회도서관에 소장되어 있습니다.

🔗 [국회도서관 소장정보](https://dl.nanet.go.kr/detail/MONO12026000015571)

### 1차년도 결과

- 기존 예보 프로그램 개선
- AI 에이전트의 전체 흐름 설계
- 에이전트를 구성하는 모델 연구
- 기관별 연구 결과의 연결 지점 정리
- 차년도 작업을 위한 마일스톤과 인수 맥락 정리

### 배운 점

요구사항이 모호할 때 기능 목록부터 만드는 것으로는 문제가 풀리지 않았습니다.

사용자가 실제로 어떻게 일하는지, 각 기술이 그 과정에서 어떤 역할을 하는지, 서로 다른 조직이 무엇을 전제로 말하는지를 먼저 이해해야 했습니다.

**충분히 이해하고 서로의 언어를 연결하면, 멈춰 있던 개발도 다시 움직일 수 있습니다.**

[↑ Top](#top)

---

<a id="devlog"></a>
## Devlog

### 01. Domain Onboarding

기상 도메인 학습

- 기상 용어 정리
- 예보 업무 구조 확인
- 기존 예보 프로그램 분석
- 예보관 workflow 파악
- 기능별 사용 시점 정리

낯선 도메인 → 프로젝트 판단 가능 수준까지 구조화

### 02. Existing Workflow

기존 사용자 업무 흐름 정리

```text
Forecast Data
      ↓
Existing Program
      ↓
Forecaster Review
      ↓
Human Judgment
      ↓
Forecast Output
```

AI 적용 이전 업무 구조 기준점 확보

### 03. Requirement Structuring

도메인 이해 기반 요구사항 정리

- 기능 단위 요구사항
- 사용자 목적
- 기능 사용 맥락
- Input / Output
- Implementation direction
- Acceptance context

기능 목록보다 `Why` 중심 요구사항 연결

### 04. Technical Direction

기획 요구사항 → 개발 판단 기준 변환

- 기능 목적 정리
- 사용 시점 정리
- 기술 제약 반영
- 구현 방향 검토
- 개발 질문 대응
- 프로젝트 의도 기준 의사결정

PM / Developer 간 context loss 최소화

### 05. Milestone / WBS

프로젝트 단계 및 작업 구조화

- Yearly milestone
- Monthly milestone
- Task
- Owner
- Priority
- Schedule
- Dependency
- Deliverable
- Status

업무 단위 진행상황 가시화

### 06. Meeting Operations

정기 회의 운영 구조

- Weekly agenda
- Monthly agenda
- Action item
- Decision log
- Schedule
- Follow-up

회의 → 실제 task 연결 구조 정리

### 07. Developer Onboarding

신규 개발자 프로젝트 온보딩

전달 범위

- Project goal
- Meteorological context
- Existing workflow
- Technical direction
- Current architecture
- Assigned task
- Related dependency

Task 단위가 아닌 전체 context 기준 온보딩

### 08. Multi-organization Coordination

4개 기관 간 커뮤니케이션 조율

- Requirement alignment
- Schedule coordination
- Issue sharing
- Deliverable review
- Meeting preparation
- Decision tracking

기관별 다른 관점 → 공통 프로젝트 기준으로 정리

### 09. Issue Response

개발 / 운영 이슈 대응

- Issue identification
- Context collection
- Priority classification
- Stakeholder 확인
- Resolution direction
- Follow-up

단순 전달보다 영향 범위 기준 정리

### 10. Deliverables

문서 / 산출물 관리

- Intermediate report
- Final report
- Presentation deck
- Common template
- Development document
- Deliverable checklist

문서 형식 및 전달 기준 통일

### 11. Handoff

1차년도 프로젝트 마무리 및 차년도 연결

- 주요 산출물 정리
- 중간 / 최종보고 완료
- 차년도 monthly milestone 구성
- 진행 context 유지
- Next-phase task 정리

[↑ Case Study](#case-study)

</div>

<div class="lang-en" style="display:none">

<a id="top-en"></a>

**Period** · Sep 2025–Feb 2026  
**Role** · PM / Planning / Coordination  
**Company** · Ineeji (Lead Organization)  
**Client** · National Institute of Meteorological Sciences  
**Collaboration** · Two university research labs

[**Case Study ↓**](#case-study-en) · [**Official Report ↗**](https://dl.nanet.go.kr/detail/MONO12026000015571)

---

<a id="case-study-en"></a>
## Case Study

### Defining What We Needed to Build

The goal was to build **an AI agent that supports a forecaster's initial judgment**.

Instead of requiring forecasters to review every prediction produced by multiple language models, the agent would identify the results most relevant to the time and weather conditions and explain them in meteorological terms.

The research objective was clear, but the implementation scope was not. The client knew what forecasters needed from their workflow, while the technical approach still had to be worked out with the development consortium. Each organization in the consortium was focused on its own research, but there was no shared picture of how those pieces should work together as one agent.

As a PM, I connected the two sides.

### Learning the Forecaster's Work

Meteorology and the existing forecasting software were both new to me. With no formal onboarding, documents alone were not enough to understand how the system was actually used.

I visited the National Institute of Meteorological Sciences to learn how the software worked and how forecasts were produced. I followed the order in which forecasters reviewed information, where human judgment entered the process, and when an AI-generated explanation would be useful.

> **What do forecasters actually need?**  
> **What role does each research output play in that process?**  
> **How can separate research efforts become one service?**

I used those questions to review both the client's requirements and the work underway across the consortium.

### Turning Separate Efforts into One Flow

I mapped the purpose of each organization's work and the points where they needed to connect. The specific models and architecture are confidential, but I defined how each research output should enter the overall flow and what it needed to pass to the next stage.

I brought the developers together and used a whiteboard to explain the project's purpose and how their work fit together. Developers then began coming to me to confirm implementation decisions, and the teams could make specific requests when collaboration across organizations was needed.

A workstream that had been stalled for about three months was completed within the following one to two months after the direction was aligned.

### Translating Between Meteorology and Technology

The conflict did not come from either side being wrong.

The client spoke from the forecasting process, while the consortium spoke from the development process. They were working toward the same goal from different assumptions, which made requirements difficult to interpret and weakened trust.

I confirmed requirements with the client as they evolved and translated their intent into technical context for researchers and developers. In the other direction, I explained technical constraints and development approaches in terms the client could evaluate.

Once the two sides shared the same frame of reference, the client could see where the development was heading, and developers could understand what to build without first becoming meteorologists themselves.

### Making the Project Visible

I also managed progress across the four participating organizations.

- Identified each organization's work and cross-team dependencies
- Created an integrated milestone plan and WBS
- Managed weekly and monthly agendas and follow-up tasks
- Onboarded new developers with the project purpose and workflow
- Tracked schedules, issues, decisions, and deliverables
- Collected each organization's deliverables and managed the final submission

For the first year, I kept the work aligned through the improvement of the existing forecasting software, the agent design, and research on its component models.

### Reviewing the Official Report

As the PM for the lead organization, I managed the preparation and submission of the first-year final report.

- Reviewed Ineeji's sections
- Checked technical descriptions against the implemented work
- Consolidated contributions from participating organizations
- Organized the table of contents and overall narrative
- Checked wording, formatting, figures, tables, numbers, and citations
- Reviewed the final submission

The first-year report, **Development of AI-Based Technology for Weather Forecasting, Vol. 1**, was published by the National Institute of Meteorological Sciences in 2025 and is held by the National Assembly Library of Korea.

🔗 [National Assembly Library Catalog](https://dl.nanet.go.kr/detail/MONO12026000015571)

### First-Year Outcomes

- Improvements to the existing forecasting software
- Overall flow for the AI agent
- Research on the agent's component models
- Defined connections between participating research efforts
- Milestones and handoff context for the following year

### What I Learned

When requirements are unclear, starting with a feature list does not solve the underlying problem.

I first had to understand how the user actually worked, what role each technology played in that process, and what assumptions each organization brought to the conversation.

**Once the context was clear and the languages of both sides were connected, stalled development could move again.**

[↑ Top](#top-en)

---

<a id="devlog-en"></a>
## Devlog

### 01. Domain Onboarding

Meteorology domain onboarding

- Terminology
- Forecasting workflow
- Existing forecasting software
- Forecaster workflow
- Feature usage timing

Unfamiliar domain → working project model

### 02. Existing Workflow

Baseline user workflow

```text
Forecast Data
      ↓
Existing Program
      ↓
Forecaster Review
      ↓
Human Judgment
      ↓
Forecast Output
```

Reference point before AI integration

### 03. Requirement Structuring

Requirements grounded in domain context

- Feature-level requirements
- User goals
- Usage context
- Input / output
- Implementation direction
- Acceptance context

`Why` attached to each feature rather than a bare feature list

### 04. Technical Direction

Planning requirements translated into implementation guidance

- Feature purpose
- Usage timing
- Technical constraints
- Implementation review
- Developer Q&A
- Decisions anchored to project intent

Reduced context loss between planning and development

### 05. Milestones / WBS

Project stages and work breakdown

- Yearly milestones
- Monthly milestones
- Tasks
- Owners
- Priorities
- Schedule
- Dependencies
- Deliverables
- Status

Clearer project-level progress visibility

### 06. Meeting Operations

Recurring meeting structure

- Weekly agenda
- Monthly agenda
- Action items
- Decision log
- Schedule
- Follow-up

Meetings connected directly to execution

### 07. Developer Onboarding

New-developer onboarding

Context package

- Project goal
- Meteorology context
- Existing workflow
- Technical direction
- Current architecture
- Assigned task
- Related dependencies

Full-context onboarding instead of isolated task handoff

### 08. Multi-organization Coordination

Coordination across four organizations

- Requirement alignment
- Schedule coordination
- Issue sharing
- Deliverable review
- Meeting preparation
- Decision tracking

Different organizational perspectives aligned around a shared project context

### 09. Issue Response

Development and operational issue handling

- Issue identification
- Context collection
- Priority classification
- Stakeholder mapping
- Resolution direction
- Follow-up

Impact-based handling rather than simple message relay

### 10. Deliverables

Documentation and deliverable management

- Intermediate report
- Final report
- Presentation deck
- Shared templates
- Development documents
- Deliverable checklist

Consistent documentation and handoff standards

### 11. Handoff

First-year wrap-up and next-phase transition

- Major deliverables
- Intermediate / final reporting
- Following-year monthly milestones
- Project-context continuity
- Next-phase tasks

[↑ Case Study](#case-study-en)

</div>
