---
title: "🌦️ Alpha Weather · AI 예보지원"
title_en: "🌦️ Alpha Weather · AI Forecast Support"
type: page
aliases:
  - /portfolio/ai-forecast-support/
description: "기관 간 협력 연구에서 고객의 의사결정과 예측 모델의 목표를 맞추고, 멈춰 있던 작업 흐름을 다시 연결한 PM 경험."
description_en: "PM work on a multi-organization research project: aligning a prediction model's goal with the client's actual decision and reconnecting a stalled workstream."
---

<div class="lang-ko">

<a id="top"></a>

**기간** · 근무 2025.08–2026.02 · 프로젝트 2025.09–2026.02  
**역할** · PM / 기획 / 기관 간 협업 조율  
**소속** · INEEJI(주관기관)  
**고객사** · 국립기상과학원  
**협업** · 대학 연구실 2곳 · 총 4개 기관  
**상태** · 국가 연구 사업 1차년도 완료(기관 간 공동 결과)

[**Case Study ↓**](#case-study) · [**공식 보고서 ↗**](https://dl.nanet.go.kr/detail/MONO12026000015571)

---

<a id="case-study"></a>
## Case Study

### 무엇을 만들어야 하는지부터 찾아야 했습니다

Alpha Weather 프로젝트의 목표는 **예보관의 초기 판단을 돕는 AI 에이전트**를 만드는 것이었습니다.  
여러 예측 결과를 예보관이 일일이 확인하는 대신, 시기와 기상 환경에 맞는 결과를 찾아 예보에 활용할 수 있도록 기상 용어로 설명해주는 서비스입니다.

연구 목표는 있었지만 구체적인 구현 범위는 정해져 있지 않았습니다.  
국립기상과학원은 예보 업무에서 무엇이 필요한지 알고 있었지만, 이를 어떤 기술로 구현할지는 개발 기관과 함께 찾아야 했습니다.  
컨소시엄의 각 기관은 맡은 연구에 집중하고 있었지만, 그 결과가 어떻게 연결되어 하나의 에이전트가 되는지는 공유되지 않은 상태였습니다.

저는 주관기관의 PM으로서 그 사이를 연결했습니다.  
프로젝트의 산출물은 4개 기관이 함께 만든 결과입니다.

### 현업의 흐름을 먼저 이해했습니다

기상학도, 기존 예보 프로그램도 처음 접하는 분야였습니다.  
별도의 온보딩이 없어 문서만으로는 실제 사용 맥락을 알기 어려웠습니다.

국립기상과학원에 방문해 연구진과 담당자에게 프로그램의 작동 방식과 예보 업무의 흐름을 설명 들었고, 이후에도 지속적으로 소통하며 예보관이 어떤 정보를 어떤 순서로 활용하는지, AI의 설명이 언제 필요한지를 확인했습니다.  
예보관을 직접 관찰하거나 인터뷰하지는 않았습니다.

> **고객이 실제로 내려야 하는 판단은 무엇인가?**  
> **각 연구기관의 결과는 그 과정에서 어떤 역할을 하는가?**  
> **서로 떨어진 연구 결과를 어떻게 하나의 서비스로 연결할 것인가?**

### 예측 모델의 목표와 요구사항의 불일치

에이전트는 여러 요소로 구성되며, 그중 하나가 에이전트에 결과를 제공하는 별도의 예측 모델이었습니다.  
고객사의 설명을 정리해 보니, 예보관에게 필요한 것은 여러 예측 결과 중 **어떤 결과를 참고할지 고르는 판단**이었습니다.  
반면 당시 예측 모델은 값을 직접 맞히는 회귀 문제로 설정되어 있었습니다.

저는 이 차이를 팀에 공유하고, 예측 모델의 작업 방향을 회귀에서 분류로 바꾸는 방안을 제안했습니다.  
고객의 판단 단위가 '어느 결과를 선택할 것인가'라면 모델의 출력도 그 선택을 직접 돕는 형태가 더 적합하다고 봤기 때문입니다.  
이 제안은 에이전트 전체의 유형을 바꾸는 것이 아니라 그 안의 예측 모델이 풀어야 할 문제를 다시 정의하는 것이었고, 팀 내 논의를 거쳐 이 제안이 채택되어 예측 모델의 개발 방향이 분류로 바뀌었고, 모델의 실제 구현은 해당 연구를 맡은 연구진과 개발자가 담당했습니다.

### 흩어진 작업을 하나의 흐름으로 정리했습니다

각 기관이 진행하던 작업의 의미와 연결 지점을 파악하고, 이들이 하나의 에이전트로 동작하기 위한 전체 흐름을 그렸습니다.  
보안과 계약상 구체적인 모델명과 구조는 공개할 수 없지만, 각 연구 결과가 어디에서 입력되고 다음 단계에 무엇을 전달해야 하는지, 어떤 작업이 어떤 작업에 의존하는지를 정리했습니다.

개발자들을 모아 화이트보드에 흐름을 그려가며 프로젝트의 목적과 각 작업의 관계를 설명했습니다.  
이후 개발자들은 구현 방향을 확인할 때 저를 찾아왔고, 기관 간 협력이 필요한 지점도 구체적으로 요청할 수 있게 됐습니다.

### 4개 기관 사이의 병목 해소

갈등의 원인은 어느 한쪽이 틀렸기 때문이 아니었습니다.  
국립기상과학원은 실제 예보 과정을 기준으로, 개발 기관은 기술 개발 과정을 기준으로 이야기했습니다.  
같은 목표를 두고도 서로 다른 전제에서 출발하다 보니 요구사항이 제대로 전달되지 않았고, 불신도 쌓이고 있었습니다.

저는 국립기상과학원과 요구사항을 수시로 확인해 연구진과 개발자가 판단할 수 있는 기술적 맥락으로 바꿔 전달하고, 반대로 기술적 제약과 개발 방식을 고객사가 이해할 수 있는 언어로 설명했습니다.  
기관 간 데이터 전달과 전문가 검토 일정도 조율했습니다.

이 과정을 통해 약 3개월 동안 정체되어 있던 작업이 다시 진행되기 시작했고, 팀은 이후 2개월 안에 해당 작업을 마무리했습니다.  
작업을 완료한 것은 각 기관의 연구진과 개발자이며, 제 역할은 멈춰 있던 의존 관계를 풀고 방향을 맞추는 것이었습니다.

### 프로젝트가 보이도록 운영했습니다

- 기관별 작업과 협력이 필요한 지점 확인
- 통합 마일스톤과 WBS 정리
- 주간 · 월간 회의 아젠다와 후속 작업 관리
- 신규 개발자에게 프로젝트 목적과 업무 흐름 설명
- 일정, 이슈, 의사결정과 산출물 추적
- 기관별 산출물 취합과 최종 제출본 관리

### 공식 보고서 검수

주관기관 PM으로서 1차년도 최종보고서의 작성과 제출 과정을 관리했습니다.  
회사 담당 내용 검토, 기술 설명과 실제 구현 내용 대조, 기관별 작성 내용 취합, 목차와 전체 흐름 정리, 문장 · 형식 · 수치 · 그림 · 표 · 인용 확인, 최종 제출본 검수를 맡았습니다.

보고서 『AI기반 예보지원 기술개발 1』은 2025년 기상청 국립기상과학원에서 발행됐으며 국회도서관에 소장되어 있습니다.

🔗 [국회도서관 소장정보](https://dl.nanet.go.kr/detail/MONO12026000015571)

### 별도 운영 개선: 사내 도구의 자발적 확산

이 프로젝트를 운영하며 Notion 템플릿 1개와 Google Apps Script 기반 WBS 도구 1개를 만들었습니다.  
두 도구는 세트로 배포한 것이 아니라 각각 독립적으로 만든 것이며, 다른 팀과 프로젝트가 필요에 따라 자발적으로 가져가 사용했습니다.  
제가 확인한 범위에서는 일주일 안에 적어도 4곳의 다른 사내 프로젝트가 이 도구들을 사용했습니다.

이 사내 도구는 제가 D3F!B에서 운영하는 Notion 워크스페이스나 DIA 자동화와는 별개의 작업입니다.

### 1차년도 결과(기관 공동)

- 기존 예보 프로그램 개선
- AI 에이전트의 전체 흐름 설계
- 에이전트를 구성하는 모델 연구
- 기관별 연구 결과의 연결 지점 정리
- 차년도 작업을 위한 마일스톤과 인수 맥락 정리

### 제약과 배운 점

계약상 보고서 · 다이어그램 · 모델 구성의 세부 내용은 공개하지 않고, 일반화한 업무 흐름과 판단 근거만 기록했습니다.

요구사항이 모호할 때 기능 목록부터 만드는 것으로는 문제가 풀리지 않았습니다.  
고객이 실제로 어떤 판단을 내려야 하는지, 각 기술이 그 과정에서 어떤 역할을 하는지, 서로 다른 조직이 무엇을 전제로 말하는지를 먼저 이해해야 했습니다.

[↑ Top](#top)

---

<a id="devlog"></a>
## Devlog

### 01. Domain Onboarding

기상 도메인 학습

- 기상 용어 정리
- 예보 업무 구조 확인
- 기존 예보 프로그램 분석
- 예보관의 정보 활용 흐름 파악(연구진 · 담당자 설명 기반)
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

**Period** · Employed Aug 2025–Feb 2026 · Project Sep 2025–Feb 2026  
**Role** · PM / Planning / Cross-organization Coordination  
**Company** · INEEJI (Lead Organization)  
**Client** · National Institute of Meteorological Sciences (NIMS)  
**Collaboration** · Two university research labs · four organizations in total  
**Status** · First year of a national research program completed (joint result)

[**Case Study ↓**](#case-study-en) · [**Official Report ↗**](https://dl.nanet.go.kr/detail/MONO12026000015571)

---

<a id="case-study-en"></a>
## Case Study

### Defining what we needed to build

The goal of Alpha Weather was **an AI agent that supports a forecaster's initial judgment**.  
Instead of requiring forecasters to review every prediction result, the agent would identify the results most relevant to the time and weather conditions and explain them in meteorological terms.

The research objective was clear, but the implementation scope was not.  
NIMS knew what forecasters needed, while the technical approach still had to be worked out with the development consortium.  
Each organization was focused on its own research, but there was no shared picture of how those pieces should work together as one agent.

As PM for the lead organization, I connected the two sides.  
The project's outputs were produced jointly by four organizations.

### Understanding the workflow first

Meteorology and the existing forecasting software were both new to me, and there was no formal onboarding.

I visited NIMS, where researchers and staff walked me through the software and the forecasting workflow.  
Through ongoing communication with them, I learned which information forecasters use, in what order, and when an AI-generated explanation would help.  
I did not observe or interview forecasters directly.

> **What decision does the client actually need to make?**  
> **What role does each research output play in that process?**  
> **How can separate research efforts become one service?**

### A mismatch between the requirement and the model's goal

The agent had several components, one of which was a separate prediction model feeding results into it.  
When I organized what the client had told us, the forecasters' real need was to **choose which of several prediction results to rely on**.  
The prediction model, however, had been framed as a regression problem that estimated values directly.

I raised this gap with the team and proposed reframing the prediction model's task from regression to classification: if the client's decision is which result to pick, the model's output should support that choice directly.  
This was a change to the problem one component model solved, not to the type of the agent as a whole.  
After discussion, the team adopted the proposal and moved the model's development to classification; the researchers and developers responsible for that model handled its implementation.

### Turning separate efforts into one flow

I mapped the purpose of each organization's work and the points where they needed to connect.  
The specific models and architecture are confidential, but I defined how each research output should enter the overall flow, what it needed to pass to the next stage, and which tasks depended on which.

I brought the developers together and used a whiteboard to explain the project's purpose and how their work fit together.  
Developers then began coming to me to confirm implementation decisions, and the teams could make specific requests when cross-organization work was needed.

### Clearing a bottleneck across four organizations

The conflict did not come from either side being wrong.  
NIMS spoke from the forecasting process, while the developers spoke from the development process.  
Working toward the same goal from different assumptions made requirements hard to interpret and weakened trust.

I checked requirements with NIMS as they evolved and translated them into technical context for researchers and developers, and explained technical constraints back to the client in terms they could evaluate.  
I also coordinated data handoffs and expert reviews between the organizations.

A workstream that had been stalled for about three months started moving again, and the team completed it within the following two months.  
The researchers and developers at each organization did that work; my part was to untangle the dependencies and align the direction.

### Making the project visible

- Identified each organization's work and cross-team dependencies
- Created an integrated milestone plan and WBS
- Managed weekly and monthly agendas and follow-up tasks
- Onboarded new developers with the project purpose and workflow
- Tracked schedules, issues, decisions, and deliverables
- Collected each organization's deliverables and managed the final submission

### Reviewing the official report

As PM for the lead organization, I managed the preparation and submission of the first-year final report: reviewing INEEJI's sections, checking technical descriptions against the implemented work, consolidating each organization's contributions, organizing the structure, checking wording, figures, tables, numbers, and citations, and reviewing the final submission.

The report, **Development of AI-Based Technology for Weather Forecasting, Vol. 1**, was published by NIMS in 2025 and is held by the National Assembly Library of Korea.

🔗 [National Assembly Library Catalog](https://dl.nanet.go.kr/detail/MONO12026000015571)

### A separate operations improvement: internal tools adopted by other teams

While running this project, I built one Notion template and one WBS tool based on Google Apps Script.  
They were separate tools, not a package, and other teams picked them up on their own as they needed them.  
Within a week, at least four other internal projects that I know of were using them.

These internal tools are separate from the Notion workspace and DIA automation I run for D3F!B.

### First-year outcomes (joint)

- Improvements to the existing forecasting software
- Overall flow for the AI agent
- Research on the agent's component models
- Defined connections between participating research efforts
- Milestones and handoff context for the following year

### Constraints and lessons

Under the contract, details of the report, diagrams, and model configuration stay private; this page records only the generalized workflow and the reasoning behind decisions.

When requirements are unclear, starting with a feature list does not solve the underlying problem.  
I first had to understand what decision the client actually needed to make, what role each technology played in it, and what assumptions each organization brought to the conversation.

[↑ Top](#top-en)

---

<a id="devlog-en"></a>
## Devlog

### 01. Domain Onboarding

Meteorology domain onboarding

- Terminology
- Forecasting workflow
- Existing forecasting software
- How forecasters use information (as explained by NIMS staff)
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

---

[← Projects](/projects/) · [Portfolio](/portfolio/)
