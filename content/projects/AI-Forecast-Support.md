---
title: "Alpha Weather · AI 예보지원"
title_en: "Alpha Weather · AI Forecast Support"
type: page
aliases:
  - /portfolio/ai-forecast-support/
description: "4개 기관이 함께한 AI 예보지원 연구에서 PM으로 요구사항을 정리하고 멈춰 있던 작업을 다시 연결한 경험"
description_en: "PM work on a four-organization AI forecasting research program: clarifying requirements and restarting a stalled workstream"
---

<div class="lang-ko">

**기간** · 근무 2025.08–2026.02 · 프로젝트 2025.09–2026.02  
**역할** · PM · 기획 · 기관 간 조율  
**소속** · INEEJI (주관기관)  
**고객사** · 국립기상과학원  
**협업** · 대학 연구실 2곳 · 총 4개 기관  
**상태** · 국가 연구 사업 1차년도 완료 (기관 공동 결과)

[공식 보고서 (국회도서관)](https://dl.nanet.go.kr/detail/MONO12026000015571)

## 요약

- **문제** · 목표는 "예보관의 초기 판단을 돕는 AI 에이전트"였지만 구현 범위가 정해지지 않았고, 기관 간 작업은 약 3개월간 정체
- **한 일** · 고객의 실제 판단과 예측 모델 목표의 불일치 발견, 회귀 → 분류 전환 제안, 기관별 입출력 · 의존성 정리, 요구사항 통역과 일정 조율
- **결과** · 제안 채택, 작업 재개 후 팀이 2개월 안에 마무리. 별도로 만든 Notion 템플릿 · WBS 도구를 사내 다른 프로젝트들이 자발적으로 도입

## 과정

### 현업 흐름부터 이해

기상학도, 기존 예보 프로그램도 처음이었습니다.  
국립기상과학원 연구진과 담당자에게 프로그램과 예보 업무 흐름을 설명 듣고, 이후 꾸준히 소통하며 이해를 넓혔습니다.  
예보관을 직접 관찰하거나 인터뷰하지는 않았습니다.

### 예측 모델의 목표 다시 정하기

- 고객에게 필요한 것: 여러 예측 결과 중 **어떤 결과를 참고할지 고르는 판단**
- 당시 예측 모델: 값을 직접 맞히는 회귀 문제
- 제안: 예측 모델의 문제를 회귀 → 분류로 재정의
- 결과: 팀 논의 후 채택. 구현은 해당 연구진과 개발자가 담당

에이전트 전체가 아니라, 그 안의 예측 모델 하나의 목표를 바꾼 제안입니다.

### 흩어진 작업을 하나의 흐름으로

- 기관별 연구 결과가 어디서 입력되고 다음 단계에 무엇을 넘기는지 정리
- 개발자들을 모아 화이트보드로 전체 흐름과 각자의 위치 설명
- 이후 개발자들이 구현 방향을 확인하러 찾아오고, 기관 간 협력 요청이 구체적으로 바뀜

계약상 모델명과 구조는 공개하지 않습니다.

### 4개 기관 사이의 병목 풀기

고객사는 예보 업무 기준으로, 개발 기관은 기술 개발 기준으로 말하고 있었습니다.  
양쪽 요구사항을 서로의 언어로 바꿔 전달하고, 기관 간 데이터 전달과 전문가 검토 일정을 조율했습니다.

약 3개월간 멈춰 있던 작업이 다시 진행됐고, 팀은 이후 2개월 안에 마무리했습니다.  
작업을 완료한 건 각 기관의 연구진과 개발자이고, 제 역할은 막힌 의존 관계를 푸는 것이었습니다.

### 운영과 보고서

- 통합 마일스톤 · WBS, 주간 · 월간 회의 아젠다와 후속 작업 관리
- 신규 개발자 온보딩, 일정 · 이슈 · 산출물 추적
- 1차년도 최종보고서 취합 · 검수 (기술 설명과 실제 구현 대조, 수치 · 그림 · 인용 확인)

### 사내 도구가 다른 팀으로

과제 운영용으로 Notion 템플릿 1개와 Google Sheets WBS 1개를 만들었습니다.  
세트가 아니라 각각 따로 쓰였고, 확인한 범위에서 일주일 안에 적어도 4곳의 다른 사내 프로젝트가 가져가 썼습니다.  
→ [두 도구의 구조](/posts/2025-11-29-pm-collaboration-template/)

## 배운 점

- 요구사항이 모호할 때는 기능 목록보다, 고객이 실제로 내려야 하는 판단을 먼저 확인해야 했음

<details>
<summary>Devlog · 업무 단위 정리</summary>

| 단계 | 내용 |
|---|---|
| 도메인 학습 | 기상 용어, 기존 프로그램, 예보관의 정보 활용 흐름(연구진 · 담당자 설명 기반) |
| 요구사항 정리 | 기능별 목적, 사용 시점, 입력 · 출력, 구현 방향 |
| 기술 방향 | 기획 요구사항을 개발 판단 기준으로 변환, 개발 질문 대응 |
| 마일스톤 · WBS | 연간 · 월간 마일스톤, 담당 · 우선순위 · 의존성 · 산출물 |
| 회의 운영 | 주간 · 월간 아젠다, 액션 아이템, 결정 기록 |
| 온보딩 | 신규 개발자에게 과제 목표와 전체 맥락 전달 |
| 기관 조율 | 요구사항 · 일정 · 이슈 · 산출물 검토 |
| 마무리 | 중간 · 최종 보고, 차년도 마일스톤과 인수 맥락 |

</details>

</div>

<div class="lang-en" style="display:none">

**Period** · Employed Aug 2025–Feb 2026 · Project Sep 2025–Feb 2026  
**Role** · PM · Planning · Cross-organization coordination  
**Company** · INEEJI (lead organization)  
**Client** · National Institute of Meteorological Sciences (NIMS)  
**Collaboration** · Two university labs · four organizations in total  
**Status** · First year of a national research program completed (joint result)

[Official report (National Assembly Library)](https://dl.nanet.go.kr/detail/MONO12026000015571)

## Summary

- **Problem** · The goal was "an AI agent that supports a forecaster's first judgment," but the scope was undefined and cross-organization work had stalled for about three months
- **What I did** · Found a mismatch between the client's real decision and the prediction model's goal, proposed moving it from regression to classification, mapped each organization's inputs, outputs, and dependencies, and translated requirements between client and developers
- **Result** · The proposal was adopted; work resumed and the team finished it within two months. Separately, a Notion template and a WBS tool I built were picked up by other internal projects

## Process

### Learning the workflow first

Meteorology and the existing software were both new to me.  
NIMS researchers and staff walked me through the software and the forecasting workflow, and I kept in regular contact with them.  
I did not observe or interview forecasters directly.

### Reframing the prediction model

- What the client needed: to **choose which of several prediction results to rely on**
- What the model did: regression, estimating values directly
- Proposal: reframe that model's task from regression to classification
- Outcome: adopted after team discussion; the responsible researchers and developers implemented it

This changed the goal of one component model, not the agent as a whole.

### Turning separate efforts into one flow

- Mapped where each organization's output entered the flow and what it passed on
- Walked the developers through the whole flow at a whiteboard
- Developers then came to me to check direction, and cross-organization requests became specific

Model names and architecture stay confidential under the contract.

### Clearing the bottleneck

The client spoke from the forecasting process; the developers spoke from the development process.  
I translated requirements in both directions and coordinated data handoffs and expert reviews between organizations.

A workstream stalled for about three months started moving again, and the team finished it within two months.  
The researchers and developers did that work; my part was untangling the dependencies.

### Operations and the report

- Integrated milestones and WBS, weekly and monthly agendas, follow-ups
- Onboarding new developers; tracking schedules, issues, and deliverables
- Compiling and reviewing the first-year final report (technical descriptions vs. implementation, figures, citations)

### Internal tools other teams adopted

I built one Notion template and one Google Sheets WBS to run the project.  
They were used separately, not as a set, and within a week at least four other internal projects that I know of had picked them up.  
→ [How the two tools are structured](/posts/2025-11-29-pm-collaboration-template/)

## What I learned

- When requirements are vague, the decision the client actually has to make matters more than a feature list

<details>
<summary>Devlog · work breakdown</summary>

| Stage | Content |
|---|---|
| Domain onboarding | Terminology, existing software, how forecasters use information (as explained by NIMS staff) |
| Requirements | Purpose, timing, inputs and outputs, implementation direction per feature |
| Technical direction | Turning planning requirements into development criteria; developer Q&A |
| Milestones · WBS | Yearly and monthly milestones; owners, priorities, dependencies, deliverables |
| Meetings | Weekly and monthly agendas, action items, decision log |
| Onboarding | Giving new developers the goal and full context |
| Coordination | Requirements, schedules, issues, deliverable reviews |
| Wrap-up | Interim and final reports; next-year milestones and handoff |

</details>

</div>

---

[← Projects](/projects/) · [Portfolio](/portfolio/)
