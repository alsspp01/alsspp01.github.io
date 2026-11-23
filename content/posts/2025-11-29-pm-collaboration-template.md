---
title: "PM용 협업 도구 두 가지: Notion 템플릿과 WBS"
title_en: "Two PM Tools: A Notion Template and a WBS Sheet"
date: 2025-11-29
description: "여러 기관이 함께하는 과제를 관리하려고 만든 Notion 템플릿과 Google Sheets WBS의 구조"
description_en: "The structure of a Notion template and a Google Sheets WBS I built to run a multi-organization project"
type: "post"
tags: ["PM", "Notion", "Google Apps Script", "INEEJI"]
---

<div class="lang-ko">

원래 개인적으로 쓰던 Notion 템플릿을, INEEJI에서 여러 기관이 참여하는 과제에 맞게 고쳐 쓴 것입니다.  
처음에는 제 과제에만 썼는데, 다른 사내 프로젝트들이 필요에 따라 가져가 쓰기 시작했습니다.

아래 내용은 템플릿 구조만 다룹니다. 실제 과제 정보는 포함하지 않았습니다.

## 만든 이유

과제 하나에 기관, 세부 과제, 부서, 팀원이 얽혀 있었습니다.  
문서는 흩어지고, 누가 무엇을 언제까지 하는지 한 화면에서 보기 어려웠습니다.

목표는 세 가지였습니다.

- PM이 전체 일정과 업무를 한 곳에서 관리
- 팀원은 자기와 관련된 것만 간단히 확인
- 외부 기관과는 필요한 것만 안전하게 공유

## 구조

### 원본은 두 개만

모든 화면은 **회의록**과 **Task** 두 데이터베이스를 원본으로 씁니다.  
나머지는 같은 데이터를 다르게 보여주는 보기입니다.  
문서가 어디 있는지 헷갈려도 원본 두 곳만 보면 찾을 수 있습니다.

### 속성으로 연결

문서마다 **기관 / 세부 / 부서 / 팀원** 속성을 붙입니다.  
이 속성이 필터 역할을 해서, 같은 Task가 PM 화면과 각자의 Page에 동시에 나타납니다.

### 시작 화면

| 영역 | 역할 |
|---|---|
| Info | 과제 개요, 담당 PM, 주요 문서 링크 |
| Milestone | 끝나지 않은 일정을 날짜순으로 |
| 외부 공유 | 지금 외부에 공유 중인 Page와 DB 목록 |
| 주요 일정 | PM 일정과 개발 일정을 나눠서 표시 |
| Dashboard | 팀 내부에서만 보는 공간 |

### 팀원 · 기관별 Page

Page마다 그 주인의 Task만 보이도록 필터를 걸어 둡니다.  
칸반, 타임라인, 관련 회의록, 고객사 참여 회의록을 한 페이지에서 확인할 수 있습니다.

### PM Note

- 상태는 개발자가 바꾸고, 종료는 PM이 판단해서 수동으로 처리
- 타임라인에서 일정 간 종속성 표시
- 마감을 적으면 시작 화면의 주요 일정에 자동으로 반영

## 외부 공유를 안전하게

외부 기관용 Page는 공유할 내용만 남기고 나머지는 지웁니다.

- 보기 잠금으로 필터를 바꾸거나 다른 일정을 볼 수 없게 함
- 원본 데이터베이스 잠금으로 원본 자체에 접근하지 못하게 함

Notion은 Page 안의 데이터베이스를 따로 공유해야 내용이 보입니다.  
이 부분을 놓치기 쉬워서 사용 설명서에 공유 순서를 따로 적어 두었습니다.

## 같이 만든 것: Google Sheets WBS

같은 시기에 Google Sheets와 Apps Script로 WBS 도구도 만들었습니다.  
템플릿과 세트는 아니었고, 필요한 팀이 각각 따로 가져가 썼습니다.

- 업무를 레벨 1~4로 쪼개 적고, 하위 일정을 바꾸면 상위 일정이 따라 바뀜
- 산출물, 담당, 시작 · 종료일, 지연일, 작업일, 진행률, 완료 체크를 한 줄에
- 기간을 정하면 주 단위 간트 차트를 자동으로 그림

![WBS 예시 화면 (템플릿 예시 데이터)](/image/pm-tools/wbs-gantt.png)

시트 상단 메뉴 세 개로 반복 작업을 줄였습니다.

| 메뉴 | 하는 일 |
|---|---|
| Init | 기간에 맞춰 간트 틀을 새로 만듦 |
| Trim | 완료 항목 정리, 상위 일정 동기화, 서식 일괄 정리 |
| Export Timeline | 부서별 일정표를 탭으로 뽑아냄 |

## 돌아보면

- 기능을 늘리기보다 원본을 줄이는 쪽이 관리가 쉬웠음
- 설정 순서를 지키지 않으면 연동이 깨져서, 사용 설명서를 "필독"부터 시작하게 구성
- 다른 프로젝트가 자발적으로 쓰기 시작한 것이 가장 확실한 피드백이었음

관련 사례: [Alpha Weather](/projects/ai-forecast-support/)

</div>

<div class="lang-en" style="display:none">

This is a Notion template I had been using personally, adapted at INEEJI to run a research project shared by several organizations.  
At first I used it only on my own project; other internal projects later picked it up on their own.

This post covers the structure only. No project data is included.

## Why

One project involved several organizations, sub-projects, departments, and people.  
Documents were scattered, and it was hard to see who was doing what by when.

I wanted three things:

- The PM manages every schedule and task in one place
- Each member sees only what concerns them
- External partners get only what they need, safely

## Structure

### Two sources of truth

Every view reads from two databases: **Meeting notes** and **Tasks**.  
Everything else is a different view of the same data, so anything lost can be found in one of the two.

### Linked by properties

Each document carries **Organization / Sub-project / Department / Member** properties.  
They act as filters, so a task shows up on the PM's board and on each person's page at once.

### Home screen

| Area | Purpose |
|---|---|
| Info | Overview, PM, key document links |
| Milestones | Open milestones by date |
| External sharing | Pages and databases currently shared outside |
| Key dates | PM dates and development dates, shown separately |
| Dashboard | Internal-only space |

### A page per member or organization

Each page is filtered to its owner's tasks.  
Kanban, timeline, related meeting notes, and client meetings are all on one page.

### PM Note

- Developers update status; the PM decides when a task is closed
- Dependencies between tasks are shown on the timeline
- Setting a deadline adds it to the home screen's key dates

## Sharing externally

A partner-facing page keeps only what is shared and removes the rest.

- View locks stop visitors from changing filters or seeing other schedules
- The source databases are locked so they cannot be opened directly

In Notion, databases inside a shared page must be shared separately or the page looks empty.  
That is easy to miss, so the guide lists the sharing steps explicitly.

## Built alongside: a Google Sheets WBS

Around the same time I built a WBS tool with Google Sheets and Apps Script.  
It was not bundled with the template; teams adopted each one separately.

- Work is broken into levels 1–4, and changing a child's dates updates its parent
- Deliverable, owner, start and end, delay, working days, progress, and done on one row
- Setting the period draws a weekly Gantt chart automatically

![WBS sample (template example data)](/image/pm-tools/wbs-gantt.png)

Three menu commands handle the repetitive parts.

| Command | What it does |
|---|---|
| Init | Rebuilds the Gantt frame for the project period |
| Trim | Filters finished items, syncs parent dates, cleans up formatting |
| Export Timeline | Generates a schedule tab per department |

## Looking back

- Fewer sources were easier to manage than more features
- Setup order mattered, so the guide starts with a "read first" section
- Other projects adopting it on their own was the clearest feedback

Related case study: [Alpha Weather](/projects/ai-forecast-support/)

</div>
