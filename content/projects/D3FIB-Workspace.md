---
title: "D3F!B Team Operations & Automation"
title_en: "D3F!B Team Operations & Automation"
type: page
aliases:
  - /portfolio/d3fib-workspace/
description: "활동 시간이 다른 10명 원격 게임 제작팀의 협업 구조와 운영 자동화"
description_en: "Collaboration structure and automation for a ten-person remote game team on different schedules"
---

<div class="lang-ko">

**기간** · 2024.11–현재  
**역할** · 창립자 · 대표 · 팀장 · 운영 체계와 자동화 설계  
**팀** · 약 10명, 완전 원격 (기획 · 개발 · 아트)

## 요약

- **문제** · 학생과 직장인이 섞여 활동 시간이 모두 다른 팀. 자료와 결정이 흩어지고, 질문이 대표 한 명에게 몰림
- **한 일** · Discord · Notion · Drive 역할 분리, 권한 구조 설계, Notion 워크스페이스 구축과 탐색 구조 개선, 회의록 · 개발일지 자동화 적용
- **결과** · 회의 기록과 온보딩이 문서로 이어지고, 같은 설명을 반복하는 시간이 줄어 방향과 의사결정에 집중

## 과정

### 도구마다 역할 나누기

새 도구를 늘리기보다 이미 쓰던 도구의 역할부터 정했습니다.

| 도구 | 역할 |
|---|---|
| Discord | 실시간 대화, 회의, 공지, 자동 알림 |
| Notion | 기획, 일정, 회의록, 개발일지, 팀 가이드, 부서별 마일스톤 |
| Google Drive | 함께 쓰는 최종 제작 리소스 |

채널 · 역할 체계, Notion 데이터베이스, Drive 폴더와 파일명 규칙, 회의 방식, 온보딩 문서까지 직접 설계했습니다.

### 권한은 필요한 만큼만

- 운영 · 계약 문서는 대표만 접근, 신규 팀원은 합류 시 보안서약
- 최종 리소스는 팀원 누구나 열람 가능
- 세부 작업 과정은 부서 채널 안에서만 공유, GitHub 알림은 개발 채널로

### 쓰고 있던 워크스페이스를 더 찾기 쉽게

Notion 워크스페이스는 빈 페이지에서 시작해 처음부터 만들었고, 팀은 초기 구조도 사용하고 있었습니다.  
다만 필요한 정보까지 여러 번 클릭해야 했습니다.

- 기획 · 개발 · 아트별 메인 페이지를 강화
- 각 페이지에서 마일스톤, 작업 문서, 자주 쓰는 정보로 바로 이동하도록 탐색 깊이 축소
- 개편 후 부서 페이지 활용을 Notion 트래픽으로 확인

### 반복 업무 자동화

개인 사이드 프로젝트 [DIA](/projects/dia/)에서 만든 도구를 팀 운영에 적용했습니다. DIA는 D3F!B와 별개 프로젝트입니다.

- [Secretary4Discord](https://github.com/alsspp01/Secretary4Discord) · Discord 음성 회의를 안건 단위로 요약해 Notion 회의록으로 저장
- [Notion2Discord](https://github.com/alsspp01/Notion2Discord) · 새 개발일지를 Discord에 자동 알림
- 예약 메시지로 정기 공지, GitHub 웹훅으로 저장소 변경 공유

요구사항 정의, 흐름 설계, 기술 구조 결정, API 연결, 테스트, 운영은 제가 했고, 코드는 Claude의 도움을 받아 작성했습니다.

## 배운 점

- 운영 구조는 관리자가 보기 좋은 것보다 팀원이 빨리 찾아 쓰는 쪽이 맞음
- 정보가 한 사람을 거치지 않게 만드는 것이 원격 팀 규모를 유지하는 조건

</div>

<div class="lang-en" style="display:none">

**Period** · Nov 2024–present  
**Role** · Founder · Team lead · Operations and automation design  
**Team** · About 10, fully remote (planning · development · art)

## Summary

- **Problem** · Students and full-time workers on different schedules. Documents and decisions were scattered, and questions kept routing through one person
- **What I did** · Gave Discord, Notion, and Drive distinct roles, designed permissions, built the Notion workspace and later improved its navigation, applied meeting-note and devlog automation
- **Result** · Meeting records and onboarding moved into documents, and less time went into repeating explanations, leaving more for direction and decisions

## Process

### A role for each tool

Instead of adding tools, I defined what each existing one was for.

| Tool | Role |
|---|---|
| Discord | Live talk, meetings, announcements, automated alerts |
| Notion | Plans, schedules, meeting notes, devlogs, team guide, milestones by discipline |
| Google Drive | Final production assets for everyone |

I designed the channel and role structure, Notion databases, Drive folders and naming rules, meeting practice, and onboarding documents.

### Permissions, only as needed

- Operations and contract documents are representative-only; new members sign an NDA
- Final assets are open to every member
- Work in progress stays in discipline channels; GitHub alerts go to the dev channel

### Making an active workspace easier to navigate

I built the Notion workspace from a blank page, and the team was already using it.  
Reaching what you needed, though, often took several clicks.

- Strengthened a main page for each discipline
- Cut navigation depth so milestones, working documents, and common links were one step away
- Confirmed through Notion traffic that members used the new pages

### Automating repeated work

I applied tools from [DIA](/projects/dia/), a separate personal side project, to the team's operations.

- [Secretary4Discord](https://github.com/alsspp01/Secretary4Discord) · Summarizes Discord voice meetings by agenda item into Notion notes
- [Notion2Discord](https://github.com/alsspp01/Notion2Discord) · Posts new devlogs to Discord automatically
- Scheduled messages for recurring notices; GitHub webhooks for repository updates

I defined the requirements, designed the flows, chose the structure, connected the APIs, tested, and run the tools; the code was written with Claude's help.

## What I learned

- An operating structure should fit how members look things up, not how tidy it looks to the admin
- A remote team scales only when information stops passing through one person

</div>

---

[← Projects](/projects/) · [Portfolio](/portfolio/)
