---
title: "DIA — Do It, AI"
title_en: "DIA — Do It, AI"
type: page
description: "반복되는 일을 AI와 함께 도구로 만들어 온 개인 사이드 프로젝트 모음"
description_en: "A personal side project: small tools I build with AI to take repeated work off my hands"
---

<div class="lang-ko">

**성격** · 개인 사이드 프로젝트, 진행 중  
**역할** · 요구사항 정의, 흐름 · 구조 설계, API 연결, 테스트, 운영  
**방식** · 코드는 Claude의 도움을 받아 작성

## 요약

- **출발점** · 팀 운영과 개인 작업에서 같은 일이 반복됨
- **한 일** · 반복 작업을 하나씩 골라 작은 도구로 만들고, 실제로 쓰면서 고침
- **결과** · 블로그 배포, 회의록, 개발일지 알림은 지금도 쓰는 중. 일부는 [D3F!B 팀 운영](/projects/d3fib-workspace/)에 적용

하나하나를 프로젝트라 부르기엔 작아서, DIA라는 이름으로 묶어 기록합니다.

## 만든 것

### AI Party
<small>Obsidian 플러그인 · 1편</small>

여러 AI를 캐릭터 탭으로 나란히 띄워 함께 작업하는 플러그인.  
AI마다 권한 단계를 따로 두고, 노트 쓰기는 플러그인이 같은 기준으로 대신 처리합니다.

- [AI Party - 여러 AI를 한 볼트에서](/posts/ai-party-obsidian-plugin/)

### 집사
<small>Discord 봇 · 2편 · [GitHub](https://github.com/alsspp01/Secretary4Discord)</small>

Discord 음성 회의를 요약해 Notion 회의록으로 남기고, 정기 공지를 예약 발송합니다.

- [PART 01. 회의록](/posts/집사-part-01-회의록/)
- [PART 02. 메시지 예약](/posts/집사-part-02-메세지-예약/)

### Notion → Discord
<small>웹훅 서버 · 1편 · [GitHub](https://github.com/alsspp01/Notion2Discord)</small>

Notion 개발일지에 새 글이 생기면 Discord에 알립니다.

- [자동화 PART 01. Notion 알림 Discord로 보내기](/posts/자동화-part-01-notion-알림-discord로-보내기/)

### 블로그 활성화
<small>이 블로그 · 3편</small>

Hugo와 GitHub Actions로 자동 배포를 만들고, 글 쓰는 과정을 줄였습니다.

- [PART 01. Github Blog](/posts/블로그-활성화-part-01-github-blog/)
- [PART 02. Blog Editor](/posts/블로그-활성화-part-02-blog-editor/)
- [PART 03. AI blogging manual](/posts/블로그-활성화-part-03-ai-blogging-manual/)

## AI와 일을 나누는 방식

- 무엇을 왜 만들지, 어떤 흐름으로 돌아가야 하는지는 제가 정함
- 코드 작성은 Claude가 돕고, 구조 선택과 외부 서비스 연결, 테스트는 제가 확인
- 실제로 써 보고 불편한 점을 다음 요구사항으로 넘김

## 한계

- 개인 · 소규모 팀 사용 기준. 많은 사용자를 전제로 설계하지 않음
- 글 본문 일부는 AI가 정리한 기록이라 문체가 다름

[DIA 태그 전체 보기](/tags/dia/)

</div>

<div class="lang-en" style="display:none">

**Type** · Personal side project, ongoing  
**Role** · Requirements, flow and structure design, API integration, testing, operation  
**How** · Code written with Claude's help

## Summary

- **Starting point** · The same chores kept coming back in team operations and my own work
- **What I did** · Picked repeated tasks one at a time, turned each into a small tool, and kept fixing it while using it
- **Result** · Blog deployment, meeting notes, and devlog alerts are still in daily use; some run in [D3F!B team operations](/projects/d3fib-workspace/)

Each tool is too small to call a project on its own, so I keep them together under the name DIA.

## What I built

### AI Party
<small>Obsidian plugin · 1 post</small>

A plugin that runs several AIs side by side as character tabs.  
Each AI gets its own permission level, and the plugin writes notes on their behalf under one shared rule.

- [AI Party: several AIs in one vault](/posts/ai-party-obsidian-plugin/)

### Jipsa
<small>Discord bot · 2 posts · [GitHub](https://github.com/alsspp01/Secretary4Discord)</small>

Summarizes Discord voice meetings into Notion notes and sends scheduled announcements.

- [Part 01: Meeting Records](/posts/집사-part-01-회의록/)
- [Part 02: Scheduled Messages](/posts/집사-part-02-메세지-예약/)

### Notion → Discord
<small>Webhook server · 1 post · [GitHub](https://github.com/alsspp01/Notion2Discord)</small>

Posts to Discord when a new Notion devlog appears.

- [Automation Part 01: Sending Notion Updates to Discord](/posts/자동화-part-01-notion-알림-discord로-보내기/)

### Bringing the blog back
<small>This blog · 3 posts</small>

Set up auto-deploy with Hugo and GitHub Actions, and cut down the steps for writing a post.

- [Part 01: GitHub Blog](/posts/블로그-활성화-part-01-github-blog/)
- [Part 02: Blog Editor](/posts/블로그-활성화-part-02-blog-editor/)
- [Part 03: AI Blogging Manual](/posts/블로그-활성화-part-03-ai-blogging-manual/)

## How I split the work with AI

- I decide what to build, why, and how it should flow
- Claude helps write the code; I check the structure choices, external integrations, and tests
- I use each tool for real and turn what bothers me into the next requirement

## Limits

- Built for personal and small-team use, not for many users
- Parts of the post bodies are AI-organized records, so the writing style differs

[All posts tagged DIA](/tags/dia/)

</div>

---

[← Projects](/projects/) · [Portfolio](/portfolio/)
