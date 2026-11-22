---
title: "Arcanum Nights"
title_en: "Arcanum Nights"
type: page
description: "두 시점의 정보를 합쳐 별자리를 완성하는 1~2인 퍼즐 게임. 기획 리드"
description_en: "A one- or two-player puzzle game where two viewpoints combine into a constellation. Lead game designer"
dated: true
period_start: "2025-01"
tags: ["D3F!B"]
---

<div class="lang-ko">

**기간** · 2025.01–현재  
**팀** · D3F!B · 기획 2 · 개발 4 · 아트 4  
**역할** · 대표 · 팀장 · 게임 기획 리드  
**상태** · Steam Early Access (2026.02.20, 싱글 플레이) · 멀티플레이 개발 중

[Steam](https://store.steampowered.com/app/3453760/Arcanum_Nights/)

## 요약

- **문제** · 참고할 게임이 없는 오리지널 퍼즐 규칙을, 팀이 구현할 수 있고 처음 보는 플레이어도 이해할 수 있게 만들어야 함
- **한 일** · 규칙을 와이어프레임 · 데이터 테이블로 명세화, Python 맵 제작 · 검증 도구, PlayX4 관찰로 UX 개선, 출시 범위 결정
- **결과** · PlayX4 2025 · 2026 시연, 2026년 2월 싱글 플레이로 Steam Early Access 출시(팀)

## 과정

### X에서 시작한 규칙

`X`는 두 선이 만나서 생깁니다.  
따로 보면 의미 없는 두 패턴이 한 시점에서 겹칠 때 모양이 완성되는 퍼즐로 발전시켰고, 결과물을 별자리로 정했습니다.  
두 플레이어는 서로 다른 정보를 보고, 대화로 답을 찾습니다.

### 설명할 레퍼런스가 없을 때

처음 설명했을 때 개발자들도 게임이 어떻게 풀리는지 그리기 어려워했습니다.

- 게임을 작은 기능 단위로 나눠 와이어프레임 작성
- 상태와 조건을 자료구조 · 데이터 테이블로 정리
- 기믹 전체를 이해하기 전에도 기능 단위로 프로토타입을 만들 수 있게 함

플레이 가능한 프로토타입이 나오고 나서야 팀 전체가 같은 그림을 보게 됐습니다.

### 맵 제작 · 검증 도구

- 기획자가 맵을 그리고, 답이 하나뿐인지 검증한 뒤 CSV로 내보내는 Python 도구
- CSV 구조를 몰라도 게임에 바로 넣을 수 있어 변환 실수가 사라짐
- 이 도구로 만든 데이터가 실제 게임 전반에 쓰임

### 플레이를 보고 고친 것

PlayX4에서 클리어 여부뿐 아니라 표정, 자세, 클릭 속도, 반복 행동을 관찰했습니다.

| 관찰 | 수정 |
|---|---|
| 이동 애니메이션이 길어 같은 곳을 여러 번 클릭 | 애니메이션 단축, 이동 중 다음 입력을 기억해 이어서 이동 |
| 튜토리얼에서 히든 엔딩 조건을 알려주자 힌트를 참다 지쳐서 이탈 | 조건을 처음에 보여주지 않음 → 힌트로 규칙을 익힌 뒤 스스로 줄임 |
| 규칙을 만든 나에게만 쉬운 퍼즐 | 답 대신 다시 추리할 지점을 알려주는 단계별 힌트 추가 |

### 직접 구현해 본 부분

개발자가 만든 메인 장면을 바탕으로 튜토리얼을 Unity/C#으로 직접 구현했습니다.  
문서로는 보이지 않던 UI 불편을 찾아 고쳤고, 구현 비용을 알고 기획 결정을 내리는 데 도움이 됐습니다.

### 출시 범위 정하기

처음부터 2인 멀티플레이 게임으로 설계했지만, 긴 개발로 팀 피로도가 높았습니다.

- 두 시점을 번갈아 보는 싱글 플레이에서도 핵심 경험이 유지되는지 먼저 확인
- 데모로 완주 가능 여부를 검증
- 싱글 플레이로 먼저 출시하고, 멀티플레이는 이후 개발로 분리

## 배운 점

- 오리지널 아이디어는 팀이 구현할 수 있는 구조로 설명해야 게임이 됨
- 처음 보는 플레이어에게 다시 보여주고, 팀이 끝낼 수 있는 범위로 줄이는 것까지가 기획

<details>
<summary>Devlog · 기획 범위</summary>

| 영역 | 내용 |
|---|---|
| 핵심 경험 | 1~2인 퍼즐, 해 · 달 · 별자리 테마, 협력 상호작용 |
| 시스템 | 코어 루프, 플레이어 상태, 스테이지 흐름, 성공 · 실패 조건 |
| UI/UX | 첫 플레이 기준 정보 노출 순서, 상호작용 피드백 |
| 튜토리얼 | 설명보다 플레이로 배우는 순서 |
| 데이터 | 시스템 · 스테이지 · 스토리 데이터, 상태 정의, 예외 상황 |
| 제작 범위 | 개발 가능성, 아트 리소스 비용, 일정, 우선순위 |
| 데모 빌드 | 화면에 안 보이는 원본 스토리 데이터가 빌드에 포함된 문제 → 데모용 데이터 분리 |

</details>

</div>

<div class="lang-en" style="display:none">

**Period** · Jan 2025–present  
**Team** · D3F!B · planning 2 · development 4 · art 4  
**Role** · Studio head · Team lead · Lead game designer  
**Status** · Steam Early Access (Feb 20, 2026, single-player) · Multiplayer in development

[Steam](https://store.steampowered.com/app/3453760/Arcanum_Nights/)

## Summary

- **Problem** · An original puzzle rule with no reference game had to be buildable by the team and understandable to first-time players
- **What I did** · Specified the rules through wireframes and data tables, built a Python map tool with solution checks, improved UX from observation at PlayX4, and set the release scope
- **Result** · Showcased at PlayX4 2025 and 2026; the team released the single-player version in Steam Early Access in February 2026

## Process

### A rule that started with an X

An `X` appears where two lines meet.  
I turned that into a puzzle where two patterns that mean nothing alone form a shape when seen from one viewpoint, and made the shape a constellation.  
The two players see different information and have to talk to find the answer.

### Explaining without a reference

When I first described it, even the developers struggled to picture how it played.

- Broke the game into small features and drew wireframes
- Defined states and conditions as data structures and tables
- Let the team prototype feature by feature before the whole mechanic clicked

The team shared the same picture only after a playable prototype existed.

### Map creation and validation tool

- A Python tool for designers to draw a map, check it has exactly one solution, and export CSV
- No need to know the CSV format, so conversion mistakes disappeared
- Data from the tool is used throughout the game

### Fixes from watching people play

At PlayX4 I watched expressions, posture, click speed, and repeated actions, not just whether players cleared a stage.

| Observed | Changed |
|---|---|
| Long move animations led to repeated clicking | Shorter animation; the next click is queued and played right after |
| Revealing the hidden-ending condition made players avoid hints until they quit | Stopped revealing it up front → players learned with hints, then cut back on their own |
| Puzzles were easy only for me, who made the rules | Step-by-step hints that point to where to rethink, not to the answer |

### Building part of it myself

I implemented the tutorial in Unity/C# on top of the developers' main scene.  
It surfaced UI friction that documents missed and helped me plan with a sense of implementation cost.

### Choosing what to ship

The game was designed for two players, but the long production was wearing on the team.

- Checked first that single-player, switching between both views, kept the core experience
- Confirmed with a demo that the game could be completed
- Shipped single-player first and split multiplayer into later work

## What I learned

- An original idea becomes a game only once the team can build from its structure
- Showing it to first-time players and cutting scope the team can finish are part of design too

<details>
<summary>Devlog · design scope</summary>

| Area | Content |
|---|---|
| Core experience | One- or two-player puzzle, sun · moon · constellation theme, cooperation |
| Systems | Core loop, player states, stage flow, success and failure conditions |
| UI/UX | What a first-time player sees and when; interaction feedback |
| Tutorial | Learning through play rather than text |
| Data | System, stage, and story data; state definitions; edge cases |
| Scope | Feasibility, art cost, schedule, priority |
| Demo build | Original story data hidden in the build → split demo-only data |

</details>

</div>

---

[← Projects](/projects/) · [Portfolio](/portfolio/)
