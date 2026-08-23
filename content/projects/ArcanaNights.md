---
title: "🎮 Arcanum Nights"
type: page
description: "해와 달, 별자리를 소재로 한 싱글~2인 퍼즐게임."
dated: true
period_start: "2025-01"
tags: ["D3F!B"]
---

<div class="lang-ko">

<a id="top"></a>

**기간** · 2025.01–현재  
**팀** · D3F!B, 약 10명  
**역할** · 대표 / 팀장 / 게임 기획 리드  
**플랫폼** · PC / 1~2인 플레이  
**상태** · Steam Early Access

[**Case Study ↓**](#case-study) · [**Steam ↗**](https://store.steampowered.com/app/3453760/Arcanum_Nights/)

---

<a id="case-study"></a>
## Case Study

### X에서 시작한 퍼즐

Arcanum Nights의 퍼즐은 단순한 `X`에서 시작했습니다.

X는 두 개의 선이 만나 만들어집니다. 그렇다면 서로 떨어진 두 개의 패턴과 그 관계를 알면, 원래의 형태를 다시 만들 수 있지 않을까 생각했습니다.

이 생각을 **각자의 시점에서는 의미 없어 보이는 두 패턴이 하나의 시점에서 만날 때 특별한 형태로 완성되는 퍼즐**로 발전시켰습니다. 두 플레이어는 서로 다른 정보를 보고, 대화를 통해 하나의 답을 찾아야 합니다.

별자리도 비슷하다고 느꼈습니다. 우주의 다른 곳에서는 의미 없는 별의 배치가 지구라는 한 시점에서 봤을 때 모양과 이야기를 갖습니다. 그래서 퍼즐의 결과를 별자리로 정하고, 타로와 점성술을 게임의 세계관에 연결했습니다.

이 기믹은 참고한 게임이나 레퍼런스 없이 처음부터 직접 설계했습니다.

### 레퍼런스가 없으면 설명부터 만들어야 했습니다

처음 기믹을 설명했을 때는 개발자들도 게임이 어떻게 풀리는지 이해하기 어려워했습니다. 비교해서 보여줄 기존 게임이 없었기 때문에 아이디어만으로는 구현을 시작할 수 없었습니다.

그래서 게임을 작은 기능 단위로 나눠 와이어프레임을 만들고, 필요한 상태와 조건을 자료구조와 데이터 테이블로 정리했습니다. 개발자가 기믹 전체를 바로 이해하지 못하더라도 각 기능을 따라 프로토타입을 만들 수 있도록 했습니다.

플레이 가능한 프로토타입이 나오자 팀도 기믹의 의도와 전체 흐름을 이해하기 시작했습니다. 추상적인 아이디어가 팀이 함께 논의할 수 있는 게임으로 바뀐 순간이었습니다.

### 맵을 만들고 검증하는 도구

퍼즐 맵은 정답이 하나만 나오도록 검증해야 했고, 완성된 형태를 눈으로 확인하면서 게임 데이터로 옮길 수 있어야 했습니다.

이를 위해 기획자가 맵을 만들고 검증한 뒤 CSV 데이터로 내보낼 수 있는 Python 도구를 제작했습니다. CSV 구조를 몰라도 게임에 바로 적용할 파일을 만들 수 있어, 기획팀이 데이터 형식과 변환 실수를 걱정하지 않고 퍼즐 설계에 집중할 수 있었습니다.

이 도구로 만든 데이터는 실제 게임 제작 전반에 사용됐습니다.

### 나에게 쉽다고 플레이어에게 쉬운 것은 아니었습니다

처음에는 이론적으로 풀 수 있는 퍼즐이니 별도의 힌트가 많지 않아도 된다고 생각했습니다. 하지만 그 규칙을 만든 저에게만 쉬웠습니다.

기획팀의 의견과 플레이어 테스트를 통해 여러 단계의 힌트를 추가했습니다. 답을 바로 알려주기보다 플레이어가 막힌 지점에서 다시 추리할 수 있도록 돕는 방향으로 조정했습니다.

PlayX4에서는 플레이어의 표정과 자세뿐 아니라 클릭하는 속도와 반복 행동도 관찰했습니다. 퍼즐 마니아들이 처음 보는 신선한 기믹이라고 평가해준 덕분에 새로운 퍼즐에 대한 수요를 확인할 수 있었고, 동시에 낯선 규칙을 더 쉽게 전달해야 한다는 과제도 분명해졌습니다.

### 플레이를 보면 문서에서 보이지 않던 문제가 보입니다

게임은 클릭한 위치로 이동하는 방식입니다. 이동 애니메이션이 길자 플레이어들은 같은 곳을 여러 번 클릭했고, 퍼즐을 고민하기보다 이동을 기다리는 데 답답함을 느꼈습니다.

그래서 이동 애니메이션을 줄이고, 이동 중 들어온 다음 입력을 기억했다가 현재 이동이 끝난 직후 이어서 움직이도록 바꿨습니다. 조작의 끊김이 줄어들자 플레이어가 맵을 살피고 퍼즐을 고민하는 흐름도 자연스러워졌습니다.

힌든 엔딩 조건도 처음에는 튜토리얼에서 알려줬습니다. 플레이어들은 처음부터 조건을 만족하려고 힌트 사용을 참았고, 한 스테이지에 오래 머물다 지쳐 게임을 그만두기도 했습니다.

조건을 바로 보여주지 않도록 바꾸자 플레이어들은 먼저 힌트를 사용해 규칙을 익혔습니다. 이후에는 더 잘 풀기 위해 스스로 힌트 사용을 줄였습니다.

**모든 정보를 미리 주는 것이 항상 친절한 설계는 아니었습니다.**

### 직접 구현하며 기획을 확인했습니다

개발자가 만든 메인 게임 장면을 바탕으로 대사와 기믹 설명이 포함된 튜토리얼을 직접 구현했습니다. 이 과정에서 문서로 볼 때는 알기 어려웠던 UI의 불편함을 발견해 수정했습니다.

코드 최적화와 리팩터링 과정에도 참여해 기능이 어떤 흐름으로 동작해야 하는지 설명했습니다. 개발자의 역할을 대신하려는 것이 아니라, 구현 구조와 비용을 이해한 상태에서 기획 결정을 내리기 위해서였습니다.

### 완벽을 기다리지 않고 출시 범위를 정했습니다

Arcanum Nights는 처음부터 2인 멀티플레이를 중심으로 설계했습니다. 하지만 게임 개발을 처음부터 공부하며 진행한 사이드 프로젝트였고, 장기간 개발로 팀의 피로도도 높아지고 있었습니다.

대표로서 팀이 계속 나아가기 위해서는 실제 출시라는 결과가 필요하다고 판단했습니다.

두 플레이어의 시점을 번갈아 보는 싱글 플레이에서도 정보 조합이라는 핵심 경험이 유지되는지 먼저 확인했습니다. 데모를 공개해 플레이 가능한 수준을 검증한 뒤, 싱글 플레이를 중심으로 2026년 2월 20일 Steam Early Access를 출시했습니다.

멀티플레이는 싱글 플레이에서 정리된 시스템과 추가 사용자 테스트를 바탕으로 개발하고 있습니다.

### 대표, 팀장, 게임 기획 리드

D3F!B는 기획 3명, 개발 4명, 아트 3명으로 구성된 약 10명 규모의 팀입니다.

대표로서 외부 네트워킹, 시연 행사와 공모전 출품에 책임을 지고 있습니다. 팀장으로서는 일정과 업무를 배분하고 각 직군의 결정을 검토해 최종 방향을 정합니다.

게임 기획 리드로서는 초기 기획과 핵심 기믹을 만들고, 개발과 아트가 실제로 제작할 수 있도록 기능의 범위와 규모를 조정합니다.

- 핵심 퍼즐 기믹과 게임 방향 설계
- 시스템, UI/UX, 튜토리얼과 밸런스 기획
- 와이어프레임, 자료구조와 데이터 테이블 작성
- Python 맵 제작 · 검증 도구 개발
- Unity/C# 기반 튜토리얼 구현 참여
- 일정 관리, 업무 배분과 최종 의사결정
- PlayX4 2025 · 2026 참가 및 Steam 출시 결정

### 세계관에도 다음 이야기를 남겼습니다

플레이어 캐릭터 Helian과 Oenothera의 이름은 꽃의 학명에서 가져왔습니다. 두 이름은 Arcanum Nights의 이야기뿐 아니라 식물 아포칼립스를 다룰 차기 세계관으로 이어집니다.

### 배운 점

오리지널 기믹을 만드는 것만으로는 게임이 완성되지 않았습니다.

팀이 구현할 수 있는 구조로 설명하고, 처음 보는 플레이어가 이해할 수 있도록 다시 보여주고, 팀이 끝까지 만들 수 있는 범위로 조정해야 했습니다.

**새로운 아이디어를 떠올리는 것부터 다른 사람이 실제로 즐길 수 있는 게임으로 내놓는 것까지가 기획의 일이라고 생각합니다.**

🔗 [Steam에서 Arcanum Nights 보기](https://store.steampowered.com/app/3453760/Arcanum_Nights/)

[↑ Top](#top)

---

<a id="devlog"></a>
## Devlog

### 01. Core Experience

게임 핵심 경험 정의

- Single ~ 2 Player Puzzle
- Sun / Moon / Constellation Theme
- Cooperative Interaction
- Stage Progression
- Puzzle Rule

초기 컨셉 → 플레이 가능한 시스템 단위로 구조화

### 02. System Planning

게임 시스템 설계

- Core Loop
- System Rule
- Player State
- Stage Flow
- Puzzle Condition
- Success / Failure Condition
- Interaction Rule

정상 흐름 + 예외 상태 동시 정리

### 03. UI / UX

Player flow 기준 UI 구조 설계

- Main Flow
- In-game UI
- Interaction Feedback
- Tutorial UI
- Stage Transition
- Information Priority

첫 플레이 기준 정보 노출 순서 정리

### 04. Scenario / Tutorial

플레이 시나리오 및 튜토리얼 구성

- First-play Scenario
- Tutorial Sequence
- Interaction Introduction
- Puzzle Rule Introduction
- Stage Progression
- Story Delivery Timing

설명보다 플레이를 통한 학습 중심 구성

### 05. Data Structure

개발 전달용 데이터 구조 정리

- System Data Table
- Stage Data
- Story Data
- State Definition
- Condition / Branch
- Edge Case
- Data Relation

기획 문서와 실제 구현 데이터 연결

### 06. Edge Case

구현 전 상태 분기 사전 검토

- Missing Value
- Duplicate Input
- Simultaneous Condition
- Mid-state Change
- Invalid Interaction
- Unexpected Player Action

Happy path 외 상태 선행 정의

### 07. Unity / C#

기획-개발 간 간극 축소 목적의 직접 구현

- Unity project structure 학습
- C# 구조 학습
- Tutorial implementation
- Data flow 확인
- Runtime 구조 검증

개발 대체가 아닌 구현 비용 파악 목적

### 08. Python Map Tool

맵 제작 반복 작업 보조 도구 제작

- Map data input
- Data conversion
- Repetitive task reduction
- Planning / development handoff 지원

맵 제작 workflow 단순화

### 09. Cross-functional Planning

개발 / 아트 제약 기준 기능 범위 조정

- Development feasibility
- Art resource cost
- Schedule
- Production scope
- Priority
- Implementation complexity

기획 의도 유지 + 제작 비용 최소화 방향 조정

### 10. PlayX4 User Test

현장 플레이 관찰

관찰 항목

- 표정
- 자세
- Keyboard / mouse input rhythm
- 집중 구간
- Pacing drop-off
- Interaction hesitation

반영 항목

- Game tempo
- Animation speed
- Player flow
- Tutorial timing
- Interaction feedback

### 11. Demo Build Review

데모 배포 데이터 검토

문제

- 화면상 stage 비노출
- Original story data 빌드 포함

조치

- Demo 전용 story data 분리
- 원본 콘텐츠 불필요 포함 방지

배포 화면 외 build 내부 데이터까지 검토

[↑ Case Study](#case-study)

</div>

<div class="lang-en" style="display:none">

<a id="top-en"></a>

**Period** · Jan 2025–Present  
**Team** · D3F!B, around 10 people  
**Role** · Studio Head / Team Lead / Lead Game Designer  
**Platform** · PC / 1–2 players  
**Status** · Steam Early Access

[**Case Study ↓**](#case-study-en) · [**Steam ↗**](https://store.steampowered.com/app/3453760/Arcanum_Nights/)

---

<a id="case-study-en"></a>
## Case Study

### A Puzzle That Started with an X

The puzzle in Arcanum Nights began with a simple `X`.

An X is formed when two lines meet. That led me to wonder whether a shape could be reconstructed from two separate patterns and the relationship between them.

I developed that thought into **a puzzle where two patterns that appear meaningless from separate viewpoints form a recognizable shape when brought together**. Each player sees different information, and they must communicate to find one answer.

Constellations felt like a natural match. A group of stars that means nothing from elsewhere in space gains a shape and a story when seen from Earth. That idea became the link between the puzzle, constellations, tarot, and the game's world.

I designed the mechanic from scratch without an existing game or reference to follow.

### With No Reference, I Had to Build the Explanation

When I first introduced the mechanic, the developers could not see how the game was meant to work. There was no similar game I could point to, so the idea alone was not enough to begin implementation.

I broke the game into small functions, created wireframes, and defined its states and conditions through data structures and tables. Even before the team fully understood the mechanic, they could follow those pieces to build a prototype.

Once the prototype became playable, the team began to see the intent and overall flow. An abstract idea had become a game we could discuss together.

### A Tool for Building and Validating Maps

Each puzzle map needed to have exactly one answer. The planning team also needed to inspect the completed shape visually and turn it into data the game could use.

I built a Python tool that let planners create and validate maps, then export them as CSV data. They could produce game-ready files without knowing the CSV structure, which removed conversion errors and let them focus on puzzle design.

The resulting data was used throughout development and in the released game.

### What Was Easy for Me Was Not Easy for the Player

At first, I assumed that because the puzzle could be solved logically, players would not need many hints. It turned out to be easy mainly because I had designed the rules myself.

Feedback from the planning team and playtests led us to add several layers of guidance. Rather than revealing the answer, the hints help players resume their reasoning at the point where they became stuck.

At PlayX4, I watched not only players' expressions and posture but also their clicking speed and repeated actions. Puzzle enthusiasts described the mechanic as fresh and unlike anything they had played before. Their response confirmed the appeal of a new kind of puzzle, while their behavior showed us where its unfamiliar rules needed clearer guidance.

### Play Reveals What Documents Cannot

Players move by clicking a destination. When the movement animation took too long, they repeatedly clicked the same spot. Instead of thinking about the puzzle, they were waiting for the character to move.

We shortened the animation and queued the first new destination clicked during movement, so the character would continue there as soon as the current movement ended. With less friction in navigation, players could stay focused on exploring the map and solving the puzzle.

The tutorial also used to reveal the condition for a hidden ending. Players tried to meet it on their first attempt by avoiding hints, even when they were already tired. Some spent too long on a single stage and stopped playing.

Once we stopped revealing the condition up front, players used hints to learn the rules first. Later, they began limiting hint use on their own as they tried to solve the puzzles more cleanly.

**Giving players every piece of information in advance is not always the most helpful design.**

### Checking the Plan Through Implementation

Using the developers' main game scene, I implemented the tutorial with its dialogue and mechanic explanations. Working directly in the game also revealed UI friction that was difficult to see in a document.

I participated in code optimization and refactoring discussions by defining how each feature should behave. The purpose was not to replace the developers, but to make planning decisions with a practical understanding of implementation structure and cost.

### Choosing a Scope We Could Ship

Arcanum Nights was designed around two-player multiplayer from the beginning. It was also a side project built by a team learning game development as we went, and the long production cycle was taking a toll on the team.

As the studio head, I decided that shipping a real release was important for keeping the team moving.

We first confirmed that switching between both viewpoints in single-player preserved the core experience of combining information. After releasing a demo and verifying that the game could be completed, we launched the single-player version in Steam Early Access on February 20, 2026.

Multiplayer is being developed on top of the systems established in single-player and will require its own round of user testing.

### Studio Head, Team Lead, and Lead Game Designer

D3F!B is a team of around ten: three planners, four developers, and three artists.

As studio head, I represent the team in external networking, showcases, and competitions. As team lead, I assign work, manage the schedule, review decisions across disciplines, and make the final calls.

As lead game designer, I created the initial concept and core mechanic, set the direction of the game, and adjusted its scope so that development and art could produce it.

- Original puzzle mechanic and game direction
- Systems, UI/UX, tutorial, and balance design
- Wireframes, data structures, and data tables
- Python map creation and validation tool
- Hands-on Unity/C# tutorial implementation
- Scheduling, work allocation, and final decisions
- PlayX4 2025 and 2026 participation and Steam release decision

### A Link to the Next World

The names Helian and Oenothera come from the scientific names of flowers. They connect the story of Arcanum Nights to a future game world built around a plant apocalypse.

### What I Learned

Inventing an original mechanic was not enough to make a game.

I had to explain it in a structure the team could implement, present it in a way a first-time player could understand, and adjust the scope so the team could finish and release it.

**To me, planning covers the entire path from imagining something new to putting it in the hands of someone who can actually enjoy it.**

🔗 [View Arcanum Nights on Steam](https://store.steampowered.com/app/3453760/Arcanum_Nights/)

[↑ Top](#top-en)

---

<a id="devlog-en"></a>
## Devlog

### 01. Core Experience

Core experience definition

- Single-to-two-player puzzle
- Sun / moon / constellation theme
- Cooperative interaction
- Stage progression
- Puzzle rules

Early concept → playable system structure

### 02. System Planning

Game-system design

- Core loop
- System rules
- Player states
- Stage flow
- Puzzle conditions
- Success / failure conditions
- Interaction rules

Happy path and edge states defined together

### 03. UI / UX

UI structure around player flow

- Main flow
- In-game UI
- Interaction feedback
- Tutorial UI
- Stage transitions
- Information priority

Information order for first-time players

### 04. Scenario / Tutorial

Player scenario and tutorial structure

- First-play scenario
- Tutorial sequence
- Interaction introduction
- Puzzle-rule introduction
- Stage progression
- Story-delivery timing

Learning through play rather than explanation

### 05. Data Structure

Implementation-facing data structure

- System data tables
- Stage data
- Story data
- State definitions
- Conditions / branches
- Edge cases
- Data relationships

Planning documents connected to implementation data

### 06. Edge Cases

Pre-implementation state review

- Missing values
- Duplicate input
- Simultaneous conditions
- Mid-state changes
- Invalid interactions
- Unexpected player actions

Exceptional states defined beyond the happy path

### 07. Unity / C#

Hands-on implementation for tighter planning / development alignment

- Unity project structure
- C# structure
- Tutorial implementation
- Data-flow checks
- Runtime validation

Implementation-cost awareness rather than replacing development

### 08. Python Map Tool

Map-production support tool

- Map-data input
- Data conversion
- Repetitive-task reduction
- Planning / development handoff

Simplified map-production workflow

### 09. Cross-functional Planning

Scope adjustment around production constraints

- Development feasibility
- Art-resource cost
- Schedule
- Production scope
- Priority
- Implementation complexity

Core intent preserved with lower production cost

### 10. PlayX4 User Test

On-site player observation

Observed

- Facial expressions
- Posture
- Keyboard / mouse input rhythm
- Focus
- Pacing drop-off
- Interaction hesitation

Adjusted

- Game tempo
- Animation speed
- Player flow
- Tutorial timing
- Interaction feedback

### 11. Demo Build Review

Demo-build data review

Issue

- Hidden stages in the UI
- Original story data still bundled

Action

- Demo-specific story-data split
- Removal of unnecessary original-content exposure

Review extended beyond visible UI into shipped build data

[↑ Case Study](#case-study-en)

</div>
