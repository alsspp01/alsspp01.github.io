---
title: "우리 정글 머함? 우정머"
title_en: "Woojeongmeo: Jungle Path Recommender"
type: page
description: "라인별 상성 승률로 게임 초반 정글 동선과 오브젝트 우선순위를 추천하는 Python 프로그램. 4인 팀 조장."
description_en: "A Python program that recommends early-game jungle paths and objective priorities from lane matchup win rates. Led a four-person team."
dated: true
period_start: "2021-09"
period_end: "2021-12"
summary: "승률 검색 사이트에서 크롤링한 상대적인 승률을 바탕으로 게임 초반 정글 동선을 추천해주는 프로그램.\n\nWith crawled data of the relative win rate from existing site,\nthis program recommends a jungler's early-game path."
---

<div class="lang-ko">

**기간** · 2021.09–2021.12 (대학 1학년)  
**역할** · 조장 · 알고리즘 설계 · 데이터 형식 · UI 연결  
**팀** · 4명  
**상태** · 수업 프로젝트 완료 · Python / PyQt

## Case Study

### Problem

챔피언 티어 정보는 많았지만, **우리 팀과 상대 팀의 라인별 상성**을 보고 정글러가 초반에 어디로 가야 하는지 알려주는 도구는 찾기 어려웠습니다.

### My Contribution

- **진행** · 아이디어 발표 → 다수결 주제 선정 → 역할 분담 순으로 회의를 운영하고, 매주 회의록을 Discord 스레드로 정리
- **알고리즘 설계** · 정글 · 미드 · 탑 · 바텀의 유불리 조합을 의사결정 트리로 그리고, 버프 방향 · 갱 위치 · 오브젝트 우선순위 변수로 구조화
- **데이터** · 승률 데이터 저장 형식(구분자)을 정하고 텍스트 파일을 Python으로 읽어오는 부분, 바텀 2인 조합 승률 계산 담당
- **UI 연결 · 디버깅** · 승률 데이터가 없을 때 경고창이 뜨지 않던 문제를, 경고 호출을 알고리즘 함수에서 버튼 핸들러로 옮겨 해결
- **발표 준비** · 시연 시나리오 4가지(모두 유리 · 일부 불리 · 모두 불리 · 데이터 없음)와 데이터 추가 흐름을 설계

![의사결정 트리 스케치](/image/LRP/decision-tree.jpg)

### Outcome

| 입력 | 출력 |
|---|---|
| 양 팀 10명의 챔피언 | 정버프 / 역버프 추천 |
| 라인별 상성 승률 | 갱 갈 라인 (승률이 같으면 미드 → 탑 → 바텀) |
| | 바위게 · 용 · 전령 우선순위 |

![출력 예시](/image/LRP/buffs.jpg)

### Constraints

- 데이터가 많아 **픽률 상위 15개 챔피언**만 임시로 지원하고, 없는 조합은 사용자가 직접 추가하도록 설계
- 유불리 기준(승률 60% 등)은 데이터 분석이 아닌 팀의 게임 지식으로 정한 규칙
- 승률 데이터는 기존 통계 사이트에서 가져와 정리

이때 느낀 데이터의 한계는 이후 Riot API를 직접 다룬 [Chemi.lol](/projects/chemi/)과 [League of Legends Data Analysis](/projects/lda/)로 이어졌습니다.

<details>
<summary>알고리즘 설계 메모</summary>

![알고리즘 스케치](/image/LRP/algorithm-sketch.jpg)

</details>

[![GitHub](/image/profile/github-mark.png)](https://github.com/alsspp01/LRP.git)

</div>

<div class="lang-en" style="display:none">

**Period** · Sep–Dec 2021 (first year of university)  
**Role** · Team Lead · Algorithm Design · Data Format · UI Integration  
**Team** · 4 members  
**Status** · Course project completed · Python / PyQt

## Case Study

### Problem

Champion tier lists were everywhere, but there was little to tell a jungler where to go early on based on **lane-by-lane matchups between the two teams**.

### My Contribution

- **Running the team** · Ran meetings from idea pitches to a vote to role assignment, and kept weekly minutes in Discord threads
- **Algorithm design** · Drew the favorable/unfavorable combinations of jungle, mid, top, and bot as a decision tree, then structured them into variables for buff side, gank target, and objective priority
- **Data** · Defined the storage format for win-rate data, loaded it from text files in Python, and calculated bot-lane duo win rates
- **UI and debugging** · Fixed a warning dialog that never appeared when data was missing by moving the call from the algorithm function to the button handler
- **Demo** · Planned four demo scenarios (all favorable, some unfavorable, all unfavorable, missing data) plus the flow for adding data

![Decision-tree sketch](/image/LRP/decision-tree.jpg)

### Outcome

| Input | Output |
|---|---|
| Ten champions across both teams | Which buff to start |
| Lane matchup win rates | Which lane to gank (ties: mid → top → bot) |
| | Priority among scuttle crabs, dragon, and herald |

![Sample output](/image/LRP/buffs.jpg)

### Constraints

- To keep the data manageable, only the **top 15 champions by pick rate** were supported, and users could add missing matchups themselves
- Thresholds such as a 60% win rate came from the team's game knowledge, not data analysis
- Win-rate data was collected from existing stats sites

The data limits I hit here led to working directly with the Riot API in [Chemi.lol](/projects/chemi/) and [League of Legends Data Analysis](/projects/lda/).

<details>
<summary>Algorithm design notes</summary>

![Algorithm sketch](/image/LRP/algorithm-sketch.jpg)

</details>

[![GitHub](/image/profile/github-mark.png)](https://github.com/alsspp01/LRP.git)

</div>
