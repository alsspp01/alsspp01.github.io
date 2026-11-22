---
title: "League of Legends Data Analysis"
title_en: "League of Legends Data Analysis"
type: page
description: "LoL 연속 플레이와 인지피로 연구를 위한 실험 · 수집 · 분석 · 리포트 도구"
description_en: "Experiment, collection, analysis, and reporting tools for a study on consecutive play and cognitive fatigue"
dated: true
period_start: "2024-03"
period_end: "2025-06"
tags: ["DKU"]
---

<div class="lang-ko">

**기간** · 2024.03–2025.06  
**소속** · 단국대학교 스포츠심리학 연구실  
**역할** · 연구 개발 아르바이트 · 개발 전반  
**상태** · 연구실 내부 실험 · 분석 도구

[LDA](https://github.com/alsspp01/LDA) · [TestResultAnalysis](https://github.com/alsspp01/TestResultAnalysis)

## 요약

- **문제** · 연구진이 정한 실험 · 분석 조건을 실제로 돌아가는 도구로 옮겨야 함
- **한 일** · Stroop Test 앱, 약 6개월간 Riot API 수집, 연속 플레이 표본 추출, 참가자별 리포트 도구
- **결과** · 실험 참가자 약 40명이 사용하고 개인 리포트를 받음

연구 기획과 가설 설정에는 참여하지 않았습니다. 조건을 받아 구현 방식과 데이터 흐름을 정하는 일을 맡았습니다.

## 과정

### 실험 도구

- PySide6 · Pygame으로 Stroop Test 제작: 연습 세션 + 본 실험 3블록, 자극 순서 무작위
- 블록별 정답 여부와 반응시간을 같은 형식으로 저장해, 분석 코드가 바로 읽도록 연결

![집단별 Stroop Test 결과](/image/LDA/stroop_test_by_group.png)

### 6개월간의 플레이 데이터

- Riot API 호출을 모듈화하고 요청 간격을 제어하며 티어별 무작위 수집
- 경기 시작 · 종료 시각으로 휴식 5분 · 10분 이내를 연속 플레이로 분류
- 연속 경기 수(6 · 8경기 등)별로 승률, 분당 CS, 분당 골드 변화 비교

정의를 조금만 바꿔도 표본이 수십 건에서 수만 건까지 달라졌습니다.

### 자연어 조건을 코드로

"5분 간격 5게임에 대한 지표"는 두 가지로 읽혔습니다.

- 5경기 **이상** 이어서 한 사람의 첫 5경기
- **정확히** 5경기를 하고 쉰 사람

임의로 고르지 않고 두 해석을 문서로 정리해 연구진과 기준을 맞췄습니다.

### 내 코드의 버그

Riot API의 최근 경기 목록은 최신순인데, 초기 처리 코드가 다시 정렬하지 않아 변화량을 거꾸로 계산하고 있었습니다.  
분석 결과를 검토하다 발견해 타임스탬프 기준으로 고쳤습니다.

### 수집할 수 있어도 하지 않은 것

공개 API 밖에서 게임 내 위치 데이터를 모으는 방법은 기술적으로 가능했지만, Riot 운영정책에 문제가 될 수 있다고 판단했습니다.  
연구진에게 보고하고 연구에서 제외했습니다.

### 참가자용 리포트

[TestResultAnalysis](https://github.com/alsspp01/TestResultAnalysis)로 Stroop 결과와 플레이 데이터를 참가자 단위로 묶었습니다.  
반응시간 · 정확도 변화, 집단 내 위치, 시간대 · 휴식 시간별 승률을 시각화해 개인 리포트로 만들었습니다.

## 관련 논문

같은 주제를 다룬 연구실의 질적 연구입니다. 저자로 참여하지 않았고, 제가 만든 도구가 이 논문의 방법이나 결과에 쓰이지 않았습니다.

[「e스포츠 참여자들의 인지피로 경험에 관한 현상학적 연구」](https://doi.org/10.21097/ksw.2025.2.20.1.303) · 한국웰니스학회지 · 2025

<details>
<summary>Devlog · 구성 요소</summary>

| 구성 | 내용 |
|---|---|
| Stroop Test | PySide6 UI, Pygame 진행, Pandas · NumPy 분석 |
| 수집 POC | 기존 전적 서비스의 요청 · 응답 · 페이지네이션 구조 분석 |
| Riot API 모듈 | 티어별 플레이어, PUUID, Match ID, 상세 정보, 타임라인 |
| 연속 플레이 표본 | 5 · 10분 간격, 6 · 8경기 조건별 CSV |
| 조건 정의 | 경기 목록 → 시간순 정렬 → 간격 계산 → 시퀀스 판별 |
| 위치 데이터 POC | JavaScript 로거, Python 위치 기록기 (연구 미적용) |
| 리포트 | 참가자 단위 통계, 시각화, 리포트 출력 |

</details>

</div>

<div class="lang-en" style="display:none">

**Period** · Mar 2024–Jun 2025  
**Organization** · Dankook University Sport Psychology Lab  
**Role** · Research developer (part-time) · End-to-end development  
**Status** · Internal lab tools for experiments and analysis

[LDA](https://github.com/alsspp01/LDA) · [TestResultAnalysis](https://github.com/alsspp01/TestResultAnalysis)

## Summary

- **Problem** · The researchers' experiment and analysis conditions had to become working tools
- **What I did** · A Stroop Test app, about six months of Riot API collection, consecutive-play sampling, and per-participant reports
- **Result** · Used by around 40 study participants, each of whom received a report

I did not design the research or its hypotheses; I turned the given conditions into implementation and data flow.

## Process

### Experiment tool

- Built a Stroop Test with PySide6 and Pygame: a practice session plus three blocks, randomized stimuli
- Stored accuracy and reaction time per block in one format so the analysis code could read it directly

![Stroop Test results by group](/image/LDA/stroop_test_by_group.png)

### Six months of gameplay data

- Modularized Riot API calls with request pacing; sampled players across tiers
- Classified consecutive play by breaks under five or ten minutes, using match start and end times
- Compared win rate, CS per minute, and gold per minute by session length (six, eight games, etc.)

Small changes in the definition moved the sample from dozens to tens of thousands.

### From a sentence to a rule

"Metrics for five games at five-minute intervals" could mean two things:

- the first five games of players who played **at least** five in a row
- players who played **exactly** five and then stopped

I wrote down both readings and settled the definition with the researchers instead of picking one.

### A bug in my own code

Riot's recent-match list is newest first, and my early code did not re-sort it, so changes were computed backward.  
I caught it while reviewing results and fixed it by sorting on timestamps.

### Data I chose not to collect

Collecting in-game position data outside the public API was technically possible but could conflict with Riot's policies.  
I reported it to the researchers and it was left out.

### Participant reports

[TestResultAnalysis](https://github.com/alsspp01/TestResultAnalysis) combined Stroop results and gameplay data per participant,  
visualizing reaction time and accuracy, position within the group, and win rate by time of day and break length.

## Related paper

A separate qualitative study from the lab on the same topic. I was not an author, and my tools were not part of its methods or results.

["A Phenomenological Study on Cognitive Fatigue Experiences of eSports Participants"](https://doi.org/10.21097/ksw.2025.2.20.1.303) · Journal of Korea Society for Wellness · 2025

<details>
<summary>Devlog · components</summary>

| Component | Content |
|---|---|
| Stroop Test | PySide6 UI, Pygame flow, Pandas · NumPy analysis |
| Collection POC | Request, response, and pagination of existing match-history services |
| Riot API module | Players by tier, PUUID, match IDs, details, timelines |
| Consecutive-play samples | CSVs by 5 · 10-minute gap and 6 · 8-game conditions |
| Condition rules | Match list → sort by time → gaps → sequence detection |
| Position-data POC | JavaScript logger, Python position recorder (not used in the study) |
| Reports | Per-participant statistics, charts, report output |

</details>

</div>

---

[← Projects](/projects/) · [Portfolio](/portfolio/)
