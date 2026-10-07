---
title: "📊 League of Legends Data Analysis"
type: page
description: "LoL 연속 플레이와 인지피로 연구를 위한 실험 도구, 데이터 수집 · 분석 및 개인 리포트 개발."
dated: true
period_start: "2024-03"
period_end: "2025-06"
tags: ["DKU"]
---

<div class="lang-ko">

<a id="top"></a>

**기간** · 2024.03–2025.06  
**소속** · 단국대학교 스포츠심리학 연구실  
**역할** · 연구 개발 아르바이트 · 개발 전반  
**주요 작업** · 인지과제 프로그램 · Riot API 수집 · 조건별 데이터 분석 · 참가자 리포트

[🔗 LDA GitHub](https://github.com/alsspp01/LDA)  
[🔗 TestResultAnalysis GitHub](https://github.com/alsspp01/TestResultAnalysis)

### 바로가기
[**Case Study ↓**](#case-study) · [**Devlog ↓**](#devlog)

---

<a id="case-study"></a>
## Case Study

### 정해진 연구 조건을 실행 가능한 시스템으로 바꾸기

연구진은 League of Legends의 연속 플레이와 인지피로의 관계를 살펴보기 위한 연구를 기획했습니다. 저는 연구 기획이나 가설 수립에는 참여하지 않았고, 연구실의 개발 아르바이트로 합류해 정해진 실험 조건을 실제로 실행할 수 있는 시스템으로 만드는 일을 맡았습니다.

연구자가 필요한 실험과 분석 조건을 전달하면, 저는 구현 방식과 데이터 흐름을 정하고 프로그램으로 옮겼습니다. 인지기능 변화를 측정하는 Stroop Test부터 약 6개월간의 플레이 데이터 수집, 연속 경기 조건 분류, 통계 처리와 참가자별 결과 리포트까지 개발 전반을 담당했습니다.

### 약 40명이 사용한 인지과제 프로그램

인지피로에 따른 반응속도 변화를 확인하기 위해 PySide6와 Pygame으로 Stroop Test 프로그램을 제작했습니다. 참가자 ID를 입력하면 연습 세션과 세 개의 본 실험 블록을 진행하고, 자극의 제시 순서를 무작위화해 정답 여부와 반응시간을 기록하도록 만들었습니다.

실험 전 · 중 · 후 결과를 비교할 수 있도록 블록별 데이터를 분리해 저장하고 CSV 분석으로 연결했습니다. 약 40명의 참가자가 이 프로그램으로 인지피로를 측정하고 개인 결과 리포트를 받았습니다.

연구실에서 반복 실험에 사용하는 도구였기 때문에 화면이 동작하는 것만으로는 부족했습니다. 참가자별 데이터가 같은 형식으로 저장되고, 이후 분석 코드가 별도의 수작업 없이 읽을 수 있도록 실험부터 결과 처리까지 연결했습니다.

### 6개월간의 플레이 데이터를 연구 조건으로 분류하기

실험 참가자만으로는 장기간 반복 플레이의 경향을 충분히 보기 어려웠습니다. 그래서 Riot API 호출을 모듈화하고, 약 6개월 동안 전체 LoL 이용자를 대상으로 티어별 플레이 데이터를 무작위로 수집했습니다.

플레이어, PUUID, match ID, 경기 상세 정보와 타임라인을 단계별로 수집했으며 API 호출 제한에 맞춰 요청 간격을 제어했습니다. 수집한 경기의 시작 · 종료 시각을 바탕으로 경기 사이 휴식이 5분 또는 10분 이내인 경우를 연속 플레이로 분류했습니다. 6경기 · 8경기를 포함한 여러 연속 경기 수 조건을 나누어 승률, 분당 CS와 분당 골드 변화 등을 비교했습니다.

조건에 따라 확보되는 시퀀스는 수십 건부터 수만 건까지 차이가 났습니다. 같은 데이터라도 휴식 시간과 연속 경기 수를 어떻게 정의하느냐에 따라 표본의 크기와 의미가 크게 달라졌습니다.

TFT에서도 같은 접근이 가능한지 수집 코드를 만들어 검토했지만, 최종 연구에서는 제외했습니다.

### 자연어 조건에서 생긴 두 가지 해석

연구자가 요청한 “5분 간격 5게임에 대한 지표”는 두 가지로 해석할 수 있었습니다.

- 휴식이 5분 미만인 상태로 **5경기 이상 플레이한 사람의 첫 5경기**
- 휴식이 5분 미만인 상태로 **정확히 5경기를 마치고 쉰 사람**

두 집단은 같은 의미가 아닙니다. 첫 번째 조건에는 이후에도 계속 플레이한 사람이 포함되고, 두 번째 조건은 5경기에서 세션을 끝낸 사람만 포함합니다. 저는 모호한 요청을 임의로 구현하지 않고 가능한 해석을 문서로 정리해 연구진과 기준을 다시 맞췄습니다.

연구 개념을 코드로 옮길 때는 조건문을 작성하는 것보다, 그 조건이 실제로 어떤 사람을 표본에 포함하는지 확인하는 일이 더 중요했습니다.

### 시간의 방향이 뒤집힌 오류

분석 과정에서는 Riot API의 최근 경기 목록이 최신순으로 반환된다는 점 때문에, 연속 경기 데이터가 실제 플레이 순서와 반대로 정렬된 오류를 발견했습니다. 게임을 시작한 순서가 아니라 마지막 게임부터 거꾸로 변화량을 계산하고 있었기 때문에, 그대로 두면 경기 수가 늘어날수록 지표가 어떻게 변하는지 반대로 해석할 수 있었습니다.

타임스탬프를 기준으로 데이터를 다시 시간순으로 정렬하고, 분석 과정 전반에서 경기 순서를 일관되게 사용하도록 수정했습니다. 데이터가 모두 존재하는지뿐 아니라 시간축이 연구 질문과 같은 방향인지 검증해야 한다는 것을 배운 사례였습니다.

### 수집할 수 있어도 수집하지 않은 데이터

공개 Riot API 밖에서 게임 내 위치와 플레이 패턴을 수집할 수 있는지도 기술적으로 검토했습니다. JavaScript 로거와 Python 위치 기록기, 분석용 노트북으로 가능성을 확인했지만 Riot 운영정책에 문제가 될 수 있다고 판단했습니다.

이 위험을 연구진에게 보고했고 해당 데이터는 연구 대상에서 제외했습니다. 기술적으로 가능하다는 이유만으로 연구에 사용하지 않고, 수집 방법의 적절성과 운영정책을 먼저 판단했습니다.

### 참가자가 이해할 수 있는 결과로 바꾸기

후반에는 별도의 [TestResultAnalysis](https://github.com/alsspp01/TestResultAnalysis)를 만들어 Stroop Test와 플레이 데이터를 참가자 단위로 취합했습니다.

블록별 반응시간과 정확도, 시간에 따른 변화 기울기, 전체 참가자 안에서의 순위와 z-score를 계산했습니다. 여기에 플레이 시간대 · 휴식 시간별 승률과 빈도, 연속 경기 중 승률 · CS · 골드 변화를 시각화해 개인 리포트로 만들었습니다.

연구자가 분석할 수 있는 숫자를 만드는 데서 끝내지 않고, 참가자가 자신의 결과를 이해할 수 있는 형태까지 연결한 작업이었습니다.

### 관련 논문

이 개발 업무는 단국대학교 스포츠심리학 연구진의 e스포츠 인지피로 연구 맥락에서 진행됐습니다. 아래 논문은 같은 연구 주제를 다룬 연구실의 질적 연구입니다. 저는 논문의 저자나 연구 기획 · 면담 · 질적 분석에 참여하지 않았으며, 제가 개발한 도구가 해당 논문의 방법이나 결과에 사용된 것은 아닙니다.

🔗 [「e스포츠 참여자들의 인지피로 경험에 관한 현상학적 연구」](https://doi.org/10.21097/ksw.2025.2.20.1.303)  
한국웰니스학회지 · 2025 · 20(1) · 303–313

[↑ Top](#top)

---

<a id="devlog"></a>
## Devlog

### 01. Stroop Test

인지피로 측정용 실험 도구 제작

- UI · PySide6
- Test flow · Pygame
- Data · Pandas
- Analysis · NumPy
- Visualization

피험자별 결과 저장 및 분석 연결

![Stroop test result](/image/LDA/stroop_test_by_group.png)

### 02. Match-history Exploration

기존 전적 분석 서비스의 데이터 수집 구조 분석

- Request 구조 확인
- JSON response 분석
- Pagination 방식 확인
- 연속 플레이 데이터 수집 가능성 검토

초기 데이터 수집 POC

### 03. Riot API Module

`API.py` 기준 Riot API 호출 모듈화

- Tier별 player 수집
- PUUID 조회
- Riot ID / TagLine 처리
- Match ID 수집
- Match detail 조회
- Timeline 조회
- Request interval 처리

연구용 반복 수집 구조 구성

### 04. Consecutive-play Sampling

Timestamp 기반 연속 플레이 판별 로직 구성

Sampling 조건

- 5-minute gap
- 10-minute gap
- 6-game sequence
- 8-game sequence

Match timestamp 기준 sequence 생성

조건 충족 sample 별도 CSV 저장

### 05. Research Condition Encoding

자연어 연구 조건을 프로그램 규칙으로 변환

```text
Player
  ↓
Match History
  ↓
Sort by Timestamp
  ↓
Calculate Gap
  ↓
Sequence Detection
  ↓
Research Sample
```

`연속 플레이` 개념의 자동 판별 구조 정리

### 06. Dynamic Analysis

Public Riot API 외 데이터 수집 가능성 검토

`DynamicAnalysis` 구성

- JavaScript console logger
- Python position logger
- Position CSV
- Champion reference
- Position reference
- Analysis notebook

게임 내부 위치 데이터 수집 POC

최종 연구 적용 여부와 별개로 기술 가능 범위 확인

### 07. Data Analysis

수집 결과 분석 및 시각화

- Pandas
- NumPy
- Statistical processing
- Group comparison
- Visualization

연구 조건별 dataset 비교 구조 정리

### 08. Participant Report

연구 후반 별도 분석 도구 제작

`TestResultAnalysis`

- Stroop Test 결과 처리
- Gameplay result 통합
- Participant 단위 통계
- Visualization
- Report output

공개 저장소로 관리 · [🔗 TestResultAnalysis GitHub](https://github.com/alsspp01/TestResultAnalysis)

### 09. Final Structure

연구 전체 흐름

```text
Research Question
       ↓
Experiment Tool
       ↓
Game Data Collection
       ↓
Sample Extraction
       ↓
Analysis
       ↓
Participant Report
```

단발성 script보다 반복 가능한 research pipeline 중심 정리

[🔗 GitHub](https://github.com/alsspp01/LDA)
[↑ Case Study](#case-study)

</div>

<div class="lang-en" style="display:none">

<a id="top-en"></a>

**Period** · Mar 2024–Jun 2025  
**Organization** · Dankook University Sports Psychology Laboratory  
**Role** · Research Developer (Part-time) · End-to-End Development  
**Scope** · Cognitive Task Application · Riot API Collection · Conditional Data Analysis · Participant Reports

[🔗 LDA GitHub](https://github.com/alsspp01/LDA)  
[🔗 TestResultAnalysis GitHub](https://github.com/alsspp01/TestResultAnalysis)

### Jump to
[**Case Study ↓**](#case-study-en) · [**Devlog ↓**](#devlog-en)

---

<a id="case-study-en"></a>
## Case Study

### Turning defined research requirements into a working system

The research team planned a study on consecutive League of Legends play and cognitive fatigue. I did not participate in defining the research question, hypothesis, or methodology. I joined the laboratory as a part-time developer to turn the researchers' requirements into a system they could use.

When the researchers provided the experimental and analytical requirements, I decided how to implement them and structure the data flow. I was responsible for the development work end to end: the Stroop Test application, six months of gameplay-data collection, consecutive-play classification, statistical processing, and participant-level reporting.

### A cognitive task used by around 40 participants

I built a Stroop Test application with PySide6 and Pygame to measure changes in response time associated with cognitive fatigue. Participants entered an ID and completed a practice session followed by three experimental blocks. Stimuli were presented in randomized order, and the application recorded both response accuracy and reaction time.

The data from each block was stored separately so that results from before, during, and after the task could be compared and passed directly into the CSV-based analysis workflow. Around 40 participants completed the cognitive-fatigue assessment and received an individual report.

Because the application was used for repeated research sessions, a working interface was not enough. Participant data had to be stored consistently and remain readable by the downstream analysis without manual reformatting.

### Classifying six months of gameplay by research criteria

The participant study alone could not provide enough observations of extended play. I modularized Riot API requests and collected randomly sampled, tier-stratified League of Legends gameplay data for roughly six months.

The pipeline retrieved players, PUUIDs, match IDs, match details, and timelines in stages while respecting the API request interval. Match start and end timestamps were then used to classify consecutive play under five- and ten-minute break thresholds. I compared several sequence lengths, including six and eight matches, across win rate, CS per minute, and gold per minute.

Depending on the threshold and sequence length, the available samples ranged from dozens to tens of thousands. The same source data could represent very different populations depending on how a break and a consecutive session were defined.

I also built collection code to test whether the approach could be extended to Teamfight Tactics, but TFT was excluded from the final research scope.

### Two meanings hidden in one requirement

A request for “metrics for five games with a five-minute interval” had two possible meanings:

- the first five games of players who continued for **at least five consecutive matches**
- players who played **exactly five consecutive matches and then stopped for a break**

These were not equivalent samples. The first included people who continued playing, while the second isolated sessions that ended after the fifth match. Instead of choosing an interpretation silently, I documented both and aligned the definition with the researchers before revising the analysis.

Translating a research concept into code required more than implementing a condition. I had to verify which people that condition actually included in the sample.

### An error that reversed the direction of time

During analysis, I found that Riot's recent-match endpoint returned matches newest first. The consecutive-play records were therefore ordered in the opposite direction from the actual sessions. If left unchanged, the analysis could have interpreted changes across a session backwards.

I reordered the data chronologically by timestamp and made the rest of the analysis use the same time direction. The lesson was that complete data is not necessarily correctly structured data, especially when the research question depends on change over time.

### Data I chose not to collect

I explored whether in-game position and play-pattern data could be collected outside Riot's public API. A JavaScript logger, Python position recorder, and analysis notebook showed that it was technically possible, but I identified a potential conflict with Riot's operating policies.

I reported the concern to the researchers, and the approach was removed from the study. Technical feasibility did not override whether the collection method was appropriate.

### Turning analysis into participant-facing reports

Later in the project, I built [TestResultAnalysis](https://github.com/alsspp01/TestResultAnalysis) to combine Stroop Test and gameplay data at the participant level.

The tool calculated reaction time and accuracy by block, trends over time, ranks within the participant group, and z-scores. It also visualized win rate and play frequency by time of day and break length, along with changes in win rate, CS, and gold during consecutive matches.

The work extended beyond producing statistics for researchers. It turned the results into reports that participants could understand and take away.

### Related paper

This development work took place within Dankook University's broader research on cognitive fatigue in esports. The paper below is a separate qualitative study from the laboratory on the same subject. I was not an author and did not participate in its research planning, interviews, or qualitative analysis. The tools I developed were not part of the paper's methods or results.

🔗 [“Phenomenological Study on Cognitive Fatigue Experiences of eSports Participants”](https://doi.org/10.21097/ksw.2025.2.20.1.303)  
Journal of Korea Society for Wellness · 2025 · 20(1) · 303–313

[↑ Top](#top-en)

---

<a id="devlog-en"></a>
## Devlog

### 01. Stroop Test

Experimental tool for cognitive-fatigue measurement

- UI · PySide6
- Test flow · Pygame
- Data · Pandas
- Analysis · NumPy
- Visualization

Per-participant result storage and analysis flow

![Stroop test result](/image/LDA/stroop_test_by_group.png)

### 02. Match-history Exploration

Data-collection flow of existing match-history services

- Request structure
- JSON response
- Pagination behavior
- Feasibility of consecutive-play sampling

Initial collection POC

### 03. Riot API Module

Reusable Riot API wrapper in `API.py`

- Tier-based player collection
- PUUID lookup
- Riot ID / TagLine handling
- Match ID collection
- Match detail retrieval
- Timeline retrieval
- Request-interval handling

Reusable collection flow for research runs

### 04. Consecutive-play Sampling

Timestamp-based sequence detection

Sampling conditions

- 5-minute gap
- 10-minute gap
- 6-game sequence
- 8-game sequence

Sequence generation from match timestamps

Matching samples exported as CSV

### 05. Research Condition Encoding

Research criteria translated into deterministic program rules

```text
Player
  ↓
Match History
  ↓
Sort by Timestamp
  ↓
Calculate Gap
  ↓
Sequence Detection
  ↓
Research Sample
```

Automated definition of `consecutive play`

### 06. Dynamic Analysis

Exploration beyond the public Riot API

`DynamicAnalysis`

- JavaScript console logger
- Python position logger
- Position CSV
- Champion reference
- Position reference
- Analysis notebook

In-game position-data POC

Technical feasibility check independent of final research adoption

### 07. Data Analysis

Collected-data analysis and visualization

- Pandas
- NumPy
- Statistical processing
- Group comparison
- Visualization

Comparison structure across research-condition datasets

### 08. Participant Report

Separate reporting utility for the later research stage

`TestResultAnalysis`

- Stroop Test processing
- Gameplay-result integration
- Participant-level statistics
- Visualization
- Report output

Public repository · [🔗 TestResultAnalysis GitHub](https://github.com/alsspp01/TestResultAnalysis)

### 09. Final Structure

Research pipeline

```text
Research Question
       ↓
Experiment Tool
       ↓
Game Data Collection
       ↓
Sample Extraction
       ↓
Analysis
       ↓
Participant Report
```

Repeatable research pipeline over one-off scripts

[🔗 GitHub](https://github.com/alsspp01/LDA)
[↑ Case Study](#case-study-en)

</div>
