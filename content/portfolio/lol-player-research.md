---
title: "LoL 플레이어 데이터 연구"
title_en: "LoL Player Research"
type: page
description: "인지피로 연구를 위한 LoL 데이터 수집, 실험 도구, 분석 및 리포트 파이프라인."
description_en: "A research-engineering project spanning LoL data collection, experiment tooling, analysis, and reporting."
---

<div class="lang-ko">

**Period** · 2024.03 – 2025.06  
**Context** · Dankook University Sports Psychology Laboratory  
**Role** · POC / Development Design / Data Engineering

[🔗 LDA Repository](https://github.com/alsspp01/LDA)

## From Research Question to Data

League of Legends 플레이에 따른 인지 피로를 연구하려면  
누가, 어느 정도 간격으로, 몇 경기를 연속 플레이했는지를 연구 조건에 맞게 찾아야 했습니다.

Riot API 호출을 모듈화하고 티어별 플레이어, match ID, 경기 데이터를 자동 수집했습니다.

## Research Conditions as Code

경기 시작/종료 timestamp를 이용해 연속 플레이 여부를 판단했습니다.

실험 코드에서는 다음 조건을 탐색했습니다.

- 경기 간격 **5분 / 10분**
- 연속 **6게임 / 8게임**

조건을 만족하는 sequence는 별도 dataset으로 저장했습니다.

**“연속 플레이”라는 연구 개념을 프로그램이 판단할 수 있는 규칙으로 바꾸는 작업**이었습니다.

## Looking Beyond the API

Riot API 밖에서 얻을 수 있는 정보도 탐색했습니다.

Repository의 `DynamicAnalysis`에는:

- JavaScript console logger
- Python position logger
- position dataset
- analysis notebook

이 남아 있습니다.

모든 방식이 최종 연구에 사용된 것은 아니지만, 기술적으로 가능한 선택지를 확인하는 POC였습니다.

## Stroop Test

시간에 따른 인지 피로 변화를 측정하기 위한 Stroop Test도 구현했습니다.

- UI · PyQt
- Test flow · Pygame
- Data · Pandas
- Analysis · NumPy
- Visualization

## From Numbers to Participant Reports

연구 후반에는 별도의 **TestResultAnalysis** 도구를 만들어  
실험 결과와 플레이 데이터를 통계 처리·시각화하고 피험자에게 보여줄 수 있는 리포트로 변환했습니다.

해당 저장소는 private research utility라 공개 링크는 제공하지 않습니다.

## What I Learned

이 프로젝트에서 가장 재미있었던 것은 API 자체보다  
**연구자의 질문을 데이터 구조와 프로그램의 조건으로 바꾸는 과정**이었습니다.

> **“데이터를 수집했습니다.”**

에서 끝나는 것이 아니라,

> **“이 데이터가 다음 분석과 설명에 어떻게 다시 사용될 수 있는가?”**

까지 생각해야 했습니다.

[🔗 LDA Repository](https://github.com/alsspp01/LDA)  
[🔗 Portfolio](/portfolio/)

</div>

<div class="lang-en" style="display:none">

**Period** · Mar 2024 – Jun 2025  
**Context** · Dankook University Sports Psychology Laboratory  
**Role** · POC / Development Design / Data Engineering

[🔗 LDA Repository](https://github.com/alsspp01/LDA)

## From Research Question to Data

Researching cognitive fatigue around League of Legends required more than pulling a few recent matches.

We needed to identify who had played, how close the games were to one another, and whether a sequence matched the research definition of consecutive play.

I modularized Riot API calls and automated collection of players, match IDs, and match data across tiers.

## Research Conditions as Code

Match start/end timestamps were used to determine continuity.

The experimental code explored:

- **5-minute / 10-minute** gaps
- **6-game / 8-game** consecutive sequences

Matching sequences were stored as separate datasets.

It was the work of translating **“consecutive play” from a research concept into a rule software could apply consistently.**

## Looking Beyond the API

I also explored data that was harder to obtain through Riot's public API.

The repository's `DynamicAnalysis` directory contains:

- a JavaScript console logger
- a Python position logger
- position datasets
- analysis notebooks

Not every path became part of the final study, but the experiments clarified what was technically possible.

## Stroop Test

I also implemented a Stroop Test for measuring changes in cognitive fatigue over time.

- UI · PyQt
- Test flow · Pygame
- Data · Pandas
- Analysis · NumPy
- Visualization

## From Numbers to Participant Reports

Later, I built a separate **TestResultAnalysis** utility that processed and visualized experiment outputs and play data into participant-facing reports.

The repository is private research tooling, so it is intentionally not linked publicly.

## What I Learned

The most interesting part was not learning another API.

It was **turning a researcher's question into data structures, collection rules, and reusable tooling.**

The process could not stop at:

> **“The data has been collected.”**

It had to continue to:

> **“How can this data be reused in the next analysis and explained to another person?”**

[🔗 LDA Repository](https://github.com/alsspp01/LDA)  
[🔗 Portfolio](/portfolio/)

</div>
