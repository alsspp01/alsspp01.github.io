---
title: "📊 League of Legends Data Analysis"
title_en: "📊 League of Legends Data Analysis"
type: page
description: "단국대학교 스포츠심리학 연구실 인지피로 연구를 위한 데이터 수집 및 실험 도구."
description_en: "Data collection and experimental tooling for cognitive-fatigue research."
dated: true
period_start: "2024-03"
period_end: "2025-06"
tags: ["DKU"]
---

<div class="lang-ko">

**Dankook University Sports Psychology Laboratory**

[🔗 Portfolio Case Study](/portfolio/lol-player-research/) · [🔗 GitHub](https://github.com/alsspp01/LDA)

## 1. Stroop Test
- UI · PyQt
- Test · Pygame
- Data · Pandas
- Analysis · NumPy
- Visualization

![Stroop test result](/image/LDA/stroop_test_by_group.png)

## 2. Match-history Exploration
기존 전적 사이트의 request/JSON 구조와 pagination을 분석해 연속 플레이 데이터 수집 가능성을 확인했습니다.

## 3. Riot API Pipeline
티어별 플레이어, match ID, 경기 정보를 자동 수집하고 timestamp로 연속 플레이 sequence를 판별했습니다.

실험 코드에는:
- 5분 / 10분 간격
- 6게임 / 8게임 연속 sequence

조건이 포함되어 있습니다.

## 4. Dynamic Data Exploration
`DynamicAnalysis`에 JavaScript logger, Python position logger, position dataset, notebook이 남아 있습니다.

## 5. Participant Report Utility
연구 후반에는 별도의 private 도구로 실험 결과를 통계 처리·시각화해 피험자에게 보여줄 리포트를 만들었습니다.

</div>

<div class="lang-en" style="display:none">

**Dankook University Sports Psychology Laboratory**

[🔗 Portfolio Case Study](/portfolio/lol-player-research/) · [🔗 GitHub](https://github.com/alsspp01/LDA)

## 1. Stroop Test
- UI · PyQt
- Test · Pygame
- Data · Pandas
- Analysis · NumPy
- Visualization

![Stroop test result](/image/LDA/stroop_test_by_group.png)

## 2. Match-history Exploration
I inspected request/JSON structures and pagination behavior from existing match-history services.

## 3. Riot API Pipeline
I automated collection of players, match IDs, and match data across tiers, then used timestamps to identify consecutive-play sequences.

Experimental conditions included:
- 5 / 10 minute gaps
- 6 / 8 game sequences

## 4. Dynamic Data Exploration
`DynamicAnalysis` contains JavaScript/Python loggers, position datasets, and analysis notebooks.

## 5. Participant Report Utility
Later, I built a separate private utility that processed and visualized experiment outputs into participant-facing reports.

</div>
