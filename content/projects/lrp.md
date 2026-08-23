---
title: "우리 정글 머함? 우정머"
type: page
description: "League of Legends Recommendation Program"
dated: true
period_start: "2021-09"
period_end: "2021-12"
summary: "승률 검색 사이트에서 크롤링한 상대적인 승률을 바탕으로 게임 초반 정글 동선을 추천해주는 프로그램.\n\nWith crawled data of the relative win rate from existing site,\nthis program recommends a jungler's early-game path."
---

## Overview
당시 League of Legends(LOL)의 챔피언 티어는 정리가 되어 있었으나, 챔피언 상성을 고려하여 정글 동선을 직접적으로 짜주는 프로그램은 없었다.  
이에 따라 우리는 챔피언 상성을 바탕으로 분기문을 만들어 정글 동선을 추천하는 프로그램을 만들고자 하였다.  
또한 승률에 기반한 추천 챔프를 콤보박스로 알려주는 기능 또한 넣어 밴픽 편의성을 증대하였다.

## Simulation

1. 아무것도 출력되기 전
   
  ![협곡사진](/image/LRP/rift.jpg)


2. 전 라인 상성이 좋은 경우 사진
   
  ![출력물](/image/LRP/buffs.jpg)

  [ 출력 구문 ]
  ```python
  역버프를 추천드립니다. # 정글 챔프 승률 기반 - 승률이 높으면 정버프 / 낮으면 역버프
  미드갱을 추천드립니다. # 승률이 똑같이 모두 높을 경우 라인 우선순위 미드, 탑, 바텀 순서
  모든 오브젝트 싸움에서 이길 확률이 높습니다. 최대한 챙기세요. # 모든 경우의 수를 고려하여 하나하나 책정
  ```

## 잘 된 점
1. 당시에 승률 기반 밴픽 추천 기능이나, 정글 동선 추천 기능이 없었음.
2. 상대 승률을 정글 동선 추천으로 연결하려 했다는 점이 당시 프로젝트의 가장 독특한 부분이었습니다.
3. 이를 공식화하려 한 시도는 굉장히 참신했고, 정글링에 대한 이해도가 없는 플레이어들에게 도움이 될 수 있을만한 프로그램이었을 것이라 생각함.


## 아쉬운 점

1. 1학년 때 만든 간단한 프로그램이라 크롤링에 의존한 데이터 수집을 할 수 밖에 없던 점
2. 데이터에 기반한 기준이 아닌 임의의 기준을 사용한 점
3. 당시 영어를 좀 더 잘했으면 Riot Developer Portal에서 소환사의 협곡 맵이나 오브젝트들의 사진을 가져올 수 있었을 텐데, 잘 몰라서 직접 가져와야 했음

이때 느낀 한계는 이후 Riot API 데이터를 사용한 Chemi.lol과 League of Legends Data Analysis 프로젝트로 이어졌습니다.

- [Chemi.lol](/projects/chemi/)
- [League of Legends Data Analysis](/projects/lda/)


---

## CODE

[![GITHUB](/image/profile/github-mark.png)](https://github.com/alsspp01/LRP.git)
&nbsp;  
&nbsp;  
> 그림을 클릭하면 GitHub 저장소로 연결됩니다.
