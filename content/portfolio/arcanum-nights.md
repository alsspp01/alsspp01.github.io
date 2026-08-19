---
title: "Arcanum Nights"
title_en: "Arcanum Nights"
type: page
description: "아이디어를 시스템, 데이터, UI/UX와 실제 플레이 경험으로 구체화한 게임 프로젝트."
description_en: "A game project where an early idea became systems, data, UI/UX, and a playable experience."
---

<div class="lang-ko">

**Period** · 2024.11 – 2026.08  
**Team** · D3F!B  
**Role** · Team Lead / Lead Planning

## From Blank Page

해와 달, 별자리를 소재로 한 싱글~2인 퍼즐게임입니다.

초기 경험 설계부터 플레이 흐름, 시스템, UI/UX, 데이터 구조와 구현 방식까지 구체화했습니다.

> **개발할 수 있는가?**  
> **아트 리소스로 표현할 수 있는가?**  
> **정해진 시간 안에 만들 수 있는가?**  
> **플레이어가 실제로 이해할 수 있는가?**

를 함께 봤습니다.

## Design Before Questions Arrive

정상적인 흐름뿐 아니라 예상 밖의 행동, 조건 충돌, 값이 없는 상태, 중간 상태 변경처럼 구현 단계에서 나올 질문을 미리 생각했습니다.

자료구조 설명 문서와 데이터 테이블을 만들어 개발자가 구현 방향을 이해할 수 있는 기반도 함께 정리했습니다.

> **“이 경우에는 어떻게 해요?”**

라는 질문에

> **“그 경우에는 이렇게 처리하면 됩니다.”**

라고 답할 수 있는 기획을 좋아합니다.

## Technical Enough to Make It Real

Unity와 C# 구조를 익혔고 튜토리얼 구현과 Python 기반 맵 툴 제작에도 참여했습니다.

목적은 기획을 개발로 대체하는 것이 아니라  
**기획 단계에서 구현 비용과 구조를 현실적으로 상상하는 것**이었습니다.

## Watching the Player

PlayX4 시연에서는 클리어 여부만 보지 않았습니다.

표정과 자세, 키보드·마우스 입력 빈도와 속도, 플레이가 늘어지는 구간을 관찰해  
게임 템포, 애니메이션 속도, 플레이 흐름을 다시 검토했습니다.

## One More Edge Case

데모 빌드에서 화면상 숨겨진 콘텐츠의 원본 스토리 데이터가 함께 포함된 것을 발견해  
데모 전용 데이터를 분리했습니다.

**정상 화면뿐 아니라 실제 배포 시 무엇이 함께 따라가는지까지 생각하는 편입니다.**

## What I Learned

예전에는 기능이 많을수록 좋은 기획이라고 생각했습니다.

지금은 반대로 묻습니다.

> **이 기능이 정말 필요한가?**  
> **복잡하게 만든 만큼 사용자 경험이 좋아지는가?**

**뒤에서는 복잡하게 고민하되, 앞에서는 단순하게 보이게 만드는 것.**

[관련 글 →](/posts/2026-08-19-designing-for-edge-cases/)  
[D3F!B Collaboration System →](/portfolio/d3fib-workspace/)  
[← Portfolio](/portfolio/)

</div>

<div class="lang-en" style="display:none">

**Period** · Nov 2024 – Aug 2026  
**Team** · D3F!B  
**Role** · Team Lead / Lead Planning

## From Blank Page

Arcanum Nights is a single-to-two-player puzzle game built around the sun, moon, and constellations.

I worked from the early experience concept through player flow, systems, UI/UX, data structures, and implementation considerations.

> **Can we build it?**  
> **Can the art team express it with available resources?**  
> **Can it fit the schedule?**  
> **Will a player understand it without us standing next to them?**

## Design Before Questions Arrive

I tried to think through more than the happy path: unexpected behavior, conflicting conditions, missing values, and state changes during an action.

I also created data-structure notes and tables so implementation did not have to start from an abstract idea.

I prefer reaching a point where:

> **“What happens in this case?”**

can often be answered with:

> **“Then we handle it this way.”**

## Technical Enough to Make It Real

I learned the Unity/C# structure and contributed to tutorial implementation and a Python-based map tool.

The point was not to replace development with planning.  
It was to **estimate implementation cost and technical structure more realistically while planning.**

## Watching the Player

At PlayX4, I watched more than whether players finished the demo.

Expressions, posture, keyboard/mouse rhythm, and where the experience started to drag all fed back into tempo, animation speed, and player flow.

## One More Edge Case

In a demo build, I noticed hidden content whose original story data was still bundled. I separated demo-specific data instead.

**I care not only about the expected screen, but also about what travels with the product when it is actually shipped.**

## What I Learned

I used to think more features meant a better plan.

Now I ask:

> **Does this feature actually need to exist?**  
> **Does the added complexity improve the experience enough to justify itself?**

**Think through the complexity in the background; keep the surface simple.**

[Related Post →](/posts/2026-08-19-designing-for-edge-cases/)  
[D3F!B Collaboration System →](/portfolio/d3fib-workspace/)  
[← Portfolio](/portfolio/)

</div>
