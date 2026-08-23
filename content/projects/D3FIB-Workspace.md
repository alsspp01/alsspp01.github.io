---
title: "⚡ D3F!B Team Operations & Automation"
title_en: "⚡ D3F!B Team Operations & Automation"
type: page
aliases:
  - /portfolio/d3fib-workspace/
description: "활동 시간이 다른 10명 규모 원격 게임 제작팀을 창립하고 운영하며 구축한 정보 구조와 자동화."
description_en: "Team operations, information architecture, and automation built for a ten-person remote game development team working on different schedules."
---

<div class="lang-ko">

<a id="top"></a>

**기간** · 2024.11–현재  
**역할** · 창립자 · 대표 · 팀장 · 팀 운영 체계 및 자동화 설계  
**팀** · 약 10명, 완전 원격  
**구성** · 기획 3 · 개발 4 · 아트 3  
**주요 작업** · 조직 운영 · 정보 구조 · 권한 설계 · 업무 흐름 · AI 보조 자동화 개발

[**Case Study ↓**](#case-study)

---

<a id="case-study"></a>
## Case Study

### 플레이어의 심장을 다시 뛰게 하는 게임

D3F!B는 여러 장르의 게임 시리즈를 만들기 위해 시작한 게임 제작팀입니다. 이름은 심장 제세동을 뜻하는 `defibrillation`에서 가져왔고, `3`과 `!`는 leetspeak로 표현했습니다.

비슷한 문법을 반복하는 게임에 지친 플레이어에게 다시 심장이 뛰는 듯한 새로운 경험을 주고 싶다는 포부를 담았습니다. 첫 작품은 Arcanum Nights이며, 현재 차기작도 준비하고 있습니다. 앞으로 제작하고 싶은 사이버펑크 게임까지 하나의 세계관으로 연결할 계획입니다.

저는 팀을 만들고 사람을 모은 창립자이자 대표입니다. 약 10명의 팀원이 기획, 개발, 아트로 나뉘어 완전 원격으로 활동하고 있습니다. 학생과 직장인이 섞여 있어 활동 시간이 모두 다르고, 한 달에 한 번 전원이 모여 진행 상황을 공유하는 것 외에는 각자의 일정에 맞춰 작업합니다.

### 도구가 아니라 일하는 방식을 설계하기

팀이 커지면서 운영상의 문제가 한꺼번에 나타났습니다. 자료와 결정 사항을 찾기 어려웠고, 회의가 끝나면 논의 결과가 사라졌습니다. 담당자와 마감일이 불분명한 일도 생겼고, 채널과 문서는 늘어났지만 어디에 무엇을 남겨야 하는지는 더 모호해졌습니다. 접근 권한 요청도 반복됐습니다.

무엇보다 대부분의 질문과 전달이 대표인 저를 거쳐야 했습니다. 제가 모든 내용을 기억하고 다시 설명하는 방식으로는 팀의 규모를 유지할 수 없다고 판단했습니다. 그래서 새 도구를 하나 더 도입하기보다, Discord · Notion · Google Drive의 역할을 먼저 나눴습니다.

- **Discord** · 실시간 대화, 회의, 공지와 자동 알림
- **Notion** · 기획, 일정, 회의록, 개발일지, 팀 가이드와 부서별 마일스톤
- **Google Drive** · 팀원이 함께 열람하고 활용할 수 있는 최종 제작 리소스

Discord 서버와 채널, 역할 체계부터 Notion 데이터베이스, Drive 폴더와 파일명 규칙, 회의 및 기록 방식, 일정과 업무 배분, 신규 팀원 온보딩 문서까지 직접 설계하고 운영했습니다.

### 보안과 접근성을 함께 고려한 권한 구조

운영 · 계약 관련 문서는 대표인 저만 접근할 수 있도록 분리하고, 새로운 팀원이 합류하면 먼저 보안서약을 체결합니다. 반면 Drive에 올라온 최종 리소스는 팀원이 서로 열람하고 활용하는 데 제한을 두지 않았습니다.

Discord에는 기획 · 개발 · 아트 부서별 채널을 두어 세부 작업 과정은 해당 부서 안에서 공유하게 했습니다. 다른 부서가 모든 중간 과정을 따라가느라 정보에 묻히지 않으면서도, 최종 결과물과 협업에 필요한 정보에는 접근할 수 있도록 한 구조입니다. GitHub 웹훅 알림도 개발팀 전용 채널에 연결했습니다.

권한을 단순히 많이 열거나 잠그는 문제가 아니라, 각 역할에 필요한 정보가 자연스럽게 보이도록 설계하는 문제로 다뤘습니다.

### 사용하지 않는 페이지를 다시 설계한 과정

처음 만든 Notion 페이지는 필요한 정보가 있어도 여러 번 클릭해야 도달할 수 있었습니다. 구조는 정돈되어 있었지만 팀원들이 잘 사용하지 않았습니다.

이를 확인한 뒤 기획 · 개발 · 아트가 각자 즐겨찾기해 사용할 수 있는 부서별 메인 페이지를 만들었습니다. 각 페이지에서 해당 부서의 마일스톤, 작업 문서와 필요한 정보에 바로 접근할 수 있도록 탐색 깊이를 줄였습니다. 개편 이후 팀원들이 부서 페이지를 실제로 사용하고 있다는 것을 Notion 트래픽으로 확인했습니다.

운영 시스템은 관리자가 보기에 완성된 구조보다 팀원이 실제로 찾아 쓰는 구조여야 한다는 것을 배웠습니다.

### 별도 사이드 프로젝트를 팀 운영에 적용하기

반복되는 운영 업무는 제 별도 사이드 프로젝트인 DIA의 자동화 도구를 적용해 줄였습니다. DIA가 D3F!B에 속한 프로젝트인 것은 아니며, 한 사이드 프로젝트에서 만든 도구를 다른 프로젝트의 실제 운영 환경에 적용한 사례입니다.

#### 회의 기록 자동화

[🔗 Secretary4Discord](https://github.com/alsspp01/Secretary4Discord)는 Discord 음성 회의를 녹음하고, 긴 회의를 안건 단위로 나누어 Gemini로 요약한 뒤 참석자와 부서 정보를 포함한 Notion 회의록을 생성합니다. 회의 중 휴식과 재개, 요약 재작성, 임시 파일 관리도 Discord 안에서 처리할 수 있습니다.

이를 통해 회의 내용을 제가 전부 기억했다가 다시 전달할 필요가 줄었고, 참석하지 못한 팀원도 정리된 기록을 확인할 수 있게 됐습니다.

#### 개발일지 공유 자동화

[🔗 Notion2Discord](https://github.com/alsspp01/Notion2Discord)는 Notion 개발일지 데이터베이스에 새 페이지가 생기면 웹훅을 받아 Discord 채널에 알립니다. 팀원들은 별도로 공지문을 작성하지 않고도 새 개발일지를 공유할 수 있고, Discord에서 해당 글을 바로 인용해 대화를 이어갈 수 있습니다.

#### 반복 공지와 개발 상황 공유

Discord 예약 메시지를 만들어 정해진 시간의 공지를 자동으로 전송하도록 했습니다. GitHub 활동은 웹훅을 통해 개발팀 채널에 전달해 저장소의 변경 사항을 별도로 옮겨 적지 않아도 확인할 수 있게 했습니다.

이 도구들은 실제 팀 운영에 사용하고 있습니다. 저는 문제와 요구사항 정의, 기능 및 사용자 흐름 설계, 기술 구조 결정, Claude에 대한 구현 지시, API 연결, 테스트와 오류 재현, 수정 방향 결정, 운영 및 유지보수를 담당했습니다. 코드는 Claude의 도움을 받아 작성한 AI 보조 개발 방식이었습니다.

### 설명을 반복하지 않아도 움직이는 팀

시스템을 적용한 뒤 회의 내용을 일일이 기억할 필요가 줄었고, 신규 팀원의 온보딩도 구두 설명 대신 문서로 진행할 수 있게 됐습니다. 개발일지는 Discord에서 바로 공유하고 인용할 수 있게 되었으며, 리소스 목록을 Notion에 정리한 뒤에는 어떤 리소스가 필요한지 묻는 일도 줄었습니다.

가장 큰 변화는 대표인 제가 같은 내용을 반복해서 설명하는 시간이 줄었다는 점입니다. 팀원은 각자의 활동 시간에 필요한 기록과 자료를 찾고, 저는 모든 정보 전달의 중간에 서는 대신 방향과 의사결정에 집중할 수 있게 됐습니다.

[↑ Top](#top)

</div>

<div class="lang-en" style="display:none">

<a id="top-en"></a>

**Period** · Nov 2024–Present  
**Role** · Founder · Representative · Team Lead · Team Operations and Automation Design  
**Team** · Around 10 members, fully remote  
**Disciplines** · Planning 3 · Development 4 · Art 3  
**Scope** · Team Operations · Information Architecture · Access Design · Workflows · AI-Assisted Automation

[**Case Study ↓**](#case-study-en)

---

<a id="case-study-en"></a>
## Case Study

### Games that make players' hearts race again

D3F!B is a game development team I founded to create a series of games across different genres. The name comes from `defibrillation`, expressed with `3` and `!` as leetspeak.

It reflects our goal of making unfamiliar experiences that can bring excitement back to players who have grown tired of formulaic games. Arcanum Nights is our first title, and we are already preparing a follow-up project. The longer-term plan is to connect these games, including a cyberpunk title I want to make, through a shared universe.

I founded the team, recruited its members, and continue to run it as its representative. Around ten people work remotely across planning, development, and art. The team includes both students and full-time employees with different working hours. Apart from a monthly all-hands progress meeting, members work asynchronously around their own schedules.

### Designing how the team works

As the team grew, several operational problems appeared at once. Documents and decisions were difficult to find, meeting outcomes disappeared into chat history, and ownership and deadlines were not always clear. We accumulated more channels and documents without a shared understanding of where information belonged. Access requests also became repetitive.

Most questions and updates eventually passed through me. A team of this size could not depend on its representative remembering and re-explaining everything, so I first gave each existing tool a clear role instead of adding another platform.

- **Discord** · Live communication, meetings, announcements, and automated notifications
- **Notion** · Plans, schedules, meeting notes, development logs, team guides, and department milestones
- **Google Drive** · Final production assets available for team-wide use

I designed and operated the Discord server, channel and role structure, Notion databases, Drive folders and naming conventions, meeting and documentation practices, task allocation, schedules, and onboarding documentation.

### Balancing security and access

Operational and contract documents are restricted to me as the team representative, and new members sign a confidentiality agreement when they join. Final assets placed in Drive, however, are available for members to review and reuse without unnecessary restrictions.

Discord channels for planning, development, and art keep detailed work in progress within the relevant discipline. This prevents other departments from being overwhelmed by every intermediate step while preserving access to final assets and information needed for collaboration. GitHub webhook notifications are likewise routed to a development-only channel.

I treated permissions as a workflow-design problem: each role should naturally see what it needs without exposing unrelated information or creating repeated access requests.

### Redesigning a workspace people were not using

The first Notion workspace was organized, but important information sat several clicks below the top level. Team members rarely used it.

I replaced the single deep structure with dedicated home pages for planning, development, and art. Each page gave its department direct access to milestones, working documents, and frequently needed information. Members bookmarked and began using these pages, which I verified through Notion traffic rather than assuming the redesign had worked.

The experience taught me that an operational system must be designed around how people retrieve information, not around how tidy its hierarchy looks to the administrator.

### Applying a separate side project to team operations

I reduced recurring administrative work by applying tools from DIA, a separate automation side project of mine. DIA is not a subproject of D3F!B; this was a case of using tools from one side project in the live environment of another.

#### Automated meeting records

[🔗 Secretary4Discord](https://github.com/alsspp01/Secretary4Discord) records Discord voice meetings, splits longer sessions into agenda-sized segments, summarizes the audio with Gemini, and creates Notion meeting notes with attendee and department information. Breaks, resumptions, summary regeneration, and temporary-file cleanup can also be handled from Discord.

This reduced the need for me to remember and relay every discussion, while giving members who missed a meeting a durable record to consult.

#### Automated development-log sharing

[🔗 Notion2Discord](https://github.com/alsspp01/Notion2Discord) receives a webhook whenever a page is added to the Notion development-log database and posts the update to Discord. Members no longer need to write a separate announcement, and the team can cite and discuss the log directly from Discord.

#### Scheduled communication and development visibility

Scheduled Discord messages handle announcements that must be sent at a particular time. GitHub webhooks deliver repository activity to the development channel, removing the need to relay every update manually.

These tools are used in the team's day-to-day operations. I defined the problems and requirements, designed the functions and user flows, decided the technical structure, directed Claude's implementation, connected the APIs and runtime environment, tested and reproduced failures, decided how they should be fixed, and now operate and maintain the tools. The code itself was produced through an AI-assisted development process with Claude.

### A team that can move without repeated explanations

The system reduced the need to remember every meeting detail and made onboarding less dependent on verbal explanations. Development logs became easier to share and cite from Discord. After I documented the available assets in Notion, members also stopped needing to ask which resources were available.

The most meaningful result was a reduction in the time I spent explaining the same information repeatedly. Members can find the records and resources they need on their own schedules, while I can focus on direction and decisions instead of acting as the team's central relay.

[↑ Top](#top-en)

</div>
