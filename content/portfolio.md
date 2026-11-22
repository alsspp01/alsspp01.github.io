---
title: "Portfolio"
title_en: "Portfolio"
type: page
description: "주요 프로젝트 & 역할"
description_en: "Selected work & roles"
url: "/portfolio/"
---

<div class="lang-ko">

현업의 문제를 이해하고, 모호한 요구사항을 실행 가능한 기술적 결정과 작업 흐름으로 연결해 왔습니다. PM, 연구 도구 개발, 원격 팀 운영, 게임 기획, 데이터 분석에서 제가 직접 맡은 역할과 팀의 결과를 구분해 정리했습니다.

프로젝트 이름을 클릭하면 자세한 Case Study를 볼 수 있습니다.

## 주요 사례

### 🌦️ [INEEJI · Alpha Weather (AI 예보지원)](/projects/ai-forecast-support/)

**2025.09–2026.02 · PM / 기획 / 기관 간 조율 · 4개 기관 공동 연구**

- **문제** · 고객의 실제 판단과 예측 모델의 목표가 어긋나 있었고, 기관 간 작업은 약 3개월간 정체
- **행동** · 회귀 → 분류로 예측 모델의 방향 전환 제안 · 채택, 기관별 입력 · 출력 · 의존성 정리, 요구사항 통역과 일정 조율
- **결과** · 작업이 다시 진행되어 팀이 이후 2개월 안에 마무리, 별도로 만든 Notion 템플릿 · WBS 도구를 사내 다른 프로젝트들이 자발적으로 도입

---

### 📊 [단국대 스포츠심리학 연구실 · 연구 도구 개발](/projects/lda/)

**2024.03–2025.06 · 연구 개발 아르바이트 · 개발 전반**

- **문제** · 연속 플레이와 인지피로 연구의 조건을 실행 가능한 실험 · 수집 · 분석 도구로 옮겨야 함
- **행동** · Stroop Test 앱, 약 6개월간의 Riot API 수집, 연속 플레이 표본 추출, 모호한 연구 조건의 해석 정리
- **결과** · 실험 참가자 약 40명이 사용, 참가자별 개인 리포트 도구 제작

---

### ⚡ [D3F!B · 협업 구조와 자동화](/projects/d3fib-workspace/)

**2024.11–현재 · 창립자 / 팀장 / 운영 체계 · 자동화 설계**

- **문제** · 활동 시간이 다른 10명 원격 팀에서 정보와 질문이 대표 한 명에게 몰림
- **행동** · Discord · Notion · Drive 역할과 권한 설계, Notion 워크스페이스 구축 · 탐색 구조 개선, 회의록 · 개발일지 자동화 적용
- **결과** · 회의 기록과 온보딩이 문서로 이어지고, 반복 설명에 쓰던 시간 감소

---

### 🎮 [Arcanum Nights](/projects/arcananights/)

**2025.01–현재 · 대표 / 팀장 / 게임 기획 리드 · D3F!B 팀 프로젝트**

- **문제** · 레퍼런스가 없는 오리지널 퍼즐 규칙을 팀이 구현할 수 있는 형태로 전달해야 함
- **행동** · 와이어프레임 · 데이터 테이블로 규칙 명세화, Python 맵 제작 · 유일해 검증 도구, PlayX4 관찰 기반 UX 개선
- **결과** · 2026.02.20 싱글 플레이 Steam Early Access 출시(팀), 멀티플레이 개발 중

---

### 🧪 [Chemi.lol](/projects/chemi/)

**2024.12–2025.05 · 프로젝트 발의 / 팀장 / 통계 분석 · 예측 실험 · 5인 팀**

- **문제** · 두 플레이어의 듀오 승률과 궁합을 데이터로 보여줄 수 있는지
- **행동** · 서비스 기획, 플레이어 통계 분석, 변수 설계, 회귀 모델 실험
- **결과** · 팀은 과거 기록 기반 듀오 승률 조회 기능을 출시, 예측 기능은 데이터 편향과 낮은 설명력(R² 약 0.2)을 근거로 출시하지 않고 종료 제안 → 팀 합의

---

## 보안과 시스템

### 🗄️ [RocksDB Chrono Break DB](/projects/rocksdb-chronobreak/)

**2025 1학기 · 팀 리더 / 구상 / 발표 · 3인 팀**

RocksDB의 Compaction · TTL · Merge 분석을 4주에 걸쳐 게임 로그를 특정 시점으로 되돌리는 로그 DB로 발전시켰습니다.

- 주차별 실험 결과를 다음 질문으로 연결해 Chrono Break라는 응용 시나리오로 구상
- 팀은 key 형식별 read 성능 비교(약 2.3배 차이), Snapshot 복원(2,933개 엔트리 일치)까지 구현 · 검증

---

### 🛡️ [SQLi Parsing AI](/projects/sqli-parsing-ai/)

**2024.03–2024.06 · AI 연구 · 개발 / 보안 도구**

SQL Injection 문장을 토큰 단위로 전처리하고 공격 구문을 분류하는 모델을 개발했습니다.

- 팀장으로서 프로젝트 기획과 AI 연구 · 개발 주도
- TensorFlow 기반 LSTM 모델 구현 및 성능 평가

---

### 🔍 [SIAM-W](/projects/siam-w/)

**2024.03–2024.06 · 졸업 프로젝트 / 보안 도구 · 3인 팀**

SQLmap 분석 결과를 바탕으로 웹사이트의 SQL Injection 가능성과 대응 방안을 보여주는 도구를 팀으로 제작했습니다.

- SQLi 기법 자료 조사, SQLmap 결과 파싱과 PySide6 GUI 개발 담당
- 보안 비전공 개발자가 이해할 수 있도록 유형별 대응 방법을 화면에 안내

---

🔗 [모든 프로젝트](/projects/)  
  
🔗 [About Me](/about_me/)  
  
🔗 [게시글](/posts/)  
  
🔗 [GitHub](https://github.com/alsspp01)

</div>

<div class="lang-en" style="display:none">

I turn ambiguous requirements and cross-functional input into practical technical plans and working tools. The cases below cover project management, research tooling, remote team operations, game design, and data analysis, and separate what I did from what the team delivered.

Click a project name to read the full case study.

## Selected Work

### 🌦️ [INEEJI · Alpha Weather (AI Forecast Support)](/projects/ai-forecast-support/)

**Sep 2025–Feb 2026 · PM / Planning / Cross-organization Coordination · Four-organization research program**

- **Problem** · The prediction model's goal did not match the decision the client actually needed to make, and cross-organization work had stalled for about three months
- **Action** · Proposed reframing the prediction model from regression to classification (adopted by the team), mapped each organization's inputs, outputs, and dependencies, and translated requirements between client and developers
- **Result** · Work resumed and the team completed it within two months; separately, a Notion template and a WBS tool I built were adopted by other internal projects on their own

---

### 📊 [Dankook University Sport Psychology Lab · Research Tooling](/projects/lda/)

**Mar 2024–Jun 2025 · Research Developer (Part-time) · End-to-End Development**

- **Problem** · Research conditions on consecutive play and cognitive fatigue had to become working experiment, collection, and analysis tools
- **Action** · Built a Stroop Test app, collected about six months of Riot API data, extracted consecutive-play samples, and documented ambiguous research conditions before implementing them
- **Result** · Used by around 40 study participants; built a tool that generated each participant's report

---

### ⚡ [D3F!B · Collaboration Structure & Automation](/projects/d3fib-workspace/)

**Nov 2024–Present · Founder / Team Lead / Operations & Automation Design**

- **Problem** · In a ten-person remote team on different schedules, information and questions kept routing through one person
- **Action** · Defined roles and permissions for Discord, Notion, and Drive; built the Notion workspace and later improved its navigation; applied meeting-note and devlog automation
- **Result** · Meeting records and onboarding moved into documents, and less time went into repeating explanations

---

### 🎮 [Arcanum Nights](/projects/arcananights/)

**Jan 2025–Present · Studio Head / Team Lead / Lead Game Designer · D3F!B team project**

- **Problem** · An original puzzle rule with no reference game had to be explained in a form the team could build
- **Action** · Specified the rules through wireframes and data tables, built a Python tool to create maps and verify a unique solution, and improved UX from observations at PlayX4
- **Result** · The team launched the single-player version in Steam Early Access on Feb 20, 2026; multiplayer is in development

---

### 🧪 [Chemi.lol](/projects/chemi/)

**Dec 2024–May 2025 · Initiator / Team Lead / Statistical Analysis & Prediction Experiments · Five-person team**

- **Problem** · Could data show the win rate and compatibility of two players as a duo?
- **Action** · Product planning, player statistics analysis, feature engineering, and regression experiments
- **Result** · The team launched a duo win-rate lookup based on shared match history; I proposed not releasing the prediction feature because of dataset bias and low explanatory power (R² ≈ 0.2), and the team agreed to end the project

---

## Security & Systems

### 🗄️ [RocksDB Chrono Break DB](/projects/rocksdb-chronobreak/)

**Spring 2025 · Team Lead / Concept / Presenter · Three-person team**

Over four weeks, we took an analysis of RocksDB's Compaction, TTL, and Merge and turned it into a game-log DB that rolls a session back to a point in time.

- Linked each week's results to the next question and framed the project around the Chrono Break scenario
- The team built and verified it, including a key-format read comparison (about 2.3×) and a snapshot restore of 2,933 matching entries

---

### 🛡️ [SQLi Parsing AI](/projects/sqli-parsing-ai/)

**Mar 2024–Jun 2024 · AI Research & Development / Security Tool**

I developed a model that tokenized SQL Injection statements and classified attack syntax.

- Led project planning and AI research and development
- Implemented and evaluated an LSTM classifier with TensorFlow

---

### 🔍 [SIAM-W](/projects/siam-w/)

**Mar 2024–Jun 2024 · Graduation Project / Security Tool · Three-person team**

Our team built a tool that used SQLmap results to explain potential SQL Injection vulnerabilities and appropriate responses.

- I researched SQLi techniques and built the SQLmap output parsing and the PySide6 GUI
- Showed type-specific countermeasures for developers without a security background

---

🔗 [All Projects →](/projects/)  
  
🔗 [About Me →](/about_me/)  
  
🔗 [Posts →](/posts/)  
  
🔗 [GitHub →](https://github.com/alsspp01)  

</div>
