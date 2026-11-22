---
title: "SIAM-W"
title_en: "SIAM-W"
type: page
description: "SQLmap 결과로 웹사이트의 SQL Injection 위험과 대응 방법을 안내하는 도구"
description_en: "A tool that turns SQLmap results into SQL Injection risks and fixes"
dated: true
period_start: "2024-03"
period_end: "2024-06"
summary: "오픈 소스 코드 사용: SQLmap\n웹사이트의 SQL injection 공격 가능성 여부를 진단 후 각 공격에 대한 대응 방법을 알려주는 보안 비전공 개발자를 위한 프로그램.\n\nUses SQLmap to identify possible SQL injection vulnerabilities and explain appropriate responses to developers without a security background."
tags: ["DKU"]
---

<div class="lang-ko">

## 요약

- **문제** · 보안 비전공 개발자도 자신의 웹사이트에 SQL Injection 위험이 있는지 확인하고 대응할 수 있도록 돕기
- **팀** · 3명, 2024년 1학기 산업보안학과 졸업(캡스톤) 프로젝트
- **역할** · SQLi 기법 자료 조사, SQLmap 결과 파싱(`crawling.py`)과 PySide6 GUI 개발
- **구조** · URL 입력 → SQLmap 자동 실행(`--batch`) → 탐지된 SQLi 유형 · Payload 추출 → 유형별 대응 방법(Prepared Statement, 입력값 검증 등) 안내
- **설계 기준** · 사이트마다 구성이 달라 강제 지침 대신 "추천식" 안내로, 보안 초급자도 따라 할 수 있는 수준으로 정함
- **검증** · 공개 테스트 사이트와 로컬 더미 서버(Ubuntu · nginx · PHP · MariaDB)에서 동작 확인
- **한계** · 탐지는 SQLmap에 의존하며, 대응 안내는 유형별로 미리 정리한 규칙 기반 설명

</div>

<div class="lang-en" style="display:none">

## Summary

- **Problem** · Help developers without a security background check their sites for SQL Injection and respond
- **Team** · 3 members, Industrial Security graduation (capstone) project, spring 2024
- **Role** · Researched SQLi techniques; built SQLmap output parsing (`crawling.py`) and the PySide6 GUI
- **Flow** · Enter a URL → run SQLmap automatically (`--batch`) → extract detected SQLi types and payloads → show type-specific countermeasures (prepared statements, input validation, etc.)
- **Design choice** · Since every site is built differently, the tool recommends fixes rather than enforcing rules, at a level a security beginner can follow
- **Validation** · Tested against a public test site and a local dummy server (Ubuntu · nginx · PHP · MariaDB)
- **Limits** · Detection relies on SQLmap; guidance is rule-based text prepared per SQLi type

</div>

---

### 시연 영상
#### [YouTube Demo](https://bit.ly/SIAM-W)

[![QR](/image/SIAM-W/QRcode.png)](https://bit.ly/SIAM-W)



#### < 영상 요약 >
1. 취약점 진단
   
![ex1](/image/SIAM-W/ex1.png)

2. 예상되는 취약점 안내
   
![ex2](/image/SIAM-W/ex2.png)

3. 코드 보안 수정 사항 지시

![ex3](/image/SIAM-W/ex3.png)


4. 안전한 코드인 경우의 출력

![ex4](/image/SIAM-W/ex4.png)


---

### Poster
#### 한국어
![Poster](/image/SIAM-W/Poster_kr.jpg)   

#### English
![Poster](/image/SIAM-W/Poster_en.jpg)  

---

### CODE
[GitHub 저장소 · Repository](https://github.com/hanja1500/SIAM-W)

---

[← Projects](/projects/) · [Portfolio](/portfolio/)
