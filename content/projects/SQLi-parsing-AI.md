---
title: "SQLi parsing AI"
type: page
---

### SQL injection이란?

웹 애플리케이션과 데이터베이스 간의 연동에서 발생하는 취약점을 이용해,  
공격자가 입력 폼에 악의적으로 조작된 쿼리를 삽입해  
데이터베이스 정보를 불법적으로 열람하거나 조작하는 공격  


### Overview

1. 구문 분석을 위해 Secquence Data로 전처리
2. 전처리 시 문법적인 부분을 강조하기 위해 기호, SQL 문법 구성 단어를 기준으로 feature 추출
3. 기호, SQL언어, 변수, 처리되지 않은 일반 string 순으로 분류하여 토큰화, 정규화
4. Sezuence Data를 처리할 수 있는 모델 - LSTM(Long Short-Term Memory) 사용
5. F1 Score 등을 통해 검증

### 실행결과

