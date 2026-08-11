---
title: "자동화 - PART 01. Notion 알림 Discord로 보내기"
date: 2026-08-11
description: "Project DIA(Do It, AI) #5"
image: 
type: "post"
tags: ["DIA", "DEFIB", "Node.js", "Notion", "Discord", "Webhook"]
---

## 1. 개요 및 프로젝트 목적

Notion의 '개발 일지' 데이터베이스에 새 페이지가 생성될 때, Discord 채널로 알림 임베드(Embed)를 자동으로 전송하는 독립형 파이프라인을 구축했다. 본 프로젝트는 `d3fib/Devlog2Discord` 단일 저장소로 구성된다. 이는 기존에 회의록 등을 기록하던 상시 실행형 디스코드 봇 프로젝트인 `secretary`와는 완전히 별개로 동작하는 소규모 독립 시스템이다.

## 2. 아키텍처 설계: GitHub Actions에서 자체 웹훅 서버로의 전환

초기에는 **GitHub Actions**를 트리거로 사용하는 아키텍처를 계획했다. 하지만 조사와 실무 검증 과정에서 구조적 한계가 드러나, 최종적으로는 홈서버(노트북)에서 상시 실행되는 **경량 Node.js 웹훅 서버**로 전환했다.

### GitHub Actions 설계 폐기 이유

1. **Inbound Webhook 엔드포인트의 부재**: GitHub Actions는 외부 시스템의 단순 웹훅 요청을 직접 수신할 수 없다. 외부에서 워크플로우를 깨우는 유일한 방법은 `repository_dispatch` API를 사용하는 것뿐이다. 그러나 이 API는 반드시 `{'event_type': ..., 'client_payload': {...}}` 형태의 고정된 JSON 본문을 요구한다.
2. **Notion 자동화 웹훅의 고정 포맷 한계**: Notion이 제공하는 'Send webhook' 액션(2026년 1월 출시)은 커스텀 헤더는 지원하지만, JSON 본문은 사용자가 수정할 수 없는 고정 포맷(`{ source: {...}, data: <Notion page object> }`)으로만 전송된다. Notion 측에서 `event_type`과 같은 커스텀 키를 포함하도록 페이로드를 가공할 방법이 없다.
3. **연동 실패 및 자동화 일시 중지**: 결과적으로 Notion에서 GitHub로 직접 전송되는 요청은 스키마 검증 단계에서 항상 **422 Unprocessable Entity** 오류로 거부되었다. Notion은 이러한 실패가 반복되면 해당 자동화를 '예기치 않은 오류로 인해 자동화 일시 중지됨' 상태로 전환하며 비활성화한다.

[ Notion 자동화 설정 중 webhook 전송 실패 경고 화면 ]

이러한 근본적 한계를 확인한 후, GitHub Actions 및 Notion API 재조회(폴링) 방식을 모두 걷어냈다. Notion의 페이로드를 직접 수신해 처리하는 단일 파일 구조의 `server.js` 기반 서버를 구축했다. Notion이 전송하는 페이로드(`data`) 내에 이미 페이지 제목, 속성, URL이 모두 포함되어 있으므로 `@notionhq/client` 라이브러리를 통한 추가 API 호출이나 커서 상태 관리도 불필요해졌다.

## 3. 외부 노출 및 네트워크 구성 (Tailscale Funnel)

홈서버는 공인 IP나 도메인이 없는 환경이다. 외부의 Notion 서버로부터 요청을 받기 위해서는 공인 HTTPS URL이 필요했다. 처음에는 ngrok의 무료 도메인을 검토했으나, 이미 해당 호스트에 구축되어 작동 중인 **Tailscale Funnel**을 재사용하기로 결정했다.

새로운 계정 등록이나 토큰 발급 없이, 기존 Funnel 설정에 `/notion-relay/devlog` 경로만 추가하여 비용과 공수를 줄였다.

* 릴레이 컨테이너는 호스트나 LAN에 노출되지 않도록 `127.0.0.1:####`에만 포트를 바인딩한다.
* 다음 명령어를 통해 외부 트래픽을 내부 포트로 전달한다.
  `tailscale funnel --set-path=/notion-relay/devlog http://127.0.0.1:####`

## 4. 코드 구현 및 구조

시스템은 의존성을 최소화하여 가볍고 견고하게 구현했다.

### server.js (핵심 런타임 파일)

외부 의존성 라이브러리 없이, Node.js 내장 `http` 모듈과 글로벌 `fetch` API만을 사용하여 작성했다. 주요 모듈식 기능은 다음과 같다.

* **getPageTitle**: 데이터베이스마다 제목 속성의 이름이 다를 수 있으므로, 하드코딩된 속성명 대신 속성 배열 내에서 `type === 'title'`인 대상을 동적으로 탐색하여 제목을 추출한다.
* **formatPropertyValue**: select, multi_select, rich_text, status, number, checkbox, url, date 등 Notion의 다양한 데이터 타입을 안전하게 문자열로 변환하는 범용 포맷터다. 알림 대상인 Project, Tag, Description 속성을 이름 기준으로 탐색하며, 데이터가 없거나 비어 있을 때는 출력을 조용히 생략한다.
* **handleNotionPage**: Discord 임베드 메시지를 작성한다. 알림 제목은 정적 텍스트('📝 Devlog updated')로 유지하되, Notion 페이지 제목은 `[**제목**](page.url)` 형태의 마크다운 링크로 구성하여 클릭 시 바로 Notion 페이지로 이동할 수 있도록 설계했다. 각 속성 정보는 `**Label:** 값` 형태로 빈 줄 없이 세로로 정렬하여 가독성을 높였다. 제목과 속성 목록 사이에는 빈 줄 하나를 추가하여 시각적 구분을 주었다.
* **HTTP Server**: 보안을 위해 사전에 정의된 `RELAY_PATH`로 들어오는 POST 요청만 허용한다. 요청 헤더의 `x-relay-secret` 값이 환경 변수의 `RELAY_SECRET`과 정확히 일치하는지 확인하여 허가되지 않은 불법 요청을 차단한다.

### 인프라 및 설정 파일

* **Dockerfile**: `node:20-alpine` 기반으로 빌드된다. 의존성 라이브러리가 없어 `npm install` 과정을 거치지 않으므로 빌드 및 배포 속도가 매우 빠르다.
* **start.sh / stop.sh**: 빌드, 포트 포워딩을 통한 컨테이너 구동, Tailscale Funnel 경로 매핑 등록을 자동화한다. `stop.sh` 실행 시에는 다른 서비스에 영향을 주지 않도록 Tailscale 설정은 유지하고 컨테이너만 정지한다.
* **.env 및 .env.example**: `RELAY_SECRET`, `DISCORD_WEBHOOK_URL`, `RELAY_PORT`, `RELAY_PATH`, `FUNNEL_PATH` 등의 환경 변수를 템플릿화하여 관리한다.

## 5. 운영 인시던트 및 해결 과정

구축 및 실운영 테스트 도중 두 가지 주요 네트워크 인시던트가 발생했다.

### 인시던트 1: Tailscale serve와 funnel의 비호환 문제

Funnel이 구동 중인 호스트에서 신규 경로를 추가하기 위해 `tailscale serve --set-path=...` 명령을 실행했다. 이 과정에서 호스트 전체의 Funnel(외부 공개) 설정이 내부 tailnet 전용으로 강등되는 문제가 발생했다. 이로 인해 동일 호스트의 루트 경로(`/`)를 사용하던 블로그 좋아요 API까지 외부 접근이 차단되었다.

* **원인**: `serve` 명령어가 기존 `funnel` 설정을 덮어쓰면서 발생한 비호환 현상이다.
* **해결**: 대상 서비스들을 원래 포트와 설정으로 복구하고 `tailscale funnel --bg 443`을 통해 재실행했다.
* **교훈**: 기존 공개 Funnel 경로에 새로운 경로를 얹을 때는 반드시 `serve`가 아닌 `funnel --set-path` 명령을 사용해야 한다. 이 수칙은 `~/portInfo.md` 및 프로젝트 `README.md`에 경고문으로 기록해 두었다.

### 인시던트 2: Tailscale Funnel의 경로 접두어(Prefix) 제거 동작

설정 직후 모든 외부 요청이 '404 Not Found'로 거부되는 현상이 발견되었다.

* **원인**: Tailscale Funnel은 외부 유입 URL이 `/notion-relay/devlog`이더라도, 내부 백엔드로 요청을 전달할 때 접두어 경로를 무시하고 `req.url`을 `/`로 변경하여 전달한다. 웹서버가 내부적으로 `RELAY_PATH`를 `/notion-relay/devlog`로 기대하고 있었기에 경로 매칭 실패가 일어난 것이다.
* **해결**: 임시 Python 에코 서버를 구성하여 실제 인입되는 웹훅 페이로드와 HTTP 헤더를 확인했다. 검증 결과를 바탕으로 `server.js` 내부의 `RELAY_PATH` 기본값을 `'/'`로 수정하여 해결했다.

### 실전 테스트 정책 수립

운영 도중 검증 목적으로 운영용 Discord 채널에 불필요한 알림이 여러 차례 발송되는 문제가 발생했다. 채널 오염을 방지하기 위해 실전 채널 대상의 라이브 테스트를 완전히 금지하는 방침을 수립했다. 향후 모든 기능 검증은 로컬 환경에서의 curl 호출 테스트와 컨테이너 로그 모니터링, 그리고 텍스트 미리보기를 통해서만 수행하도록 절차를 체계화했다.

## 6. 보안 조치 및 저장소 이원화

프로젝트를 커뮤니티에 공개 배포하기 위해 민감 정보를 격리하고 소스 코드를 사니타이즈(Sanitize)했다.

### 프라이빗 및 퍼블릭 레포지토리 관리

* **프라이빗 저장소 (`d3fib/Devlog2Discord`)**: 초기의 온갖 시행착오 이력과 실제 홈서버의 도메인 주소, 내부 포트가 커밋 로그에 그대로 기록되어 있으므로 비공개 상태를 유지한다.
* **퍼블릭 저장소 (`alsspp01/Notion2Discord`)**: 민감 정보를 철저히 제거하기 위해 `git checkout --orphan` 명령을 활용하여 새로운 이력을 생성했다. 기존 커밋 로그를 전부 스쿼시(Squash)하여 깨끗한 단일 커밋으로 시작하도록 구성한 뒤 공개 저장소에 푸시했다.

### 포트 및 호스트 환경 변수화

하드코딩되어 있던 실제 포트와 Tailscale Funnel 도메인 정보를 모두 제거하고 `.env` 환경 변수로 전환했다.

* 공용 템플릿의 기본 대기 포트는 Node 개발 환경에서 흔히 사용하는 `3000`이나 `8080` 대신, 포트 충돌 위험이 적은 번호로 새로 지정했다.
* Notion이 호출하는 외부 URL 경로(`FUNNEL_PATH`)와 서버 내부에서 처리하는 수신 경로(`RELAY_PATH`)가 기능적으로 서로 다름을 인지하고, 이를 환경 변수 내에서 완전히 분리 설계하여 잠재적 경로 일치 오류를 원천 차단했다.
