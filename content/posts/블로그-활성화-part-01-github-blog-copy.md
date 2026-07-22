---
title: "블로그 활성화 - PART 01. Github Blog"
date: 2026-07-22
description: "Project GIA(Give It to AI) #2"
image: 
type: "post"
tags: ["GIA", "github-pages", "hugo", "ci-cd"]
---

## 1. 개요

본 문서는 GitHub Pages와 정적 사이트 생성기(Static Site Generator)인 Hugo를 활용하여 개인 기술 블로그를 구축한 내역을 정리합니다. 프로젝트 개발 환경 설정, Hugo 디렉터리 구조, 기본 설정 파일(`hugo.toml`), GitHub Actions 기반의 자동 배포(CI/CD) 파이프라인 구축과 더불어, 블로그 사용자 경험(UX) 향상을 위해 추가한 커스텀 기능 구현 사항을 다룹니다.

---

## 2. 저장소 구조 및 환경 구성

블로그 소스코드는 GitHub의 `<username>.github.io` 저장소에서 관리하며, 디렉터리 구조는 아래와 같습니다.

```text
. 
├── .github/
│   └── workflows/
│       └── deploy.yml      # GitHub Actions CI/CD 워크플로우
├── archetypes/
├── content/                # 마크다운 포스트 및 프로젝트 정보 저장
│   ├── posts/
│   └── projects/
├── layouts/                # 커스텀 HTML 템플릿
├── static/                 # 이미지, 파비콘 등 정적 자원
├── themes/                 # 적용된 Hugo 테마 (Submodule 관리)
└── hugo.toml               # Hugo 글로벌 설정 파일
```

---

## 3. Hugo 설정 (`hugo.toml`)

사이트 전체의 동작 및 메타데이터를 정의하는 `hugo.toml` 구성 코드입니다.

```toml
baseURL = 'https://alsspp01.github.io/'
languageCode = 'ko-kr'
defaultContentLanguage = 'ko'
title = 'Dev Log'
theme = 'PaperMod'

enableInlineShortcodes = true
enableRobotsTXT = true
buildDrafts = false
buildFuture = false
buildExpired = false

[minify]
  minifyOutput = true

[params]
  env = 'production'
  description = '개인 개발 및 프로젝트 기록 블로그'
  dateFormat = '2006-01-02'
  ShowReadingTime = true
  ShowShareButtons = false
  ShowPostNavLinks = true
  ShowCodeCopyButtons = true

[taxonomies]
  tag = 'tags'
  category = 'categories'
```

---

## 4. CI/CD 파이프라인 구축 (`deploy.yml`)

`main` 브랜치에 코드가 `push`될 때 자동으로 Hugo 사이트를 빌드하고 GitHub Pages로 배포하는 GitHub Actions 워크플로우 스크립트입니다.

```yaml
name: Deploy Hugo Site to GitHub Pages

on:
  push:
    branches: [ "main" ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Source Code
        uses: actions/checkout@v4
        with:
          submodules: recursive
          fetch-depth: 0

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: 'latest'
          extended: true

      - name: Build Hugo Site
        run: hugo --minify

      - name: Upload Pages Artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./public

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

---

## 5. 주요 커스텀 기능 및 UX 개선

블로그의 사용성과 완성도를 높이기 위해 구현한 주요 추가 기능은 다음과 같습니다.

- **통합 검색창 및 태그 검색(AND/OR)**: 전용 검색창을 구현하고, 태그 조건 검색(AND/OR)을 지원하여 원하는 포스트를 더욱 정교하게 탐색할 수 있도록 구성했습니다.
- **좋아요(Like) 기능 및 어뷰징 방지**: 게시글별 좋아요 버튼을 추가했습니다. 클릭 후 취소가 불가능하도록 설계하여 무분별한 클릭 연타 등 POD(Spam/Abuse) 공격 영향을 줄이고 데이터의 신뢰성을 확보했습니다.
- **클립보드 링크 공유(Share)**: 공유 버튼 클릭 시 현재 포스트의 페이지 링크가 클립보드에 자동으로 복사되어 간편하게 공유할 수 있습니다.
- **이전/다음 글 대형 네비게이션 버튼**: 포스트 하단의 이전글/다음글 이동 영역을 대형 버튼 형태로 개선하고 어떤 글인지 제목이 명확히 보이도록 디자인하여 탐색 편의성을 극대화했습니다.
- **동적 가로 레이아웃(Dynamic Width)**: 화면 폭을 동적으로 할당하도록 스타일을 적용하여 다양한 디바이스 환경에서 더 시각적으로 정돈되고 깔끔한 레이아웃을 제공합니다.

---

## 6. 결론 및 요약

- Hugo 프레임워크를 기반으로 블로그 구조를 체계화함.
- `deploy.yml`을 통해 Git 커밋 후 `push` 시 자동 빌드 및 배포되도록 구성함.
- 검색(AND/OR 태그 필터링), 좋아요, 링크 공유, 대형 글 이동 버튼, 동적 가로 레이아웃 등 주요 UX 커스텀 기능을 추가하여 가독성과 사용자 편의성을 크게 개선함.
