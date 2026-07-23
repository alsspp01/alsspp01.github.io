---
title: "블로그 활성화 - PART 01. Github Blog"
date: 2026-07-22
description: "Project DIA(Do It, AI) #1"
image: 
type: "post"
tags: ["DIA", "Hugo", "github-pages", "CI/CD"]
---

## 1. 개요

본 문서는 Hugo 정적 사이트 생성기(Static Site Generator)와 GitHub Pages, GitHub Actions를 활용하여 개인 블로그를 구축하고 자동 배포 파이프라인을 설정한 과정을 정리한다. 외부 라이브러리 의존성을 최소화하고 CI/CD 파이프라인을 통해 배포 과정을 자동화하는 것을 목표로 한다.

---

## 2. Hugo 사이트 구조 및 설정

블로그 디렉터리 구조는 표준 Hugo 레이아웃을 따른다.

```text
.
├── config.toml
├── content/
│   ├── posts/
│   └── projects/
├── themes/
└── .github/
    └── workflows/
        └── deploy.yml
```

기본 설정 파일인 `config.toml`은 다음과 같이 작성했다.

```toml
baseURL = 'https://alsspp01.github.io/'
languageCode = 'ko-kr'
title = 'Dev Blog'
theme = 'custom-theme'

[markup.goldmark.renderer]
  unsafe = true
```

---

## 3. GitHub Actions 기반 자동 배포 (CI/CD)

`main` 브랜치에 코드가 push되면 Hugo 빌드를 거쳐 `github-pages` 환경으로 정적 파일이 자동 배포되도록 워크플로를 설계했다.

```yaml
name: Deploy Hugo Site to GitHub Pages

on:
  push:
    branches:
      - main

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
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: true
          fetch-depth: 0

      - name: Setup Pages
        uses: actions/configure-pages@v5

      - name: Build with Hugo
        run: |
          hugo --minify

      - name: Upload artifact
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

[ GitHub Actions 워크플로 배포 성공 화면 스크린샷 ]

---

## 4. 로컬 테스트 및 서빙 스크립트

별도의 `pip` 패키지 설치 없이 Python 표준 라이브러리의 `http.server` 모듈을 활용하여 로컬에서 빌드된 정적 파일을 바로 확인할 수 있는 서빙 스크립트를 작성했다.

```python logic.py
import http.server
import socketserver
import subprocess

PORT = 8000

def build_hugo():
    subprocess.run(["hugo", "--buildDrafts"], check=True)

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="public", **kwargs)

if __name__ == "__main__":
    build_hugo()
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()
```

---

## 5. 주요 커스텀 기능 및 UX 개선

블로그의 사용성과 완성도를 높이기 위해 다음과 같은 추가 기능을 구현했다.

- **통합 검색창 및 태그 검색(AND/OR)**: 전용 검색 기능을 구현하고, 태그 조건 검색(AND/OR)을 지원하여 원하는 포스트를 더욱 정교하게 탐색할 수 있도록 설계했다.
- **좋아요(Like) 기능 및 어뷰징 방지**: 게시글별 좋아요 버튼을 추가했다. 클릭 후 취소가 불가능하도록 설계하여 무분별한 연타 등 어뷰징(Abuse) 공격의 영향을 줄이고 데이터의 신뢰성을 확보했다.
- **클립보드 링크 공유(Share)**: 공유 버튼 클릭 시 현재 포스트의 링크가 클립보드에 자동으로 복사되도록 구현하여 편의성을 높였다.
- **이전/다음 글 대형 네비게이션 버튼**: 포스트 하단의 이동 영역을 눈에 띄는 대형 버튼 형태로 개선하고, 제목이 명확히 보이도록 디자인하여 탐색 편의성을 극대화했다.
- **동적 가로 레이아웃(Dynamic Width)**: 화면 폭을 동적으로 할당하도록 스타일을 구성하여 다양한 디바이스 환경에서 한층 더 정돈된 레이아웃을 제공한다.

---

## 6. 요약

- **Hugo**: 빠른 정적 사이트 컴파일 수행
- **GitHub Actions**: `main` 브랜치 푸시 시 자동 빌드 및 GitHub Pages 호스팅 완료
- **Python Standard Library**: 별도 의존성 없는 가벼운 로컬 검증 및 서빙 환경 구성
