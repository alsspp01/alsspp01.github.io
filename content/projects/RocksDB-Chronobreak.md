---
title: "RocksDB Chrono Break DB"
title_en: "RocksDB Chrono Break DB"
type: page
description: "RocksDB 기능 분석에서 출발해 게임 로그를 특정 시점으로 되돌리는 로그 DB까지 만든 4주 팀 프로젝트"
description_en: "A four-week team project from analyzing RocksDB features to a game-log DB that rolls a session back in time"
tags: ["DKU"]
---

<div class="lang-ko">

**기간** · 2025년 1학기, 4주 (주차별 발표)  
**과목** · 단국대학교 오픈소스SW분석(빅데이터) · 4조  
**역할** · 팀장 · 주제 제안 · 발표  
**팀** · 3명 (실험 · 벤치마크, 자료 분석 · 발표자료는 팀원 담당)

[ChronobreakFilter 코드](https://github.com/alsspp01/RocksDB_ChronobreakFilter) · [db_bench 실험용 RocksDB fork](https://github.com/alsspp01/OpensourceBigdata)

## 요약

- **문제** · RocksDB가 LevelDB보다 실제로 어디서 더 쓸모 있는지 4주 안에 확인
- **한 일** · "RocksDB로 대전 게임 로그 저장 시스템 만들기"(전투 로그 Merge, 세션 저장 · 복구) 주제 제안, 주차별 결과를 다음 실험으로 연결, 발표
- **결과** · 팀이 게임 로그를 특정 시점으로 되돌리는 Chrono Break DB를 구현 · 검증

Chrono Break는 e스포츠 경기 중 치명적인 버그가 생겼을 때 경기를 이전 시점으로 되돌리는 기능입니다.  
코드 구현과 벤치마크는 팀원이 맡았고, 아래 결과는 팀 전체의 결과입니다.

## 주차별 흐름

| 주차 | 질문 | 결과 |
|---|---|---|
| 1 | Compaction · TTL · Merge는 어떻게 동작하나 | Merge와 Put 방식 비교 벤치마크 |
| 2 | 로그를 저장한다면 key를 어떻게 잡나 | Compaction Filter 영향 측정, key 형식 비교 |
| 3 | key 형식이 read 성능에 어떤 영향을 주나 | key 형식 3종 비교, Endian 불일치 발견 |
| 4 | 이걸로 Chrono Break DB를 만들 수 있나 | LogDB API, TTL 필터, Snapshot, 시점 복구 구현 |

## 근거

### 1주차 · Merge와 Put

| 항목 | Merge (stringappend) | Merge (uint64add) | Put (updaterandom) | Put (appendrandom) |
|---|---:|---:|---:|---:|
| 처리량 (ops/sec) | 28,694 | 28,724 | 43,007 | 39,574 |
| Compaction 횟수 | 88 | 78 | 74 | 182 |
| P99 latency | 570.48 | 569.85 | 1183.78 | 1053.55 |

- Put은 평균이 빠르지만 P99 지연이 크게 튐
- Merge는 느리지만 지연이 일정하고, 문자열 덧붙이기에선 Compaction이 절반 이하

![MergeOperator 벤치마크](/image/rocksdb-chronobreak/merge-benchmark.jpg)

### 2주차 · 같은 실험, 다른 결과

같은 db_bench가 팀원마다 10~60초, 1~2초로 달랐습니다.  
원인은 `--threads=10` 파라미터 하나였고, 이후 실험은 파라미터를 명시해 조건을 맞췄습니다.

### 3주차 · key 형식별 read 성능

1. key = `sessionId`, value = `(timestamp, log)[]` (Merge로 누적)
2. key = `(sessionId, timestamp)`, value = `log`
3. 2번 + `sessionId` prefix bloom filter

- 세션 전체를 읽을 때 1번이 가장 빠름, 2 · 3번과 throughput 약 2.3배 차이
- 한 세션에 값이 몰린 조건에서 prefix bloom filter 효과는 거의 없음
- db_bench는 big endian, `DecodeFixed64`는 little endian을 가정하는 불일치 확인

![key 형식별 read 성능](/image/rocksdb-chronobreak/keyformat-read.jpg)

### 4주차 · Chrono Break DB

- key를 `(uint32 sessionId, uint32 timestamp, string type)`으로, 사전순이 시간순과 같도록 big endian 인코딩
- value는 JSON 대신 MessagePack
- TTL Compaction Filter, 게임 상태 Snapshot 생성 · 조회 · 복원
- `DeleteRange`와 범위 제한 Iterator로 특정 시점 이후 로그 삭제

검증: 실제 경기 로그 300개 저장 후 30,000ms 전으로 되돌림 → 원본 65번째 로그까지 일치, 이후 저장도 정상  
Snapshot 복원은 2,933개 엔트리가 모두 원본과 일치했습니다.

![Chrono Break 직전 dump 비교](/image/rocksdb-chronobreak/chronobreak-dump.jpg)

</div>

<div class="lang-en" style="display:none">

**Period** · Spring 2025, four weeks (weekly presentations)  
**Course** · Open-Source SW Analysis (Big Data), Dankook University · Team 4  
**Role** · Team lead · Topic proposal · Presenter  
**Team** · 3 members (teammates ran experiments and benchmarks, analysis, and slides)

[ChronobreakFilter code](https://github.com/alsspp01/RocksDB_ChronobreakFilter) · [RocksDB fork for db_bench](https://github.com/alsspp01/OpensourceBigdata)

## Summary

- **Problem** · Find out, in four weeks, where RocksDB is actually more useful than LevelDB
- **What I did** · Proposed "a game-log storage system on RocksDB" (merging combat logs, saving and restoring sessions), linked each week's results to the next experiment, presented
- **Result** · The team built and verified Chrono Break DB, which rolls game logs back to a point in time

Chrono Break is what esports uses to rewind a live match after a critical bug.  
Teammates wrote the code and ran the benchmarks; the results below belong to the whole team.

## Week by week

| Week | Question | Result |
|---|---|---|
| 1 | How do Compaction, TTL, and Merge work? | Merge vs. Put benchmarks |
| 2 | How should keys be shaped for log data? | Compaction Filter cost, key-format comparison |
| 3 | How does key format affect reads? | Three formats compared; endianness mismatch found |
| 4 | Can this become a Chrono Break DB? | LogDB API, TTL filter, snapshots, point-in-time rollback |

## Evidence

### Week 1 · Merge vs. Put

| Metric | Merge (stringappend) | Merge (uint64add) | Put (updaterandom) | Put (appendrandom) |
|---|---:|---:|---:|---:|
| Throughput (ops/sec) | 28,694 | 28,724 | 43,007 | 39,574 |
| Compactions | 88 | 78 | 74 | 182 |
| P99 latency | 570.48 | 569.85 | 1183.78 | 1053.55 |

- Put was faster on average but had large P99 spikes
- Merge was slower but steady, and halved compactions for string appends

![MergeOperator benchmark](/image/rocksdb-chronobreak/merge-benchmark.jpg)

### Week 2 · Same benchmark, different results

The same db_bench run took 10–60 seconds for one teammate and 1–2 for another.  
The cause was a single `--threads=10` flag; from then on we pinned parameters explicitly.

### Week 3 · Read performance by key format

1. key = `sessionId`, value = `(timestamp, log)[]` (accumulated with Merge)
2. key = `(sessionId, timestamp)`, value = `log`
3. Format 2 plus a `sessionId` prefix bloom filter

- Reading a whole session, format 1 was fastest, about 2.3× the throughput of 2 and 3
- With many values per session, the prefix bloom filter barely helped
- Found that db_bench assumes big endian while `DecodeFixed64` assumes little endian

![Read performance by key format](/image/rocksdb-chronobreak/keyformat-read.jpg)

### Week 4 · Chrono Break DB

- Keys as `(uint32 sessionId, uint32 timestamp, string type)`, big-endian so byte order matches time order
- Values in MessagePack instead of JSON
- A TTL compaction filter; game-state snapshots with create, read, and restore
- Point-in-time rollback using `DeleteRange` and bounded iterators

Check: stored 300 real match-log entries, rolled back 30,000 ms → matched the original through entry 65, and later writes were stored correctly.  
A snapshot restore recovered 2,933 entries, all matching the original.

![Dump before Chrono Break](/image/rocksdb-chronobreak/chronobreak-dump.jpg)

</div>

---

[← Projects](/projects/) · [Portfolio](/portfolio/)
