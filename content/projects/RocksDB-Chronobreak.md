---
title: "RocksDB Chrono Break DB"
title_en: "RocksDB Chrono Break DB"
type: page
description: "RocksDB의 Compaction · TTL · Merge 분석에서 출발해, 게임 로그를 특정 시점으로 되돌리는 로그 DB를 설계 · 구현한 4주 팀 프로젝트."
description_en: "A four-week team project that started with analyzing RocksDB's Compaction, TTL, and Merge and ended with a game-log DB that can roll a session back to a point in time."
tags: ["DKU"]
---

<div class="lang-ko">

<a id="top"></a>

**기간** · 2025년 1학기, 4주(주차별 발표)  
**과목** · 단국대학교 오픈소스SW분석(빅데이터) · 4조  
**역할** · 팀 리더 · 프로젝트 구상 · 발표  
**팀** · 3명 (실험 · 벤치마크, 실험 자료 분석 · 발표자료는 팀원 담당)  
**상태** · 수업 프로젝트 완료 · C++ / RocksDB

[**Case Study ↓**](#case-study) · [**코드 ↗**](https://github.com/alsspp01/RocksDB_ChronobreakFilter)

---

<a id="case-study"></a>
## Case Study

### Problem

RocksDB가 LevelDB보다 실제로 어떤 점에서 더 실용적인지, 그리고 그 기능들이 실제 시스템에서 어떻게 쓰일 수 있는지를 4주 동안 확인하는 과제였습니다.  
기능을 하나씩 비교하는 데서 끝나지 않고, 매주 결과를 바탕으로 다음 주의 질문을 정해 하나의 응용 시나리오로 이어가는 방식을 택했습니다.

응용 시나리오는 e스포츠의 **Chrono Break**였습니다.  
경기 도중 치명적인 버그가 생겼을 때 경기를 특정 시점으로 되돌리는 기능으로, 이를 위해서는 그 순간의 모든 상태를 기록하고 원하는 시점까지 정확히 되돌릴 수 있는 로그 저장소가 필요합니다.

### My Contribution

- 주제를 정할 때 "RocksDB로 대전 게임 로그 저장 시스템 만들기"(전투 로그 Merge, 세션 저장 · 복구)를 제안했고, 이 아이디어가 4주차 Chrono Break DB의 바탕이 됐습니다.
- 팀 리더로서 주차별 목표와 실험 방향을 정했습니다.
- 매주 실험 결과를 해석해 다음 주의 질문으로 정리했습니다. 예를 들어 2주차의 key format 논의를 3주차의 read 성능 비교 실험으로, 3주차 결과를 4주차의 DB 설계로 이어갔습니다.
- 주차별 발표를 맡아 팀의 분석과 구현 결과를 설명했습니다.

벤치마크 실행, 코드 구현, 실험 자료 분석과 발표자료 제작은 팀원들이 맡았습니다.  
아래 결과는 팀 전체의 결과입니다.

### 주차별 흐름

| 주차 | 질문 | 결과 |
|---|---|---|
| 1 | Compaction · TTL · Merge는 각각 어떻게 동작하는가 | MergeOperator와 Put 방식의 처리 속도 · 지연 · Compaction 횟수 비교 |
| 2 | 실제 로그를 저장한다면 어떤 key 형식이 맞는가 | Compaction Filter 영향 측정, `sessionId` / `sessionId:timestamp` key 설계 비교 |
| 3 | key 형식과 prefix bloom filter는 read 성능에 어떤 영향을 주는가 | key 형식 3종의 read · write 성능 비교, Endian 불일치 문제 확인 |
| 4 | 지금까지의 기능으로 Chrono Break DB를 만들 수 있는가 | LogDB API, TTL Compaction Filter, Snapshot, Iterator 기반 dump와 Chrono Break 구현 · 검증 |

### Outcome & Evidence

**1주차 · Merge와 Put의 차이**

| 항목 | Merge (stringappend) | Merge (uint64add) | Put (updaterandom) | Put (appendrandom) |
|---|---:|---:|---:|---:|
| 처리량 (ops/sec) | 28,694 | 28,724 | 43,007 | 39,574 |
| Compaction 횟수 | 88 | 78 | 74 | 182 |
| P99 latency | 570.48 | 569.85 | 1183.78 | 1053.55 |

Put 방식이 평균 처리 속도는 빨랐지만 P99 지연이 크게 튀었고, Merge 방식은 느린 대신 지연이 일정했습니다.  
문자열을 덧붙이는 경우 Merge는 변경분만 기록해 Compaction 횟수가 절반 이하였습니다.

![MergeOperator 벤치마크](/image/rocksdb-chronobreak/merge-benchmark.jpg)

**2주차 · 같은 벤치마크, 다른 결과**

팀원마다 같은 db_bench 실험의 실행 시간이 10~60초와 1~2초로 크게 달랐습니다.  
원인을 추적한 결과 `--threads=10` 파라미터 하나가 차이를 만든다는 것을 확인했고, 이후 실험에서는 파라미터를 명시해 조건을 맞췄습니다.

**3주차 · key 형식에 따른 read 성능**

세 가지 key 형식을 비교했습니다.

1. key = `sessionId`, value = `(timestamp, log)[]` (Merge로 누적)
2. key = `(sessionId, timestamp)`, value = `log`
3. 2번과 같은 key에 `sessionId` prefix bloom filter 적용

한 세션의 로그 전체를 읽는 경우 1번이 가장 빨랐고, 2 · 3번과의 throughput 차이는 약 2.3배였습니다.  
3번은 prefix가 동작했지만, 한 세션에 많은 값이 몰려 있는 조건에서는 2번과 차이가 거의 없었습니다.  
이 과정에서 RocksDB의 db_bench는 big endian, `DecodeFixed64`는 little endian을 가정한다는 불일치도 확인했습니다.

![key 형식별 read 성능](/image/rocksdb-chronobreak/keyformat-read.jpg)

**4주차 · Chrono Break DB**

- key를 `(uint32 sessionId, uint32 timestamp, string type)`으로 정하고, 사전순 정렬이 시간순과 일치하도록 big endian으로 인코딩
- value는 JSON 대신 MessagePack으로 저장
- TTL 기반 Compaction Filter, 게임 상태 Snapshot 생성 · 조회 · 복원
- `DeleteRange`와 범위 제한 Iterator로 특정 시점 이후의 로그를 지우는 `ChronoBreak` / `ChronoBreakAbsolute` 구현

검증에서는 실제 경기 로그를 300번째까지 저장한 뒤 30,000ms 전으로 Chrono Break를 실행했습니다.  
실행 직후 dump는 원본의 65번째 로그까지와 일치했고, 이후 301번째부터 다시 저장한 로그도 이어서 기록됐습니다.  
Snapshot 복원 테스트에서는 2,933개 엔트리를 복원해 원본과 모두 일치하는 것을 확인했습니다.

![Chrono Break 직전 dump 비교](/image/rocksdb-chronobreak/chronobreak-dump.jpg)

### Constraints & Trade-offs

- 수업 기간 안에 진행한 4주 프로젝트로, 단일 머신의 db_bench와 자체 테스트 기준의 결과입니다. 실제 서비스 규모의 부하 테스트는 하지 않았습니다.
- 3주차 비교는 "한 세션의 로그 전체를 읽는다"는 가정 아래의 결과이며, 다른 읽기 패턴에서는 결론이 달라질 수 있습니다.
- 성능 수치는 위의 실험 조건에서만 의미가 있고, RocksDB 일반의 성능 비교로 확장하지 않습니다.

[ChronobreakFilter 코드](https://github.com/alsspp01/RocksDB_ChronobreakFilter) · [db_bench 실험용 RocksDB fork](https://github.com/alsspp01/OpensourceBigdata)

[↑ Top](#top)

</div>

<div class="lang-en" style="display:none">

<a id="top-en"></a>

**Period** · Spring 2025, four weeks (weekly presentations)  
**Course** · Open-Source SW Analysis (Big Data), Dankook University · Team 4  
**Role** · Team Lead · Project Concept · Presenter  
**Team** · 3 members (teammates handled experiments and benchmarks, data analysis, and slides)  
**Status** · Course project completed · C++ / RocksDB

[**Case Study ↓**](#case-study-en) · [**Code ↗**](https://github.com/alsspp01/RocksDB_ChronobreakFilter)

---

<a id="case-study-en"></a>
## Case Study

### Problem

The assignment was to spend four weeks finding out what actually makes RocksDB more practical than LevelDB and how its features could be used in a real system.  
Instead of comparing features in isolation, we used each week's results to set the next week's question and built toward one application scenario.

That scenario was esports **Chrono Break**: rolling a live match back to an earlier moment after a critical bug.  
It requires a log store that records every state change and can return precisely to a chosen point in time.

### My Contribution

- When choosing a topic, I proposed "a game-log storage system built on RocksDB" (merging combat logs, saving and restoring sessions), which became the basis of the week 4 Chrono Break DB.
- As team lead, I set the weekly goals and experiment direction.
- I turned each week's results into the next week's question, for example taking the week 2 key-format discussion into a week 3 read-performance experiment, and the week 3 results into the week 4 DB design.
- I gave the weekly presentations explaining the team's analysis and implementation.

My teammates ran the benchmarks, wrote the code, analyzed the experiment data, and built the slides.  
The results below belong to the whole team.

### Week by week

| Week | Question | Result |
|---|---|---|
| 1 | How do Compaction, TTL, and Merge work? | Throughput, latency, and compaction counts for MergeOperator vs. Put |
| 2 | Which key format fits real log data? | Measured Compaction Filter overhead; compared `sessionId` and `sessionId:timestamp` keys |
| 3 | How do key format and a prefix bloom filter affect reads? | Read and write performance of three key formats; found an endianness mismatch |
| 4 | Can these features make a Chrono Break DB? | Built and verified a LogDB API, TTL compaction filter, snapshots, iterator dumps, and Chrono Break |

### Outcome & Evidence

**Week 1 · Merge vs. Put**

| Metric | Merge (stringappend) | Merge (uint64add) | Put (updaterandom) | Put (appendrandom) |
|---|---:|---:|---:|---:|
| Throughput (ops/sec) | 28,694 | 28,724 | 43,007 | 39,574 |
| Compactions | 88 | 78 | 74 | 182 |
| P99 latency | 570.48 | 569.85 | 1183.78 | 1053.55 |

Put was faster on average but had large P99 spikes; Merge was slower but steadier.  
For string appends, Merge wrote only the change and needed less than half as many compactions.

![MergeOperator benchmark](/image/rocksdb-chronobreak/merge-benchmark.jpg)

**Week 2 · Same benchmark, different results**

The same db_bench run took 10–60 seconds for one teammate and 1–2 seconds for another.  
We traced the gap to a single `--threads=10` parameter and specified parameters explicitly from then on.

**Week 3 · Read performance by key format**

We compared three formats:

1. key = `sessionId`, value = `(timestamp, log)[]` (accumulated with Merge)
2. key = `(sessionId, timestamp)`, value = `log`
3. Format 2 with a `sessionId` prefix bloom filter

Reading a whole session's logs, format 1 was fastest, with about 2.3× the throughput of formats 2 and 3.  
The prefix filter in format 3 worked, but made almost no difference when many values sat under one session.  
We also found that RocksDB's db_bench assumes big endian while `DecodeFixed64` assumes little endian.

![Read performance by key format](/image/rocksdb-chronobreak/keyformat-read.jpg)

**Week 4 · Chrono Break DB**

- Keys as `(uint32 sessionId, uint32 timestamp, string type)`, big-endian encoded so lexicographic order matches time order
- Values stored as MessagePack instead of JSON
- A TTL-based compaction filter, plus snapshot creation, reads, and restore for game state
- `ChronoBreak` / `ChronoBreakAbsolute`, which delete logs after a point in time with `DeleteRange` and bounded iterators

To verify it, we stored a real match log up to its 300th entry and ran Chrono Break back 30,000 ms.  
The dump right after matched the original through the 65th log, and logs stored afterward from entry 301 were recorded correctly.  
A snapshot restore test recovered 2,933 entries, all matching the original.

![Dump comparison before Chrono Break](/image/rocksdb-chronobreak/chronobreak-dump.jpg)

### Constraints & Trade-offs

- A four-week course project, measured with db_bench and our own tests on a single machine; no production-scale load testing.
- The week 3 comparison assumes reading a whole session's logs; other read patterns could change the conclusion.
- The numbers hold only under these experiment conditions and are not a general RocksDB performance claim.

[ChronobreakFilter code](https://github.com/alsspp01/RocksDB_ChronobreakFilter) · [RocksDB fork used for db_bench experiments](https://github.com/alsspp01/OpensourceBigdata)

[↑ Top](#top-en)

</div>

---

[← Projects](/projects/) · [Portfolio](/portfolio/)
