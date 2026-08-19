---
title: "🪨 전투 로그를 RocksDB에 저장한다면"
title_en: "🪨 Designing Combat-Log Storage with RocksDB"
date: 2026-08-19
description: "Raw JSON, 압축, MsgPack, Flatten 전략을 비교하며 저장 포맷보다 시스템 동작을 먼저 보게 된 실험."
description_en: "What comparing Raw JSON, compression, MsgPack, and flattened storage taught me about looking beyond file formats."
type: "post"
tags: ["RocksDB", "C++", "Database", "Benchmark"]
---

<div class="lang-ko">

전투 로그를 저장한다고 하면 처음에는 포맷이 먼저 떠오릅니다.

JSON 그대로 저장할까?  
압축할까?  
MsgPack으로 바꿀까?  
아예 key-path 형태로 펼칠까?

저도 처음에는 **어떤 포맷이 제일 빠르고 작을까**가 핵심 질문이라고 생각했습니다.

RocksDB를 붙여 직접 실험해보니 그 질문만으로는 부족했습니다.

---

## 네 가지 저장 전략

### Raw JSON
JSON 문자열을 거의 그대로 저장합니다. 가장 단순하고 디버깅하기 쉽습니다.

### Compressed JSON
JSON을 zlib으로 압축해 저장합니다. 직관적으로는 디스크 사용량이 가장 작을 것 같았습니다.

### MsgPack
JSON 구조를 binary serialization 형태로 변환합니다. 문자열 표현의 일부 overhead를 줄일 수 있습니다.

### Flatten JSON
중첩 JSON을 key-path 구조로 펼칩니다. 특정 path 단위 구조를 다루기 쉽지만 변환과 merge 비용이 생깁니다.

---

## “압축하면 더 작겠지”가 항상 맞지 않았던 이유

재미있었던 결과 중 하나는 Compressed 전략이 기대만큼 단순하게 이기지 않았다는 점입니다.

압축에는 CPU 비용이 있습니다.

그리고 RocksDB는 내가 넣은 value만 그대로 파일 하나에 저장하는 시스템이 아닙니다.

MemTable, SST, compaction, metadata와 같은 내부 동작이 있고, 작은 데이터가 반복적으로 기록되고 merge되는 상황에서는 압축 포맷의 장점과 데이터베이스의 동작이 예상과 다르게 만날 수 있습니다.

즉,

> **“JSON 값이 30% 줄었으니 DB도 30% 작아질 것이다.”**

라고 바로 연결할 수 없었습니다.

---

## MergeOperator를 건드리면서 보인 것

기존 로그에 새로운 내용을 합치는 시나리오도 다뤘습니다.

그래서 각 포맷에 맞는 RocksDB `MergeOperator`를 구현하며 merge 비용을 비교했습니다.

- Raw JSON은 parsing이 필요하고
- 압축 데이터는 merge 전에 decompress가 필요하며
- MsgPack은 binary decode/encode가 필요하고
- Flatten은 path 처리와 구조 변환 비용이 생깁니다

**디스크 포맷 하나를 고르는 문제가 read/write/merge 전체 경로의 문제가 되었습니다.**

---

## TTL과 Compaction

전투 로그는 오래된 데이터를 영구 보관할 필요가 없는 경우가 많습니다.

그래서 TTL도 살펴봤습니다.

여기서 중요한 점은

> **“시간이 지나면 key가 바로 사라진다.”**

라고 생각하면 안 된다는 것이었습니다.

RocksDB의 TTL은 compaction과 연결되어 동작하기 때문에 논리적 만료와 실제 디스크 공간 회수가 같은 순간이 아닐 수 있습니다.

자연스럽게 compaction 방식과 데이터 lifecycle을 같이 보게 되었습니다.

---

## Benchmark에서 보고 싶었던 것

비교 항목은 하나가 아니었습니다.

- write
- merge
- read
- disk usage

어떤 전략은 쓰기가 빠르고, 어떤 전략은 읽을 때 decode 비용이 생기고, 어떤 전략은 표현은 작아도 merge가 비쌌습니다.

MsgPack이 여러 실험에서 좋은 균형을 보여준 경우가 있었지만  
그 결과도 **“MsgPack이 항상 정답”**이라는 뜻은 아니었습니다.

데이터 형태와 workload가 바뀌면 결과도 달라집니다.

---

## 저장 포맷보다 먼저 물어야 할 것

이 실험 이후 DB 포맷을 고를 때 먼저 보는 질문이 달라졌습니다.

> **데이터는 얼마나 자주 쓰이는가?**  
> **기존 값과 얼마나 자주 merge되는가?**  
> **전체를 읽는가, 일부만 읽는가?**  
> **얼마나 오래 보관하는가?**  
> **디스크와 CPU 중 무엇이 더 비싼가?**

포맷 이름만 비교해서는 답이 나오지 않습니다.

좋은 저장 방식은 **데이터 구조 + workload + lifecycle**이 같이 맞아야 합니다.

RocksDB를 공부하면서 가장 재미있었던 부분도 결국 그 지점이었습니다.

[🔗 GitHub Repository](https://github.com/alsspp01/OpensourceBigdata)

</div>

<div class="lang-en" style="display:none">

When someone says “store combat logs,” the first question often sounds like a format question.

Keep the JSON?  
Compress it?  
Use MsgPack?  
Flatten the structure into key paths?

That was my first instinct too:

**Which format is the fastest and smallest?**

After experimenting with RocksDB, that question turned out to be incomplete.

---

## Four storage strategies

### Raw JSON
Store the JSON representation almost as-is. Simple and easy to inspect.

### Compressed JSON
Compress JSON with zlib before storing it. The intuitive expectation is obvious: smaller values should mean less disk usage.

### MsgPack
Serialize the same structure into a binary format, reducing some of JSON's textual overhead.

### Flatten JSON
Turn nested JSON into key-path style structures. That can make path-oriented manipulation easier, but introduces transformation and merge costs.

---

## Why “compressed means smaller” was not the whole story

One of the more interesting results was that compression did not simply dominate the disk metric.

Compression has CPU cost.

More importantly, RocksDB is not a file where each value is written once and left alone.

There are MemTables, SST files, compaction, metadata, and repeated writes/merges.

With small values and repeated operations, the benefits of a compressed representation can interact with the storage engine in non-obvious ways.

So:

> **“The JSON value is 30% smaller, therefore the database will be 30% smaller.”**

was not a safe assumption.

---

## What became visible through MergeOperator

The experiment also involved appending new log information to existing values.

That led me to implement RocksDB `MergeOperator`s for the different representations.

- Raw JSON still needs parsing
- compressed data needs decompression before merge
- MsgPack needs binary decode/encode
- flattened structures need path and transformation work

A format decision became a question about the entire **read/write/merge path**.

---

## TTL and Compaction

Combat logs often have a natural lifetime.

Old data may not need to live forever, so I also explored TTL behavior.

A key lesson was that TTL should not be imagined as:

> **“The timestamp expires and the key instantly disappears from disk.”**

In RocksDB, TTL behavior is tied to compaction.

Logical expiration and physical space reclamation are not necessarily the same moment.

That pushed the experiment from serialization formats into the lifecycle of the storage engine itself.

---

## What I wanted from the benchmark

I compared more than write speed:

- write
- merge
- read
- disk usage

Some strategies paid more CPU during writes. Others added decode cost on reads. Some were compact in representation but more expensive during merge.

MsgPack often showed a useful balance in the tested datasets, but that still did not mean **“MsgPack is always the right answer.”**

Change the data shape or workload and the tradeoff changes too.

---

## The questions before choosing a format

After the experiment, I started asking different questions first:

> **How often is this data written?**  
> **How often is it merged with an existing value?**  
> **Do we read the whole object or only pieces of it?**  
> **How long should it live?**  
> **Which is more expensive for this system: disk or CPU?**

A format name cannot answer those questions by itself.

A good storage decision has to fit **the data structure, workload, and lifecycle together.**

That system-level view was the part of RocksDB I ended up enjoying most.

[🔗 GitHub Repository](https://github.com/alsspp01/OpensourceBigdata)

</div>
