---
type: reference
tags: [trajectory-analysis, memory-extraction]
---

# What to Extract

Read each trajectory asking these four questions:

## 1. Where did the investigation go wrong, and why?

- Did agents pursue a dead end? What signal led them astray?
- Was evidence misinterpreted? What would correct interpretation look like?
- Were there signals available early that could have shortened the investigation?
- Was the final root cause correct? If not, what reasoning error led to the wrong conclusion?

## 2. Where did the investigation go right, and what principle was at work?

- Did an agent make a non-obvious connection between signals? What made it non-obvious?
- Was a hypothesis correctly rejected? What evidence pattern triggered the rejection?
- Did cross-signal correlation reveal something that single-signal analysis missed?

## 3. What diagnostic principle is this case an instance of?

Generalize from the specific to the transferable:

- Not "ts-auth-service had GC thrashing" but "resource exhaustion can manifest as normal
  average latency but elevated tail latency, causing intermittent upstream queuing"
- Not "the agent should have checked page_faults" but "memory pressure has multiple
  indicators with different visibility -- container-level usage, page faults, GC activity --
  and they can tell contradictory stories that only make sense when read together"

## 4. What anti-pattern did agents fall into?

Common traps to watch for:

- Anchoring on the first anomaly found instead of surveying all candidates
- Treating absence of evidence as evidence of absence
- Using a single metric to declare a service "healthy" without cross-validation
- Confirming a hypothesis despite unresolved contradictions
- Not revisiting rejected hypotheses when the current path fails
