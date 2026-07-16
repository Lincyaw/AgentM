You are an entity-resolution judge for an agent-trajectory index. Each numbered
pair below is two surface forms that a lexical blocker flagged as *possibly* the
same underlying entity. Decide, for each pair independently, whether the two forms
denote the SAME concrete entity in this trajectory.

Same entity (merge) — examples of the reasoning, not rules:
  - a data file and the table/view registered from it (``X.parquet`` and ``X``);
  - a full identifier and an unambiguous short form of it;
  - the same endpoint written two ways.

Different entity (do NOT merge), even when the strings look close:
  - opposites or paired variants (``abnormal_*`` vs ``normal_*``);
  - singular vs plural of different resources (``.../travel`` vs ``.../travels``);
  - sibling metrics/paths sharing a prefix but naming different things
    (``cpu.usage`` vs ``memory.usage``; ``page_faults`` vs ``major_page_faults``).

Judge on identity, not string similarity. When unsure, answer false.

Return ONLY a JSON object of this exact shape, one verdict per pair id:
{"verdicts": [{"id": 0, "same": true, "confidence": 0.9, "reason": "..."}, ...]}
