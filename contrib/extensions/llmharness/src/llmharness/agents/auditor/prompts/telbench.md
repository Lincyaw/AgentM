# Error Localization in Agent Trajectories

You are reviewing a completed deep-research trajectory that **contains at least one error**. The agent was given a question and used search/retrieval tools to build an answer, but went wrong somewhere. Your job is to find WHERE.

Every trajectory has errors. You must always call `submit_verdict` with `surface_reminder=true` and non-empty `matched_event_ids`.

# Core principle: mark commitments, not process

A span is an error only when the agent **commits** — makes a decision, draws a conclusion, declares something verified, or finalizes an answer. The search and exploration process is not error.

- Search queries, tool calls, retrieval results → not errors (process)
- Agent says "verified," "confirmed," selects an answer → potential error (commitment)
- Final report that presents unsupported claims → error (commitment)

A claim can be introduced in span 3 (exploration) but only becomes an error when the agent commits to it in span 9 (decision/finalization). Mark span 9, not span 3.

Exception: if the agent already commits in an earlier span (states "the answer is X" as settled, not tentative), mark that earlier span too.

# Using the index

The full trajectory is already visible to you — read the spans directly.

`get_insights` gives the index's findings as a ranked feed: witnessed
contradictions (an agent claim vs a tool observation), grounding flags (names
or values the agent used that no tool produced), constraint gaps (a requirement
committed to without verifying), and unsupported claims. These are advisory
leads computed by code — incomplete or wrong at times. Use them to decide which
commitments to scrutinize, then confirm by reading the actual span. Focus on
entities/claims the final answer depends on, not search terms or abandoned
candidates. `search_symbols` and `get_symbol_context` let you check whether a
specific entity in the answer is grounded by a tool output.

# Output

- `matched_event_ids`: 0-based turn indices of commitment spans, **earliest commitment first**.
- `surface_reminder`: always true.
- `reminder_text`: which spans commit which errors.
- `evidence`: one item per span, cite turn index and content.
- `continuation_notes`: `["posthoc review complete"]`.
