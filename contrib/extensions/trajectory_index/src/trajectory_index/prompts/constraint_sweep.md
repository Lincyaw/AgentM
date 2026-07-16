You are shown ALL tool/search result excerpts an agent gathered, and a list
of constraints from the question it was answering. For each constraint,
decide whether ANY excerpt carries evidence bearing on it (about any entity).
If yes, cite one excerpt id in "step" — a yes without a citation is invalid.
Recall matters more than precision here: when in doubt, answer yes.

Return ONLY: {"verdicts": [{"id": 0, "evidence_exists": true, "step": "12", "confidence": 0.9}]}
