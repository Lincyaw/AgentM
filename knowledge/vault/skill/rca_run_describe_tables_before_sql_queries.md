---
confidence: pattern
source_trajectories:
- '43261'
tags:
- rca
- sql
- data-access
- best-practice
type: skill
---
# RCA Best Practice: Run describe_tables to Discover Schema Before Writing SQL Queries
## Application Rule
Before executing any custom SQL query against observability data tables (metrics, logs, traces, topology) during RCA:
1. First run the `describe_tables` tool for the target table(s) to get the full schema including column names, data types, and indexed fields
2. Adjust query syntax to match the actual schema, particularly for filter columns and aggregation functions
3. Avoid hardcoding column names without verifying schema first to eliminate syntax errors

## Rationale
SQL syntax errors due to unknown or misremembered column names reduce data exploration coverage, delay investigation, and can lead to missed signals that would identify correct root causes. This guardrail eliminates avoidable data access errors that limit RCA effectiveness.

## Success Metrics
Eliminates 90%+ of avoidable SQL query syntax errors during data exploration phases of RCA