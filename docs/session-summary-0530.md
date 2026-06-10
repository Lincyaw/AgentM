# Session Summary: 2026-05-30

## 背景

延续 cognitive-audit harness 的 investigation（`docs/cognitive-audit-investigation.md`）。
之前 400 次干预实验（Doubao auditor + Doubao agent）结果是 helped=0。

## 本次做了什么

### 1. 全维度数据分析

对已有实验数据做了 10 个维度的系统分析（不只是 cherry-pick 几个 case）：

- 介入次数 vs 结果
- Reminder 主题（coverage_nudge / contradiction_flag / error_hint 等）vs 精度
- Fork turn timing vs 结果
- 跨模型对比（Doubao vs GLM-5.1 同 case）
- GT service 提及率 vs 精度
- Auditor firing 序列质量
- Agent 答案结构（服务数量、false positive 分布）
- Auditor focus drift 轨迹

**关键发现**：contradiction_flag 是唯一与 helped 相关的 reminder 类型，但 Doubao auditor
极少产生它（4/99 cases）。

### 2. DeepSeek-V4-pro 10 case 对比实验

用 DSv4pro 同时作为 auditor 和 continuation agent 跑了 10 个 case（skip-extractor 模式）。

**结果**：

| Model | fired | ctrl_exact | iv_exact | helped | harmed | delta | avg_prec | avg_recall |
|---|---|---|---|---|---|---|---|---|
| doubao-noext | 10/10 | 2 | 1 | 0 | 1 | **-1** | 0.733 | 0.550 |
| glm51 | 9/10 | 2 | 2 | 1 | 1 | **+0** | 0.950 | 0.600 |
| dsv4pro-noext | 7/10 | 2 | 3 | 1 | 0 | **+1** | 1.000 | 0.650 |

DSv4pro 是唯一有正向 delta 的模型。precision=1.000（零 false positive）。

**关键 case KQHDBBF**（GT=recommendation+profile）：Doubao 和 GLM 干预后都 harmed（丢了
profile），DSv4pro 干预后找到了 profile+recommendation（exact match）。DSv4pro auditor 的
reminder 精准指出了 agent 在浪费时间探索 schema，而不是泛泛地说"你还没看 X service"。

**DSv4pro auditor 的 continuation_notes 质量远超 Doubao**：维护结构化的 open-question 列表，
跟踪每个 turn 的具体发现，标记 resolved/pending。

数据已上传到 aegis blob：`shared:cases/dsv4pro-noext-10/`

详细报告：`docs/dsv4pro-10case-comparison.md`

### 3. CLI 改造：--agent-model 走 config.toml

之前 `--agent-model` 用 OPENAI_* env vars 解析，`--harness-model` 用 config.toml profile。
改成两者统一走 config.toml，去掉 `--agent-provider` 参数。消除 env var 泄漏风险。

**文件**：
- `contrib/scenarios/rca/src/agentm_rca/eval/replay_fork/cli.py`
- `contrib/scenarios/rca/src/agentm_rca/eval/agent.py`
- `~/.agentm/config.toml` 新增 `litellm-dsv4pro` profile

### 4. Event-driven trigger 系统（实现 + review + simplify）

**问题**：auditor 用固定 cadence（turn % 5 == 0）触发，agent 在非 cadence turn 提交时
auditor 永远不会被触发。DSv4pro 的 continuation_notes 里明确记录了问题但来不及 fire。

**方案**：pluggable trigger atom 替代固定 cadence。`TriggerRegistry` 发布在 `ExtensionAPI`，
trigger atoms 注册 `should_fire(ctx)` 谓词，runner 每 turn 做 OR 组合评估。

**实现**（在 worktree `worktree-agent-a5cde61c4afa33947`）：

新文件：
- `llmharness/audit/triggers.py` — TriggerContext, TriggerDecision, Trigger Protocol, TriggerRegistry
- `llmharness/extensions/trigger_cadence.py` — 每 N turn 触发（替代硬编码 cadence）
- `llmharness/extensions/trigger_on_submission.py` — agent 调 submit_* 时触发

修改：HarnessRunner, offline_driver, fork_tree, adapter（线程 trigger_registry）

**Review findings（已修复）**：
- live 路径没传 tool_names_called → 修复，sync/async 都传了
- evaluate() 没 try/except → 修复，和 AuditCheckRegistry.run_all 对齐
- terminal tool 检测取最后一个 → 改成 frozenset[str] 全集

**Simplify findings（已修复）**：
- TriggerContext.messages 和 .latest_assistant_message 是 dead state → 删掉
- tool_names_from_message() 重复 → 提取共享 helper
- registered_triggers() 分配 tuple 做 truthiness check → 加 __bool__
- offline_driver messages[:prefix_len] 双重 slice → 复用 prefix
- evaluate() 返回 reasons list 没人用 → 简化为 (bool, bool)，内部 log

**验证**：trigger_on_submission 在 KQHREG case 上成功触发（之前 cadence-only 没触发），
auditor 得以在 agent 提交时 review 最终答案。

状态：3 commits 在 worktree 分支，已 cherry-pick 到 replay-fork worktree。
待 merge 到 main。

### 5. Reminder preamble 改成普通语言

`[harness advisory — meta-injection from cognitive audit, not from the human user]`
→ `[system reminder — automated review of your investigation so far]`

主 agent 无法感知 "harness advisory" / "cognitive audit" 这些内部术语。

commit: `310c5300` on main

### 6. max_tool_calls_per_turn

DSv4pro thinking model 在某些 turn 爆发式发出 479 个 parallel tool calls，
导致 context 从 38K 暴涨到 284K tokens，单 turn 卡 30 分钟。

在 `LoopConfig` 加了 `max_tool_calls_per_turn`。超出部分不执行，每个 dropped tool call
返回 error result 告诉 agent "拆分到多个 turn"。replay-fork CLI 里设了 20。

commit: `345372d2` on main

### 7. 带 extractor 的 DSv4pro 实验（失败）

尝试用 DSv4pro 做 extractor + auditor + agent。发现 DSv4pro 做 extractor 太慢：
- 第 1 次 firing（~3K input）：75 秒
- 第 8 次 firing（~8K input）：5 分 51 秒
- 原因：reasoning tokens 占 86.7%（10K reasoning vs 1.5K 有用 output）
- 信息抽取任务不需要 deep reasoning

**结论**：extractor 应该用快模型（Doubao），auditor 用 DSv4pro。代码已有
`provider_extractor` / `provider_auditor` 分离支持。

### 8. aegis-ui 部署

为 skip-extractor 模式加了 graph viewport fallback（"No-extractor mode" 提示
替代误导性的 "No graph snapshot" 错误）。

deployed: `byte-20260530-noext-graph-fallback-r4`

## 待做

1. **Merge trigger 分支到 main** — 3 commits 在 worktree
2. **拆分 harness model CLI** — 加 `--extractor-model` / `--auditor-model` 两个 flag，
   支持 extractor=Doubao + auditor=DSv4pro 的配置
3. **跑完整 100 case** — DSv4pro auditor + DSv4pro agent, skip-extractor, concurrency 15
4. **跑 DSv4pro auditor + Doubao agent** — 分离 auditor 质量 vs agent 能力的贡献
5. **跑 Doubao extractor + DSv4pro auditor** — 验证 extractor 用快模型 + auditor 用强模型的组合

## 文件索引

| 路径 | 描述 |
|---|---|
| `docs/cognitive-audit-investigation.md` | 原始调查报告（更新中） |
| `docs/dsv4pro-10case-comparison.md` | DSv4pro 10 case 对比报告 |
| `docs/session-summary-0530.md` | 本文件 |
| `runs/dsv4pro-noext-10/` | DSv4pro skip-extractor 实验数据 |
| `runs/dsv4pro-trigger-1case/` | trigger 系统验证实验 |
| `runs/dsv4pro-ext-10/` | DSv4pro with-extractor 实验（失败/不完整） |
| `~/.agentm/config.toml` | 新增 litellm-dsv4pro profile |
| aegis blob `shared:cases/dsv4pro-noext-10/` | 上传到平台的可视化数据 |
