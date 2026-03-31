# Claude Code 学习 × AgentM SDK 改造分析报告

**Date**: 2026-03-31
**Sources**: Claude Code CLI 源码逆向分析 (src/), AgentM SDK 接口审计
**Scope**: SDK 层改进方向，不含场景(Scenario)特定逻辑

---

## 一、Claude Code Prompt 设计模式（核心学习内容）

### 1. 结构化技巧清单

| 技巧 | 用法 | 效果 |
|------|------|------|
| **`=== CRITICAL ===` 标记** | 三等号大写标记绝对约束 | 视觉醒目，建立"告警级别"——普通指令用 markdown 标题，关键约束用 `=== ===` |
| **编号章节** | Coordinator prompt 用 `## 1. Your Role` 编号 | 长 prompt 中帮助 LLM 准确引用特定章节 |
| **XML Tag** | `<task-notification>`、`<example>`、`<commentary>` | 充当"类型标签"，使 LLM 区分用户消息和系统通知 |
| **决策表格** | Continue vs Spawn 决策矩阵 | 将模糊判断转化为可查表的规则 |
| **Good/Bad 对比** | 反面示例标注 `Anti-pattern`，正面标注 `Good` | 比单纯给正例更有效——LLM 学习"不要做什么" |
| **反合理化清单** | 预测 LLM 会说的具体借口，逐条打脸 | 比抽象地说"不要偷懒"有效 10 倍 |
| **威慑声明** | "caller may spot-check by re-running your commands" | 即使不真的校验，也能改善行为 |
| **Purpose Statement** | 委派任务附带目的说明 | 让 worker 自主判断深度和重点 |
| **强制输出格式** | `VERDICT: PASS/FAIL/PARTIAL` 机器可解析 | 结构化输出用于自动化流水线 |

### 2. 关键 Prompt 原文（可直接参考）

#### 2.1 Coordinator — "Never delegate understanding"

```
Never write "based on your findings" or "based on the research." These phrases 
delegate understanding to the worker instead of doing it yourself. You never 
hand off understanding to another worker.
```

> **AgentM 启示**：orchestrator prompt 应明确禁止懒惰委派。当前 RCA orchestrator prompt 里没有这条约束。

#### 2.2 Verification — 反合理化清单

```
=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
You will feel the urge to skip checks. These are the exact excuses you reach for:
- "The code looks correct based on my reading" — reading is not verification. Run it.
- "The implementer's tests already pass" — the implementer is an LLM. Verify independently.
- "This is probably fine" — probably is not verified. Run it.
- "I don't have a browser" — did you actually check for tools? If present, use them.
- "This would take too long" — not your call.
If you catch yourself writing an explanation instead of a command, stop. Run the command.
```

> **AgentM 启示**：AgentM 的 worker 也会出现类似的"偷懒"行为。可以在 worker prompt 模板中加入场景特定的反合理化清单。

#### 2.3 Read-only Agent — 三层权限声明

```
第一层 (Prompt):  === CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
                  - Creating new files (no Write, touch, or file creation of any kind)
                  - Modifying existing files (no Edit operations) ...

第二层 (Tool):    disallowedTools: [FILE_EDIT, FILE_WRITE, NOTEBOOK_EDIT]

第三层 (Reminder): criticalSystemReminder_EXPERIMENTAL (每轮重复注入)
```

> **AgentM 启示**：AgentM 当前 worker 的权限控制完全靠 tools 列表。应增加 prompt 层声明 + 中间件层重复注入。

#### 2.4 Coordinator — 上下文隔离声明

```
Workers can't see your conversation. Every prompt must be self-contained with 
everything the worker needs.
```

```
Brief the agent like a smart colleague who just walked into the room — it hasn't 
seen this conversation, doesn't know what you've tried, doesn't understand why 
this task matters.
```

> **AgentM 启示**：AgentM 的 orchestrator prompt 需要更强的上下文隔离意识教育。

#### 2.5 Explore Agent — 速度优先声明

```
NOTE: You are meant to be a fast agent that returns output as quickly as possible.
- Make efficient use of the tools
- Wherever possible you should try to spawn multiple parallel tool calls
```

> **AgentM 启示**：不同 task_type 的 worker 可以有不同的性能期望声明。scout 类似 Explore（速度优先），deep_analyze 类似 Plan（深度优先）。

---

## 二、SDK 接口改造建议

### 改造 1: 声明式 Agent 配置 — Markdown Frontmatter 格式

**现状**：AgentM 的 agent 配置在 `scenario.yaml` 的 `agents.worker` 块中，system prompt 在独立的 `.j2` 文件中。二者分离，不直观。

**Claude Code 做法**：

```markdown
---
name: code-reviewer
description: Review code for quality issues
tools: [Read, Grep, Glob, Bash]
disallowedTools: [Agent, Edit, Write]
model: sonnet
permissionMode: plan
maxTurns: 50
memory: project
color: red
background: true
---

你是一个代码审查专家...（system prompt body）
```

**建议方案**：

在 `config/scenarios/<name>/agents/` 目录下支持 `.md` 格式：

```markdown
---
name: scout
description: "Fast reconnaissance worker for initial investigation"
task_type: scout
tools: ["*"]
disallowed_tools: [dispatch_agent, check_tasks]
model: ${WORKER_MODEL}
max_steps: 15
timeout: 120
include_think_tool: true
---

You are a reconnaissance agent. Your job is to quickly investigate...
```

**SDK 接口变更**：

```python
# 新增: src/agentm/config/agent_loader.py

@dataclass(frozen=True)
class AgentDefinition:
    """Declarative agent definition, loaded from markdown or YAML."""
    name: str
    description: str
    task_type: str
    system_prompt: str
    tools: list[str] | None = None          # None = all tools
    disallowed_tools: list[str] | None = None
    model: str | None = None                 # None = inherit from scenario
    max_steps: int | None = None
    timeout: float | None = None
    include_think_tool: bool = True
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    memory_scope: MemoryScope | None = None

def load_agent_definitions(scenario_dir: Path) -> dict[str, AgentDefinition]:
    """Load agent definitions from agents/ subdirectory (markdown or yaml)."""
    ...
```

`WorkerLoopFactory` 改为接受 `AgentDefinition`：

```python
class WorkerLoopFactory:
    def __init__(
        self,
        agent_definitions: dict[str, AgentDefinition],  # 替代 scenario_config
        tool_registry: ToolRegistry,
        ...
    ) -> None

    def create_worker(self, agent_id: str, task_type: str) -> SimpleAgentLoop:
        defn = self.agent_definitions[task_type]
        tools = self._resolve_tools(defn)  # 应用白名单/黑名单
        prompt = defn.system_prompt  # 已从 .md body 加载
        ...
```

---

### 改造 2: 工具白名单/黑名单机制

**现状**：`WorkerLoopFactory` 通过 `tool_registry.get_tools(config.tools)` 获取工具列表，再加 `extra_tools`。没有黑名单。

**Claude Code 做法**：三层过滤——全局禁止 → 黑名单 → 白名单。

**建议方案**：

```python
# 新增: src/agentm/harness/tool_filter.py

# SDK 全局禁止列表 — worker 不能使用调度类工具（防止递归）
WORKER_DISALLOWED_TOOLS: frozenset[str] = frozenset({
    "dispatch_agent",
    "check_tasks",
    "inject_instruction",
    "abort_task",
})

def resolve_tools(
    available_tools: list[Tool],
    *,
    allowed: list[str] | None = None,       # None or ["*"] = all
    disallowed: list[str] | None = None,     # 从 allowed 中移除
    global_disallowed: frozenset[str] = WORKER_DISALLOWED_TOOLS,
) -> list[Tool]:
    """
    Three-layer tool filtering:
    1. Remove global_disallowed (SDK-level safety)
    2. Remove disallowed (agent-level blacklist)
    3. If allowed is specified (not None and not ["*"]), keep only those
    """
    result = [t for t in available_tools if t.name not in global_disallowed]

    if disallowed:
        deny_set = set(disallowed)
        result = [t for t in result if t.name not in deny_set]

    if allowed is not None and allowed != ["*"]:
        allow_set = set(allowed)
        result = [t for t in result if t.name in allow_set]

    return result
```

在 `WorkerLoopFactory.create_worker()` 中使用：

```python
all_tools = self.tool_registry.get_all_tools() + self.extra_tools
tools = resolve_tools(
    all_tools,
    allowed=defn.tools,
    disallowed=defn.disallowed_tools,
)
```

---

### 改造 3: 权限模式 (PermissionMode)

**现状**：AgentM 没有权限控制概念。

**Claude Code 做法**：

| 模式 | 行为 |
|------|------|
| `default` | 每次危险操作需审批 |
| `plan` | 只读，agent 只规划不执行 |
| `acceptEdits` | 自动接受文件编辑 |
| `bypassPermissions` | 跳过所有权限检查 |
| `bubble` | 权限请求向上冒泡到 orchestrator |

**建议方案（简化版）**：

AgentM 作为 SDK，主要关注 agent 的行为模式约束，而非 UI 交互审批：

```python
# 新增: src/agentm/harness/permissions.py

class PermissionMode(str, Enum):
    """Agent permission mode — controls behavioral constraints."""
    DEFAULT = "default"           # Normal operation
    READONLY = "readonly"         # No tool calls that modify state
    SUPERVISED = "supervised"     # Tools execute but results are logged for review
    UNRESTRICTED = "unrestricted" # No constraints

@dataclass(frozen=True)
class PermissionPolicy:
    """Runtime permission policy for an agent."""
    mode: PermissionMode
    readonly_tools: frozenset[str] = frozenset()  # Tools treated as read-only
    
    def can_execute(self, tool_name: str) -> bool:
        if self.mode == PermissionMode.UNRESTRICTED:
            return True
        if self.mode == PermissionMode.READONLY:
            return tool_name in self.readonly_tools
        return True  # DEFAULT and SUPERVISED allow all
```

在 `SimpleAgentLoop` 的 tool execution 阶段注入检查：

```python
# In SimpleAgentLoop._execute_tool():
if not self._permission_policy.can_execute(tool_name):
    return f"Permission denied: tool '{tool_name}' is not allowed in {self._permission_policy.mode} mode"
```

---

### 改造 4: Agent Memory（Per-Agent 持久化记忆）

**现状**：AgentM 有 `MemoryVault`（全局共享），但没有 per-agent memory。

**Claude Code 做法**：

```
三层 scope:
- user:    ~/.claude/agent-memory/<agentType>/     (跨项目)
- project: <cwd>/.claude/agent-memory/<agentType>/ (git tracked, 团队共享)
- local:   <cwd>/.claude/agent-memory-local/<agentType>/ (机器特定)

memory prompt 在 getSystemPrompt() 闭包中尾部追加:
  systemPrompt + '\n\n' + loadAgentMemoryPrompt(agentType, scope)

loadAgentMemoryPrompt 生成内容:
  1. 行为指令（怎么读/写/组织 memory）
  2. Scope 注释（user → 保持通用 / project → 项目针对性）
  3. MEMORY.md 当前索引内容（上限 200 行）
```

**建议方案**：

```python
# 新增: src/agentm/harness/agent_memory.py

class MemoryScope(str, Enum):
    AGENT = "agent"       # Per-agent, per-scenario run
    SCENARIO = "scenario" # Shared across agents within scenario
    PROJECT = "project"   # Persistent across runs (git tracked)

@dataclass(frozen=True)
class AgentMemoryConfig:
    scope: MemoryScope
    base_dir: Path
    max_index_lines: int = 200

def load_agent_memory_prompt(
    agent_type: str,
    config: AgentMemoryConfig,
) -> str:
    """Build memory prompt with behavioral instructions + current index."""
    memory_dir = config.base_dir / agent_type
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = memory_dir / "MEMORY.md"
    index_content = ""
    if index_file.exists():
        lines = index_file.read_text().splitlines()
        if len(lines) > config.max_index_lines:
            index_content = "\n".join(lines[:config.max_index_lines])
            index_content += f"\n... (truncated, {len(lines) - config.max_index_lines} lines omitted)"
        else:
            index_content = "\n".join(lines)

    scope_note = {
        MemoryScope.AGENT: "This memory is agent-scoped — only you access it.",
        MemoryScope.SCENARIO: "This memory is scenario-scoped — shared with other agents in this run.",
        MemoryScope.PROJECT: "This memory is project-scoped — persists across runs.",
    }[config.scope]

    return f"""# Persistent Agent Memory

You have a persistent memory system at `{memory_dir}/`.
{scope_note}

## Current Memory Index
{index_content if index_content else "(empty)"}

## How to Use
- Write memory files as markdown with YAML frontmatter
- Update MEMORY.md as the index (one line per entry, under 150 chars)
- Only save non-obvious learnings that will help in future runs
"""
```

在 `WorkerLoopFactory.create_worker()` 中注入：

```python
if defn.memory_scope is not None:
    memory_prompt = load_agent_memory_prompt(task_type, memory_config)
    system_prompt = system_prompt + "\n\n" + memory_prompt
```

---

### 改造 5: Prompt 模板改进 — 应用 Claude Code 模式

**现状问题**：AgentM 的 `.j2` prompt 模板缺少结构化约束技巧。

**建议在 prompt 模板中引入的模式**：

#### 5.1 Orchestrator Prompt 增加四条核心原则

```jinja2
{# config/scenarios/<name>/prompts/orchestrator_system.j2 #}

## Core Principles

1. **Never delegate understanding.** When a worker reports findings, YOU must 
   synthesize them before directing follow-up work. Never write "based on your 
   findings, do X" — write prompts that prove you understood: include specific 
   details, file paths, evidence.

2. **Parallelism is your superpower.** Launch independent workers concurrently.
   Read-only tasks run in parallel freely. Write-heavy tasks serialize per resource.

3. **Workers can't see your conversation.** Every dispatch_agent task must be 
   self-contained. Brief workers like a colleague who just walked into the room.

4. **Include a purpose statement.** Tell workers WHY you need this, so they can 
   calibrate depth:
   - "This is initial reconnaissance — report what you find, don't go deep."
   - "I need this to make a final decision — be thorough and precise."
```

#### 5.2 Worker Prompt 增加权限边界声明

```jinja2
{# For read-only workers (scout) #}
=== CRITICAL: READ-ONLY MODE ===
You are a reconnaissance agent. You MUST NOT:
- Modify any system state
- Execute commands that change configurations
- Write to any persistent storage

Your role is EXCLUSIVELY to investigate and report findings.

{# For write-capable workers (deep_analyze, verify) #}
=== OPERATIONAL BOUNDARIES ===
You may execute diagnostic commands and temporary modifications.
You MUST NOT:
- Make permanent configuration changes without orchestrator approval
- Restart production services
```

#### 5.3 增加反合理化清单（按 task_type 定制）

```jinja2
{# For verify task_type #}
=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
You will feel the urge to skip verification steps:
- "The logs look normal" — normal at what time window? Check the incident time.
- "This metric is within range" — within range for this service under this load?
- "The previous agent already checked this" — you are the independent verifier. Verify independently.
- "There's too much data to analyze" — sample strategically, don't skip.
If you catch yourself narrating instead of querying, stop. Run the query.
```

---

## 三、SDK 现有问题修复建议（来自 sdk-review agent）

以下问题不直接来自 Claude Code 学习，但在审计过程中发现的 SDK 缺陷：

### CRITICAL

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| 1 | `Middleware` Protocol vs `MiddlewareBase` class 不一致 | `protocols.py` vs `middleware.py` | 删除 `protocols.py` 中的 `Middleware` Protocol，或让 `SimpleAgentLoop` 依赖 Protocol 而非具体类 |
| 2 | `CompressionMiddleware._summarize_messages` 同步调用阻塞事件循环 | `middleware.py` | 改用 `await model.ainvoke()` |

### HIGH

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| 3 | `asyncio.get_event_loop()` 已废弃 | `runtime.py` `wait_any` | 改用 `asyncio.get_running_loop()` |
| 4 | `RunConfig.timeout` 字段从未被执行 | `types.py`, `simple.py` | 在 `SimpleAgentLoop.stream()` 中加 `asyncio.timeout()` |
| 5 | `AgentEvent.timestamp` 永远是空字符串 | `types.py` | 在 `_emit_event` 中填充 `datetime.now(UTC).isoformat()` |
| 6 | `dispatch_agent` fire-and-forget task 异常被静默吞没 | `orchestrator.py` | 添加 `task.add_done_callback` 记录异常 |
| 7 | `SetupContext.vault` 用具体类而非 Protocol | `scenario.py` | 改为 `NoteReader | None` |

### MEDIUM

| # | Issue | Fix |
|---|-------|-----|
| 8 | `Tool` dataclass 可变，应 `frozen=True` | 加 `frozen=True` |
| 9 | `Message` type alias 泄漏 LangChain 依赖 | 定义纯 SDK `Message` 类型 |
| 10 | `Tool.ainvoke` 对同步函数未做线程卸载 | 用 `asyncio.to_thread()` |
| 11 | `AgentRuntime._agents` 无 GC | 完成的 agent 在可配置时间后清理 |
| 12 | `check_tasks.request` 参数是死代码 | 删除或使用 |

---

## 四、优先级排序

### Phase 1: 立即可做（纯增量，不破坏现有接口）

1. **工具黑名单** — 在 `WorkerLoopFactory` 中加 `resolve_tools()` 三层过滤
2. **Prompt 模板增强** — 在现有 `.j2` 模板中加入 Claude Code 的结构化技巧
3. **修复 CRITICAL/HIGH 问题** — Protocol 一致性、阻塞调用、废弃 API

### Phase 2: 短期改造（需要新增接口但兼容现有）

4. **声明式 Agent 定义** — `AgentDefinition` dataclass + markdown loader
5. **权限模式** — `PermissionMode` enum + `PermissionPolicy` 检查
6. **Agent Memory** — per-agent memory scope + prompt 注入

### Phase 3: 中期演进（需要设计文档和 TDD）

7. **Orchestrator prompt 中的 "Continue vs Spawn" 框架** — 让 LLM 主动选择复用还是新建 worker
8. **Worker 主动上报** — `report_to_orchestrator` 工具替代纯轮询
9. **`criticalSystemReminder` 机制** — 每轮 LLM 调用前重复注入关键约束

---

## 五、Prompt 设计模式速查表

供编写 AgentM prompt 模板时参考：

```
┌─────────────────────────────────────────────────────┐
│               PROMPT 结构层级                        │
├─────────────────────────────────────────────────────┤
│ 1. 角色定位（一句话）                                 │
│    "You are a verification specialist."             │
│                                                     │
│ 2. === CRITICAL === 权限边界                         │
│    绝对不能违反的约束，每个放一行                      │
│                                                     │
│ 3. ## 编号章节 — 任务指引                             │
│    ## 1. Your Role / ## 2. Your Tools               │
│                                                     │
│ 4. 表格 — 决策矩阵                                   │
│    | Situation | Action | Why |                     │
│                                                     │
│ 5. Good/Bad 对比示例                                 │
│    // Anti-pattern — ...                            │
│    // Good — ...                                    │
│                                                     │
│ 6. 反合理化清单                                      │
│    "你会想说 X — 不要。做 Y。"                        │
│                                                     │
│ 7. 强制输出格式                                      │
│    ### Check: / **Command run:** / VERDICT: PASS    │
│                                                     │
│ 8. 威慑声明（可选）                                   │
│    "The caller may spot-check your output."         │
└─────────────────────────────────────────────────────┘
```
