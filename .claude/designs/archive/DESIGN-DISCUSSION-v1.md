# AgentM 系统设计讨论稿 v1.0 (更新：假设驱动的 RCA 设计)

**Status**: DRAFT - 准备与用户初始化讨论
**Date**: 2026-03-07
**Focus**: Multi-Agent Orchestration Framework for Operational RCA with Hypothesis-Driven Reasoning

---

## 核心设计理念：假设驱动的 RCA（最新补充）

你补充的核心思想非常重要，完全改变了 Orchestrator 的角色定义：

### 之前的理解（有问题）
```
Orchestrator 是"协调者" → Sub-Agent 执行诊断并输出推理结论
（Orchestrator 只是调度，每个 Agent 都独立推理）
```

### 正确的理解（你的补充）
```
Orchestrator 是"推理者" → Sub-Agent 是"数据收集者"

流程：
1. 调查阶段: Sub-Agents 收集数据（只返回数据，不推理）
2. 推理阶段: Orchestrator 分析数据，生成假设
3. 验证阶段: 逐一验证假设，更新假设状态和置信度
4. 确认阶段: 找到确认的根因

关键：Orchestrator 维护一个"诊断笔记本" (Notebook)
- 记录所有假设及其演变过程
- 记录每次验证如何改变了假设
- 追踪整个推理过程
```

---

## 整体架构设计 (更新版本)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Root StateGraph                           │
│  ExecutorState: {                                                │
│    messages,                                                     │
│    notebook (Orchestrator的诊断笔记本) ← ⭐ 新增！             │
│  }                                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ User Input: "API 响应时间从 200ms → 5s"                        │
│     ↓                                                             │
│ [Phase 1] 初步全面调查 (Comprehensive Exploration)            │
│     Orchestrator 并行分配数据收集任务:                         │
│     ├─ Infrastructure Agent → 只收集 CPU/内存/磁盘 数据       │
│     ├─ Log Agent → 只收集错误日志数据                         │
│     └─ DB Agent → 只收集连接池/查询数据                       │
│     ↓ (Sub-Agents 不进行推理！)                               │
│ Notebook.collected_data = {所有原始数据}                      │
│     ↓                                                             │
│ [Phase 2] 生成初始假设 (Hypothesis Generation)                │
│     Orchestrator 分析数据，使用 LLM 生成 3-5 个假设           │
│     Notebook.hypotheses = {                                    │
│       "H1": Hypothesis(desc="DB连接池耗尽", confidence=0.85), │
│       "H2": Hypothesis(desc="CPU压力过高", confidence=0.6),  │
│       "H3": Hypothesis(desc="网络延迟", confidence=0.3),      │
│     }                                                            │
│     Notebook.hypothesis_order = ["H1", "H2", "H3"]            │
│     ↓                                                             │
│ [Phase 3] 逐一验证假设 (Hypothesis Verification)              │
│     for each hypothesis (sorted by confidence):                │
│         验证任务 = generate_verification_task(H)              │
│         Orchestrator → dispatch_to_agent(验证任务)            │
│         ↓                                                         │
│         Sub-Agent 返回: verdict = "confirmed" / "rejected" / "partial"
│         ↓                                                         │
│         Orchestrator 更新假设:                                 │
│         - confirmed: H.status = "confirmed", H.confidence = 0.99
│         - rejected: H.status = "rejected", H.evidence = []     │
│         - partial: H.confidence *= 0.7, 修改描述             │
│         ↓                                                         │
│ Notebook.exploration_history.append(ExplorationStep(...))     │
│     ↓                                                             │
│ [Phase 4] 确认根因 (Root Cause Confirmation)                  │
│     if notebook.confirmed_hypothesis:                          │
│         输出根因和修复建议                                    │
│     else:                                                        │
│         继续更深入的验证 (Phase 3.2)                          │
│     ↓                                                             │
│ Output: {                                                        │
│   "root_cause": "数据库连接池耗尽 + N+1查询",               │
│   "confidence": 0.95,                                           │
│   "recommendations": ["增加连接池", "优化查询"],              │
│   "notebook": { 完整的诊断过程 },  ← 用于学习和调试        │
│   "hypothesis_evolution": { 假设如何演变的 }  ← 用于RL训练  │
│ }                                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 关键概念：Orchestrator Notebook

### Notebook 的职责

Notebook 不仅是"日志"，而是 Orchestrator 的"工作记忆"和"小抄"：

```python
@dataclass
class DiagnosticNotebook:
    # 1. 数据管理：记录从 Sub-Agents 收集的所有原始数据
    collected_data: dict[str, dict]
    # {
    #   "infrastructure": {"cpu": 0.85, "memory": 0.4, ...},
    #   "logs": ["[ERROR] DB connection timeout", ...],
    #   "database": {"connection_pool": "100/100", ...}
    # }

    # 2. 假设管理：追踪所有假设的状态
    hypotheses: dict[str, Hypothesis]
    hypothesis_order: list[str]  # 按置信度排序

    # 3. 探索历史：完整的诊断过程记录
    exploration_history: list[ExplorationStep]
    # 每一步记录: 做了什么、假设如何变化、学到了什么

    # 4. 当前状态：我现在在做什么？
    current_phase: Literal["exploration", "hypothesis_generation", "verification", "completed"]

    # 5. 最终结果
    confirmed_hypothesis: Optional[str]  # 最终确认的根因
    root_cause: Optional[str]
```

### Notebook 的实际价值

#### 1️⃣ 记录进度和探索状态

```python
# Orchestrator 可以随时检查自己做了什么
print(notebook.current_phase)  # "verification"
print(notebook.hypothesis_order)  # ["H1", "H2", "H3"]
print([h.status for h in notebook.hypotheses.values()])  # ["confirmed", "rejected", "active"]

# 知道还需要做什么
unverified = [h for h in notebook.hypotheses.values() if h.status == "active"]
if unverified:
    next_step = sorted(unverified, key=lambda h: h.confidence, reverse=True)[0]
    print(f"接下来验证: {next_step.id}")
```

#### 2️⃣ 避免重复探索

```python
# 查看已经被尝试过的假设
rejected = [h for h in notebook.hypotheses.values() if h.status == "rejected"]
print(f"已排除的假设（不需要重新验证）: {[h.description for h in rejected]}")

# 如果用户要求"再看一次网络"，Orchestrator 可以说
# "我已经验证过网络延迟（H3），它被排除了。确定要重新验证吗？"
```

#### 3️⃣ 完整的推理过程

```python
# 追踪假设如何随时间演变
for step in notebook.exploration_history:
    print(f"Step {step.step_number}: {step.action}")
    if step.hypotheses_updated:
        for update in step.hypotheses_updated:
            print(f"  {update['hypothesis_id']}: "
                  f"{update['old_confidence']:.2f} → {update['new_confidence']:.2f}")
            print(f"  原因: {update['reason']}")

# 这就是 Orchestrator 的"思维过程"，清晰可追踪
```

#### 4️⃣ 故障恢复和中断恢复

```python
# 如果执行被中断，Notebook 可以序列化保存
checkpoint = {
    "notebook": notebook,
    "current_phase": notebook.current_phase,
    "next_action": "验证 H2"
}

# 恢复时，Orchestrator 继续从 H2 开始
# "我之前已经确认了 H1，现在继续验证 H2..."
```

#### 5️⃣ 学习和改进

```python
# 所有 insights 被记录，可以用于改进未来的诊断
all_insights = []
for step in notebook.exploration_history:
    all_insights.extend(step.insights)

# 例如:
# - "DB 连接池问题通常与错误日志相关，应该同时查看"
# - "CPU 高不一定是根本原因，需要深入验证"

# 这些可以用于强化学习，学习"什么时候该做什么诊断"
```

---

## 四个 Phase 详解

### Phase 1: 初步全面调查

**特点**：并行收集数据，Sub-Agent 仅返回原始数据，不进行推理

```python
# Orchestrator 的行为
exploration_tasks = [
    {"agent": "infrastructure_agent", "task": "快速扫描: CPU, 内存, 磁盘, 网络"},
    {"agent": "log_agent", "task": "最近1小时的错误日志"},
    {"agent": "db_agent", "task": "当前数据库状态（连接数、慢查询、锁）"},
]

results = await parallel_dispatch(exploration_tasks)
notebook.collected_data = results

# ⭐ Sub-Agent 的约束
# GOOD: 返回 {"cpu": 0.85, "memory": 0.4, ...}
# BAD: 返回 "CPU 高，可能是问题"
```

**Notebook 在此阶段后**：
```python
notebook.collected_data = {
    "infrastructure": {"cpu": 0.85, "memory": 0.4, "disk_io": 0.5},
    "logs": ["[ERROR] DB connection timeout", "[WARN] Slow query"],
    "database": {"connection_pool": "100/100", "active_connections": 100}
}
# collected_data 是 Notebook 的核心，所有后续推理都基于它
```

---

### Phase 2: 生成初始假设

**特点**：Orchestrator 分析 Phase 1 的数据，使用 LLM 生成假设

```python
# Orchestrator 使用 LLM 进行推理
reasoning_prompt = f"""
Based on the data: {notebook.collected_data}
Generate 3-5 hypotheses about the root cause.

Output as JSON:
{{
    "hypotheses": [
        {{
            "description": "Database connection pool exhaustion",
            "confidence": 0.85,
            "supporting_evidence": ["connection_pool: 100/100", "DB timeout errors"],
            "counter_evidence": ["CPU usage is high"]
        }},
        ...
    ]
}}
"""

response = await llm.ainvoke(reasoning_prompt)
hypotheses_data = json.loads(response)

# 转换为 Hypothesis 对象
for i, h in enumerate(hypotheses_data["hypotheses"]):
    notebook.hypotheses[f"H{i+1}"] = Hypothesis(
        id=f"H{i+1}",
        description=h["description"],
        confidence=h["confidence"],
        evidence=h["supporting_evidence"],
        counter_evidence=h["counter_evidence"],
        status="active"
    )

# 按置信度排序
notebook.hypothesis_order = sorted(
    notebook.hypotheses.keys(),
    key=lambda hid: notebook.hypotheses[hid].confidence,
    reverse=True
)
```

**Notebook 在此阶段后**：
```python
notebook.hypotheses = {
    "H1": {"description": "DB连接池耗尽", "confidence": 0.85, "status": "active"},
    "H2": {"description": "CPU压力过高", "confidence": 0.6, "status": "active"},
    "H3": {"description": "网络延迟", "confidence": 0.3, "status": "active"}
}
notebook.hypothesis_order = ["H1", "H2", "H3"]
notebook.current_phase = "verification"
```

---

### Phase 3: 逐一验证假设

**特点**：按置信度逐一验证，根据验证结果更新假设

```python
# 验证循环
for hypothesis_id in notebook.hypothesis_order:
    hypothesis = notebook.hypotheses[hypothesis_id]

    if hypothesis.status in ["confirmed", "rejected"]:
        continue

    # 生成验证任务
    verification_task = generate_verification_task(hypothesis)
    # 例如: "深度检查数据库连接池"

    # 分配给 Sub-Agent
    verification_result = await dispatch_to_agent(
        agent=verification_task["agent"],
        task=verification_task["description"]
    )

    # ⭐ 解析验证结果的 VERDICT
    # 这是 Sub-Agent 最重要的输出
    verdict = verification_result.verdict  # "confirmed" / "rejected" / "partial"

    # 更新假设
    if verdict == "confirmed":
        hypothesis.status = "confirmed"
        hypothesis.confidence = 0.99
        notebook.confirmed_hypothesis = hypothesis_id
        break  # 可以提前结束

    elif verdict == "rejected":
        hypothesis.status = "rejected"
        hypothesis.evidence = []

    elif verdict == "partial":
        hypothesis.status = "partial"
        hypothesis.confidence *= 0.7
        # 修改假设描述（如果需要）

    # ⭐ 记录这一步如何改变了假设
    notebook.exploration_history.append(ExplorationStep(
        step_number=len(notebook.exploration_history) + 1,
        action=f"验证 {hypothesis_id}",
        hypotheses_before=deepcopy(notebook.hypotheses),
        hypotheses_after=deepcopy(notebook.hypotheses),
        hypotheses_updated=[{
            "hypothesis_id": hypothesis_id,
            "old_status": "active",
            "new_status": hypothesis.status,
            "old_confidence": 0.85,
            "new_confidence": hypothesis.confidence,
            "reason": f"验证结果: {verdict}"
        }],
        insights=[
            f"{hypothesis_id} 的验证结果: {verdict}",
            f"当前最有可能的假设: {notebook.hypothesis_order[0]}"
        ]
    ))
```

**验证任务的例子**：
```python
# 对于 "DB连接池耗尽" 的假设
VerificationTask(
    description="深度检查数据库连接池是否是瓶颈",
    focus_areas=["pool_limit", "active_connections", "connection_wait_time"],
    depth="detail",
    return_format="data_only"  # ⭐ 仅返回数据，不推理！
)

# Sub-Agent 返回
VerificationResult(
    verdict="confirmed",  # ⭐ 这是关键
    data={
        "pool_size": 100,
        "active_connections": 100,
        "connection_wait_time_ms": 45,
        "query_causing_bottleneck": "SELECT * FROM orders WHERE user_id = ?"
    },
    # ⚠️ 注意：Sub-Agent 不提供推理，只是说 "confirmed"
    # Orchestrator 负责解释为什么
)
```

**Notebook 在验证过程中的演变**：
```
Step 1: 初步调查
  Notebook.collected_data = {...}

Step 2: 生成假设
  Notebook.hypotheses = {H1, H2, H3}
  Notebook.hypothesis_order = ["H1", "H2", "H3"]

Step 3: 验证 H1
  H1.status: active → confirmed
  H1.confidence: 0.85 → 0.99
  Notebook.confirmed_hypothesis = "H1"

输出: 根因已确认
```

---

### Phase 4: 确认根因

```python
if notebook.confirmed_hypothesis:
    root_hypothesis = notebook.hypotheses[notebook.confirmed_hypothesis]
    output = {
        "root_cause": root_hypothesis.description,
        "confidence": root_hypothesis.confidence,
        "evidence": root_hypothesis.evidence,
        "recommendations": generate_recommendations(root_hypothesis)
    }
else:
    # 如果没有确认，可能需要更深入的验证
    # 或者放宽确认标准
    pass
```

---

## State Schema 的更新

### Root ExecutorState

```python
class ExecutorState(TypedDict):
    # 基本
    messages: Annotated[list, operator.add]
    task_id: str
    task_description: str

    # ⭐ 新增：Orchestrator 的诊断笔记本
    notebook: DiagnosticNotebook

    # 控制流
    current_phase: str
    pending_agent: Optional[str]

    # 历史
    agent_execution_history: Annotated[list[dict], operator.add]
```

### 为什么要单独的 notebook 字段？

**原因 1**：不要把假设逻辑混在 messages 里
```python
# ❌ 不好：假设逻辑隐藏在 messages 中
state["messages"] = [
    HumanMessage(content="我有一个假设..."),
    AIMessage(content="根据数据，我认为..."),
]

# ✅ 好：假设明确表示
state["notebook"].hypotheses = {
    "H1": Hypothesis(...)
}
```

**原因 2**：Notebook 是 Orchestrator 的"内存"，需要独立管理
```python
# Orchestrator 可以随时查询
if state["notebook"].confirmed_hypothesis:
    print("已找到根因")
else:
    print("还在验证中")
```

**原因 3**：便于序列化和导出
```python
# 导出 Notebook 用于学习和调试
export_notebook(state["notebook"])

# 导出假设演变轨迹用于 RL 训练
export_hypothesis_evolution(state["notebook"])
```

---

## 消息传递的规范

### Sub-Agent 只返回"数据"，不返回"推理"

**Phase 1：初步调查**
```python
# Orchestrator → Infrastructure Agent
{
    "task": "快速扫描系统资源",
    "depth": "overview"
}

# Infrastructure Agent → Orchestrator
# ✅ 正确：只返回数据
{
    "cpu_usage": 0.85,
    "memory_usage": 0.4,
    "disk_io": 0.5,
    "network_latency_ms": 10
}

# ❌ 错误：不应该包含推理
{
    "analysis": "CPU 高，可能是性能瓶颈",
    "recommendation": "增加 CPU"
}
```

**Phase 3：验证假设**
```python
# Orchestrator → DB Agent
{
    "hypothesis_being_tested": "DB连接池耗尽",
    "task": "深度检查连接池",
    "focus_areas": ["pool_limit", "active_connections", "wait_time"]
}

# DB Agent → Orchestrator
# ✅ 正确：返回 verdict + 数据
{
    "verdict": "confirmed",  # ⭐ 关键：是 / 否 / 部分正确
    "data": {
        "pool_size": 100,
        "active_connections": 100,
        "wait_time_ms": 45
    }
}

# ❌ 错误：不应该返回推理
{
    "analysis": "连接池确实是瓶颈，因为...",
    "root_cause": "连接池耗尽"
}
```

**关键的改变**：
- Phase 1 的数据用来"生成假设"
- Phase 3 的 verdict 用来"验证假设"
- Orchestrator 负责"推理"
- Sub-Agents 只负责"收集数据"

---

## 轨迹记录：假设演变 (新增)

除了标准的 Checkpoint 轨迹外，还需要记录**假设如何随时间演变**。这是 RL 训练的关键数据。

### 假设演变轨迹的导出格式

```json
{
  "task_id": "rca_001",
  "timeline": [
    {
      "step_number": 1,
      "action": "初步全面调查",
      "phase": "exploration",
      "hypotheses": null,
      "changes": []
    },
    {
      "step_number": 2,
      "action": "生成初始假设",
      "phase": "hypothesis_generation",
      "hypotheses": {
        "H1": {
          "description": "DB连接池耗尽",
          "confidence": 0.85,
          "status": "active",
          "evidence": ["connection_pool: 100/100", "DB timeout errors"]
        },
        "H2": {
          "description": "CPU压力过高",
          "confidence": 0.6,
          "status": "active"
        }
      },
      "changes": [
        {
          "type": "hypothesis_created",
          "hypothesis_id": "H1",
          "confidence": 0.85
        }
      ]
    },
    {
      "step_number": 3,
      "action": "验证 H1: DB连接池",
      "phase": "verification",
      "hypotheses": {
        "H1": {
          "description": "DB连接池耗尽",
          "confidence": 0.95,
          "status": "confirmed"  # ⭐ 状态改变了
        },
        "H2": {
          "confidence": 0.6,
          "status": "active"
        }
      },
      "changes": [
        {
          "type": "status_changed",
          "hypothesis_id": "H1",
          "old_status": "active",
          "new_status": "confirmed",
          "reason": "验证步骤返回 verdict=confirmed"
        },
        {
          "type": "confidence_updated",
          "hypothesis_id": "H1",
          "old_confidence": 0.85,
          "new_confidence": 0.95,
          "reason": "确认后提升置信度"
        }
      ]
    }
  ],
  "final_root_cause": {
    "hypothesis_id": "H1",
    "description": "数据库连接池耗尽 + N+1查询",
    "confidence": 0.95
  }
}
```

### 这个轨迹用于 RL 训练

```python
# 强化学习可以学到：
# "当看到 CPU 高 + 错误日志中有 DB 超时，应该优先检查连接池"
# "当验证返回 confirmed，置信度应该从 0.85 跳到 0.99"
# "当有多个假设时，按置信度验证能更快找到根因"

# 这些策略可以通过 reward signal 进行优化
```

---

## 与前一个设计的关系

### 完全兼容，但思想更清晰

| 方面 | 之前 | 现在 |
|------|------|------|
| Orchestrator 角色 | 路由 + 决策 | **推理 + 假设验证** |
| Sub-Agent 角色 | 执行诊断 | **收集数据**（不推理） |
| 假设管理 | 隐含在 messages | **显式的 Notebook** |
| 假设演变 | 无法追踪 | **完整的 ExplorationHistory** |
| 推理过程 | 分散在各 Agent | **集中在 Orchestrator** |

### 这个改变带来的好处

1. ✅ **清晰的职责分离** → 更容易理解和调试
2. ✅ **完整的推理过程** → 可以看到 Orchestrator 如何思考
3. ✅ **假设演变可追踪** → 用于学习和改进
4. ✅ **Notebook 作为记忆** → Orchestrator 有"小抄"
5. ✅ **避免重复探索** → 记住已验证的假设
6. ✅ **适合 RL 训练** → 学习"什么时候做什么决策"

---

## 关键设计决策

| 决策 | 理由 |
|------|------|
| **Notebook 作为 Orchestrator 的记忆** | 记录进度、支持中断恢复、追踪假设演变 |
| **Sub-Agent 只返回数据** | 推理集中在 Orchestrator，更清晰 |
| **四个 Phase 的划分** | exploration → generation → verification → confirmation |
| **verdict 字段的重要性** | "是/否/部分" 决定如何更新假设 |
| **hypothesis_order 按置信度排序** | 优先验证最可能的假设，更高效 |
| **ExplorationHistory 记录每一步** | 完整的推理过程，用于学习 |

---

## 待讨论的问题

### 核心问题
1. ✅ **假设驱动的设计思想**是否符合你的期望？
2. ❓ **四个 Phase** 的划分是否合理？
3. ❓ **Notebook 的结构**是否足够详细？
4. ❓ **Sub-Agent 的约束**（只返回数据，不推理）是否可行？

### 细节问题
5. ❓ 在 Phase 3 中，当一个假设是 "partial" 时，应该如何处理？
   - A) 修改描述，继续验证
   - B) 拆分为多个假设
   - C) 其他

6. ❓ 如果 Notebook 中没有任何假设被确认（都是 rejected），应该怎么办？
   - A) 进入"深入调查"阶段，收集更多数据
   - B) 降低确认标准（从 0.99 改为 0.8）
   - C) 其他

7. ❓ **Notebook 的序列化和恢复**：
   - A) 仅保存关键字段（notebook）
   - B) 保存完整的 state
   - C) 其他

---

## 下一步建议

**立即讨论**：
- ✅ 假设驱动设计是否正确理解了你的需求
- ✅ Notebook 结构是否合理
- ✅ Phase 的划分是否清晰

**后续讨论**：
- 配置系统如何支持不同的"验证策略"
- 轨迹导出的具体格式
- 前端如何展示假设演变
- RL 训练如何使用这些轨迹

---

希望这个补充的设计能准确反映你的思想！期待你的反馈。


你的需求可以归纳为 **6 个方面**：

1. ✅ **Orchestrator 和 Agent 的 Prompt 自由配置**
2. ✅ **场景切换只需换 Prompt/Config，不改代码**
3. ✅ **Agent 间通过 TypedDict 传递参数进行定制化**
4. ✅ **Orchestrator 可以监控和干预 Sub-Agent 的执行**
5. ✅ **支持层次化轨迹记录和 RL 训练导出**
6. ✅ **支持故障恢复和调试重放**

---

## 整体架构设计 (LangGraph Supervisor + Subgraph)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Root StateGraph                           │
│  ExecutorState: {messages, diagnosis_history, current_agent}    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  START → [Orchestrator Node]                                   │
│           (LLM-based decision)                                  │
│              ↓                                                   │
│         Route to Sub-Agent:                                     │
│         ├─ [Infrastructure Agent] ← create_react_agent         │
│         ├─ [Log Agent]           ← create_react_agent         │
│         ├─ [DB Agent]            ← create_react_agent         │
│         └─ [Analyzer Agent]      ← create_react_agent         │
│              ↓                                                   │
│         Return to Orchestrator                                  │
│              ↓ (multi-round loop)                               │
│         [END]                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 关键设计要点

#### 1. Orchestrator 的职责
- **决策者**：基于 diagnosis_history，使用 LLM 决定下一个要调用的 Sub-Agent
- **监控者**：实时流式监听 Sub-Agent 的执行状态（stream + subgraphs=True）
- **干预者**：可以在关键点中断执行、修改指令、注入新信息
- **协调者**：管理整个执行流程，决定何时结束

**Orchestrator 的行为示例（RCA 场景）：**

```
Round 1:
  用户: "API响应时间从200ms → 5s，找根本原因"
  Orchestrator → Infrastructure Agent

Round 2:
  诊断结果: "CPU 85%，内存充足"
  Orchestrator → Log Agent

Round 3:
  诊断结果: "发现大量DB连接超时"
  Orchestrator → DB Agent

Round 4:
  诊断结果: "连接池耗尽，N+1查询"
  Orchestrator → Analyzer Agent (综合分析)

Output: 详细的根因分析和修复建议
```

#### 2. Sub-Agent 的架构
- **独立性**：每个 Sub-Agent 是独立的 create_react_agent，有自己的 State Schema
- **状态隔离**：只通过 messages 字段与 Root 通信（同名字段自动映射）
- **工具集独立**：每个 Agent 拥有自己的工具列表（从 registry 动态加载）
- **中断能力**：可配置 interrupt_before，允许 Orchestrator 审核关键步骤

**Sub-Agent 的内部流程（ReAct 循环）：**

```
Input (messages + task definition)
    ↓
[LLM Node] 推理
    ├─ 有 Tool Call? → [Tools Node] → 执行工具 → Loop Back
    └─ 输出最终答案? → Return
    ↓
Output (messages + results)
```

#### 3. 消息传递机制
- **消息格式**：JSON 编码的 DiagnosisTask / DiagnosisResult 数据类
- **传递方式**：通过 state["messages"] 中的 HumanMessage / AIMessage
- **类型安全**：使用 @dataclass 定义参数和结果，不依赖 Prompt 文本

**消息流示例：**

```python
# Orchestrator → Sub-Agent
HumanMessage(content=json.dumps({
    "task": DiagnosisTask(
        task_id="diag_001",
        agent_id="db_agent",
        target_system="database",
        target_metrics=["query_time", "connection_count"],
        time_range=("2026-03-07 10:00", "2026-03-07 11:00"),
        strategy="aggressive"
    )
}))

# Sub-Agent → Orchestrator
AIMessage(content=json.dumps({
    "result": DiagnosisResult(
        task_id="diag_001",
        findings=["连接池100%使用"],
        hypotheses=[{"cause": "连接池耗尽", "confidence": 0.95}],
        recommendations=["增加连接池大小"],
        confidence=0.9
    )
}))
```

---

## 配置系统设计

### 层级配置结构

```
config/
├── system.yaml              # 系统全局配置 (model provider, checkpointer 等)
├── orchestrator.yaml        # Orchestrator 配置
├── agents.yaml              # Sub-Agent 配置（所有 Agent 的通用参数）
└── scenarios/
    ├── rca.yaml                    # RCA 场景特定配置
    ├── performance_optimization.yaml
    └── security_audit.yaml

templates/
├── orchestrator/
│   ├── rca_system.txt              # RCA 场景的 Orchestrator 系统提示
│   ├── performance_system.txt
│   └── security_system.txt
└── agents/
    ├── infrastructure_system.txt    # Infrastructure Agent 系统提示
    ├── log_system.txt
    ├── db_system.txt
    └── analyzer_system.txt
```

### 配置文件示例

#### config/orchestrator.yaml
```yaml
orchestrator:
  llm:
    model: "gpt-4"
    temperature: 0.2
    max_tokens: 1024

  prompts:
    system: "templates/orchestrator/rca_system.txt"
    decision_format: "json"  # 强制 JSON 输出格式

  available_agents:
    - infrastructure_agent
    - log_agent
    - db_agent
    - analyzer_agent

  max_rounds: 10  # 最多诊断轮数
  timeout: 600    # 总超时时间
```

#### config/agents.yaml
```yaml
agents:
  infrastructure_agent:
    enabled: true
    description: "检查系统基础设施（CPU、内存、磁盘、网络）"
    llm:
      model: "gpt-4-mini"
      temperature: 0.2
      max_tokens: 2048

    tools:
      - get_cpu_metrics
      - get_memory_metrics
      - get_disk_metrics

    tool_settings:
      get_cpu_metrics:
        interval: "5m"
        granularity: "1m"

    execution:
      max_steps: 15
      timeout: 120
      interrupt_before: []  # 不需要中断

  db_agent:
    enabled: true
    description: "检查数据库性能"
    llm:
      model: "gpt-4"
      temperature: 0.1  # 最低温，SQL 精确性最重要
      max_tokens: 3000

    tools:
      - get_db_metrics
      - analyze_slow_queries
      - check_connections
      - explain_query

    execution:
      max_steps: 20
      timeout: 300
      interrupt_before: ["tools"]  # ⭐ 允许 Orchestrator 审核 DB 查询

  analyzer_agent:
    enabled: true
    description: "综合分析所有诊断结果"
    llm:
      model: "gpt-4"
      temperature: 0.3
      max_tokens: 4000

    tools: []  # 分析 Agent 不调用外部工具

    execution:
      max_steps: 10
      timeout: 60
```

#### config/scenarios/rca.yaml
```yaml
scenario: "root_cause_analysis"

orchestrator:
  system_prompt: "templates/orchestrator/rca_system.txt"
  decision_options:
    - "infrastructure_agent"
    - "log_agent"
    - "db_agent"
    - "analyzer_agent"
    - "END"

  methodology:
    - "理解问题症状"
    - "逐层诊断（基础设施 → 日志 → 数据库）"
    - "分析因果关系"
    - "确定根本原因"
    - "提出修复建议"

agent_prompts:
  infrastructure_agent: "templates/agents/infrastructure_system.txt"
  log_agent: "templates/agents/log_system.txt"
  db_agent: "templates/agents/db_system.txt"
  analyzer_agent: "templates/agents/analyzer_system.txt"
```

### 场景切换示例

**切换到性能优化场景，只需 3 步：**

1. 创建 `config/scenarios/performance_optimization.yaml`
2. 创建 `templates/orchestrator/performance_system.txt`
3. 创建 `templates/agents/optimizer_system.txt` (如需新 Agent)
4. 代码完全不变！

```python
# 代码永远是这样的
orchestrator = create_orchestrator(
    scenario="performance_optimization",  # 只改这一个参数！
    config_file="config/scenarios/performance_optimization.yaml"
)
```

---

## Prompt 配置管理

### Prompt 模板系统（Jinja2）

#### templates/orchestrator/rca_system.txt
```
你是一个高级运维工程师，特化于系统根因分析 (RCA)。

你的诊断方法论：
{% for step in methodology %}
{{ loop.index }}. {{ step }}
{% endfor %}

可用的诊断工具（通过 Sub-Agents）：
{% for agent in available_agents %}
- {{ agent.name }}: {{ agent.description }}
{% endfor %}

当前诊断进度：
{{ diagnosis_progress }}

根据当前进度，决策下一步应该执行哪个诊断。

回复格式（必须是 JSON）：
{
    "thinking": "你的分析思路",
    "decision": "选择的 Agent",
    "reasoning": "为什么选择这个 Agent",
    "instruction": "给该 Agent 的具体指令"
}
```

#### templates/agents/db_system.txt
```
你是一个数据库性能诊断专家。

你的诊断方法：
1. 理解诊断任务（从消息中解析 DiagnosisTask）
2. 执行诊断（使用指定的工具，获取目标指标）
3. 分析数据（寻找异常和瓶颈）
4. 形成假设（按置信度排序）
5. 建议行动（给出可修复的建议）

诊断任务格式：
{
  "task": {
    "target_system": "database",
    "target_metrics": ["query_time", "connection_count"],
    "time_range": ["2026-03-07 10:00", "2026-03-07 11:00"],
    "strategy": "aggressive"
  }
}

基于 strategy 的执行方式：
- "aggressive": 尽可能多地收集数据
- "conservative": 快速诊断，只收集关键指标
- "auto": 根据初步发现动态调整

可用工具：
- get_db_metrics: 获取数据库性能指标
- analyze_slow_queries: 分析慢查询
- check_connections: 检查连接池状态
- explain_query: 分析 SQL 执行计划

输出格式（JSON）：
{
  "findings": ["发现1", "发现2"],
  "hypotheses": [
    {"cause": "...", "confidence": 0.95, "evidence": ["..."]},
    {"cause": "...", "confidence": 0.6, "evidence": ["..."]}
  ],
  "recommendations": ["建议1", "建议2"],
  "confidence": 0.85
}
```

---

## 监控与干预机制

### 实时监控（Streaming + Hierarchical Namespace）

```python
# 使用 stream() 实时监听执行
async for namespace, mode, data in graph.stream(
    {"messages": [HumanMessage(content="API响应时间增加，找根本原因")]},
    config,
    stream_mode=["updates", "custom"],
    subgraphs=True,  # ⭐ 捕获层次结构
):
    agent_path = " > ".join(namespace) if namespace else "[Main]"

    if mode == "updates":
        print(f"[{agent_path}] Update: {data}")
    elif mode == "custom":
        print(f"[{agent_path}] Custom Event: {data}")
```

**输出示例：**
```
[Main] Update: {'orchestrator': {...}}
[Main > infrastructure_agent] Update: {'agent': {...}}
[Main > infrastructure_agent] Update: {'messages': [...]}
[Main > log_agent] Update: {'agent': {...}}
[Main > db_agent] Update: {'messages': [...]}
```

### 中断与干预

#### 1. Sub-Agent 级别的中断 (interrupt_before)
```yaml
# db_agent 在执行工具前中断，允许 Orchestrator 审核
db_agent:
  execution:
    interrupt_before: ["tools"]
```

**工作流：**
```
Orchestrator: "开始 DB 诊断"
    ↓
DB Agent: 计划执行 [explain_query(table_a), check_locks()]
    ⏸️  PAUSE (interrupt_before=["tools"])
    ↓
Orchestrator: 收到暂停信号，审核工具调用
    ✅ 批准所有工具调用 → Agent 继续执行
    或
    ❌ 拒绝某些工具 → Agent 修改计划
```

#### 2. Orchestrator 级别的控制 (update_state + inject_instruction)

```python
# 在任何时刻，Orchestrator 可以注入新指令
graph.update_state(
    config,
    {
        "messages": [
            HumanMessage(content="[Orchestrator Instruction] 优先关注连接池问题，忽略其他")
        ]
    }
)

# 恢复执行
result = graph.invoke(None, config)
```

---

## 轨迹记录与导出

### 层次化轨迹捕获

使用 LangGraph 的 Checkpoint 系统 + Streaming：

```python
# 1. 实时捕获执行事件
trace = ExecutionTrace(thread_id="task-1")

async for namespace, mode, data in graph.stream(
    input_data,
    config,
    stream_mode=["updates", "custom"],
    subgraphs=True,
):
    agent_path = list(namespace) if namespace else ["main"]
    trace.add_event(agent_path, node, event_type, data)

# 2. 从 Checkpoint 导出完整轨迹
history = list(graph.get_state_history(config))
trajectory = extract_trajectory(history)

# 3. 保存为不同格式
# - 可视化: JSON (Web UI 用)
# - RL 训练: JSONL (RLTransition 格式)
# - 调试: Markdown (人类可读)
```

### RL 训练数据导出格式

```python
@dataclass
class RLTransition:
    state: dict              # 当前状态快照
    action: dict             # 节点执行（Orchestrator 的决策）
    reward: float            # 奖励信号
    next_state: dict         # 执行后的状态
    agent: str               # 哪个 Agent 执行
    step: int                # 步骤号
    metadata: dict           # 其他信息

# 导出格式
# transitions.jsonl:
{"state": {...}, "action": {...}, "reward": 1.0, "next_state": {...}, "agent": "orchestrator", "step": 1}
{"state": {...}, "action": {...}, "reward": 0.5, "next_state": {...}, "agent": "infrastructure_agent", "step": 2}
...
```

---

## 故障恢复与调试重放

### 从故障点恢复

```python
# 执行过程中出现错误
try:
    result = graph.invoke({"messages": [...]}, config)
except Exception as e:
    print(f"错误: {e}")

# 检查失败点的状态
state = graph.get_state(config)
print(f"停止于: {state.next}")  # 正在执行的节点

# 修复状态并重试
graph.update_state(config, {
    "messages": state.values["messages"] + [
        HumanMessage(content="[System] Retry with adjusted parameters")
    ]
})

# 恢复执行
result = graph.invoke(None, config)
```

### 调试重放（从特定 Checkpoint）

```python
# 浏览所有 Checkpoint
history = list(graph.get_state_history(config))

# 找到想要调试的步骤
target = history[3]  # 第 3 步

# 从该点重新开始
graph.update_state(target.config, {
    "messages": [HumanMessage(content="[Debug] Alternative approach")]
})

# 创建新的执行分支
debug_result = graph.invoke(None, target.config)
```

---

## 前端架构设计

### 需要展示的信息

1. **Agent 结构拓扑**
   - Root Orchestrator 节点
   - 所有 Sub-Agent 节点
   - 节点间的路由关系

2. **执行进度监控**
   - 当前正在执行的节点
   - 每个 Agent 的状态（待命、执行中、已完成）
   - 实时消息流（使用 WebSocket）

3. **对话记录与轨迹**
   - Orchestrator 的推理过程
   - 每个 Agent 的诊断结果
   - 完整的执行树形结构

4. **调试工具**
   - Checkpoint 浏览器（查看历史状态）
   - 轨迹重放（从特定点重新执行）
   - 状态检查器（查看当前的 messages, diagnosis_history 等）

### 后端 API 设计

```python
# WebSocket 事件流
/ws/execute/{thread_id}
  - event: "node_start", data: {"node": "infrastructure_agent", "path": ["main"]}
  - event: "node_end", data: {"node": "infrastructure_agent", "output": {...}}
  - event: "message_update", data: {"messages": [...]}
  - event: "interrupt", data: {"node": "db_agent", "pending_tools": [...]}

# REST API
GET /api/execution/{thread_id}/state         # 当前执行状态
GET /api/execution/{thread_id}/history       # Checkpoint 历史
POST /api/execution/{thread_id}/interrupt    # 中断执行
POST /api/execution/{thread_id}/resume       # 恢复执行
POST /api/execution/{thread_id}/update-state # 修改状态并恢复
GET /api/execution/{thread_id}/trajectory    # 导出完整轨迹
```

---

## 关键设计决策总结

| 方面 | 设计选择 | 理由 |
|------|--------|------|
| **架构模式** | Supervisor + Subgraph | LangGraph 原生支持，状态管理清晰 |
| **Orchestrator 决策** | LLM + JSON 输出 | 灵活适应不同场景，无需改代码 |
| **Sub-Agent 实现** | create_react_agent | 预构建的 ReAct 循环，开箱即用 |
| **消息传递** | JSON + TypedDict | 类型安全，不依赖 Prompt 文本 |
| **Prompt 管理** | YAML + Jinja2 模板 | 完全参数化，支持动态渲染 |
| **轨迹记录** | Checkpoint + Streaming | 完整的执行历史，支持时间旅行 |
| **监控干预** | stream() + interrupt_before | 实时监听，细粒度控制 |
| **场景切换** | Config 文件 + 无代码改动 | 最大化代码复用 |

---

## 待讨论的设计问题

### 1️⃣ Orchestrator 的决策模型
- **当前设计**：基于 LLM 和 diagnosis_history 进行多轮推理
- **可选**：是否需要显式的"诊断计划"阶段？（即 Orchestrator 先制定诊断步骤，再逐步执行？）
- **你的想法**？

### 2️⃣ Sub-Agent 的配置粒度
- **当前设计**：agents.yaml 统一管理所有 Agent 参数
- **可选**：是否需要为不同场景提供不同的 agents 配置？（例如 rca 场景用 agents-rca.yaml）
- **你的想法**？

### 3️⃣ 中断与干预的触发机制
- **当前设计**：支持 interrupt_before（工具执行前）和 update_state（任意时刻）
- **可选**：是否需要 interrupt_after（工具执行后，允许 Orchestrator 审核结果）？
- **你的想法**？

### 4️⃣ 轨迹导出的细粒度
- **当前设计**：导出 Root 级和 Sub-Agent 级的两层轨迹
- **可选**：是否需要更多细节？（例如每个 Tool Call 的输入/输出）
- **你的想法**？

### 5️⃣ 故障恢复的自动化程度
- **当前设计**：手动检查 state，手动调用 update_state 恢复
- **可选**：是否需要自动重试策略？（例如限流错误自动退避，超时错误自动重试）
- **你的想法**？

### 6️⃣ 前端可视化的优先级
- **当前设计**：基本的拓扑展示、执行进度、对话记录
- **可选**：是否需要高级特性？（例如诊断结果的可视化对比、根因树状图）
- **你的想法**？

---

## 下一步

基于上述设计，我建议的实施顺序是：

**Phase 1: Agent 核心设计（已完成）**
- ✅ Orchestrator 设计和 Sub-Agent 设计
- ✅ 消息传递和配置系统
- ✅ 场景切换验证

**Phase 2: 支持系统设计**
- [ ] 详细设计配置管理系统（YAML 加载、验证、热更新）
- [ ] 详细设计轨迹记录系统（Checkpoint 链、RL 导出）
- [ ] 详细设计监控干预系统（Streaming、中断点）
- [ ] 详细设计故障恢复系统（错误分类、重试策略）

**Phase 3: 前端架构设计**
- [ ] WebSocket 事件流设计
- [ ] REST API 设计
- [ ] UI 组件设计（拓扑、进度、轨迹等）

**Phase 4: 实现**
- [ ] 核心框架编码
- [ ] 配置系统实现
- [ ] 轨迹系统实现
- [ ] 前端实现

---

## 问题清单

请你看完这份设计稿后，回答以下问题：

- [ ] **整体架构**是否符合你的预期？有没有需要调整的地方？
- [ ] **配置系统**的设计是否足够灵活？三层配置（system.yaml、orchestrator.yaml、agents.yaml）合理吗？
- [ ] **Prompt 管理**（YAML + Jinja2）的方案是否满足你的需求？
- [ ] **消息传递**（TypedDict + JSON）的设计是否足够清晰？有没有额外的需求？
- [ ] **监控干预**（stream + interrupt_before + update_state）的设计是否足够强大？
- [ ] **轨迹记录**（Checkpoint + Streaming）的方案能否满足你的 RL 训练需求？
- [ ] **故障恢复**的机制是否满足调试需求？
- [ ] **6 个待讨论的设计问题**中，哪些对你最重要？有其他关键问题吗？

---

## 附录：常见使用场景示例

### 场景 1: RCA 运维诊断

```python
# 初始化
orchestrator = create_orchestrator(
    scenario="rca",
    config_file="config/scenarios/rca.yaml"
)

# 执行
result = await graph.ainvoke(
    {"messages": [HumanMessage(content="API 响应时间从 200ms → 5s，找根本原因")]},
    config
)

# 输出
print(result["messages"][-1].content)
# [Analyzer] 根本原因: 数据库连接池耗尽（置信度 95%）
# 建议: 1) 增加连接池大小；2) 优化 N+1 查询
```

### 场景 2: 性能优化诊断

```python
# 只需改配置，代码不变
orchestrator = create_orchestrator(
    scenario="performance_optimization",
    config_file="config/scenarios/performance_optimization.yaml"
)

# 其他代码完全相同
result = await graph.ainvoke(...)
```

### 场景 3: 监控与干预

```python
# 实时监听执行
async for namespace, mode, data in graph.astream(
    input_data,
    config,
    stream_mode=["updates"],
    subgraphs=True,
):
    # 在某个 Agent 执行时中断
    if namespace == ("db_agent",) and mode == "updates":
        # 检查工具调用
        if "pending_tools" in data:
            # 可以修改或拒绝工具调用
            graph.update_state(config, {...})
            # 恢复执行
            result = graph.invoke(None, config)
```

### 场景 4: 故障恢复

```python
# 从失败点恢复
try:
    result = graph.invoke(input_data, config)
except Exception:
    # 检查失败点
    state = graph.get_state(config)

    # 修复并重试
    graph.update_state(config, {...})
    result = graph.invoke(None, config)
```

### 场景 5: 调试重放

```python
# 浏览历史
history = list(graph.get_state_history(config))

# 从第 3 步重新开始
target = history[3]
graph.update_state(target.config, {...})
debug_result = graph.invoke(None, target.config)
```

---

期待你的反馈！
