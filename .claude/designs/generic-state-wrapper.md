# Design: 通用的 State 管理 SDK Wrapper

**Status**: DRAFT
**Last Updated**: 2026-03-07

## 核心思想

你提出的关键洞察：

> 在工程实现上，不同的 Agent 系统（假设驱动、顺序诊断、决策树等）的区别，其实就是**中间记录的状态的区别**。所以应该设计一个通用的 SDK Wrapper，只改变少部分代码就可以支持不同用途的 Agent System。

这个设计思想解决了什么问题？

```
❌ 问题：为每个不同的 Agent 系统写不同的代码
RCA (假设驱动)  → 一套代码
性能优化 (顺序) → 另一套代码
决策树        → 再一套代码

✅ 解决方案：写一个通用的框架，通过改变配置支持多种系统
核心框架：一套代码
├─ 假设驱动 RCA    → 配置 + 少量代码
├─ 顺序诊断        → 配置 + 少量代码
└─ 决策树分类      → 配置 + 少量代码
```

---

## SDK Wrapper 的结构

### 概览

```
┌─────────────────────────────────────────────┐
│     Generic SDK Wrapper (通用框架)           │
├─────────────────────────────────────────────┤
│                                             │
│  1. State Management (通用状态管理)         │
│     - StateSchema 定义                     │
│     - StateReducer 定义                    │
│     - StateValidator                       │
│                                             │
│  2. Phase Management (通用阶段管理)         │
│     - Phase 定义（exploration, etc）      │
│     - 阶段转换逻辑                         │
│     - 阶段检查点                           │
│                                             │
│  3. Orchestrator Core (通用 Orchestrator)  │
│     - 基础路由逻辑                         │
│     - Sub-Agent 调度                      │
│     - 状态更新                             │
│                                             │
│  4. Trajectory Recording (通用轨迹记录)    │
│     - Checkpoint 管理                      │
│     - Event 流捕获                         │
│     - 导出机制                             │
│                                             │
│  5. Configuration System (通用配置系统)    │
│     - 配置加载和验证                       │
│     - 动态能力注入                         │
│                                             │
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│  具体 Agent System（通过配置区分）          │
├─────────────────────────────────────────────┤
│                                             │
│  Hypothesis-Driven RCA                     │
│  ├─ StateSchema: notebook + hypotheses     │
│  ├─ Phases: exploration, generation, ...  │
│  └─ Orchestrator Logic: 假设验证          │
│                                             │
│  Sequential Diagnosis                      │
│  ├─ StateSchema: steps + results           │
│  ├─ Phases: step-by-step                  │
│  └─ Orchestrator Logic: 顺序执行          │
│                                             │
│  Decision Tree Classification               │
│  ├─ StateSchema: decision_path + features │
│  ├─ Phases: node-by-node                  │
│  └─ Orchestrator Logic: 递归分类          │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 通用的状态管理框架

### 1. StateSchema 的参数化

**基础 StateSchema 接口**:
```python
from typing import TypedDict, Annotated, Any
import operator

class BaseExecutorState(TypedDict):
    """所有 Agent System 共有的基础状态"""
    messages: Annotated[list, operator.add]  # 通用消息
    task_id: str
    task_description: str
    current_phase: str  # 当前所在的阶段

    # ⭐ 动态字段：由具体系统定义
    # custom_state: dict  # 由配置注入


class HypothesisDrivenState(BaseExecutorState):
    """假设驱动 RCA 的状态"""
    notebook: DiagnosticNotebook  # 假设驱动特有
    current_hypothesis: Optional[str]


class SequentialDiagnosisState(BaseExecutorState):
    """顺序诊断的状态"""
    steps: Annotated[list[dict], operator.add]  # 顺序特有
    current_step_index: int


class DecisionTreeState(BaseExecutorState):
    """决策树分类的状态"""
    decision_path: list[str]  # 决策树特有
    current_node_id: str
    feature_values: dict[str, Any]
```

### 2. StateSchema 工厂

```python
class StateSchemaFactory:
    """根据配置动态生成 StateSchema"""

    @staticmethod
    def create_state_schema(system_type: str, custom_fields: dict = None) -> type:
        """
        动态创建 StateSchema

        Args:
            system_type: "hypothesis_driven", "sequential", "decision_tree"
            custom_fields: 额外的自定义字段

        Returns:
            动态生成的 TypedDict 类
        """

        # 基础字段
        base_fields = {
            "messages": Annotated[list, operator.add],
            "task_id": str,
            "task_description": str,
            "current_phase": str
        }

        # 系统特定字段
        system_fields = {
            "hypothesis_driven": {
                "notebook": DiagnosticNotebook,
                "hypothesis_order": list[str]
            },
            "sequential": {
                "steps": Annotated[list[dict], operator.add],
                "current_step_index": int,
                "completed_steps": list[str]
            },
            "decision_tree": {
                "decision_path": list[str],
                "current_node_id": str,
                "feature_values": dict
            }
        }.get(system_type, {})

        # 合并
        all_fields = {**base_fields, **system_fields}

        # 添加自定义字段
        if custom_fields:
            all_fields.update(custom_fields)

        # 动态创建 TypedDict
        return TypedDict("ExecutorState", all_fields)
```

---

## 通用的阶段管理框架

### Phase 的定义和管理

```python
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class Phase:
    """阶段定义"""
    name: str
    description: str
    handler: Callable  # 该阶段的处理函数
    next_phases: list[str]  # 可能的下一个阶段
    on_enter: Optional[Callable] = None  # 进入该阶段时的回调
    on_exit: Optional[Callable] = None   # 退出该阶段时的回调


class PhaseManager:
    """通用的阶段管理器"""

    def __init__(self, phases: dict[str, Phase], initial_phase: str):
        """
        Args:
            phases: {phase_name: Phase 对象}
            initial_phase: 初始阶段
        """
        self.phases = phases
        self.current_phase = initial_phase
        self.phase_history = [initial_phase]

    def get_current_phase(self) -> Phase:
        return self.phases[self.current_phase]

    async def execute_current_phase(self, state: dict) -> dict:
        """执行当前阶段"""
        phase = self.get_current_phase()

        # 执行 on_enter 回调
        if phase.on_enter:
            state = await phase.on_enter(state)

        # 执行阶段主逻辑
        state = await phase.handler(state)

        # 执行 on_exit 回调
        if phase.on_exit:
            state = await phase.on_exit(state)

        return state

    def transition_to(self, next_phase: str):
        """转移到下一个阶段"""
        current = self.get_current_phase()

        if next_phase not in current.next_phases:
            raise ValueError(f"Cannot transition from {self.current_phase} to {next_phase}")

        self.current_phase = next_phase
        self.phase_history.append(next_phase)

    def can_transition_to(self, next_phase: str) -> bool:
        current = self.get_current_phase()
        return next_phase in current.next_phases
```

### 具体系统的 Phase 定义

#### 假设驱动 RCA

```python
# 定义假设驱动 RCA 的 Phase

async def phase_exploration(state: dict) -> dict:
    """Phase 1: 初步全面调查"""
    notebook = state["notebook"]
    # ... 收集数据的逻辑
    return state


async def phase_hypothesis_generation(state: dict) -> dict:
    """Phase 2: 生成假设"""
    notebook = state["notebook"]
    # ... 生成假设的逻辑
    return state


async def phase_hypothesis_verification(state: dict) -> dict:
    """Phase 3: 验证假设"""
    notebook = state["notebook"]
    # ... 验证假设的逻辑
    return state


async def phase_confirmation(state: dict) -> dict:
    """Phase 4: 确认根因"""
    notebook = state["notebook"]
    # ... 输出根因的逻辑
    return state


# 使用 PhaseManager 注册这些 Phase
hypothesis_driven_phases = {
    "exploration": Phase(
        name="exploration",
        description="Initial data collection",
        handler=phase_exploration,
        next_phases=["hypothesis_generation"]
    ),
    "hypothesis_generation": Phase(
        name="hypothesis_generation",
        description="Generate candidate hypotheses",
        handler=phase_hypothesis_generation,
        next_phases=["hypothesis_verification"]
    ),
    "hypothesis_verification": Phase(
        name="hypothesis_verification",
        description="Verify hypotheses one by one",
        handler=phase_hypothesis_verification,
        next_phases=["hypothesis_verification", "confirmation"],
        # 可以循环验证或进入确认
    ),
    "confirmation": Phase(
        name="confirmation",
        description="Confirm root cause",
        handler=phase_confirmation,
        next_phases=[]  # 终止
    )
}

phase_manager = PhaseManager(
    phases=hypothesis_driven_phases,
    initial_phase="exploration"
)
```

#### 顺序诊断

```python
async def phase_step_1(state: dict) -> dict:
    """Step 1: 检查基础设施"""
    # ...
    state["steps"].append({"name": "infrastructure", "result": ...})
    state["current_step_index"] = 1
    return state


async def phase_step_2(state: dict) -> dict:
    """Step 2: 检查日志"""
    # ...
    state["steps"].append({"name": "logs", "result": ...})
    state["current_step_index"] = 2
    return state


# 更简单，是线性的
sequential_phases = {
    "step_1": Phase("step_1", "Check infrastructure", phase_step_1, ["step_2"]),
    "step_2": Phase("step_2", "Check logs", phase_step_2, ["step_3"]),
    "step_3": Phase("step_3", "Final analysis", phase_step_3, [])
}

phase_manager = PhaseManager(sequential_phases, "step_1")
```

---

## 通用的 Orchestrator 核心

### 基础 Orchestrator 类

```python
from abc import ABC, abstractmethod


class BaseOrchestrator(ABC):
    """通用 Orchestrator 基类"""

    def __init__(
        self,
        state_schema: type,
        phase_manager: PhaseManager,
        sub_agents: dict[str, CompiledGraph],
        config: dict
    ):
        self.state_schema = state_schema
        self.phase_manager = phase_manager
        self.sub_agents = sub_agents
        self.config = config

    async def execute(self, initial_input: dict) -> dict:
        """执行诊断流程"""

        # 初始化状态
        state = self._initialize_state(initial_input)

        # 执行各个 Phase
        while self.phase_manager.current_phase != "completed":
            state = await self.phase_manager.execute_current_phase(state)

            # 由具体系统决定下一个 Phase
            next_phase = await self._decide_next_phase(state)

            if self.phase_manager.can_transition_to(next_phase):
                self.phase_manager.transition_to(next_phase)
            else:
                break

        return state

    def _initialize_state(self, initial_input: dict) -> dict:
        """初始化状态"""
        return {
            "messages": [HumanMessage(content=initial_input["task"])],
            "task_id": initial_input.get("task_id", str(uuid.uuid4())),
            "task_description": initial_input["task"],
            "current_phase": self.phase_manager.current_phase
        }

    @abstractmethod
    async def _decide_next_phase(self, state: dict) -> str:
        """
        决定下一个 Phase
        由具体系统（假设驱动、顺序等）实现
        """
        pass

    async def dispatch_to_agent(self, agent_id: str, task: dict) -> dict:
        """分配任务给 Sub-Agent"""
        agent = self.sub_agents[agent_id]
        return await agent.ainvoke(task)


class HypothesisDrivenOrchestrator(BaseOrchestrator):
    """假设驱动 RCA 的 Orchestrator"""

    async def _decide_next_phase(self, state: dict) -> str:
        notebook = state["notebook"]

        if notebook.confirmed_hypothesis:
            return "confirmation"
        elif all(h.status in ["confirmed", "rejected"] for h in notebook.hypotheses.values()):
            return "confirmation"
        else:
            return "hypothesis_verification"  # 继续验证


class SequentialOrchestrator(BaseOrchestrator):
    """顺序诊断的 Orchestrator"""

    async def _decide_next_phase(self, state: dict) -> str:
        current_step = state["current_step_index"]
        total_steps = len(self.phase_manager.phases)

        if current_step >= total_steps - 1:
            return "completed"
        else:
            next_step_id = f"step_{current_step + 2}"
            return next_step_id
```

---

## SDK Wrapper 的使用方式

### 方式 1: 假设驱动 RCA

```python
from agentm.sdk import AgentSystemBuilder, StateSchemaFactory, PhaseManager

# 1. 创建 StateSchema
state_schema = StateSchemaFactory.create_state_schema("hypothesis_driven")

# 2. 创建 Phase Manager（从配置文件加载）
config = load_yaml("config/scenarios/rca.yaml")
phase_manager = PhaseManager.from_config(config)

# 3. 构建 Orchestrator
orchestrator = AgentSystemBuilder.build(
    system_type="hypothesis_driven",
    config=config,
    state_schema=state_schema,
    phase_manager=phase_manager
)

# 4. 执行
result = await orchestrator.execute({
    "task": "API 响应时间从 200ms → 5s，找根本原因"
})
```

### 方式 2: 顺序诊断（只需改配置和很少代码）

```python
# 1. 改配置（新建 config/scenarios/sequential.yaml）
config = load_yaml("config/scenarios/sequential.yaml")

# 2. 创建 StateSchema（自动）
state_schema = StateSchemaFactory.create_state_schema("sequential")

# 3. 创建 Phase Manager（从配置自动生成）
phase_manager = PhaseManager.from_config(config)

# 4. 构建 Orchestrator（同样的 API）
orchestrator = AgentSystemBuilder.build(
    system_type="sequential",
    config=config,
    state_schema=state_schema,
    phase_manager=phase_manager
)

# 5. 执行（完全相同）
result = await orchestrator.execute({
    "task": "检查系统性能"
})

# ✅ 看，代码几乎没变！只改了配置和 system_type
```

### 方式 3: 自定义系统

```python
# 完全自定义一个新的 Agent System

# Step 1: 定义自定义 State 字段
custom_fields = {
    "decision_tree": str,
    "current_node": str,
    "branch_history": list[str]
}

# Step 2: 创建 StateSchema
state_schema = StateSchemaFactory.create_state_schema(
    "custom",  # 自定义类型
    custom_fields=custom_fields
)

# Step 3: 定义 Phase 处理函数
async def phase_decision_tree(state: dict) -> dict:
    """自定义的决策树阶段"""
    # ... 自定义逻辑
    return state

# Step 4: 创建 Phase Manager
custom_phases = {
    "decision_tree": Phase(
        "decision_tree",
        "Navigate decision tree",
        phase_decision_tree,
        ["decision_tree", "completed"]
    ),
    "completed": Phase(
        "completed",
        "Done",
        lambda s: s,
        []
    )
}

phase_manager = PhaseManager(custom_phases, "decision_tree")

# Step 5: 创建 Orchestrator
class CustomOrchestrator(BaseOrchestrator):
    async def _decide_next_phase(self, state: dict) -> str:
        # 自定义决策逻辑
        return "completed"

orchestrator = CustomOrchestrator(
    state_schema=state_schema,
    phase_manager=phase_manager,
    sub_agents={...},
    config={...}
)

# ✅ 用同样的 execute() 接口
result = await orchestrator.execute(input_data)
```

---

## 配置系统的参数化

### 通用配置结构

```yaml
# config/scenarios/<system_type>.yaml

system:
  type: "hypothesis_driven"  # or "sequential", "decision_tree", "custom"
  description: "RCA with hypothesis-driven reasoning"

state_schema:
  base_fields:
    - messages
    - task_id
    - task_description
    - current_phase
  system_fields:
    notebook: "DiagnosticNotebook"  # 自动注入
    hypothesis_order: "list[str]"

phases:
  exploration:
    handler: "phase_exploration"
    next_phases:
      - hypothesis_generation
    on_enter: "init_notebook"
    on_exit: "log_exploration_complete"

  hypothesis_generation:
    handler: "phase_hypothesis_generation"
    next_phases:
      - hypothesis_verification

  hypothesis_verification:
    handler: "phase_hypothesis_verification"
    next_phases:
      - hypothesis_verification  # 可循环
      - confirmation

  confirmation:
    handler: "phase_confirmation"
    next_phases: []

orchestrator:
  decision_logic:
    class: "HypothesisDrivenOrchestrator"  # 自动加载
    methods:
      _decide_next_phase: "decide_by_notebook"

sub_agents:
  infrastructure_agent: {...}
  log_agent: {...}
  db_agent: {...}
```

---

## 好处总结

### 1️⃣ 代码复用最大化

```
假设驱动 RCA: 600 行代码（Orchestrator + Phase 处理）
顺序诊断:    200 行代码（配置 + Phase 处理）
决策树:      200 行代码（配置 + Phase 处理）
总计:        只有 1000 行，而不是 3×600 = 1800 行
```

### 2️⃣ 新系统添加变得非常简单

```
添加新的 Agent System:
1. 写配置文件（config/scenarios/new_system.yaml）
2. 实现 Phase 处理函数（可以复用很多）
3. 实现 Orchestrator 的 _decide_next_phase 方法
4. 完成！

总工作量: 50-100 行代码
```

### 3️⃣ 核心框架非常稳定

```
一旦 SDK Wrapper 完成，后续开发都是：
- 配置系统的改变
- Phase 处理函数的改变
- Orchestrator 决策逻辑的改变

核心框架（StateSchemaFactory, PhaseManager, BaseOrchestrator）不需改动
```

### 4️⃣ 易于学习和维护

```
新开发者只需理解：
1. StateSchema 如何定义（很简单）
2. Phase 如何实现（就是一个异步函数）
3. Orchestrator 如何决策（就是选择下一个 Phase）

不需要理解 3 个不同的代码库
```

---

## 实现路线图

### Phase 1: 通用框架
- [ ] StateSchemaFactory
- [ ] PhaseManager
- [ ] BaseOrchestrator
- [ ] 配置加载系统

### Phase 2: 假设驱动 RCA
- [ ] HypothesisDrivenOrchestrator
- [ ] Phase 处理函数
- [ ] 假设驱动特定的配置

### Phase 3: 顺序诊断
- [ ] SequentialOrchestrator
- [ ] Phase 处理函数
- [ ] 配置示例

### Phase 4: 决策树
- [ ] DecisionTreeOrchestrator
- [ ] Phase 处理函数
- [ ] 配置示例

### Phase 5: 轨迹和 RL
- [ ] 通用的轨迹导出系统
- [ ] 与不同系统的适配

---

## 和之前设计的关系

这个 SDK Wrapper 设计**不改变之前的架构**，只是提供一个**实现框架**，使得：

1. **假设驱动 RCA** 的设计完整支持
2. **其他系统**（顺序、决策树等）也可以轻松支持
3. **代码最大化复用**
4. **配置最大化灵活性**

核心思想是：**在配置层面区分不同系统，在代码层面复用通用框架**。
