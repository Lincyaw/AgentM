package agent

import (
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/tools"
)

// Opt configures an Agent at construction time.
type Opt func(a *Agent)

// WithDescription sets the agent's description.
func WithDescription(description string) Opt {
	return func(a *Agent) { a.description = description }
}

// WithInstruction sets the agent's system instruction.
func WithInstruction(instruction string) Opt {
	return func(a *Agent) { a.instruction = instruction }
}

// WithName overrides the agent's name.
func WithName(name string) Opt {
	return func(a *Agent) { a.name = name }
}

// WithModel sets the model identifier in effect for this agent.
func WithModel(model string) Opt {
	return func(a *Agent) { a.Model = model }
}

// WithProvider sets the provider portion of the model reference.
func WithProvider(provider string) Opt {
	return func(a *Agent) { a.Provider = provider }
}

// WithThinking sets the human-readable thinking/effort level.
func WithThinking(thinking string) Opt {
	return func(a *Agent) { a.Thinking = thinking }
}

// WithNumHistoryItems sets the number of history items the agent retains.
func WithNumHistoryItems(n int) Opt {
	return func(a *Agent) { a.numHistoryItems = n }
}

// WithSubAgents sets the agent's sub-agents and records this agent as their parent.
func WithSubAgents(subAgents ...*Agent) Opt {
	return func(a *Agent) {
		a.subAgents = subAgents
		for _, sub := range subAgents {
			sub.parents = append(sub.parents, a)
		}
	}
}

// WithHandoffs sets the agent's handoff agents.
func WithHandoffs(handoffs ...*Agent) Opt {
	return func(a *Agent) { a.handoffs = handoffs }
}

// WithToolSets sets the agent's toolsets.
func WithToolSets(toolSets ...tools.ToolSet) Opt {
	return func(a *Agent) { a.toolsets = toolSets }
}
