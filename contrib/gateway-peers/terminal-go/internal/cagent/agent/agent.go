// Package agent is a tiny data-type shim of cagent's pkg/agent.
//
// The full cagent Agent carries model/provider/cache/harness machinery driven
// by the local runtime loop. In the agentm-terminal peer that runtime is
// replaced by the gateway wire protocol, so the only consumers of this package
// are the vendored session/chat data-types. They reference the descriptive
// surface of an agent (Name, Model, Provider, Thinking, Description) plus a
// handful of accessors used when building invariant system messages
// (Name, Description, HasSubAgents, SubAgents, Handoffs, Instruction,
// NumHistoryItems).
//
// Everything provider/model-heavy (model selection, fallbacks, overrides,
// hooks, cache, harness) has been dropped. The UI sidebar reads the same
// descriptive fields off runtime.AgentDetails, not off this type.
package agent

// Agent is the minimal agent identity consumed by the session/chat seam.
//
// Descriptive fields (Model, Provider, Thinking) are exported so the adapter
// can populate them directly from wire data. Name/Description are kept behind
// accessor methods because the vendored session package calls them as methods.
type Agent struct {
	name        string
	description string
	instruction string

	// Model is the model identifier in effect for this agent (e.g. "gpt-4o").
	Model string
	// Provider is the provider portion of the model reference (e.g. "openai").
	Provider string
	// Thinking is the human-readable thinking/effort level (e.g. "high").
	Thinking string

	numHistoryItems int
	subAgents       []*Agent
	handoffs        []*Agent
	parents         []*Agent
}

// New creates a new agent with the given name and applies the options.
func New(name string, opts ...Opt) *Agent {
	a := &Agent{name: name}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Name returns the agent's name.
func (a *Agent) Name() string { return a.name }

// Description returns the agent's description.
func (a *Agent) Description() string { return a.description }

// Instruction returns the agent's system instruction.
func (a *Agent) Instruction() string { return a.instruction }

// NumHistoryItems returns the number of history items the agent retains.
func (a *Agent) NumHistoryItems() int { return a.numHistoryItems }

// SubAgents returns the list of sub-agents.
func (a *Agent) SubAgents() []*Agent { return a.subAgents }

// HasSubAgents reports whether the agent has any sub-agents.
func (a *Agent) HasSubAgents() bool { return len(a.subAgents) > 0 }

// Handoffs returns the list of handoff agents.
func (a *Agent) Handoffs() []*Agent { return a.handoffs }

// Parents returns the list of parent agents.
func (a *Agent) Parents() []*Agent { return a.parents }
