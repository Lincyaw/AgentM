// Package types holds shimmed cagent config data-types consumed by the UI
// seam. Only the symbols referenced by the carved runtime/event.go are kept.
package types

// Command represents an agent command with optional metadata.
type Command struct {
	// Description is shown in completion dialogs and help text.
	Description string `json:"description,omitempty"`

	// Instruction is the prompt sent to the agent.
	Instruction string `json:"instruction,omitempty"`

	// Agent, when set, switches the active agent to the named sub-agent.
	Agent string `json:"agent,omitempty"`
}

// DisplayText returns the text to show in completion dialogs.
func (c Command) DisplayText() string {
	if c.Description != "" {
		return c.Description
	}
	if c.Instruction != "" {
		return c.Instruction
	}
	if c.Agent != "" {
		return "Switch to " + c.Agent
	}
	return ""
}

// Commands represents a set of named prompts for quick-starting conversations.
type Commands map[string]Command
