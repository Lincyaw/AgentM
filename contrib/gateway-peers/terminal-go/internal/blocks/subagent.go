package blocks

import (
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// SubagentBlock renders a sub-agent invocation as a single status line.
type SubagentBlock struct {
	Purpose string
	Done    bool
	Error   string // empty means success
}

func (b *SubagentBlock) Kind() string        { return "subagent" }
func (b *SubagentBlock) Collapsed() bool     { return false }
func (b *SubagentBlock) SetCollapsed(_ bool) {} // no-op: always one line

func (b *SubagentBlock) Render(_ int, th *theme.Theme) string {
	var dot string
	if !b.Done {
		dot = th.AssistantDotDim.Render(theme.BlackCircle)
	} else if b.Error != "" {
		dot = th.AssistantDotErr.Render(theme.BlackCircle)
	} else {
		dot = th.AssistantDotOK.Render(theme.BlackCircle)
	}
	return dot + " " + "subagent: " + b.Purpose
}
