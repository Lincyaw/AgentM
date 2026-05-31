package blocks

import (
	"fmt"

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
	glyph := theme.ToolRunning
	if b.Done {
		if b.Error == "" {
			glyph = theme.ToolOK
		} else {
			glyph = theme.ToolError
		}
	}
	line := fmt.Sprintf("%s subagent: %s  %s", theme.PhaseGlyphMap[theme.PhaseSubagent], b.Purpose, glyph)
	return th.ToolTitle.Render(line)
}
