package blocks

import (
	"strings"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// SystemTurn renders a system message with source attribution.
type SystemTurn struct {
	Content string
	Source  string // extension name that injected the message
}

func (b *SystemTurn) Kind() string        { return "system" }
func (b *SystemTurn) Collapsed() bool     { return false }
func (b *SystemTurn) SetCollapsed(_ bool) {} // no-op

func (b *SystemTurn) Render(_ int, th *theme.Theme) string {
	label := th.SystemAttrib.Render("system")
	if b.Source != "" {
		label += th.ThinkingHint.Render(" (" + b.Source + ")")
	}

	var sb strings.Builder
	sb.WriteString(label + "\n")
	sb.WriteString("  " + b.Content)
	return sb.String()
}
