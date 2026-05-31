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

func (b *SystemTurn) Render(width int, th *theme.Theme) string {
	_ = width
	spine := th.SpineSystem.Render(theme.Spine)
	label := theme.LabelSystem
	if b.Source != "" {
		label += "  (" + b.Source + ")"
	}
	attrib := th.SystemAttrib.Render(label)

	var sb strings.Builder
	sb.WriteString(spine + " " + attrib + "\n")
	for _, line := range strings.Split(b.Content, "\n") {
		sb.WriteString(spine + "  " + line + "\n")
	}
	return strings.TrimRight(sb.String(), "\n")
}
