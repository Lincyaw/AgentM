package blocks

import (
	"strings"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
	"github.com/AoyangSpace/agentm-terminal/internal/util"
)

// UserTurn renders a user message with attribution and left spine gutter.
type UserTurn struct {
	Content string
}

func (b *UserTurn) Kind() string        { return "user" }
func (b *UserTurn) Collapsed() bool     { return false }
func (b *UserTurn) SetCollapsed(_ bool) {} // no-op: user turns are always expanded

func (b *UserTurn) Render(width int, th *theme.Theme) string {
	attrib := th.UserAttrib.Render(theme.LabelUser)
	content := util.Truncate(b.Content, contentWidth(width)*20)

	var sb strings.Builder
	sb.WriteString(attrib + "\n")
	for _, line := range strings.Split(content, "\n") {
		sb.WriteString("  " + line + "\n")
	}
	return strings.TrimRight(sb.String(), "\n")
}

// contentWidth returns the usable width after accounting for the indent.
func contentWidth(width int) int {
	w := width - 2
	if w < 20 {
		w = 20
	}
	return w
}
