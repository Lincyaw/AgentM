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
	spine := th.SpineUser.Render(theme.Spine)
	attrib := th.UserAttrib.Render(theme.LabelUser)
	content := util.Truncate(b.Content, contentWidth(width)*20) // generous limit, multi-line

	var sb strings.Builder
	sb.WriteString(spine + " " + attrib + "\n")
	for _, line := range strings.Split(content, "\n") {
		sb.WriteString(spine + "  " + line + "\n")
	}
	return strings.TrimRight(sb.String(), "\n")
}

// contentWidth returns the usable width after accounting for the spine gutter.
func contentWidth(width int) int {
	// spine (1 char rendered width ~1-2) + space = ~3 chars of gutter
	w := width - 4
	if w < 20 {
		w = 20
	}
	return w
}
