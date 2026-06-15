package blocks

import (
	"github.com/muesli/reflow/wordwrap"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
	"github.com/AoyangSpace/agentm-terminal/internal/util"
)

// UserTurn renders a user message with a background color, no label.
type UserTurn struct {
	Content string
}

func (b *UserTurn) Kind() string        { return "user" }
func (b *UserTurn) Collapsed() bool     { return false }
func (b *UserTurn) SetCollapsed(_ bool) {} // no-op: user turns are always expanded

func (b *UserTurn) Render(width int, th *theme.Theme) string {
	cw := contentWidth(width)
	content := util.Truncate(b.Content, cw*20)
	content = wordwrap.String(content, cw)
	style := th.UserMessageBg.PaddingLeft(1).PaddingRight(1)
	return style.Render(content)
}

// contentWidth returns the usable width after accounting for gutter.
func contentWidth(width int) int {
	w := width - 2
	if w < 20 {
		w = 20
	}
	return w
}
