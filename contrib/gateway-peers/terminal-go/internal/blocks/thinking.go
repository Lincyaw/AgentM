package blocks

import (
	"strings"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// ThinkingBlock renders the agent's internal reasoning, collapsible.
type ThinkingBlock struct {
	Text      string
	collapsed bool
}

// NewThinkingBlock creates a ThinkingBlock that starts collapsed.
func NewThinkingBlock() *ThinkingBlock {
	return &ThinkingBlock{collapsed: true}
}

func (b *ThinkingBlock) Kind() string        { return "thinking" }
func (b *ThinkingBlock) Collapsed() bool     { return b.collapsed }
func (b *ThinkingBlock) SetCollapsed(c bool) { b.collapsed = c }

func (b *ThinkingBlock) Render(width int, th *theme.Theme) string {
	if b.collapsed {
		return b.renderCollapsed(width, th)
	}
	return b.renderExpanded(width, th)
}

func (b *ThinkingBlock) renderCollapsed(_ int, th *theme.Theme) string {
	label := th.ThinkingLabel.Render(theme.ThinkingGlyph + " Thinking")
	hint := th.ThinkingHint.Render(" (ctrl+e to expand)")
	return label + hint
}

func (b *ThinkingBlock) renderExpanded(_ int, th *theme.Theme) string {
	header := th.ThinkingLabel.Render(theme.ThinkingGlyph + " Thinking...")
	body := th.ThinkingText.Render("  " + strings.ReplaceAll(b.Text, "\n", "\n  "))
	return header + "\n" + body
}
