package blocks

import (
	"strings"

	"github.com/muesli/reflow/wordwrap"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// ThinkingBlock renders the agent's internal reasoning, collapsible.
type ThinkingBlock struct {
	Text      string
	collapsed bool
	focused   bool
}

// NewThinkingBlock creates a ThinkingBlock that starts collapsed.
func NewThinkingBlock() *ThinkingBlock {
	return &ThinkingBlock{collapsed: true}
}

func (b *ThinkingBlock) Kind() string        { return "thinking" }
func (b *ThinkingBlock) Collapsed() bool     { return b.collapsed }
func (b *ThinkingBlock) SetCollapsed(c bool) { b.collapsed = c }

// Focused reports whether this block has keyboard focus.
func (b *ThinkingBlock) Focused() bool { return b.focused }

// SetFocused sets the keyboard focus state.
func (b *ThinkingBlock) SetFocused(f bool) { b.focused = f }

func (b *ThinkingBlock) Render(width int, th *theme.Theme) string {
	var result string
	if b.collapsed {
		result = b.renderCollapsed(width, th)
	} else {
		result = b.renderExpanded(width, th)
	}
	if b.focused {
		return applyFocusBar(result, th)
	}
	return result
}

func (b *ThinkingBlock) renderCollapsed(_ int, th *theme.Theme) string {
	label := th.ThinkingLabel.Render(theme.ThinkingGlyph + " Thinking")
	hint := th.ThinkingHint.Render(" (ctrl+e to expand)")
	return label + hint
}

func (b *ThinkingBlock) renderExpanded(width int, th *theme.Theme) string {
	header := th.ThinkingLabel.Render(theme.ThinkingGlyph + " Thinking...")
	cw := width - 4 // 2 indent + 2 margin
	if cw < 20 {
		cw = 20
	}
	wrapped := wordwrap.String(b.Text, cw)
	body := th.ThinkingText.Render("  " + strings.ReplaceAll(wrapped, "\n", "\n  "))
	return header + "\n" + body
}

// applyFocusBar prefixes every line of text with the FocusBarGlyph styled with FocusBar.
func applyFocusBar(text string, th *theme.Theme) string {
	bar := th.FocusBar.Render(theme.FocusBarGlyph)
	lines := strings.Split(text, "\n")
	var sb strings.Builder
	for i, line := range lines {
		sb.WriteString(bar)
		sb.WriteString(line)
		if i < len(lines)-1 {
			sb.WriteByte('\n')
		}
	}
	return sb.String()
}
