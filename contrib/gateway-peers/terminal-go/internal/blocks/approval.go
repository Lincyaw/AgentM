package blocks

import (
	"fmt"
	"strings"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
	"github.com/AoyangSpace/agentm-terminal/internal/util"
)

// Button represents an approval action the user can select.
type Button struct {
	Label string
	Value string
	Style string // "primary" or "danger"
}

// ApprovalBlock renders an approval prompt with action buttons.
type ApprovalBlock struct {
	Content   string
	Buttons   []Button
	collapsed bool
	focused   bool
}

// NewApprovalBlock creates an ApprovalBlock in collapsed state.
func NewApprovalBlock(content string, buttons []Button) *ApprovalBlock {
	return &ApprovalBlock{Content: content, Buttons: buttons, collapsed: true}
}

func (b *ApprovalBlock) Kind() string        { return "approval" }
func (b *ApprovalBlock) Collapsed() bool     { return b.collapsed }
func (b *ApprovalBlock) SetCollapsed(c bool) { b.collapsed = c }

// Focused reports whether this block has keyboard focus.
func (b *ApprovalBlock) Focused() bool { return b.focused }

// SetFocused sets the keyboard focus state.
func (b *ApprovalBlock) SetFocused(f bool) { b.focused = f }

func (b *ApprovalBlock) Render(width int, th *theme.Theme) string {
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

func (b *ApprovalBlock) renderCollapsed(_ int, th *theme.Theme) string {
	firstLine := firstLineOf(b.Content)
	header := th.ApprovalWarn.Render("⚠ " + util.Truncate(firstLine, 60))
	buttons := b.renderButtons(th)
	return header + "\n" + buttons
}

func (b *ApprovalBlock) renderExpanded(_ int, th *theme.Theme) string {
	header := th.ApprovalWarn.Render("⚠ Approval Required")
	body := th.ApprovalChoice.Render("  " + strings.ReplaceAll(b.Content, "\n", "\n  "))
	buttons := b.renderButtons(th)
	return header + "\n" + body + "\n" + buttons
}

func (b *ApprovalBlock) renderButtons(th *theme.Theme) string {
	var parts []string
	for i, btn := range b.Buttons {
		parts = append(parts, th.ApprovalChoice.Render(fmt.Sprintf("[%d] %s", i+1, btn.Label)))
	}
	parts = append(parts, th.ApprovalChoice.Render("[?] Details"))
	return "  " + strings.Join(parts, "  ")
}

func firstLineOf(s string) string {
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		return s[:idx]
	}
	return s
}
