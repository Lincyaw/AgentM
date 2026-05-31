package blocks

import (
	"strings"

	"github.com/charmbracelet/glamour"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// AssistantTurn is a composite block representing a full assistant response.
// It aggregates thinking, markdown text, tool calls, sub-agent invocations,
// and approval prompts into a single renderable unit.
type AssistantTurn struct {
	Thinking     *ThinkingBlock
	Text         string // markdown source
	TextDirty    bool   // true while text is still being streamed
	ThinkDirty   bool   // true while thinking is still being streamed
	GlamourStyle string // "dark", "light", or "" (defaults to auto)
	Tools        []*ToolBlock
	Children     []*SubagentBlock
	Approvals    []*ApprovalBlock
	complete     bool // true after the full assistant_text event
}

func (b *AssistantTurn) Kind() string        { return "assistant" }
func (b *AssistantTurn) Collapsed() bool     { return false }
func (b *AssistantTurn) SetCollapsed(_ bool) {} // no-op: assistant turns don't collapse as a whole

// SetComplete marks the turn as finished (no longer streaming).
func (b *AssistantTurn) SetComplete() { b.complete = true }

// Complete reports whether the assistant turn has finished streaming.
func (b *AssistantTurn) Complete() bool { return b.complete }

func (b *AssistantTurn) Render(width int, th *theme.Theme) string {
	attrib := th.AssistantAttrib.Render(theme.LabelAssistant)
	indent := "  "
	contentWidth := width - len(indent)
	if contentWidth < 20 {
		contentWidth = 20
	}

	var sb strings.Builder
	sb.WriteString(attrib + "\n")

	// Thinking block (if present)
	if b.Thinking != nil && b.Thinking.Text != "" {
		sb.WriteString(prefixLines(b.Thinking.Render(contentWidth, th), indent) + "\n")
	}

	// Main text
	if b.Text != "" {
		rendered := b.renderText(contentWidth)
		sb.WriteString(prefixLines(rendered, indent) + "\n")
	}

	// Tool blocks
	for _, tool := range b.Tools {
		sb.WriteString(prefixLines(tool.Render(contentWidth, th), indent) + "\n")
	}

	// Sub-agent blocks
	for _, child := range b.Children {
		sb.WriteString(prefixLines(child.Render(contentWidth, th), indent) + "\n")
	}

	// Approval blocks
	for _, appr := range b.Approvals {
		sb.WriteString(prefixLines(appr.Render(contentWidth, th), indent) + "\n")
	}

	return strings.TrimRight(sb.String(), "\n")
}

// renderText returns the text content rendered with glamour markdown.
// During active streaming (TextDirty && !complete), returns raw text to
// avoid re-rendering partial markdown every flush tick.
func (b *AssistantTurn) renderText(width int) string {
	if b.TextDirty && !b.complete {
		return b.Text
	}

	style := "dark"
	if b.GlamourStyle == "light" {
		style = "light"
	}
	r, err := glamour.NewTermRenderer(
		glamour.WithWordWrap(width),
		glamour.WithStandardStyle(style),
	)
	if err != nil {
		return b.Text
	}
	rendered, err := r.Render(b.Text)
	if err != nil {
		return b.Text
	}
	return strings.TrimSpace(rendered)
}

// prefixLines prepends prefix to every line of s.
func prefixLines(s, prefix string) string {
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		lines[i] = prefix + line
	}
	return strings.Join(lines, "\n")
}
