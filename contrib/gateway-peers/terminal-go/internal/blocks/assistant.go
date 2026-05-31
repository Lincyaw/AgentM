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

	glamourCache      string
	glamourCacheWidth int
	glamourCacheText  string
}

func (b *AssistantTurn) Kind() string        { return "assistant" }
func (b *AssistantTurn) Collapsed() bool     { return false }
func (b *AssistantTurn) SetCollapsed(_ bool) {} // no-op: assistant turns don't collapse as a whole

// SetComplete marks the turn as finished (no longer streaming).
func (b *AssistantTurn) SetComplete() { b.complete = true }

// Complete reports whether the assistant turn has finished streaming.
func (b *AssistantTurn) Complete() bool { return b.complete }

func (b *AssistantTurn) Render(width int, th *theme.Theme) string {
	cw := width - 2 // 2 chars for "● " prefix
	if cw < 20 {
		cw = 20
	}

	var sb strings.Builder

	// Thinking block (if present)
	if b.Thinking != nil && b.Thinking.Text != "" {
		sb.WriteString(b.Thinking.Render(width, th) + "\n")
	}

	// Main text with ● prefix on first line
	if b.Text != "" {
		rendered := b.renderText(cw)
		dot := th.AssistantDot.Render(theme.BlackCircle)
		lines := strings.Split(rendered, "\n")
		if len(lines) > 0 {
			sb.WriteString(dot + " " + lines[0] + "\n")
			for _, line := range lines[1:] {
				sb.WriteString("  " + line + "\n")
			}
		}
	}

	// Tool blocks
	for _, tool := range b.Tools {
		sb.WriteString(tool.Render(width, th) + "\n")
	}

	// Sub-agent blocks
	for _, child := range b.Children {
		sb.WriteString(child.Render(width, th) + "\n")
	}

	// Approval blocks
	for _, appr := range b.Approvals {
		sb.WriteString(appr.Render(width, th) + "\n")
	}

	return strings.TrimRight(sb.String(), "\n")
}

func (b *AssistantTurn) renderText(width int) string {
	if !b.complete {
		return b.Text
	}

	if b.glamourCache != "" && b.glamourCacheWidth == width && b.glamourCacheText == b.Text {
		return b.glamourCache
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
	result := strings.TrimSpace(rendered)
	b.glamourCache = result
	b.glamourCacheWidth = width
	b.glamourCacheText = b.Text
	return result
}
