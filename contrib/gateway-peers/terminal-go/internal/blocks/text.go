package blocks

import (
	"strings"

	"github.com/charmbracelet/glamour"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// TextBlock holds a streamed markdown text segment inside an AssistantTurn.
// It renders with the "● " assistant-dot prefix on its first line and uses
// glamour word-wrap once complete is true.
type TextBlock struct {
	Text         string
	GlamourStyle string // "dark" or "light"
	complete     bool

	glamourCache      string
	glamourCacheWidth int
	glamourCacheText  string
}

func (b *TextBlock) Kind() string        { return "text" }
func (b *TextBlock) Collapsed() bool     { return false }
func (b *TextBlock) SetCollapsed(_ bool) {}

// SetComplete marks the block as finished so glamour rendering kicks in.
func (b *TextBlock) SetComplete() { b.complete = true }

// Complete reports whether the text block has finished streaming.
func (b *TextBlock) Complete() bool { return b.complete }

func (b *TextBlock) Render(width int, th *theme.Theme) string {
	if b.Text == "" {
		return ""
	}
	cw := width - 2 // 2 chars for "● " prefix
	if cw < 20 {
		cw = 20
	}
	rendered := b.renderText(cw)
	dot := th.AssistantDot.Render(theme.BlackCircle)
	lines := strings.Split(rendered, "\n")
	var sb strings.Builder
	if len(lines) > 0 {
		sb.WriteString(dot + " " + lines[0] + "\n")
		for _, line := range lines[1:] {
			sb.WriteString("  " + line + "\n")
		}
	}
	return strings.TrimRight(sb.String(), "\n")
}

func (b *TextBlock) renderText(width int) string {
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
	out, err := r.Render(b.Text)
	if err != nil {
		return b.Text
	}
	result := strings.TrimSpace(out)
	b.glamourCache = result
	b.glamourCacheWidth = width
	b.glamourCacheText = b.Text
	return result
}
