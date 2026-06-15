package blocks

import (
	"strings"

	"github.com/charmbracelet/glamour"
	"github.com/muesli/reflow/wordwrap"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// TextBlock holds a streamed markdown text segment inside an AssistantTurn.
// It renders with the "● " assistant-dot prefix on its first line and uses
// glamour word-wrap once complete is true.
//
// During streaming (complete=false) it uses incremental rendering: the text
// is split at the last safe block boundary (a blank line outside code fences)
// and only the trailing unstable tail is re-rendered on each update. The
// rendered prefix is cached and reused as long as the input prefix matches.
type TextBlock struct {
	Text         string
	GlamourStyle string // "dark" or "light"
	complete     bool

	// glamour cache for the final (complete) render.
	glamourCache      string
	glamourCacheWidth int
	glamourCacheText  string

	// Streaming cache: during streaming the assembled output is cached so
	// the Render() path can short-circuit when the text has not changed.
	streamCache      string
	streamCacheWidth int
	streamCacheText  string

	// Incremental rendering state for streaming.
	prefixText     string
	prefixRendered string
	prefixWidth    int
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
	if b.streamCacheText == b.Text && b.streamCacheWidth == width && b.streamCache != "" {
		return b.streamCache
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
	result := strings.TrimRight(sb.String(), "\n")
	b.streamCache = result
	b.streamCacheText = b.Text
	b.streamCacheWidth = width
	return result
}

func (b *TextBlock) renderText(width int) string {
	if b.complete {
		return b.renderComplete(width)
	}
	return b.renderStreaming(width)
}

func (b *TextBlock) renderComplete(width int) string {
	if b.glamourCache != "" && b.glamourCacheWidth == width && b.glamourCacheText == b.Text {
		return b.glamourCache
	}
	result := renderGlamour(b.Text, b.glamourStyle(), width)
	b.glamourCache = result
	b.glamourCacheWidth = width
	b.glamourCacheText = b.Text
	return result
}

func (b *TextBlock) renderStreaming(width int) string {
	split := findSafeSplitPoint(b.Text)
	if split <= 0 {
		return wordwrap.String(b.Text, width)
	}

	prefix := b.Text[:split]
	tail := b.Text[split:]

	if prefix != b.prefixText || width != b.prefixWidth {
		b.prefixRendered = renderGlamour(prefix, b.glamourStyle(), width)
		b.prefixText = prefix
		b.prefixWidth = width
	}

	if tail == "" {
		return b.prefixRendered
	}

	return b.prefixRendered + "\n" + wordwrap.String(tail, width)
}

func (b *TextBlock) glamourStyle() string {
	if b.GlamourStyle == "light" {
		return "light"
	}
	return "dark"
}

func renderGlamour(text, style string, width int) string {
	r, err := glamour.NewTermRenderer(
		glamour.WithWordWrap(width),
		glamour.WithStandardStyle(style),
	)
	if err != nil {
		return text
	}
	out, err := r.Render(text)
	if err != nil {
		return text
	}
	return strings.TrimSpace(out)
}

func findSafeSplitPoint(text string) int {
	if len(text) < 2 {
		return 0
	}

	inFence := classifyFenceLines(text)
	isListLike := classifyListLikeLines(text, inFence)

	lineEnds := make([]int, 0, 64)
	for i := range len(text) {
		if text[i] == '\n' {
			lineEnds = append(lineEnds, i)
		}
	}

	for k := len(lineEnds) - 1; k > 0; k-- {
		i := lineEnds[k]
		if text[i-1] != '\n' {
			continue
		}
		if k < len(inFence) && inFence[k] {
			continue
		}
		boundary := i + 1
		if boundary >= len(text) {
			continue
		}
		if k+1 < len(inFence) && inFence[k+1] {
			continue
		}
		if (k-1 >= 0 && k-1 < len(isListLike) && isListLike[k-1]) ||
			(k+1 < len(isListLike) && isListLike[k+1]) {
			continue
		}
		return boundary
	}
	return 0
}

func classifyFenceLines(input string) []bool {
	var lines []bool
	openFence := ""
	lineStart := 0
	for i := 0; i <= len(input); i++ {
		if i < len(input) && input[i] != '\n' {
			continue
		}
		line := input[lineStart:i]
		trimmed := strings.TrimLeft(line, " \t")
		switch {
		case openFence != "":
			lines = append(lines, true)
			if strings.HasPrefix(strings.TrimSpace(trimmed), openFence) {
				openFence = ""
			}
		case strings.HasPrefix(trimmed, "```"):
			lines = append(lines, true)
			openFence = "```"
		case strings.HasPrefix(trimmed, "~~~"):
			lines = append(lines, true)
			openFence = "~~~"
		default:
			lines = append(lines, false)
		}
		lineStart = i + 1
	}
	return lines
}

func classifyListLikeLines(input string, inFence []bool) []bool {
	var lines []bool
	lineStart := 0
	idx := 0
	for i := 0; i <= len(input); i++ {
		if i < len(input) && input[i] != '\n' {
			continue
		}
		line := input[lineStart:i]
		listLike := false
		if idx < len(inFence) && !inFence[idx] {
			trimmed := strings.TrimLeft(line, " \t")
			switch {
			case strings.HasPrefix(trimmed, ">"):
				listLike = true
			case isListItemStart(trimmed):
				listLike = true
			case line != "" && (line[0] == ' ' || line[0] == '\t'):
				listLike = strings.TrimLeft(line, " \t") != ""
			}
		}
		lines = append(lines, listLike)
		lineStart = i + 1
		idx++
	}
	return lines
}

func isListItemStart(trimmed string) bool {
	if len(trimmed) < 2 {
		return false
	}
	if (trimmed[0] == '-' || trimmed[0] == '*' || trimmed[0] == '+') && trimmed[1] == ' ' {
		return true
	}
	for i := 0; i < len(trimmed); i++ {
		if trimmed[i] >= '0' && trimmed[i] <= '9' {
			continue
		}
		if trimmed[i] == '.' && i > 0 && i+1 < len(trimmed) && trimmed[i+1] == ' ' {
			return true
		}
		break
	}
	return false
}
