package blocks

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/muesli/reflow/wordwrap"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
	"github.com/AoyangSpace/agentm-terminal/internal/util"
)

const (
	inlineTruncLimit  = 2000
	summaryTruncLimit = 48
)

// ToolBlock renders a tool invocation with name, args summary, and result.
type ToolBlock struct {
	Name      string
	Args      map[string]any
	Result    string
	OK        bool
	Done      bool // false while the tool is still running
	collapsed bool
	focused   bool
}

// NewToolBlock creates a ToolBlock in collapsed, running state.
func NewToolBlock(name string, args map[string]any) *ToolBlock {
	return &ToolBlock{Name: name, Args: args, collapsed: true}
}

func (b *ToolBlock) Kind() string        { return "tool" }
func (b *ToolBlock) Collapsed() bool     { return b.collapsed }
func (b *ToolBlock) SetCollapsed(c bool) { b.collapsed = c }

// Focused reports whether this block has keyboard focus.
func (b *ToolBlock) Focused() bool { return b.focused }

// SetFocused sets the keyboard focus state.
func (b *ToolBlock) SetFocused(f bool) { b.focused = f }

func (b *ToolBlock) Render(width int, th *theme.Theme) string {
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

func (b *ToolBlock) renderCollapsed(_ int, th *theme.Theme) string {
	summary := b.summary()
	dot := b.dotStyled(th)

	// Use per-tool glyphs for specialized tools
	lower := strings.ToLower(b.Name)
	var glyph string
	switch lower {
	case "bash":
		glyph = "⚙ "
	case "read":
		glyph = "📄 "
	case "edit", "notebookedit":
		glyph = "✎ "
	case "write":
		glyph = "✎ "
	default:
		glyph = ""
	}

	label := th.ToolTitle.Render(glyph + b.Name)
	if summary != "" {
		label += "(" + summary + ")"
	}
	return dot + " " + label
}

func (b *ToolBlock) renderExpanded(width int, th *theme.Theme) string {
	title := b.renderCollapsed(width, th)

	var body string
	lower := strings.ToLower(b.Name)
	switch {
	case lower == "edit" || lower == "notebookedit":
		body = b.renderEditDiff(width, th)
	case lower == "write":
		body = b.renderWritePreview(width, th)
	case lower == "bash":
		body = b.renderShell(width, th)
	case lower == "read":
		body = b.renderFileRead(width, th)
	default:
		body = b.renderGenericBody(width, th)
	}

	var sb strings.Builder
	sb.WriteString(title + "\n")
	if body != "" {
		for _, line := range strings.Split(body, "\n") {
			sb.WriteString("  " + line + "\n")
		}
	}
	// For Bash and Read, the result is already rendered in the body
	if lower != "bash" && lower != "read" {
		if b.Done && b.Result != "" {
			resultText, truncated := truncateWithHint(b.Result, inlineTruncLimit)
			cw := width - 4
			if cw < 20 {
				cw = 20
			}
			resultText = wordwrap.String(resultText, cw)
			sb.WriteString("  " + th.ToolBody.Render(resultText) + "\n")
			if truncated {
				sb.WriteString("  " + th.ToolBody.Render(" … (press v for full)") + "\n")
			}
		}
	}
	return strings.TrimRight(sb.String(), "\n")
}

func (b *ToolBlock) dotStyled(th *theme.Theme) string {
	if !b.Done {
		return th.AssistantDotDim.Render(theme.BlackCircle)
	}
	if b.OK {
		return th.AssistantDotOK.Render(theme.BlackCircle)
	}
	return th.AssistantDotErr.Render(theme.BlackCircle)
}

func (b *ToolBlock) renderEditDiff(_ int, th *theme.Theme) string {
	oldStr, _ := AsString(b.Args["old_string"])
	newStr, _ := AsString(b.Args["new_string"])
	filePath, _ := AsString(b.Args["file_path"])

	var sb strings.Builder
	if filePath != "" {
		sb.WriteString(th.ToolPath.Render("✎ "+filePath) + "\n")
	}
	if oldStr != "" || newStr != "" {
		sb.WriteString(util.RenderDiff(oldStr, newStr, th))
	}
	return sb.String()
}

func (b *ToolBlock) renderWritePreview(_ int, th *theme.Theme) string {
	filePath, _ := AsString(b.Args["file_path"])
	content, _ := AsString(b.Args["content"])

	var sb strings.Builder
	if filePath != "" {
		sb.WriteString(th.ToolBody.Render("file: "+filePath) + "\n")
	}
	if content != "" {
		preview, truncated := truncateWithHint(content, inlineTruncLimit)
		lines := strings.Split(preview, "\n")
		for _, line := range lines {
			sb.WriteString(th.DiffAdd.Render("+ "+line) + "\n")
		}
		if truncated {
			sb.WriteString(th.ToolBody.Render(" … (press v for full)") + "\n")
		}
	}
	return strings.TrimRight(sb.String(), "\n")
}

const shellResultMaxLines = 15

func (b *ToolBlock) renderShell(width int, th *theme.Theme) string {
	command, _ := AsString(b.Args["command"])
	cw := width - 4
	if cw < 20 {
		cw = 20
	}

	var sb strings.Builder
	if command != "" {
		sb.WriteString(th.ToolCommand.Render(wordwrap.String("$ "+command, cw)) + "\n")
	}
	if b.Done && b.Result != "" {
		sb.WriteString(th.ToolBody.Render(strings.Repeat("─", min(20, cw))) + "\n")
		lines := strings.Split(b.Result, "\n")
		if len(lines) > shellResultMaxLines {
			truncated := len(lines) - shellResultMaxLines
			sb.WriteString(th.ToolBody.Render(fmt.Sprintf("... (%d lines truncated)", truncated)) + "\n")
			lines = lines[len(lines)-shellResultMaxLines:]
		}
		resultText, tooLong := truncateWithHint(strings.Join(lines, "\n"), inlineTruncLimit)
		resultText = wordwrap.String(resultText, cw)
		sb.WriteString(th.ToolBody.Render(resultText))
		if tooLong {
			sb.WriteString("\n" + th.ToolBody.Render(" … (press v for full)"))
		}
	}
	return sb.String()
}

func (b *ToolBlock) renderFileRead(width int, th *theme.Theme) string {
	filePath, _ := AsString(b.Args["file_path"])
	offset, hasOffset := toIntArg(b.Args["offset"])
	limit, hasLimit := toIntArg(b.Args["limit"])

	var sb strings.Builder
	pathDisplay := filePath
	if hasOffset || hasLimit {
		if hasOffset && hasLimit {
			pathDisplay = fmt.Sprintf("%s:%d-%d", filePath, offset, offset+limit)
		} else if hasOffset {
			pathDisplay = fmt.Sprintf("%s:%d-", filePath, offset)
		} else if hasLimit {
			pathDisplay = fmt.Sprintf("%s (limit %d)", filePath, limit)
		}
	}
	if pathDisplay != "" {
		sb.WriteString(th.ToolPath.Render(pathDisplay) + "\n")
	}
	if b.Done && b.Result != "" {
		lines := strings.Split(b.Result, "\n")
		lineCount := len(lines)
		sb.WriteString(th.ToolBody.Render(fmt.Sprintf("(%d lines)", lineCount)))
		if lineCount > 5 {
			// Show first 5 lines as preview
			preview := strings.Join(lines[:5], "\n")
			sb.WriteString("\n" + th.ToolBody.Render(preview))
			sb.WriteString("\n" + th.ToolBody.Render(" … (press v for full)"))
		} else if lineCount > 0 {
			sb.WriteString("\n" + th.ToolBody.Render(b.Result))
		}
	}
	return sb.String()
}

func (b *ToolBlock) renderGenericBody(width int, th *theme.Theme) string {
	if len(b.Args) == 0 {
		return ""
	}
	cw := width - 4
	if cw < 20 {
		cw = 20
	}
	data, err := json.MarshalIndent(b.Args, "  ", "  ")
	if err != nil {
		return th.ToolBody.Render("(args unavailable)")
	}
	text, truncated := truncateWithHint(string(data), inlineTruncLimit)
	text = wordwrap.String(text, cw)
	result := th.ToolBody.Render(text)
	if truncated {
		result += "\n" + th.ToolBody.Render(" … (press v for full)")
	}
	return result
}

// summary extracts a short description from args based on tool name.
func (b *ToolBlock) summary() string {
	var s string
	lower := strings.ToLower(b.Name)
	switch {
	case strings.EqualFold(b.Name, "bash"):
		s, _ = AsString(b.Args["command"])
	case lower == "read" || lower == "write" || lower == "edit":
		s, _ = AsString(b.Args["file_path"])
	default:
		s = firstScalarValue(b.Args)
	}
	return util.Truncate(s, summaryTruncLimit)
}

// FullContent returns the complete untruncated content suitable for the view overlay.
// For generic tools it returns pretty-printed args + result; for write it returns content + result.
func (b *ToolBlock) FullContent() string {
	var sb strings.Builder

	if strings.EqualFold(b.Name, "edit") {
		oldStr, _ := AsString(b.Args["old_string"])
		newStr, _ := AsString(b.Args["new_string"])
		filePath, _ := AsString(b.Args["file_path"])
		if filePath != "" {
			sb.WriteString("file: " + filePath + "\n\n")
		}
		sb.WriteString("--- old ---\n")
		sb.WriteString(oldStr)
		sb.WriteString("\n+++ new +++\n")
		sb.WriteString(newStr)
	} else if strings.EqualFold(b.Name, "write") {
		filePath, _ := AsString(b.Args["file_path"])
		content, _ := AsString(b.Args["content"])
		if filePath != "" {
			sb.WriteString("file: " + filePath + "\n\n")
		}
		sb.WriteString(content)
	} else {
		if len(b.Args) > 0 {
			data, err := json.MarshalIndent(b.Args, "", "  ")
			if err == nil {
				sb.WriteString("args:\n")
				sb.WriteString(string(data))
			}
		}
	}

	if b.Result != "" {
		if sb.Len() > 0 {
			sb.WriteString("\n\n")
		}
		sb.WriteString("result:\n")
		sb.WriteString(b.Result)
	}
	return sb.String()
}

// truncateWithHint truncates s to limit runes and returns whether truncation occurred.
func truncateWithHint(s string, limit int) (string, bool) {
	runes := []rune(s)
	if len(runes) <= limit {
		return s, false
	}
	return string(runes[:limit]), true
}

func AsString(v any) (string, bool) {
	if v == nil {
		return "", false
	}
	switch val := v.(type) {
	case string:
		return val, true
	case fmt.Stringer:
		return val.String(), true
	default:
		return fmt.Sprintf("%v", v), true
	}
}

// toIntArg coerces a JSON number in tool args to int.
func toIntArg(v any) (int, bool) {
	if v == nil {
		return 0, false
	}
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	default:
		return 0, false
	}
}

func firstScalarValue(m map[string]any) string {
	for _, v := range m {
		switch val := v.(type) {
		case string:
			return val
		case float64, int, bool:
			return fmt.Sprintf("%v", val)
		}
	}
	return ""
}
