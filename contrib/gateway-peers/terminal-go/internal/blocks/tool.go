package blocks

import (
	"encoding/json"
	"fmt"
	"strings"

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
	label := th.ToolTitle.Render(b.Name)
	if summary != "" {
		label += "(" + summary + ")"
	}
	return dot + " " + label
}

func (b *ToolBlock) renderExpanded(width int, th *theme.Theme) string {
	title := b.renderCollapsed(width, th)

	var body string
	if strings.EqualFold(b.Name, "edit") {
		body = b.renderEditDiff(th)
	} else if strings.EqualFold(b.Name, "write") {
		body = b.renderWritePreview(width, th)
	} else {
		body = b.renderGenericBody(width, th)
	}

	var sb strings.Builder
	sb.WriteString(title + "\n")
	if body != "" {
		for _, line := range strings.Split(body, "\n") {
			sb.WriteString("  " + line + "\n")
		}
	}
	if b.Done && b.Result != "" {
		resultText, truncated := truncateWithHint(b.Result, inlineTruncLimit)
		sb.WriteString("  " + th.ToolBody.Render(resultText) + "\n")
		if truncated {
			sb.WriteString("  " + th.ToolBody.Render(" … (press v for full)") + "\n")
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

func (b *ToolBlock) renderEditDiff(th *theme.Theme) string {
	oldStr, _ := asString(b.Args["old_string"])
	newStr, _ := asString(b.Args["new_string"])
	filePath, _ := asString(b.Args["file_path"])

	var sb strings.Builder
	if filePath != "" {
		sb.WriteString(th.ToolBody.Render("file: "+filePath) + "\n")
	}
	if oldStr != "" || newStr != "" {
		sb.WriteString(util.RenderDiff(oldStr, newStr, th))
	}
	return sb.String()
}

func (b *ToolBlock) renderWritePreview(_ int, th *theme.Theme) string {
	filePath, _ := asString(b.Args["file_path"])
	content, _ := asString(b.Args["content"])

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

func (b *ToolBlock) renderGenericBody(_ int, th *theme.Theme) string {
	if len(b.Args) == 0 {
		return ""
	}
	data, err := json.MarshalIndent(b.Args, "  ", "  ")
	if err != nil {
		return th.ToolBody.Render("(args unavailable)")
	}
	text, truncated := truncateWithHint(string(data), inlineTruncLimit)
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
		s, _ = asString(b.Args["command"])
	case lower == "read" || lower == "write" || lower == "edit":
		s, _ = asString(b.Args["file_path"])
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
		oldStr, _ := asString(b.Args["old_string"])
		newStr, _ := asString(b.Args["new_string"])
		filePath, _ := asString(b.Args["file_path"])
		if filePath != "" {
			sb.WriteString("file: " + filePath + "\n\n")
		}
		sb.WriteString("--- old ---\n")
		sb.WriteString(oldStr)
		sb.WriteString("\n+++ new +++\n")
		sb.WriteString(newStr)
	} else if strings.EqualFold(b.Name, "write") {
		filePath, _ := asString(b.Args["file_path"])
		content, _ := asString(b.Args["content"])
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

func asString(v any) (string, bool) {
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
