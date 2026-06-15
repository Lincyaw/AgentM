package app

import (
	"encoding/json"
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/blocks"
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// ApprovalOverlay is a modal overlay for tool approval requests. It shows the
// tool name, formatted arguments, and action buttons.
type ApprovalOverlay struct {
	toolName string
	args     map[string]any
	content  string // raw content from the approval event
	buttons  []blocks.Button
	selected int
	resolved bool
	chosen   string
	expanded bool // whether to show full args

	// Scroll state for expanded view.
	lines  []string
	offset int
}

// NewApprovalOverlay creates an approval modal for a tool invocation.
func NewApprovalOverlay(content string, toolName string, args map[string]any, buttons []blocks.Button) *ApprovalOverlay {
	return &ApprovalOverlay{
		toolName: toolName,
		args:     args,
		content:  content,
		buttons:  buttons,
	}
}

func (o *ApprovalOverlay) Kind() OverlayKind { return OverlayApproval }

func (o *ApprovalOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		key := msg.String()

		// Digit keys select buttons directly.
		if len(key) == 1 && key[0] >= '1' && key[0] <= '9' {
			idx := int(key[0]-'0') - 1
			if idx < len(o.buttons) {
				o.resolved = true
				o.chosen = o.buttons[idx].Value
				return o, nil, true
			}
		}

		// Letter shortcuts: y=first, n=second, a=third.
		switch key {
		case "y":
			if len(o.buttons) > 0 {
				o.resolved = true
				o.chosen = o.buttons[0].Value
				return o, nil, true
			}
		case "n":
			if len(o.buttons) > 1 {
				o.resolved = true
				o.chosen = o.buttons[1].Value
				return o, nil, true
			}
		case "a":
			if len(o.buttons) > 2 {
				o.resolved = true
				o.chosen = o.buttons[2].Value
				return o, nil, true
			}
		}

		switch key {
		case "enter":
			if o.selected < len(o.buttons) {
				o.resolved = true
				o.chosen = o.buttons[o.selected].Value
				return o, nil, true
			}

		case "tab", "right":
			if len(o.buttons) > 0 {
				o.selected = (o.selected + 1) % len(o.buttons)
			}
			return o, nil, false

		case "left":
			if len(o.buttons) > 0 {
				o.selected = (o.selected - 1 + len(o.buttons)) % len(o.buttons)
			}
			return o, nil, false

		case "esc":
			// Deny by default (second button, or first if only one).
			o.resolved = true
			if len(o.buttons) > 1 {
				o.chosen = o.buttons[1].Value
			} else if len(o.buttons) > 0 {
				o.chosen = o.buttons[0].Value
			} else {
				o.chosen = "deny"
			}
			return o, nil, true

		case "?":
			o.expanded = !o.expanded
			if o.expanded {
				o.lines = strings.Split(o.formatFullArgs(), "\n")
				o.offset = 0
			}
			return o, nil, false

		case "up", "k":
			if o.expanded && o.offset > 0 {
				o.offset--
			}
			return o, nil, false

		case "down", "j":
			if o.expanded {
				o.offset++
			}
			return o, nil, false
		}
	}
	return o, nil, false
}

// Resolved reports whether the user made a decision.
func (o *ApprovalOverlay) Resolved() bool { return o.resolved }

// Chosen returns the selected button value.
func (o *ApprovalOverlay) Chosen() string { return o.chosen }

func (o *ApprovalOverlay) View(width, height int, th *theme.Theme) string {
	innerW := width - 8
	if innerW < 30 {
		innerW = 30
	}
	if innerW > 70 {
		innerW = 70
	}

	var sb strings.Builder
	sb.WriteString(th.OverlayTitle.Render("Tool Approval"))
	sb.WriteByte('\n')
	sb.WriteByte('\n')

	// Tool name + summary.
	sb.WriteString(th.ToolTitle.Render(toolGlyph(o.toolName) + o.toolName))
	sb.WriteByte('\n')

	// Tool-specific compact display.
	summary := o.formatSummary(th)
	if summary != "" {
		sb.WriteString(summary)
		sb.WriteByte('\n')
	}

	// Expanded args view.
	if o.expanded && len(o.lines) > 0 {
		sb.WriteByte('\n')
		sb.WriteString(th.OverlayDim.Render(strings.Repeat("~", 20)))
		sb.WriteByte('\n')

		maxVisible := height - 12
		if maxVisible < 5 {
			maxVisible = 5
		}
		// Clamp offset.
		maxOff := len(o.lines) - maxVisible
		if maxOff < 0 {
			maxOff = 0
		}
		if o.offset > maxOff {
			o.offset = maxOff
		}
		end := o.offset + maxVisible
		if end > len(o.lines) {
			end = len(o.lines)
		}
		for _, line := range o.lines[o.offset:end] {
			runes := []rune(line)
			if len(runes) > innerW {
				line = string(runes[:innerW])
			}
			sb.WriteString(th.OverlayText.Render(line))
			sb.WriteByte('\n')
		}
		if len(o.lines) > maxVisible {
			sb.WriteString(th.OverlayDim.Render(fmt.Sprintf("  (%d/%d lines)", o.offset+1, len(o.lines))))
			sb.WriteByte('\n')
		}
	}

	sb.WriteByte('\n')
	sb.WriteString(th.OverlayDim.Render(strings.Repeat("~", 20)))
	sb.WriteByte('\n')
	sb.WriteByte('\n')

	// Button row.
	var btnParts []string
	for i, btn := range o.buttons {
		label := fmt.Sprintf("[%d] %s", i+1, btn.Label)
		if i == o.selected {
			btnParts = append(btnParts, th.OverlayActive.Render(label))
		} else {
			btnParts = append(btnParts, th.OverlayText.Render(label))
		}
	}
	sb.WriteString(strings.Join(btnParts, "  "))
	sb.WriteByte('\n')

	// Hint.
	sb.WriteByte('\n')
	hint := "enter=confirm  ?=details  esc=deny"
	sb.WriteString(th.OverlayDim.Render(hint))

	content := th.OverlayBorder.Render(sb.String())
	return centerOverlay(content, width, height)
}

// formatSummary produces a one-line tool-specific display.
func (o *ApprovalOverlay) formatSummary(th *theme.Theme) string {
	lower := strings.ToLower(o.toolName)
	switch lower {
	case "bash":
		cmd, _ := blocks.AsString(o.args["command"])
		if cmd != "" {
			return th.ToolCommand.Render("$ " + truncLine(cmd, 60))
		}
	case "read":
		path, _ := blocks.AsString(o.args["file_path"])
		if path != "" {
			return th.ToolPath.Render(path)
		}
	case "edit", "notebookedit":
		path, _ := blocks.AsString(o.args["file_path"])
		if path != "" {
			return th.ToolPath.Render("~ " + path)
		}
	case "write":
		path, _ := blocks.AsString(o.args["file_path"])
		if path != "" {
			return th.ToolPath.Render("+ " + path)
		}
	}
	// Fallback: use content.
	if o.content != "" {
		return th.OverlayText.Render(truncLine(o.content, 60))
	}
	return ""
}

// formatFullArgs produces the full args dump for expanded view.
func (o *ApprovalOverlay) formatFullArgs() string {
	if len(o.args) == 0 {
		return o.content
	}
	data, err := json.MarshalIndent(o.args, "", "  ")
	if err != nil {
		return o.content
	}
	return string(data)
}

func toolGlyph(name string) string {
	switch strings.ToLower(name) {
	case "bash":
		return "$ "
	case "read":
		return "# "
	case "edit", "notebookedit":
		return "~ "
	case "write":
		return "+ "
	default:
		return ""
	}
}

func truncLine(s string, max int) string {
	// Take only the first line.
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		s = s[:idx]
	}
	runes := []rune(s)
	if len(runes) > max {
		return string(runes[:max-3]) + "..."
	}
	return s
}
