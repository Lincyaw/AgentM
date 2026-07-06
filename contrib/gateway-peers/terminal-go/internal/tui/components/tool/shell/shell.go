package shell

import (
	"encoding/json"
	"strings"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/spinner"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/toolschema"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/types"
)

const maxVisibleShellOutputLines = 20

func New(msg *types.Message, sessionState service.SessionStateReader) layout.Model {
	return toolcommon.NewBase(msg, sessionState, renderShell)
}

func renderShell(msg *types.Message, s spinner.Spinner, sessionState service.SessionStateReader, width, _ int) string {
	arg := ""
	if msg.ToolCall.Function.Arguments != "" {
		arg = toolcommon.ExtractField(func(a toolschema.RunShellArgs) string { return a.Cmd })(msg.ToolCall.Function.Arguments)
	}

	if msg.Content != "" {
		if output := formatClaudeShellOutput(msg.Content, width); output != "" {
			if !msg.DirectShell {
				return formatClaudeShellHeader(arg, width) + "\n" + output
			}
			return output
		}
	}

	if msg.ToolStatus == types.ToolStatusConfirmation && arg != "" {
		return formatClaudeShellHeader(arg, width)
	}

	if msg.ToolStatus == types.ToolStatusPending || msg.ToolStatus == types.ToolStatusRunning {
		if msg.DirectShell {
			return formatDirectShellRunning(width)
		}
		return toolcommon.RenderTool(msg, s, arg, "", width, sessionState.HideToolResults())
	}
	return ""
}

func formatClaudeShellHeader(command string, width int) string {
	header := "⏺ Bash(" + strings.TrimSpace(command) + ")"
	if width <= 0 {
		return header
	}
	return toolcommon.TruncateText(header, width)
}

func formatClaudeShellOutput(output string, width int) string {
	if backgroundLine, ok := formatBackgroundTicket(output); ok {
		output = backgroundLine
	}
	output = extractShellStream(output)
	output = strings.ReplaceAll(output, "\r\n", "\n")
	output = strings.ReplaceAll(output, "\r", "\n")
	output = strings.TrimRight(output, "\n")
	if output == "" {
		return ""
	}

	const firstPrefix = "  ⎿ \u00a0"
	const nextPrefix = "     "
	availableWidth := max(width-len(firstPrefix), 10)
	lines := toolcommon.WrapLines(output, availableWidth)
	if len(lines) > maxVisibleShellOutputLines {
		lines = append([]string{"…"}, lines[len(lines)-maxVisibleShellOutputLines:]...)
	}
	for i, line := range lines {
		if i == 0 {
			lines[i] = firstPrefix + line
			continue
		}
		lines[i] = nextPrefix + line
	}
	return strings.Join(lines, "\n")
}

func formatDirectShellRunning(width int) string {
	line := "  ⎿ \u00a0Running…"
	if width <= 0 {
		return line
	}
	return toolcommon.TruncateText(line, width)
}

func formatBackgroundTicket(content string) (string, bool) {
	var payload struct {
		Status string `json:"status"`
		Note   string `json:"note"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(content)), &payload); err != nil {
		return "", false
	}
	if strings.TrimSpace(payload.Status) != "running" {
		return "", false
	}
	note := strings.TrimSpace(payload.Note)
	if note == "" || !strings.Contains(note, "background") {
		return "", false
	}
	return "Running in the background (↓ to manage)", true
}

func extractShellStream(content string) string {
	for _, marker := range []string{"\nStdout:\n", "\nStderr:\n"} {
		if before, after, ok := strings.Cut(content, marker); ok {
			_ = before
			return after
		}
	}
	return content
}
