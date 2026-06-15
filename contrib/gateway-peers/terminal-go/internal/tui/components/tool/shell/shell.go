package shell

import (
	"strings"

	builtinshell "github.com/AoyangSpace/agentm-terminal/internal/cagent/tools/builtin/shell"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/spinner"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/types"
)

const maxVisibleShellOutputLines = 20

func New(msg *types.Message, sessionState service.SessionStateReader) layout.Model {
	return toolcommon.NewBase(msg, sessionState, renderShell)
}

func renderShell(msg *types.Message, s spinner.Spinner, sessionState service.SessionStateReader, width, _ int) string {
	arg := ""
	if msg.ToolCall.Function.Arguments != "" {
		arg = toolcommon.ExtractField(func(a builtinshell.RunShellArgs) string { return a.Cmd })(msg.ToolCall.Function.Arguments)
	}

	result := ""
	if msg.Content != "" {
		result = formatShellOutput(msg.Content, width)
	}

	return toolcommon.RenderTool(msg, s, arg, result, width, sessionState.HideToolResults())
}

func formatShellOutput(output string, width int) string {
	output = strings.ReplaceAll(output, "\r\n", "\n")
	output = strings.ReplaceAll(output, "\r", "\n")
	output = strings.TrimRight(output, "\n")
	if output == "" {
		return ""
	}

	availableWidth := max(width-styles.ToolCallResult.GetHorizontalFrameSize(), 10)
	lines := toolcommon.WrapLines(output, availableWidth)
	if len(lines) > maxVisibleShellOutputLines {
		lines = append([]string{"…"}, lines[len(lines)-maxVisibleShellOutputLines:]...)
	}
	return strings.Join(lines, "\n")
}
