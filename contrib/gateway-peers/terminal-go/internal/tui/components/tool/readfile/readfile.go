package readfile

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/spinner"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/toolschema"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/types"
)

func New(msg *types.Message, sessionState service.SessionStateReader) layout.Model {
	return toolcommon.NewBase(msg, sessionState, render)
}

func render(msg *types.Message, s spinner.Spinner, sessionState service.SessionStateReader, width, _ int) string {
	path := readPath(msg)
	switch msg.ToolStatus {
	case types.ToolStatusPending, types.ToolStatusRunning, types.ToolStatusConfirmation:
		return toolcommon.RenderTool(msg, s, path, "", width, sessionState.HideToolResults())
	case types.ToolStatusCompleted, types.ToolStatusError:
		if sessionState.HideToolResults() {
			return renderCollapsed(width)
		}
		return renderDetailed(msg, path, width)
	default:
		return toolcommon.RenderTool(msg, s, path, "", width, sessionState.HideToolResults())
	}
}

func readPath(msg *types.Message) string {
	var args toolschema.ReadFileArgs
	if err := json.Unmarshal([]byte(msg.ToolCall.Function.Arguments), &args); err == nil && args.Path != "" {
		return args.Path
	}
	if msg.ToolResult == nil || msg.ToolResult.Meta == nil {
		return ""
	}
	meta, ok := msg.ToolResult.Meta.(toolschema.ReadFileMeta)
	if ok {
		return meta.Path
	}
	return ""
}

func renderCollapsed(width int) string {
	return styles.RenderComposite(
		styles.ToolMessageStyle.Width(width),
		"  Read 1 file (ctrl+o to expand)",
	)
}

func renderDetailed(msg *types.Message, path string, width int) string {
	header := renderHeader(path, width)
	result := "Read file"
	if lineCount, ok := readLineCount(msg); ok {
		label := "lines"
		if lineCount == 1 {
			label = "line"
		}
		result = fmt.Sprintf("Read %d %s", lineCount, label)
	}
	if errText := readError(msg); errText != "" {
		result = errText
	}
	content := header + "\n" + "  ⎿ \u00a0" + result
	return styles.RenderComposite(styles.ToolMessageStyle.Width(width), content)
}

func renderHeader(path string, width int) string {
	header := "⏺ Read(" + path + ")"
	lines := toolcommon.WrapLines(header, max(width, 10))
	for i := 1; i < len(lines); i++ {
		lines[i] = "  " + lines[i]
	}
	return strings.Join(lines, "\n")
}

func readError(msg *types.Message) string {
	if msg.ToolResult == nil || msg.ToolResult.Meta == nil {
		return ""
	}
	meta, ok := msg.ToolResult.Meta.(toolschema.ReadFileMeta)
	if !ok {
		return ""
	}
	return meta.Error
}

func readLineCount(msg *types.Message) (int, bool) {
	if msg.ToolResult != nil && msg.ToolResult.Meta != nil {
		meta, ok := msg.ToolResult.Meta.(toolschema.ReadFileMeta)
		if ok && meta.LineCount > 0 {
			return meta.LineCount, true
		}
	}

	firstLine, _, _ := strings.Cut(strings.TrimSpace(msg.Content), "\n")
	if strings.HasPrefix(firstLine, "(") && strings.Contains(firstLine, " lines total)") {
		fields := strings.Fields(firstLine)
		if len(fields) > 0 {
			n, err := strconv.Atoi(strings.TrimPrefix(fields[0], "("))
			if err == nil {
				return n, true
			}
		}
	}
	return 0, false
}
