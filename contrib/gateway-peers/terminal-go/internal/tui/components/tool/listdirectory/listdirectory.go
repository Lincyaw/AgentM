package listdirectory

import (
	"strings"

	pathx "github.com/AoyangSpace/agentm-terminal/internal/cagent/path"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/toolschema"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/types"
)

func New(msg *types.Message, sessionState service.SessionStateReader) layout.Model {
	return toolcommon.NewBase(msg, sessionState, toolcommon.SimpleRendererWithResult(
		toolcommon.ExtractField(func(a toolschema.ListDirectoryArgs) string { return pathx.ShortenHome(a.Path) }),
		extractResult,
	))
}

func extractResult(msg *types.Message) string {
	if msg.ToolResult == nil || msg.ToolResult.Meta == nil {
		return "empty directory"
	}
	meta, ok := msg.ToolResult.Meta.(toolschema.ListDirectoryMeta)
	if !ok {
		return "empty directory"
	}

	fileCount := len(meta.Files)
	dirCount := len(meta.Dirs)
	if fileCount+dirCount == 0 {
		return "empty directory"
	}

	var parts []string
	if fileCount > 0 {
		parts = append(parts, toolcommon.Pluralize(fileCount, "file", "files"))
	}
	if dirCount > 0 {
		parts = append(parts, toolcommon.Pluralize(dirCount, "directory", "directories"))
	}

	result := strings.Join(parts, " and ")
	if meta.Truncated {
		result += " (truncated)"
	}
	return result
}
