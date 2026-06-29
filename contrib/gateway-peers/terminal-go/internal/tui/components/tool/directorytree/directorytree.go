package directorytree

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
		toolcommon.ExtractField(func(a toolschema.DirectoryTreeArgs) string { return pathx.ShortenHome(a.Path) }),
		extractResult,
	))
}

func extractResult(msg *types.Message) string {
	if msg.ToolResult == nil || msg.ToolResult.Meta == nil {
		return ""
	}
	meta, ok := msg.ToolResult.Meta.(toolschema.DirectoryTreeMeta)
	if !ok {
		return ""
	}

	if meta.FileCount+meta.DirCount == 0 {
		return "empty"
	}

	var parts []string
	if meta.FileCount > 0 {
		parts = append(parts, toolcommon.Pluralize(meta.FileCount, "file", "files"))
	}
	if meta.DirCount > 0 {
		parts = append(parts, toolcommon.Pluralize(meta.DirCount, "dir", "dirs"))
	}

	result := strings.Join(parts, ", ")
	if meta.Truncated {
		result += " (truncated)"
	}
	return result
}
