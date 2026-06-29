// Package tool builds the TUI view for a tool call message.
//
// A small lookup table (builders) maps each tool's name to a constructor.
// Lookup order is: exact tool name, then "category:<category>", then a
// defaulttool fallback.
package tool

import (
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/api"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/defaulttool"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/directorytree"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/editfile"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/handoff"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/listdirectory"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/readfile"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/readmultiplefiles"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/searchfilescontent"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/shell"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/todotool"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/transfertask"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/userprompt"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/tool/writefile"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/toolschema"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/types"
)

// builder constructs the layout.Model for a tool message.
type builder func(msg *types.Message, sessionState service.SessionStateReader) layout.Model

// builders maps a tool name (or a "category:<name>" key) to its renderer.
// Tools sharing the same visual representation point at the same builder.
var builders = map[string]builder{
	toolschema.ToolNameTransferTask:       transfertask.New,
	toolschema.ToolNameHandoff:            handoff.New,
	toolschema.ToolNameEditFile:           editfile.New,
	toolschema.ToolNameWriteFile:          writefile.New,
	toolschema.ToolNameReadFile:           readfile.New,
	toolschema.ToolNameReadMultipleFiles:  readmultiplefiles.New,
	toolschema.ToolNameListDirectory:      listdirectory.New,
	toolschema.ToolNameDirectoryTree:      directorytree.New,
	toolschema.ToolNameSearchFilesContent: searchfilescontent.New,
	toolschema.ToolNameShell:              shell.New,
	toolschema.ToolNameUserPrompt:         userprompt.New,
	toolschema.ToolNameFetch:              api.New,
	"category:api":                        api.New,
	toolschema.ToolNameCreateTodo:         todotool.New,
	toolschema.ToolNameCreateTodos:        todotool.New,
	toolschema.ToolNameUpdateTodos:        todotool.New,
	toolschema.ToolNameListTodos:          todotool.New,
}

// New returns the appropriate tool view for the given message.
// Lookup order: exact tool name, then "category:<category>", then default.
func New(msg *types.Message, sessionState service.SessionStateReader) layout.Model {
	if b, ok := builders[msg.ToolCall.Function.Name]; ok {
		return b(msg, sessionState)
	}
	if cat := msg.ToolDefinition.Category; cat != "" {
		if b, ok := builders["category:"+cat]; ok {
			return b(msg, sessionState)
		}
	}
	return defaulttool.New(msg, sessionState)
}
