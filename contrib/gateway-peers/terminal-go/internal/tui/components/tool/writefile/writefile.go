package writefile

import (
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/tools/builtin/filesystem"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/types"
)

func New(msg *types.Message, sessionState service.SessionStateReader) layout.Model {
	return toolcommon.NewBase(msg, sessionState, toolcommon.SimpleRenderer(
		toolcommon.ExtractField(func(a filesystem.WriteFileArgs) string { return a.Path }),
	))
}
