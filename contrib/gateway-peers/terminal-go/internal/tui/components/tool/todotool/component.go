package todotool

import (
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/types"
)

// New creates a new unified todo component.
// This component handles create, create_multiple, list, and update operations.
// The TODOs themselves are displayed in the sidebar; here we only show the
// tool call header (icon + name).
func New(msg *types.Message, sessionState service.SessionStateReader) layout.Model {
	return toolcommon.NewBase(msg, sessionState, toolcommon.NoArgsRenderer)
}
