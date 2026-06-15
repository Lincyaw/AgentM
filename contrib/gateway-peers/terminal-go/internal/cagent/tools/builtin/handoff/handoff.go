package handoff

import (
	"context"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/tools"
)

const ToolNameHandoff = "handoff"

type ToolSet struct{}

var _ tools.ToolSet = (*ToolSet)(nil)

type Args struct {
	Agent string `json:"agent" jsonschema:"The name of the agent to hand off the conversation to."`
}

func New() *ToolSet {
	return &ToolSet{}
}

func (t *ToolSet) Tools(context.Context) ([]tools.Tool, error) {
	return []tools.Tool{
		{
			Name:        ToolNameHandoff,
			Category:    "handoff",
			Description: "Use this function to hand off the conversation to the selected agent.",
			Parameters:  tools.MustSchemaFor[Args](),
			Annotations: tools.ToolAnnotations{
				ReadOnlyHint: true,
				Title:        "Handoff Conversation",
			},
		},
	}, nil
}
