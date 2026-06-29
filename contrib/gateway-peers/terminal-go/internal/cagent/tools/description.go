package tools

import (
	"encoding/json"
)

const (
	// DescriptionParam is the parameter name for the description
	DescriptionParam = "description"
)

// ExtractDescription extracts the description from tool call arguments.
func ExtractDescription(arguments string) string {
	var args map[string]any
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return ""
	}

	if desc, ok := args[DescriptionParam].(string); ok {
		return desc
	}
	return ""
}
