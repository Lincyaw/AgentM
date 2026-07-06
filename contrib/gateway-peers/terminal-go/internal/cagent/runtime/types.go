package runtime

import "github.com/AoyangSpace/agentm-terminal/internal/cagent/config/types"

// PermissionsInfo contains the allow, ask, and deny patterns for tool permissions.
type PermissionsInfo struct {
	Allow []string
	Ask   []string
	Deny  []string
}

// CurrentAgentInfo describes the currently active agent. The adapter
// constructs this from the wire; the UI reads the fields for display.
type CurrentAgentInfo struct {
	Name        string
	Description string
	Commands    types.Commands
}

// ModelChoice represents a model available for selection in the model picker.
//
// JSON tags are part of the public wire format used by
// GET /api/sessions/:id/models; renaming a tag is a breaking change.
type ModelChoice struct {
	// Name is the display name (config key)
	Name string `json:"name"`
	// Ref is the model reference used internally (e.g., "my_model" or "openai/gpt-4o")
	Ref string `json:"ref"`
	// SwitchRef is the backend model profile to send when this row is selected.
	// It is UI-local metadata and is intentionally omitted from the wire format.
	SwitchRef string `json:"-"`
	// Provider is the provider name (e.g., "openai", "anthropic")
	Provider string `json:"provider,omitempty"`
	// Model is the specific model name (e.g., "gpt-4o", "claude-sonnet-4-0")
	Model string `json:"model,omitempty"`
	// IsDefault indicates this is the agent's configured default model
	IsDefault bool `json:"is_default,omitempty"`
	// IsCurrent indicates this is the currently active model for the agent
	IsCurrent bool `json:"is_current,omitempty"`
	// IsCustom indicates this is a custom model from the session history (not from config)
	IsCustom bool `json:"is_custom,omitempty"`
	// IsCatalog indicates this is a model from the models.dev catalog
	IsCatalog bool `json:"is_catalog,omitempty"`

	// The fields below are populated (best-effort) from the models.dev
	// catalog. They are optional and may all be zero/empty when no
	// catalog entry is found for the model.

	// Family is the model family (e.g., "claude", "gpt").
	Family string `json:"family,omitempty"`
	// InputCost is the price (in USD) per 1M input tokens.
	InputCost float64 `json:"input_cost,omitempty"`
	// OutputCost is the price (in USD) per 1M output tokens.
	OutputCost float64 `json:"output_cost,omitempty"`
	// CacheReadCost is the price (in USD) per 1M cached input tokens.
	CacheReadCost float64 `json:"cache_read_cost,omitempty"`
	// CacheWriteCost is the price (in USD) per 1M cache-write tokens.
	CacheWriteCost float64 `json:"cache_write_cost,omitempty"`
	// ContextLimit is the maximum context window size in tokens.
	ContextLimit int `json:"context_limit,omitempty"`
	// OutputLimit is the maximum number of tokens the model can produce
	// in a single response.
	OutputLimit int64 `json:"output_limit,omitempty"`
	// InputModalities lists the input modalities supported by the model
	// (e.g., "text", "image", "audio").
	InputModalities []string `json:"input_modalities,omitempty"`
	// OutputModalities lists the output modalities the model can produce.
	OutputModalities []string `json:"output_modalities,omitempty"`
}
