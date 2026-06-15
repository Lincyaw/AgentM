// Package provider is a minimal shim exposing only the provider-name helpers
// the TUI consumes (IsKnownProvider, AllProviders). The real cagent provider
// package pulls the full model/provider transitive dependency tree; this shim
// keeps the faithful name catalog without any of that machinery.
package provider

import (
	"iter"
	"maps"
	"slices"
	"strings"
)

// Alias represents an alias configuration.
type Alias struct {
	APIType     string // The actual API type to use (openai, anthropic, etc.)
	BaseURL     string // Default base URL for the provider
	TokenEnvVar string // Environment variable name for the API token
}

// CoreProviders lists all natively implemented provider types.
var CoreProviders = []string{
	"openai",
	"anthropic",
	"google",
	"dmr",
	"amazon-bedrock",
}

// Aliases maps provider names to their corresponding configurations.
var Aliases = map[string]Alias{
	"requesty": {
		APIType:     "openai",
		BaseURL:     "https://router.requesty.ai/v1",
		TokenEnvVar: "REQUESTY_API_KEY",
	},
	"azure": {
		APIType:     "openai",
		TokenEnvVar: "AZURE_API_KEY",
	},
	"xai": {
		APIType:     "openai",
		BaseURL:     "https://api.x.ai/v1",
		TokenEnvVar: "XAI_API_KEY",
	},
	"nebius": {
		APIType:     "openai",
		BaseURL:     "https://api.studio.nebius.com/v1",
		TokenEnvVar: "NEBIUS_API_KEY",
	},
	"mistral": {
		APIType:     "openai",
		BaseURL:     "https://api.mistral.ai/v1",
		TokenEnvVar: "MISTRAL_API_KEY",
	},
	"ollama": {
		APIType: "openai",
		BaseURL: "http://localhost:11434/v1",
	},
	"minimax": {
		APIType:     "openai",
		BaseURL:     "https://api.minimax.io/v1",
		TokenEnvVar: "MINIMAX_API_KEY",
	},
	"github-copilot": {
		APIType:     "openai",
		BaseURL:     "https://api.githubcopilot.com",
		TokenEnvVar: "GITHUB_TOKEN",
	},
}

// LookupAlias returns the Alias registered for the given name (if any).
func LookupAlias(name string) (Alias, bool) {
	alias, ok := Aliases[name]
	return alias, ok
}

// EachAlias returns an iterator over every registered (name, Alias) pair.
func EachAlias() iter.Seq2[string, Alias] {
	return func(yield func(string, Alias) bool) {
		for name, alias := range Aliases {
			if !yield(name, alias) {
				return
			}
		}
	}
}

// AllProviders returns all known provider names (core providers + aliases),
// sorted for deterministic output.
func AllProviders() []string {
	providers := slices.Concat(CoreProviders, slices.Collect(maps.Keys(Aliases)))
	slices.Sort(providers)
	return providers
}

// IsKnownProvider returns true if the provider name is a core provider or an alias.
func IsKnownProvider(name string) bool {
	if slices.Contains(CoreProviders, strings.ToLower(name)) {
		return true
	}
	_, exists := LookupAlias(strings.ToLower(name))
	return exists
}
