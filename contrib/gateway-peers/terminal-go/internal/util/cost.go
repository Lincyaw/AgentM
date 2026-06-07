package util

import "strings"

// ModelPricing holds per-model token pricing (USD per 1M tokens).
type ModelPricing struct {
	InputPerM  float64
	OutputPerM float64
}

// Known model pricing (approximate, can be updated).
var knownPricing = map[string]ModelPricing{
	"doubao":                     {InputPerM: 0.5, OutputPerM: 1.5},
	"doubao-seed-2-0-pro-260215": {InputPerM: 0.5, OutputPerM: 1.5},
	"gpt-4o":                     {InputPerM: 2.5, OutputPerM: 10.0},
	"gpt-4o-mini":                {InputPerM: 0.15, OutputPerM: 0.6},
	"claude-3-5-sonnet":          {InputPerM: 3.0, OutputPerM: 15.0},
	"claude-3-opus":              {InputPerM: 15.0, OutputPerM: 75.0},
	"claude-3-haiku":             {InputPerM: 0.25, OutputPerM: 1.25},
	"glm-5.1":                    {InputPerM: 1.0, OutputPerM: 4.0},
	"deepseek-chat":              {InputPerM: 0.27, OutputPerM: 1.1},
}

// DefaultPricing is used when the model is unknown.
var DefaultPricing = ModelPricing{InputPerM: 1.0, OutputPerM: 3.0}

// EstimateCost calculates the cost in USD for the given token counts.
func EstimateCost(model string, inputTokens, outputTokens int) float64 {
	pricing, ok := knownPricing[model]
	if !ok {
		// Try prefix match (e.g., "doubao-*" matches "doubao")
		for prefix, p := range knownPricing {
			if strings.HasPrefix(model, prefix) {
				pricing = p
				ok = true
				break
			}
		}
		if !ok {
			pricing = DefaultPricing
		}
	}

	inputCost := float64(inputTokens) / 1_000_000 * pricing.InputPerM
	outputCost := float64(outputTokens) / 1_000_000 * pricing.OutputPerM
	return inputCost + outputCost
}
