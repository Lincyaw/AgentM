// Package hooks holds shimmed cagent hook data-types consumed by the UI seam.
// Only the symbols referenced by the carved runtime/event.go are kept:
// EventType (carried by HookStartedEvent/HookFinishedEvent) and Result
// (read for Allowed/Message in HookFinished).
package hooks

// EventType identifies a hook event.
type EventType string

// Result is the aggregated outcome of dispatching one event.
type Result struct {
	// Allowed indicates if the operation should proceed.
	Allowed bool
	// Message is feedback to include in the response.
	Message string
}
