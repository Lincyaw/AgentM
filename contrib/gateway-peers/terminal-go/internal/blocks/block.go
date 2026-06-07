// Package blocks defines renderable content blocks for the AgentM terminal TUI.
// Each block type implements the Block interface with a pure Render() method
// that takes a width and theme and returns a styled string. No bubbletea dependency.
package blocks

import "github.com/AoyangSpace/agentm-terminal/internal/theme"

// Block is the common interface for all renderable content blocks.
type Block interface {
	// Kind returns the block type identifier (e.g., "user", "tool", "thinking").
	Kind() string
	// Collapsed returns whether the block is in collapsed display mode.
	Collapsed() bool
	// SetCollapsed sets the collapsed state.
	SetCollapsed(bool)
	// Render produces the styled string representation at the given terminal width.
	Render(width int, th *theme.Theme) string
}

// Focusable is implemented by blocks that can receive keyboard focus for
// expand/collapse and view-overlay operations.
type Focusable interface {
	Block
	Focused() bool
	SetFocused(bool)
}
