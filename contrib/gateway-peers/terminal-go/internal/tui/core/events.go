package core

import (
	"slices"

	tea "charm.land/bubbletea/v2"
)

// ScrollDirection represents the direction of scrolling
type ScrollDirection int

const (
	ScrollUp ScrollDirection = iota
	ScrollDown
	ScrollPageUp
	ScrollPageDown
	ScrollToTop
	ScrollToBottom
)

// KeyScrollMap maps common scroll keys to directions
var KeyScrollMap = map[string]ScrollDirection{
	"up":     ScrollUp,
	"k":      ScrollUp,
	"down":   ScrollDown,
	"j":      ScrollDown,
	"pgup":   ScrollPageUp,
	"pgdown": ScrollPageDown,
	"home":   ScrollToTop,
	"end":    ScrollToBottom,
}

// GetScrollDirection returns the scroll direction for a key press, or -1 if not a scroll key
func GetScrollDirection(msg tea.KeyPressMsg) (ScrollDirection, bool) {
	dir, ok := KeyScrollMap[msg.String()]
	return dir, ok
}

// NavigationKeys are common keys used for navigation that components might want to handle.
// Note: vim-style keys (h, j, k, l) are intentionally excluded to avoid conflicts with
// typing in completion popups where users need to type those letters.
var NavigationKeys = []string{"up", "down", "left", "right", "enter", "esc"}

// IsNavigationKey returns true if the key is a common navigation key
func IsNavigationKey(msg tea.KeyPressMsg) bool {
	return slices.Contains(NavigationKeys, msg.String())
}
