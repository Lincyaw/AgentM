package termfeatures

import "strings"

// SupportsModifiedEnter returns true for terminals that can distinguish
// Shift+Enter from Enter even when they do not report Kitty keyboard flags.
func SupportsModifiedEnter(getenv func(string) string) bool {
	if getenv == nil {
		return false
	}

	termProgram := strings.ToLower(getenv("TERM_PROGRAM"))
	term := strings.ToLower(getenv("TERM"))

	return termProgram == "wezterm" ||
		getenv("WEZTERM_PANE") != "" ||
		getenv("WEZTERM_UNIX_SOCKET") != "" ||
		strings.Contains(term, "wezterm")
}
