package components

import (
	"strings"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

const maxVisibleSuggestions = 5

// SuggestionList renders a vertical list of completions.
type SuggestionList struct {
	visible bool
	matches []string
	cursor  int
}

// Populate sets the suggestion list content and shows it.
func (s *SuggestionList) Populate(matches []string) {
	s.matches = matches
	s.cursor = 0
	s.visible = len(matches) > 0
}

// Move shifts the cursor by delta, wrapping around.
func (s *SuggestionList) Move(delta int) {
	if len(s.matches) == 0 {
		return
	}
	s.cursor = (s.cursor + delta + len(s.matches)) % len(s.matches)
}

// Current returns the currently highlighted suggestion.
func (s *SuggestionList) Current() string {
	if !s.visible || len(s.matches) == 0 {
		return ""
	}
	return s.matches[s.cursor]
}

// Hide closes the suggestion list.
func (s *SuggestionList) Hide() {
	s.visible = false
}

// Visible returns whether the suggestion list is shown.
func (s *SuggestionList) Visible() bool {
	return s.visible
}

// View renders the suggestion list.
func (s *SuggestionList) View(width int, th *theme.Theme) string {
	if !s.visible || len(s.matches) == 0 {
		return ""
	}

	_ = width

	// Determine visible window
	start := 0
	end := len(s.matches)
	if end > maxVisibleSuggestions {
		// Center the cursor in the visible window
		half := maxVisibleSuggestions / 2
		start = s.cursor - half
		if start < 0 {
			start = 0
		}
		end = start + maxVisibleSuggestions
		if end > len(s.matches) {
			end = len(s.matches)
			start = end - maxVisibleSuggestions
			if start < 0 {
				start = 0
			}
		}
	}

	var sb strings.Builder
	for i := start; i < end; i++ {
		if i > start {
			sb.WriteString("\n")
		}
		label := s.matches[i]
		if i == s.cursor {
			sb.WriteString(th.SuggestionActive.Render("▸ " + label))
		} else {
			sb.WriteString(th.SuggestionNormal.Render("  " + label))
		}
	}
	return sb.String()
}

// Count returns the number of suggestions.
func (s *SuggestionList) Count() int {
	return len(s.matches)
}
