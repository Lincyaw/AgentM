// Package util provides shared rendering helpers for the AgentM terminal TUI.
package util

import (
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// RenderDiff produces a simple line-by-line diff view.
// Lines from old are prefixed with "- " in the theme's DiffDel style,
// lines from new are prefixed with "+ " in the theme's DiffAdd style.
func RenderDiff(old, new string, th *theme.Theme) string {
	var b strings.Builder
	oldLines := splitLines(old)
	newLines := splitLines(new)
	for _, line := range oldLines {
		b.WriteString(th.DiffDel.Render("- " + line))
		b.WriteByte('\n')
	}
	for _, line := range newLines {
		b.WriteString(th.DiffAdd.Render("+ " + line))
		b.WriteByte('\n')
	}
	return strings.TrimRight(b.String(), "\n")
}

// Truncate shortens s to at most maxLen display-width columns, appending
// "..." if truncated. It iterates by rune to avoid splitting multi-byte
// characters.
func Truncate(s string, maxLen int) string {
	if maxLen <= 0 {
		return ""
	}
	w := lipgloss.Width(s)
	if w <= maxLen {
		return s
	}
	if maxLen <= 3 {
		// Not enough room for ellipsis; just take runes that fit.
		out := truncateByWidth(s, maxLen)
		return out
	}
	return truncateByWidth(s, maxLen-3) + "..."
}

// truncateByWidth returns the longest prefix of s whose display width is
// <= maxWidth. It never splits a rune.
func truncateByWidth(s string, maxWidth int) string {
	w := 0
	for i, r := range s {
		rw := lipgloss.Width(string(r))
		if w+rw > maxWidth {
			return s[:i]
		}
		w += rw
	}
	return s
}

func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(s, "\n")
}
