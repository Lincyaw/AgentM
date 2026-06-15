// Package util provides shared rendering helpers for the AgentM terminal TUI.
package util

import (
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// RenderDiff produces a word-level diff view. Consecutive pairs of
// removed+added lines are compared token-by-token so the exact changed
// portions are highlighted with DiffDelHighlight / DiffAddHighlight.
// Unpaired lines (pure additions or pure deletions) use the base
// DiffDel / DiffAdd style as before.
func RenderDiff(old, new string, th *theme.Theme) string {
	var b strings.Builder
	oldLines := splitLines(old)
	newLines := splitLines(new)

	paired := min(len(oldLines), len(newLines))

	for i := 0; i < paired; i++ {
		oldSegs, newSegs := WordDiff(oldLines[i], newLines[i])
		b.WriteString(renderSegmentLine("- ", oldSegs, th.DiffDel, th.DiffDelHighlight))
		b.WriteByte('\n')
		b.WriteString(renderSegmentLine("+ ", newSegs, th.DiffAdd, th.DiffAddHighlight))
		b.WriteByte('\n')
	}

	// Remaining unpaired old lines (pure deletions).
	for i := paired; i < len(oldLines); i++ {
		b.WriteString(th.DiffDel.Render("- " + oldLines[i]))
		b.WriteByte('\n')
	}
	// Remaining unpaired new lines (pure additions).
	for i := paired; i < len(newLines); i++ {
		b.WriteString(th.DiffAdd.Render("+ " + newLines[i]))
		b.WriteByte('\n')
	}

	return strings.TrimRight(b.String(), "\n")
}

// renderSegmentLine renders a diff line with per-segment styling. The
// prefix ("- " or "+ ") and unchanged segments use baseStyle; changed
// segments use highlightStyle so they stand out within the line.
func renderSegmentLine(prefix string, segs []DiffSegment, baseStyle, highlightStyle lipgloss.Style) string {
	var sb strings.Builder
	sb.WriteString(baseStyle.Render(prefix))
	for _, seg := range segs {
		if seg.Changed {
			sb.WriteString(highlightStyle.Render(seg.Text))
		} else {
			sb.WriteString(baseStyle.Render(seg.Text))
		}
	}
	return sb.String()
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
