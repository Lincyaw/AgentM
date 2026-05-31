package app

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// OverlayKind identifies the type of overlay currently active.
type OverlayKind int

const (
	OverlayNone      OverlayKind = iota
	OverlaySearch                // Ctrl+F search bar
	OverlayHelp                  // ? / F1 keybinding reference
	OverlayBookmarks             // Ctrl+G bookmark list
	OverlayResend                // Ctrl+R reverse-i-search
	OverlayCodeSave              // Ctrl+S code block save
)

// Overlay is a transient UI mode that temporarily takes over key handling
// and renders on top of (or inline with) the viewport.
type Overlay interface {
	Kind() OverlayKind
	// Update handles a message while the overlay is active.
	// Returns the (possibly updated) overlay, an optional command, and
	// whether the overlay has closed itself.
	Update(msg tea.Msg) (Overlay, tea.Cmd, bool)
	// View renders the overlay content at the given dimensions.
	View(width, height int, th *theme.Theme) string
}

// centerOverlay places content in a centered box within the given dimensions.
func centerOverlay(content string, width, height int) string {
	lines := strings.Split(content, "\n")
	contentHeight := len(lines)

	// Vertical centering
	topPad := (height - contentHeight) / 2
	if topPad < 0 {
		topPad = 0
	}
	bottomPad := height - contentHeight - topPad
	if bottomPad < 0 {
		bottomPad = 0
	}

	var sb strings.Builder

	// Top padding
	emptyLine := strings.Repeat(" ", width)
	for i := 0; i < topPad; i++ {
		sb.WriteString(emptyLine)
		sb.WriteByte('\n')
	}

	// Content lines, horizontally centered
	for _, line := range lines {
		lineWidth := len([]rune(line))
		leftPad := (width - lineWidth) / 2
		if leftPad < 0 {
			leftPad = 0
		}
		sb.WriteString(strings.Repeat(" ", leftPad))
		sb.WriteString(line)
		rightPad := width - leftPad - lineWidth
		if rightPad > 0 {
			sb.WriteString(strings.Repeat(" ", rightPad))
		}
		sb.WriteByte('\n')
	}

	// Bottom padding
	for i := 0; i < bottomPad; i++ {
		sb.WriteString(emptyLine)
		if i < bottomPad-1 {
			sb.WriteByte('\n')
		}
	}

	return sb.String()
}
