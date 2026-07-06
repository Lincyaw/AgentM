package statusbar

import (
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

// StatusBar displays key-binding help on the left and version info on the right.
type StatusBar struct {
	width int
	help  core.KeyMapHelp
	title string

	activity      string
	modeLine      string
	modeLineRight string

	cached     string
	cacheDirty bool
}

// Option is a functional option for configuring a StatusBar.
type Option func(*StatusBar)

// WithTitle sets a custom title for the status bar.
//
// If not provided, defaults to "AgentM Terminal".
func WithTitle(title string) Option {
	return func(s *StatusBar) {
		s.title = title
	}
}

// New creates a new StatusBar instance
func New(help core.KeyMapHelp, opts ...Option) StatusBar {
	s := StatusBar{
		help:       help,
		title:      "AgentM Terminal",
		cacheDirty: true,
	}

	for _, opt := range opts {
		opt(&s)
	}

	return s
}

// SetWidth sets the width of the status bar
func (s *StatusBar) SetWidth(width int) {
	if s.width != width {
		s.width = width
		s.cacheDirty = true
	}
}

// SetHelp sets the help provider for the status bar
func (s *StatusBar) SetHelp(help core.KeyMapHelp) {
	s.help = help
	s.cacheDirty = true
}

// SetActivity sets the compact background activity label rendered on the
// right side of the bar.
func (s *StatusBar) SetActivity(activity string) {
	if s.activity != activity {
		s.activity = activity
		s.cacheDirty = true
	}
}

// SetModeLine overrides the left help area with a context-specific status
// phrase. This supports Claude Code-style mode footers such as workflow task
// picker controls and transcript detail state.
func (s *StatusBar) SetModeLine(modeLine string) {
	if s.modeLine != modeLine {
		s.modeLine = modeLine
		s.cacheDirty = true
	}
}

func (s *StatusBar) SetModeLineRight(modeLineRight string) {
	if s.modeLineRight != modeLineRight {
		s.modeLineRight = modeLineRight
		s.cacheDirty = true
	}
}

// Height returns the rendered height of the status bar (always 1).
func (s *StatusBar) Height() int {
	return 1
}

// InvalidateCache clears all cached values.
func (s *StatusBar) InvalidateCache() {
	s.cacheDirty = true
}

// rebuild renders the full status bar line and computes click hitboxes.
func (s *StatusBar) rebuild() {
	s.cacheDirty = false

	// Build the styled right side: transient status, activity, and title.
	const pad = 1
	var rightW int
	var rightParts []string

	if s.modeLineRight != "" {
		if strings.Contains(s.modeLineRight, "\x1b[") {
			rightParts = append(rightParts, s.modeLineRight)
		} else {
			rightParts = append(rightParts, styles.SecondaryStyle.Render(s.modeLineRight))
		}
	}
	if s.activity != "" {
		rightParts = append(rightParts, styles.MutedStyle.Render(s.activity))
	}
	if s.title != "" {
		rightParts = append(rightParts, styles.MutedStyle.Render(s.title))
	}
	right := strings.Join(rightParts, "  ")
	rightW = lipgloss.Width(right)

	maxRightW := max(0, s.width-pad)
	if rightW > maxRightW {
		right = ansi.Truncate(right, maxRightW, "...")
		rightW = lipgloss.Width(right)
	}

	// Build the styled left side: help bindings (possibly truncated).
	maxHelpW := s.width - rightW - 2*pad - 1

	var left string
	var leftW int
	if s.modeLine != "" {
		if strings.Contains(s.modeLine, "\x1b[") {
			left = "  " + s.modeLine
		} else {
			left = "  " + styles.SecondaryStyle.Render(s.modeLine)
		}
		leftW = lipgloss.Width(left)
	} else if s.help != nil {
		if help := s.help.Help(); help != nil {
			var parts []string
			for _, b := range help.ShortHelp() {
				if b.Help().Key != "" && b.Help().Desc != "" {
					parts = append(parts,
						styles.HighlightWhiteStyle.Render(b.Help().Key)+
							" "+
							styles.SecondaryStyle.Render(b.Help().Desc))
				}
			}
			if len(parts) > 0 && maxHelpW > 0 {
				helpStr := strings.Join(parts, "  ")
				helpW := lipgloss.Width(helpStr)
				if helpW > maxHelpW {
					helpStr = ansi.Truncate(helpStr, maxHelpW, "...")
					helpW = lipgloss.Width(helpStr)
				}
				left = "  " + helpStr
				leftW = lipgloss.Width(left)
			}
		}
	}

	gap := max(1, s.width-leftW-rightW-2*pad)

	s.cached = left + strings.Repeat(" ", gap) + right + " "
}

// View renders the status bar.
//
// Layout: [ help text ...           activity  AgentM Terminal VERSION ]
func (s *StatusBar) View() string {
	if s.cacheDirty {
		s.rebuild()
	}
	return s.cached
}
