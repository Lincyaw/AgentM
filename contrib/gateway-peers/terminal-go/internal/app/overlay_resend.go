package app

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/components"
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// ResendOverlay provides reverse-i-search over input history.
type ResendOverlay struct {
	filter     string
	candidates []string // filtered history entries
	cursor     int
	history    []string // full history (not owned)

	resend bool   // true = re-send selected entry
	edit   bool   // true = load selected entry into input for editing
	chosen string // the selected entry text
}

// NewResendOverlay creates a resend overlay from the input history.
func NewResendOverlay(history []string) *ResendOverlay {
	o := &ResendOverlay{
		history: history,
	}
	o.refilter()
	return o
}

func (r *ResendOverlay) Kind() OverlayKind { return OverlayResend }

func (r *ResendOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			return r, nil, true

		case "up", "ctrl+p":
			if r.cursor > 0 {
				r.cursor--
			}
			return r, nil, false

		case "down", "ctrl+n":
			if r.cursor < len(r.candidates)-1 {
				r.cursor++
			}
			return r, nil, false

		case "enter":
			if len(r.candidates) > 0 {
				r.resend = true
				r.chosen = r.candidates[r.cursor]
				return r, func() tea.Msg {
					return components.InputSubmitted{Text: r.chosen}
				}, true
			}
			return r, nil, true

		case "tab":
			if len(r.candidates) > 0 {
				r.edit = true
				r.chosen = r.candidates[r.cursor]
			}
			return r, nil, true

		case "backspace":
			if len(r.filter) > 0 {
				r.filter = r.filter[:len(r.filter)-1]
				r.refilter()
			}
			return r, nil, false

		default:
			key := msg.String()
			if len(key) == 1 && key[0] >= 32 && key[0] < 127 {
				r.filter += key
				r.refilter()
			} else if len([]rune(key)) == 1 {
				r.filter += key
				r.refilter()
			}
			return r, nil, false
		}
	}
	return r, nil, false
}

func (r *ResendOverlay) refilter() {
	r.candidates = nil
	r.cursor = 0
	q := strings.ToLower(r.filter)
	// Show most recent first
	for i := len(r.history) - 1; i >= 0; i-- {
		entry := r.history[i]
		if q == "" || strings.Contains(strings.ToLower(entry), q) {
			r.candidates = append(r.candidates, entry)
		}
	}
}

// WantsEdit returns true if the user chose to load a history entry for editing.
func (r *ResendOverlay) WantsEdit() bool { return r.edit }

// Chosen returns the selected history entry text.
func (r *ResendOverlay) Chosen() string { return r.chosen }

func (r *ResendOverlay) View(width, height int, th *theme.Theme) string {
	var sb strings.Builder
	sb.WriteString(th.OverlayTitle.Render("Re-send (reverse search)"))
	sb.WriteByte('\n')
	sb.WriteByte('\n')

	// Filter input
	sb.WriteString(th.OverlayInput.Render(" filter: "))
	sb.WriteString(th.OverlayText.Render(r.filter))
	sb.WriteByte('\n')
	sb.WriteByte('\n')

	// Candidate list (show up to 10)
	maxShow := 10
	if len(r.candidates) == 0 {
		sb.WriteString(th.OverlayDim.Render("  (no matches)"))
		sb.WriteByte('\n')
	} else {
		shown := len(r.candidates)
		if shown > maxShow {
			shown = maxShow
		}
		for i := 0; i < shown; i++ {
			prefix := "  "
			style := th.OverlayText
			if i == r.cursor {
				prefix = "> "
				style = th.OverlayActive
			}
			label := r.candidates[i]
			if len(label) > 60 {
				label = label[:57] + "..."
			}
			sb.WriteString(style.Render(fmt.Sprintf("%s%s", prefix, label)))
			sb.WriteByte('\n')
		}
		if len(r.candidates) > maxShow {
			sb.WriteString(th.OverlayDim.Render(fmt.Sprintf("  ... and %d more", len(r.candidates)-maxShow)))
			sb.WriteByte('\n')
		}
	}

	sb.WriteByte('\n')
	sb.WriteString(th.OverlayDim.Render("  enter=send  tab=edit  esc=close"))

	content := th.OverlayBorder.Render(sb.String())
	return centerOverlay(content, width, height)
}
