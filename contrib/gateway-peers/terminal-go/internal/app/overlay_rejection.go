package app

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// Preset rejection reasons shown when the user rejects a tool call.
var rejectionReasons = []string{
	"Don't do this",
	"Try a different approach",
	"The command is dangerous",
	"Wrong file/path",
	"Not what I asked for",
}

// RejectionOverlay lets the user pick a reason for rejecting a tool call.
// Selecting a preset or typing a custom reason calls onResult with the text.
type RejectionOverlay struct {
	toolName string
	reasons  []string
	cursor   int
	custom   string // custom reason text
	inCustom bool   // true when editing custom reason
	onResult func(reason string)
}

// NewRejectionOverlay creates a rejection reason overlay for the given tool.
func NewRejectionOverlay(toolName string, onResult func(reason string)) *RejectionOverlay {
	return &RejectionOverlay{
		toolName: toolName,
		reasons:  rejectionReasons,
		onResult: onResult,
	}
}

func (r *RejectionOverlay) Kind() OverlayKind { return OverlayRejection }

func (r *RejectionOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		key := msg.String()

		if r.inCustom {
			return r.updateCustom(key)
		}

		switch key {
		case "esc":
			return r, nil, true

		case "up", "k":
			if r.cursor > 0 {
				r.cursor--
			}
			return r, nil, false

		case "down", "j":
			if r.cursor < len(r.reasons) {
				// len(r.reasons) is the "(Custom...)" entry
				r.cursor++
			}
			return r, nil, false

		case "enter":
			if r.cursor < len(r.reasons) {
				if r.onResult != nil {
					r.onResult(r.reasons[r.cursor])
				}
				return r, nil, true
			}
			// "(Custom...)" selected
			r.inCustom = true
			return r, nil, false

		case "tab":
			r.inCustom = true
			r.cursor = len(r.reasons)
			return r, nil, false

		default:
			// Digit shortcuts 1-5 for preset reasons
			if len(key) == 1 && key[0] >= '1' && key[0] <= '5' {
				idx := int(key[0] - '1')
				if idx < len(r.reasons) {
					if r.onResult != nil {
						r.onResult(r.reasons[idx])
					}
					return r, nil, true
				}
			}
			return r, nil, false
		}
	}
	return r, nil, false
}

// updateCustom handles input while editing the custom reason.
func (r *RejectionOverlay) updateCustom(key string) (Overlay, tea.Cmd, bool) {
	switch key {
	case "esc":
		// Go back to preset list
		r.inCustom = false
		return r, nil, false

	case "enter":
		reason := strings.TrimSpace(r.custom)
		if reason == "" {
			reason = "Rejected"
		}
		if r.onResult != nil {
			r.onResult(reason)
		}
		return r, nil, true

	case "backspace":
		if len(r.custom) > 0 {
			runes := []rune(r.custom)
			r.custom = string(runes[:len(runes)-1])
		}
		return r, nil, false

	default:
		if len(key) == 1 && key[0] >= 32 && key[0] < 127 {
			r.custom += key
		} else if len([]rune(key)) == 1 {
			r.custom += key
		}
		return r, nil, false
	}
}

func (r *RejectionOverlay) View(width, height int, th *theme.Theme) string {
	var sb strings.Builder
	sb.WriteString(th.OverlayTitle.Render("Why reject this tool call?"))
	sb.WriteByte('\n')
	if r.toolName != "" {
		sb.WriteString(th.OverlayDim.Render(fmt.Sprintf("  tool: %s", r.toolName)))
		sb.WriteByte('\n')
	}
	sb.WriteByte('\n')

	for i, reason := range r.reasons {
		prefix := "  "
		style := th.OverlayText
		if i == r.cursor && !r.inCustom {
			prefix = "> "
			style = th.OverlayActive
		}
		sb.WriteString(style.Render(fmt.Sprintf("%s%d. %s", prefix, i+1, reason)))
		sb.WriteByte('\n')
	}

	// "(Custom...)" entry
	{
		prefix := "  "
		style := th.OverlayText
		if r.cursor == len(r.reasons) && !r.inCustom {
			prefix = "> "
			style = th.OverlayActive
		}
		if r.inCustom {
			prefix = "> "
			style = th.OverlayActive
		}
		sb.WriteString(style.Render(prefix + "(Custom...)"))
		sb.WriteByte('\n')
	}

	if r.inCustom {
		sb.WriteByte('\n')
		sb.WriteString(th.OverlayInput.Render(" reason: "))
		sb.WriteString(th.OverlayText.Render(r.custom))
		sb.WriteString(th.OverlayDim.Render("_"))
		sb.WriteByte('\n')
	}

	sb.WriteByte('\n')
	if r.inCustom {
		sb.WriteString(th.OverlayDim.Render("  enter=confirm  esc=back"))
	} else {
		sb.WriteString(th.OverlayDim.Render("  enter=confirm  1-5=quick  tab=custom  esc=cancel"))
	}

	content := th.OverlayBorder.Render(sb.String())
	return centerOverlay(content, width, height)
}
