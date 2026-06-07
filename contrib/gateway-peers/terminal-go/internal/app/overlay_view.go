package app

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// ViewOverlay shows the full untruncated content of a focused block in a
// bordered, scrollable box. Up/Down or PgUp/PgDn scroll; Esc or any other
// key closes it.
type ViewOverlay struct {
	lines  []string
	offset int // top visible line index
	title  string
}

// NewViewOverlay creates a ViewOverlay for the supplied full-content text.
func NewViewOverlay(title, content string) *ViewOverlay {
	return &ViewOverlay{
		title: title,
		lines: strings.Split(content, "\n"),
	}
}

func (v *ViewOverlay) Kind() OverlayKind { return OverlayView }

func (v *ViewOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		key := msg.String()
		switch key {
		case "esc":
			return v, nil, true
		case "up", "k":
			if v.offset > 0 {
				v.offset--
			}
			return v, nil, false
		case "down", "j":
			v.offset++
			return v, nil, false
		case "pgup":
			v.offset -= 10
			if v.offset < 0 {
				v.offset = 0
			}
			return v, nil, false
		case "pgdown":
			v.offset += 10
			return v, nil, false
		default:
			// Any other key closes the overlay.
			return v, nil, true
		}
	}
	return v, nil, false
}

func (v *ViewOverlay) View(width, height int, th *theme.Theme) string {
	// Reserve space for border+padding (2 lines top/bottom, 4 chars left/right).
	innerH := height - 4
	if innerH < 1 {
		innerH = 1
	}
	innerW := width - 8
	if innerW < 20 {
		innerW = 20
	}

	// Clamp offset.
	maxOffset := len(v.lines) - innerH
	if maxOffset < 0 {
		maxOffset = 0
	}
	if v.offset > maxOffset {
		v.offset = maxOffset
	}
	if v.offset < 0 {
		v.offset = 0
	}

	end := v.offset + innerH
	if end > len(v.lines) {
		end = len(v.lines)
	}
	visible := v.lines[v.offset:end]

	var sb strings.Builder
	// Title
	sb.WriteString(th.OverlayTitle.Render(v.title))
	sb.WriteByte('\n')
	sb.WriteByte('\n')

	for _, line := range visible {
		// Truncate lines that exceed inner width to avoid wrapping.
		runes := []rune(line)
		if len(runes) > innerW {
			line = string(runes[:innerW])
		}
		sb.WriteString(th.OverlayText.Render(line))
		sb.WriteByte('\n')
	}

	// Scroll indicator
	total := len(v.lines)
	if total > innerH {
		pct := 0
		if maxOffset > 0 {
			pct = v.offset * 100 / maxOffset
		}
		sb.WriteByte('\n')
		sb.WriteString(th.OverlayDim.Render(strings.Repeat("─", 20)))
		sb.WriteString(th.OverlayDim.Render(strings.Repeat(" ", 2)))
		sb.WriteString(th.OverlayDim.Render(formatScrollInfo(v.offset+1, total, pct)))
	}

	sb.WriteByte('\n')
	sb.WriteString(th.OverlayDim.Render("esc / any key to close"))

	content := th.OverlayBorder.Render(sb.String())
	return centerOverlay(content, width, height)
}

func formatScrollInfo(line, total, pct int) string {
	return strings.Join([]string{
		"line " + itoa(line) + "/" + itoa(total),
		"(" + itoa(pct) + "%)",
	}, " ")
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	var buf [20]byte
	pos := len(buf)
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
