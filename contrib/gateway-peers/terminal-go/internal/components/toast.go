package components

import (
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// Toast represents a single timed notification.
type Toast struct {
	Text    string
	Variant string // "info", "warn", "selfmod"
	Expires time.Time
}

// ToastStack manages a list of expiring toast notifications.
type ToastStack struct {
	toasts []Toast
}

// Push adds a toast with the given TTL.
func (t *ToastStack) Push(text, variant string, ttl time.Duration) {
	t.toasts = append(t.toasts, Toast{
		Text:    text,
		Variant: variant,
		Expires: time.Now().Add(ttl),
	})
}

// Tick removes expired toasts. Returns true if any were removed.
func (t *ToastStack) Tick() bool {
	now := time.Now()
	n := 0
	for _, toast := range t.toasts {
		if toast.Expires.After(now) {
			t.toasts[n] = toast
			n++
		}
	}
	changed := n < len(t.toasts)
	t.toasts = t.toasts[:n]
	return changed
}

// View renders toasts stacked vertically, right-aligned within maxWidth.
func (t *ToastStack) View(maxWidth int, th *theme.Theme) string {
	if len(t.toasts) == 0 {
		return ""
	}

	var lines []string
	for _, toast := range t.toasts {
		style := toastStyle(toast.Variant, th)
		rendered := style.MaxWidth(maxWidth).Render(toast.Text)
		// Right-align within maxWidth
		w := lipgloss.Width(rendered)
		if w < maxWidth {
			rendered = strings.Repeat(" ", maxWidth-w) + rendered
		}
		lines = append(lines, rendered)
	}
	return strings.Join(lines, "\n")
}

// Empty returns true if there are no active toasts.
func (t *ToastStack) Empty() bool {
	return len(t.toasts) == 0
}

func toastStyle(variant string, th *theme.Theme) lipgloss.Style {
	switch variant {
	case "warn":
		return th.ToastWarn
	case "selfmod":
		return th.ToastSelfmod
	default:
		return th.ToastInfo
	}
}
