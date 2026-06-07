package components

import (
	"fmt"
	"strings"
	"time"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// StatusModel holds the data rendered by the status bar.
type StatusModel struct {
	Phase       theme.Phase
	Model       string
	Elapsed     time.Duration
	TokPerSec   float64
	TokensIn    int
	TokensOut   int
	ToolCount   int
	CtxUsed     int
	CtxTotal    int
	CostTurn    float64
	CostSession float64
	SessionKey  string
	SessionAge  time.Duration
	BudgetWarn  bool
}

// StatusBar is a pure-render component for the single-line status area.
type StatusBar struct {
	model StatusModel
}

// Update replaces the status model wholesale.
func (s *StatusBar) Update(m StatusModel) { s.model = m }

// GetModel returns a copy of the current status model.
func (s *StatusBar) GetModel() StatusModel { return s.model }

// View renders the single-line status bar at the given width.
// Format: model · context_pct% · session_key [· elapsed · tok/s]
func (s *StatusBar) View(width int, th *theme.Theme) string {
	m := s.model
	if m.CtxTotal <= 0 {
		m.CtxTotal = 131072
	}

	var parts []string

	if m.Model != "" {
		parts = append(parts, m.Model)
	}

	// Context percentage
	if m.CtxTotal > 0 {
		pct := float64(m.CtxUsed) / float64(m.CtxTotal) * 100
		parts = append(parts, fmt.Sprintf("%d%%", int(pct)))
	}

	// Session key
	if m.SessionKey != "" {
		parts = append(parts, m.SessionKey)
	}

	// In-flight: elapsed + tok/s
	if m.Phase != theme.PhaseIdle {
		parts = append(parts, formatDuration(m.Elapsed))
		if m.TokPerSec > 0 {
			parts = append(parts, fmt.Sprintf("%.0f tok/s", m.TokPerSec))
		}
	}

	line := strings.Join(parts, " · ")
	return th.StatusDim.Width(width).Render(line)
}

func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	if d < time.Hour {
		return fmt.Sprintf("%.0fm", d.Minutes())
	}
	return fmt.Sprintf("%.0fh%.0fm", d.Hours(), d.Minutes()-float64(int(d.Hours()))*60)
}
