package components

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"

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

// StatusBar is a pure-render component for the 2-line status area.
type StatusBar struct {
	model StatusModel
}

// Update replaces the status model wholesale.
func (s *StatusBar) Update(m StatusModel) { s.model = m }

// GetModel returns a copy of the current status model.
func (s *StatusBar) GetModel() StatusModel { return s.model }

// View renders the two-line status bar at the given width.
func (s *StatusBar) View(width int, th *theme.Theme) string {
	m := s.model
	if m.CtxTotal <= 0 {
		m.CtxTotal = 131072
	}

	// Line 1: phase glyph + phase . model . elapsed . tok/s
	parts := []string{
		th.StatusPhase.Render(theme.PhaseGlyph(m.Phase)),
		th.StatusBold.Render(string(m.Phase)),
		th.StatusDim.Render(m.Model),
	}
	if m.Phase != theme.PhaseIdle {
		parts = append(parts, th.StatusDim.Render(formatDuration(m.Elapsed)))
		if m.TokPerSec > 0 {
			parts = append(parts, th.StatusDim.Render(fmt.Sprintf("%.0f tok/s", m.TokPerSec)))
		}
	}
	line1 := strings.Join(parts, th.StatusDim.Render(" · "))

	// Line 2: ctx gauge . cost . session key . session age
	gauge := renderGauge(m.CtxUsed, m.CtxTotal, 10, th)
	costStyle := th.StatusDim
	if m.BudgetWarn {
		costStyle = th.StatusWarn
	}
	costStr := costStyle.Render(fmt.Sprintf("$%.2f / $%.2f", m.CostTurn, m.CostSession))

	line2Parts := []string{
		th.StatusDim.Render("ctx ") + gauge,
		costStr,
	}
	if m.SessionKey != "" {
		line2Parts = append(line2Parts, th.StatusDim.Render(m.SessionKey))
	}
	if m.SessionAge > 0 {
		line2Parts = append(line2Parts, th.StatusDim.Render(formatDuration(m.SessionAge)))
	}
	line2 := strings.Join(line2Parts, th.StatusDim.Render(" · "))

	// Pad lines to full width with a bottom border feel
	borderStyle := lipgloss.NewStyle().
		Width(width).
		BorderBottom(true).
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color("238"))

	return borderStyle.Render(line1 + "\n" + line2)
}

func renderGauge(used, total, barWidth int, th *theme.Theme) string {
	if total <= 0 {
		total = 131072
	}
	pct := float64(used) / float64(total)
	if pct > 1 {
		pct = 1
	}
	filled := int(pct * float64(barWidth))
	if filled > barWidth {
		filled = barWidth
	}
	empty := barWidth - filled

	filledStr := th.GaugeFilled.Render(strings.Repeat("█", filled))
	emptyStr := th.GaugeEmpty.Render(strings.Repeat("░", empty))
	return fmt.Sprintf("[%s%s] %d%%", filledStr, emptyStr, int(pct*100))
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
