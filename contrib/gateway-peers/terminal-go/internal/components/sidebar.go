package components

import (
	"fmt"
	"strings"
	"time"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

const (
	SidebarDefaultWidth = 28
	SidebarMinWidth     = 20
	SidebarMaxPercent   = 40
	SidebarHandleWidth  = 1
)

// Sidebar renders a right-side panel showing session metadata.
type Sidebar struct {
	width     int
	collapsed bool

	model       string
	ctxUsed     int
	ctxTotal    int
	tokensIn    int
	tokensOut   int
	costTurn    float64
	costSession float64
	tools       []string
	commands    []string
	phase       theme.Phase
	elapsed     time.Duration
}

// NewSidebar creates a sidebar with the default width, initially visible.
func NewSidebar() Sidebar {
	return Sidebar{
		width:    SidebarDefaultWidth,
		ctxTotal: 131072,
	}
}

// SetData updates the sidebar from a StatusModel snapshot.
func (s *Sidebar) SetData(sm StatusModel) {
	s.model = sm.Model
	s.ctxUsed = sm.CtxUsed
	s.ctxTotal = sm.CtxTotal
	s.tokensIn = sm.TokensIn
	s.tokensOut = sm.TokensOut
	s.costTurn = sm.CostTurn
	s.costSession = sm.CostSession
	s.phase = sm.Phase
	s.elapsed = sm.Elapsed
}

// AddTool registers a tool name (deduplicated).
func (s *Sidebar) AddTool(name string) {
	for _, t := range s.tools {
		if t == name {
			return
		}
	}
	s.tools = append(s.tools, name)
}

// AddCommand registers a command name (deduplicated).
func (s *Sidebar) AddCommand(name string) {
	for _, c := range s.commands {
		if c == name {
			return
		}
	}
	s.commands = append(s.commands, name)
}

// Toggle collapses or expands the sidebar.
func (s *Sidebar) Toggle() {
	s.collapsed = !s.collapsed
}

// Visible reports whether the sidebar is currently shown.
func (s *Sidebar) Visible() bool {
	return !s.collapsed
}

// Width returns the current sidebar width (0 if collapsed).
func (s *Sidebar) Width() int {
	if s.collapsed {
		return 0
	}
	return s.width
}

// TotalWidth returns sidebar width plus the handle separator (0 if collapsed).
func (s *Sidebar) TotalWidth() int {
	if s.collapsed {
		return 0
	}
	return s.width + SidebarHandleWidth
}

// ClampWidth enforces the max-percent constraint relative to the window width.
func (s *Sidebar) ClampWidth(windowWidth int) {
	maxW := windowWidth * SidebarMaxPercent / 100
	if maxW < SidebarMinWidth {
		maxW = SidebarMinWidth
	}
	if s.width > maxW {
		s.width = maxW
	}
	if s.width < SidebarMinWidth {
		s.width = SidebarMinWidth
	}
}

// View renders the sidebar content for the given height.
func (s *Sidebar) View(height int, th *theme.Theme) string {
	if s.collapsed {
		return ""
	}
	w := s.width

	var lines []string

	// Title
	lines = append(lines, th.SidebarHeader.Render(padRight("SESSION", w)))
	lines = append(lines, "")

	// Model
	lines = append(lines, th.SidebarDim.Render("Model"))
	modelVal := s.model
	if modelVal == "" {
		modelVal = "-"
	}
	lines = append(lines, th.SidebarValue.Render(truncSidebar(modelVal, w)))

	// Context usage bar
	lines = append(lines, th.SidebarDim.Render("Context"))
	lines = append(lines, s.renderContextBar(w, th))

	// Separator
	lines = append(lines, th.SidebarDim.Render(strings.Repeat("─", w)))

	// Tokens
	lines = append(lines, th.SidebarDim.Render("Tokens"))
	tokLine := fmt.Sprintf("↑ %s  ↓ %s", formatCount(s.tokensIn), formatCount(s.tokensOut))
	lines = append(lines, th.SidebarValue.Render(truncSidebar(tokLine, w)))

	// Cost
	costLine := fmt.Sprintf("$%.4f / $%.4f", s.costTurn, s.costSession)
	lines = append(lines, th.SidebarDim.Render(truncSidebar(costLine, w)))

	// Phase & elapsed
	if s.phase != "" && s.phase != theme.PhaseIdle {
		lines = append(lines, "")
		glyph := theme.PhaseGlyph(s.phase)
		phaseLine := fmt.Sprintf("%s %s %.1fs", glyph, string(s.phase), s.elapsed.Seconds())
		lines = append(lines, th.SidebarValue.Render(truncSidebar(phaseLine, w)))
	}

	// Separator
	lines = append(lines, th.SidebarDim.Render(strings.Repeat("─", w)))

	// Tools
	toolHeader := fmt.Sprintf("Tools (%d)", len(s.tools))
	lines = append(lines, th.SidebarDim.Render(toolHeader))
	maxToolLines := 8
	for i, tool := range s.tools {
		if i >= maxToolLines {
			remaining := len(s.tools) - maxToolLines
			lines = append(lines, th.SidebarDim.Render(fmt.Sprintf(" +%d more", remaining)))
			break
		}
		lines = append(lines, th.SidebarValue.Render(truncSidebar(" "+tool, w)))
	}

	// Separator
	lines = append(lines, th.SidebarDim.Render(strings.Repeat("─", w)))

	// Commands
	cmdHeader := fmt.Sprintf("Commands (%d)", len(s.commands))
	lines = append(lines, th.SidebarDim.Render(cmdHeader))
	maxCmdLines := 6
	for i, cmd := range s.commands {
		if i >= maxCmdLines {
			remaining := len(s.commands) - maxCmdLines
			lines = append(lines, th.SidebarDim.Render(fmt.Sprintf(" +%d more", remaining)))
			break
		}
		lines = append(lines, th.SidebarValue.Render(truncSidebar(" "+cmd, w)))
	}

	// Pad or trim to the target height
	for len(lines) < height {
		lines = append(lines, strings.Repeat(" ", w))
	}
	if len(lines) > height {
		lines = lines[:height]
	}

	return strings.Join(lines, "\n")
}

// RenderHandle renders the vertical separator between the viewport and sidebar.
func RenderHandle(height int, th *theme.Theme) string {
	line := th.SidebarBorder.Render("│")
	lines := make([]string, height)
	for i := range lines {
		lines[i] = line
	}
	return strings.Join(lines, "\n")
}

func (s *Sidebar) renderContextBar(width int, th *theme.Theme) string {
	total := s.ctxTotal
	if total <= 0 {
		total = 131072
	}
	pct := float64(s.ctxUsed) / float64(total) * 100
	if pct > 100 {
		pct = 100
	}

	barWidth := width - 7 // "[████░░] XX%"
	if barWidth < 4 {
		barWidth = 4
	}
	filled := int(pct / 100 * float64(barWidth))
	if filled > barWidth {
		filled = barWidth
	}
	empty := barWidth - filled

	bar := "[" +
		th.GaugeFilled.Render(strings.Repeat("█", filled)) +
		th.GaugeEmpty.Render(strings.Repeat("░", empty)) +
		"]"
	return bar + fmt.Sprintf(" %d%%", int(pct))
}

func formatCount(n int) string {
	if n >= 1_000_000 {
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	}
	if n >= 1_000 {
		return fmt.Sprintf("%.1fK", float64(n)/1_000)
	}
	return fmt.Sprintf("%d", n)
}

func truncSidebar(s string, maxWidth int) string {
	if len(s) <= maxWidth {
		return s
	}
	if maxWidth <= 3 {
		return s[:maxWidth]
	}
	return s[:maxWidth-3] + "..."
}

func padRight(s string, width int) string {
	if len(s) >= width {
		return s[:width]
	}
	return s + strings.Repeat(" ", width-len(s))
}
