// Package theme defines the visual vocabulary for the AgentM terminal TUI.
// It provides glyphs, labels, and lipgloss styles for two themes (dark and light).
// No bubbletea dependency -- pure styling definitions.
package theme

import "github.com/charmbracelet/lipgloss"

// Phase represents the agent's current lifecycle state.
type Phase string

const (
	PhaseIdle      Phase = "idle"
	PhaseThinking  Phase = "thinking"
	PhaseStreaming  Phase = "streaming"
	PhaseTool      Phase = "tool"
	PhaseSubagent  Phase = "subagent"
)

// PhaseGlyphMap maps each phase to its display glyph.
var PhaseGlyphMap = map[Phase]string{
	PhaseIdle:     "●",
	PhaseThinking: "◐",
	PhaseStreaming: "◑",
	PhaseTool:     "⚙",
	PhaseSubagent: "⌥",
}

// PhaseGlyph returns the glyph for a given phase (function form for callers).
func PhaseGlyph(p Phase) string {
	if g, ok := PhaseGlyphMap[p]; ok {
		return g
	}
	return "●"
}

// Claude Code style glyphs.
const (
	BlackCircle   = "●" // assistant turn marker + tool call marker
	ThinkingGlyph = "∴" // thinking block header
)

// Theme holds all lipgloss styles for rendering blocks and UI components.
type Theme struct {
	// StatusBar styles for the bottom status bar.
	StatusBar   lipgloss.Style
	StatusPhase lipgloss.Style
	StatusBold  lipgloss.Style
	StatusDim   lipgloss.Style
	StatusWarn  lipgloss.Style

	// User message style (background color, no label).
	UserMessageBg lipgloss.Style

	// Assistant dot styles.
	AssistantDot    lipgloss.Style // bold ● glyph
	AssistantDotDim lipgloss.Style // dim ● for in-progress
	AssistantDotOK  lipgloss.Style // green ● for completed
	AssistantDotErr lipgloss.Style // red ● for failed

	// System attribution.
	SystemAttrib lipgloss.Style

	// Thinking block styles.
	ThinkingText  lipgloss.Style
	ThinkingLabel lipgloss.Style // dim + italic for "∴ Thinking"
	ThinkingHint  lipgloss.Style // dim for "(ctrl+e to expand)"

	// Tool block styles.
	ToolTitle lipgloss.Style
	ToolBody  lipgloss.Style
	DiffAdd   lipgloss.Style
	DiffDel   lipgloss.Style

	// Approval block styles.
	ApprovalWarn   lipgloss.Style
	ApprovalChoice lipgloss.Style

	// Toast notification styles.
	ToastInfo    lipgloss.Style
	ToastWarn    lipgloss.Style
	ToastSelfmod lipgloss.Style

	// Gauge bar styles.
	GaugeFilled lipgloss.Style
	GaugeEmpty  lipgloss.Style

	// Suggestion list styles.
	SuggestionNormal lipgloss.Style
	SuggestionActive lipgloss.Style

	// Input prompt style.
	InputPrompt lipgloss.Style

	// Overlay styles.
	OverlayBorder   lipgloss.Style // border for centered overlay boxes
	OverlayTitle    lipgloss.Style // overlay title text
	OverlayText     lipgloss.Style // normal overlay body text
	OverlayDim      lipgloss.Style // secondary/hint text in overlays
	OverlayActive   lipgloss.Style // highlighted item in a list overlay
	OverlayInput    lipgloss.Style // inline overlay input bar
	SearchHighlight lipgloss.Style // matched text highlight in search
}

// DarkTheme returns a theme suited for dark terminal backgrounds.
func DarkTheme() *Theme {
	accent := lipgloss.AdaptiveColor{Dark: "#A78BFA", Light: "#7C3AED"}
	dim := lipgloss.AdaptiveColor{Dark: "#6B7280", Light: "#9CA3AF"}
	yellow := lipgloss.AdaptiveColor{Dark: "#FBBF24", Light: "#D97706"}
	green := lipgloss.AdaptiveColor{Dark: "#34D399", Light: "#059669"}
	red := lipgloss.AdaptiveColor{Dark: "#F87171", Light: "#DC2626"}
	warn := lipgloss.AdaptiveColor{Dark: "#FBBF24", Light: "#D97706"}

	return &Theme{
		StatusBar:   lipgloss.NewStyle().Background(lipgloss.Color("#1F2937")).Foreground(lipgloss.Color("#D1D5DB")).Padding(0, 1),
		StatusPhase: lipgloss.NewStyle().Foreground(accent).Bold(true),
		StatusBold:  lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#E5E7EB")),
		StatusDim:   lipgloss.NewStyle().Foreground(dim),
		StatusWarn:  lipgloss.NewStyle().Foreground(warn).Bold(true),

		UserMessageBg: lipgloss.NewStyle().Background(lipgloss.Color("#373737")).PaddingRight(1),

		AssistantDot:    lipgloss.NewStyle().Bold(true),
		AssistantDotDim: lipgloss.NewStyle().Faint(true),
		AssistantDotOK:  lipgloss.NewStyle().Foreground(green),
		AssistantDotErr: lipgloss.NewStyle().Foreground(red),

		SystemAttrib: lipgloss.NewStyle().Foreground(dim).Italic(true),

		ThinkingText:  lipgloss.NewStyle().Foreground(dim),
		ThinkingLabel: lipgloss.NewStyle().Foreground(dim).Italic(true),
		ThinkingHint:  lipgloss.NewStyle().Foreground(dim),

		ToolTitle: lipgloss.NewStyle().Foreground(yellow).Bold(true),
		ToolBody:  lipgloss.NewStyle().Foreground(dim),
		DiffAdd:   lipgloss.NewStyle().Foreground(green),
		DiffDel:   lipgloss.NewStyle().Foreground(red),

		ApprovalWarn:   lipgloss.NewStyle().Foreground(warn).Bold(true),
		ApprovalChoice: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Dark: "#E5E7EB", Light: "#374151"}),

		ToastInfo:    lipgloss.NewStyle().Foreground(accent).Background(lipgloss.Color("#1F2937")).Padding(0, 1),
		ToastWarn:    lipgloss.NewStyle().Foreground(warn).Background(lipgloss.Color("#1F2937")).Padding(0, 1),
		ToastSelfmod: lipgloss.NewStyle().Foreground(red).Bold(true).Background(lipgloss.Color("#1F2937")).Padding(0, 1),

		GaugeFilled: lipgloss.NewStyle().Foreground(accent),
		GaugeEmpty:  lipgloss.NewStyle().Foreground(dim),

		SuggestionNormal: lipgloss.NewStyle().Foreground(lipgloss.Color("#D1D5DB")),
		SuggestionActive: lipgloss.NewStyle().Foreground(accent).Bold(true),

		InputPrompt: lipgloss.NewStyle().Foreground(accent),

		OverlayBorder: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#6B7280")).
			Padding(1, 2),
		OverlayTitle:    lipgloss.NewStyle().Foreground(accent).Bold(true),
		OverlayText:     lipgloss.NewStyle().Foreground(lipgloss.Color("#E5E7EB")),
		OverlayDim:      lipgloss.NewStyle().Foreground(dim),
		OverlayActive:   lipgloss.NewStyle().Foreground(accent).Bold(true),
		OverlayInput:    lipgloss.NewStyle().Foreground(lipgloss.Color("#E5E7EB")).Background(lipgloss.Color("#1F2937")),
		SearchHighlight: lipgloss.NewStyle().Background(lipgloss.Color("#FBBF24")).Foreground(lipgloss.Color("#1F2937")),
	}
}

// LightTheme returns a theme suited for light terminal backgrounds.
func LightTheme() *Theme {
	accent := lipgloss.AdaptiveColor{Dark: "#7C3AED", Light: "#7C3AED"}
	dim := lipgloss.AdaptiveColor{Dark: "#9CA3AF", Light: "#6B7280"}
	yellow := lipgloss.AdaptiveColor{Dark: "#D97706", Light: "#D97706"}
	green := lipgloss.AdaptiveColor{Dark: "#059669", Light: "#059669"}
	red := lipgloss.AdaptiveColor{Dark: "#DC2626", Light: "#DC2626"}
	warn := lipgloss.AdaptiveColor{Dark: "#D97706", Light: "#D97706"}

	return &Theme{
		StatusBar:   lipgloss.NewStyle().Background(lipgloss.Color("#F3F4F6")).Foreground(lipgloss.Color("#374151")).Padding(0, 1),
		StatusPhase: lipgloss.NewStyle().Foreground(accent).Bold(true),
		StatusBold:  lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#374151")),
		StatusDim:   lipgloss.NewStyle().Foreground(dim),
		StatusWarn:  lipgloss.NewStyle().Foreground(warn).Bold(true),

		UserMessageBg: lipgloss.NewStyle().Background(lipgloss.Color("#F0F0F0")).PaddingRight(1),

		AssistantDot:    lipgloss.NewStyle().Bold(true),
		AssistantDotDim: lipgloss.NewStyle().Faint(true),
		AssistantDotOK:  lipgloss.NewStyle().Foreground(green),
		AssistantDotErr: lipgloss.NewStyle().Foreground(red),

		SystemAttrib: lipgloss.NewStyle().Foreground(dim).Italic(true),

		ThinkingText:  lipgloss.NewStyle().Foreground(dim),
		ThinkingLabel: lipgloss.NewStyle().Foreground(dim).Italic(true),
		ThinkingHint:  lipgloss.NewStyle().Foreground(dim),

		ToolTitle: lipgloss.NewStyle().Foreground(yellow).Bold(true),
		ToolBody:  lipgloss.NewStyle().Foreground(dim),
		DiffAdd:   lipgloss.NewStyle().Foreground(green),
		DiffDel:   lipgloss.NewStyle().Foreground(red),

		ApprovalWarn:   lipgloss.NewStyle().Foreground(warn).Bold(true),
		ApprovalChoice: lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Dark: "#374151", Light: "#374151"}),

		ToastInfo:    lipgloss.NewStyle().Foreground(accent).Background(lipgloss.Color("#F3F4F6")).Padding(0, 1),
		ToastWarn:    lipgloss.NewStyle().Foreground(warn).Background(lipgloss.Color("#F3F4F6")).Padding(0, 1),
		ToastSelfmod: lipgloss.NewStyle().Foreground(red).Bold(true).Background(lipgloss.Color("#F3F4F6")).Padding(0, 1),

		GaugeFilled: lipgloss.NewStyle().Foreground(accent),
		GaugeEmpty:  lipgloss.NewStyle().Foreground(dim),

		SuggestionNormal: lipgloss.NewStyle().Foreground(lipgloss.Color("#374151")),
		SuggestionActive: lipgloss.NewStyle().Foreground(accent).Bold(true),

		InputPrompt: lipgloss.NewStyle().Foreground(accent),

		OverlayBorder: lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#9CA3AF")).
			Padding(1, 2),
		OverlayTitle:    lipgloss.NewStyle().Foreground(accent).Bold(true),
		OverlayText:     lipgloss.NewStyle().Foreground(lipgloss.Color("#374151")),
		OverlayDim:      lipgloss.NewStyle().Foreground(dim),
		OverlayActive:   lipgloss.NewStyle().Foreground(accent).Bold(true),
		OverlayInput:    lipgloss.NewStyle().Foreground(lipgloss.Color("#374151")).Background(lipgloss.Color("#F3F4F6")),
		SearchHighlight: lipgloss.NewStyle().Background(lipgloss.Color("#FBBF24")).Foreground(lipgloss.Color("#1F2937")),
	}
}

// ForName returns a theme by name. Accepts "dark" or "light"; defaults to dark.
func ForName(name string) *Theme {
	if name == "light" {
		return LightTheme()
	}
	return DarkTheme()
}
