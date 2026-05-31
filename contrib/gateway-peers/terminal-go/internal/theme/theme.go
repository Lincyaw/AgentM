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

// Tool lifecycle glyphs.
const (
	ToolRunning = "⟳"
	ToolOK      = "✓"
	ToolError   = "✗"
)

// Attribution labels.
const (
	LabelUser      = "› you"
	LabelAssistant = "● assistant"
	LabelSystem    = "system → you"
)

// Thinking glyphs for collapsed/expanded state.
const (
	ThinkingCollapsed = "▸"
	ThinkingExpanded  = "▾"
)

// Spine is the left gutter character.
const Spine = "┃"

// Theme holds all lipgloss styles for rendering blocks and UI components.
type Theme struct {
	// StatusBar styles for the bottom status bar.
	StatusBar   lipgloss.Style
	StatusPhase lipgloss.Style
	StatusBold  lipgloss.Style
	StatusDim   lipgloss.Style
	StatusWarn  lipgloss.Style

	// Spine styles for the left gutter per turn type.
	SpineUser      lipgloss.Style
	SpineAssistant lipgloss.Style
	SpineTool      lipgloss.Style
	SpineSystem    lipgloss.Style

	// Attribution line styles.
	UserAttrib      lipgloss.Style
	AssistantAttrib lipgloss.Style
	SystemAttrib    lipgloss.Style

	// Thinking block styles.
	ThinkingText   lipgloss.Style
	ThinkingHeader lipgloss.Style

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

		SpineUser:      lipgloss.NewStyle().Foreground(dim),
		SpineAssistant: lipgloss.NewStyle().Foreground(accent),
		SpineTool:      lipgloss.NewStyle().Foreground(yellow),
		SpineSystem:    lipgloss.NewStyle().Foreground(dim),

		UserAttrib:      lipgloss.NewStyle().Foreground(dim),
		AssistantAttrib: lipgloss.NewStyle().Foreground(accent).Bold(true),
		SystemAttrib:    lipgloss.NewStyle().Foreground(dim).Italic(true),

		ThinkingText:   lipgloss.NewStyle().Foreground(dim),
		ThinkingHeader: lipgloss.NewStyle().Foreground(dim),

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

		SpineUser:      lipgloss.NewStyle().Foreground(dim),
		SpineAssistant: lipgloss.NewStyle().Foreground(accent),
		SpineTool:      lipgloss.NewStyle().Foreground(yellow),
		SpineSystem:    lipgloss.NewStyle().Foreground(dim),

		UserAttrib:      lipgloss.NewStyle().Foreground(dim),
		AssistantAttrib: lipgloss.NewStyle().Foreground(accent).Bold(true),
		SystemAttrib:    lipgloss.NewStyle().Foreground(dim).Italic(true),

		ThinkingText:   lipgloss.NewStyle().Foreground(dim),
		ThinkingHeader: lipgloss.NewStyle().Foreground(dim),

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
	}
}

// ForName returns a theme by name. Accepts "dark" or "light"; defaults to dark.
func ForName(name string) *Theme {
	if name == "light" {
		return LightTheme()
	}
	return DarkTheme()
}
