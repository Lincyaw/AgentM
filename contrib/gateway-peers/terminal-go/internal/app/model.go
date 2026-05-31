package app

import (
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/lipgloss"

	"github.com/AoyangSpace/agentm-terminal/internal/blocks"
	"github.com/AoyangSpace/agentm-terminal/internal/components"
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

const (
	statusBarHeight   = 3 // 2 content lines + 1 border
	doubleCtrlCWindow = 1500 * time.Millisecond
)

// Model is the root bubbletea model for the TUI.
type Model struct {
	width, height int
	ready         bool

	status      components.StatusBar
	viewport    viewport.Model
	input       components.Input
	suggestions components.SuggestionList
	toasts      components.ToastStack

	theme *theme.Theme

	transcript       []blocks.Block
	transcriptDirty  bool
	blockLineOffsets []int // line offset for each block, computed during renderTranscript

	ctrlcTime time.Time
	inFlight  bool

	commands []string
}

// NewModel creates a new Model with mock data for visual testing.
func NewModel(th *theme.Theme) Model {
	m := Model{
		theme:           th,
		input:           components.NewInput(),
		transcriptDirty: true,
		commands:        []string{"/help", "/clear", "/status", "/new", "/end", "/compact"},
	}

	// Mock status
	m.status.Update(components.StatusModel{
		Phase:       theme.PhaseIdle,
		Model:       "doubao",
		TokensIn:    1234,
		TokensOut:   567,
		CtxUsed:     102400,
		CtxTotal:    131072,
		CostTurn:    0.12,
		CostSession: 1.47,
		SessionKey:  "terminal:default",
		SessionAge:  23 * time.Minute,
	})

	// Mock transcript with realistic content using P1 block types
	m.transcript = buildMockTranscript()

	// Startup toast
	m.toasts.Push("connected to gateway", "info", 5*time.Second)

	return m
}

func buildMockTranscript() []blocks.Block {
	thinking1 := blocks.NewThinkingBlock()
	thinking1.Text = "The user wants an overview of the codebase. Let me look at the directory structure and key files to understand the architecture..."

	tool1 := blocks.NewToolBlock("Read", map[string]any{
		"file_path": "src/agentm/core/abi/__init__.py",
	})
	tool1.Done = true
	tool1.OK = true
	tool1.Result = "# ABI protocols for the pluggable architecture\nfrom .protocols import ..."

	child1 := &blocks.SubagentBlock{
		Purpose: "analyze test coverage for core module",
		Done:    true,
	}

	assistant1 := &blocks.AssistantTurn{
		Thinking: thinking1,
		Text: "The codebase follows a layered architecture:\n\n" +
			"**Core** (`src/agentm/core/`) -- The runtime substrate. Contains the ABI protocols, " +
			"extension API, session management, and the agent loop. This layer is write-protected " +
			"and should not be modified by extensions.\n\n" +
			"**Extensions** (`src/agentm/extensions/builtin/`) -- Pluggable atoms that implement " +
			"specific behaviors. Each is a single file exporting MANIFEST and install(). They " +
			"communicate with the runtime only through ExtensionAPI services.\n\n" +
			"**Scenarios** (`contrib/scenarios/`) -- YAML-driven configurations that compose atoms " +
			"into complete agent personas. Selected via `--scenario <name>`.",
		Tools:    []*blocks.ToolBlock{tool1},
		Children: []*blocks.SubagentBlock{child1},
	}

	assistant2 := &blocks.AssistantTurn{
		Text: "Tests focus on **fail-stop positions** -- invariants whose violation would cause " +
			"silent corruption rather than visible errors. The key test categories are:\n\n" +
			"1. Constitution boundary enforcement\n" +
			"2. Atom hash determinism for evidence attribution\n" +
			"3. Active-set fingerprint pairing\n" +
			"4. Catalog freeze idempotence\n" +
			"5. Extension contract validation (section 11)\n\n" +
			"Run `uv run pytest --tb=short` for the full suite. Markers `ui` and `slow` are opt-in.",
	}

	return []blocks.Block{
		&blocks.UserTurn{Content: "explain the codebase structure"},
		assistant1,
		&blocks.UserTurn{Content: "what about tests?"},
		assistant2,
	}
}

// Init returns the initial commands.
func (m Model) Init() tea.Cmd {
	return tea.Batch(
		m.input.Focus(),
		tickCmd(),
	)
}

type tickMsg time.Time

func tickCmd() tea.Cmd {
	return tea.Tick(50*time.Millisecond, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

// Update handles all messages.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		if !m.ready {
			m.viewport = viewport.New(m.width, m.viewportHeight())
			m.viewport.SetContent(m.renderTranscript())
			m.transcriptDirty = false
			m.viewport.GotoBottom()
			m.ready = true
		} else {
			m.viewport.Width = m.width
			m.viewport.Height = m.viewportHeight()
			m.transcriptDirty = true
		}
		m.input.SetWidth(m.width)
		return m, nil

	case tea.KeyMsg:
		return m.handleKey(msg)

	case tea.MouseMsg:
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		return m, cmd

	case components.InputSubmitted:
		return m.handleSubmit(msg)

	case components.HistoryNav:
		return m.handleHistoryNav(msg)

	case components.InputComplete:
		return m.handleCompletion()

	case tickMsg:
		m.toasts.Tick()
		if m.transcriptDirty && m.ready {
			m.viewport.SetContent(m.renderTranscript())
			m.transcriptDirty = false
		}
		cmds = append(cmds, tickCmd())
		return m, tea.Batch(cmds...)
	}

	return m, nil
}

func (m Model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	key := msg.String()

	// Global keys that bypass input
	switch key {
	case keyCtrlD:
		return m, tea.Quit

	case keyCtrlC:
		// If input is non-empty, clear it instead of triggering quit
		if m.input.Text() != "" {
			m.input.SetText("")
			return m, nil
		}
		now := time.Now()
		if now.Sub(m.ctrlcTime) < doubleCtrlCWindow {
			return m, tea.Quit
		}
		m.ctrlcTime = now
		m.toasts.Push("press ctrl+c again to quit", "warn", 2*time.Second)
		return m, nil

	case keyCtrlL:
		m.transcript = nil
		m.transcriptDirty = true
		m.viewport.SetContent("")
		m.viewport.GotoTop()
		return m, nil

	case keyCtrlE:
		m.toggleNearestCollapsible()
		m.transcriptDirty = true
		return m, nil

	case keyEsc:
		if m.suggestions.Visible() {
			m.suggestions.Hide()
			return m, nil
		}
		return m, nil

	case keyPgUp:
		m.viewport.HalfViewUp()
		return m, nil

	case keyPgDown:
		m.viewport.HalfViewDown()
		return m, nil
	}

	// When suggestions are visible, intercept navigation keys
	if m.suggestions.Visible() {
		switch key {
		case keyUp:
			m.suggestions.Move(-1)
			return m, nil
		case keyDown:
			m.suggestions.Move(1)
			return m, nil
		case keyTab, keyEnter:
			sel := m.suggestions.Current()
			if sel != "" {
				m.input.SetText(sel + " ")
				m.suggestions.Hide()
			}
			return m, nil
		}
	}

	// Block navigation (only when input is empty and not focused on multiline)
	if m.input.Text() == "" {
		switch key {
		case keyBracketOpen:
			m.jumpTurn(-1)
			return m, nil
		case keyBracketClose:
			m.jumpTurn(1)
			return m, nil
		}
	}

	// Delegate to input
	newInput, cmd := m.input.Update(msg)
	m.input = newInput

	// Check if input starts with "/" for suggestion triggering
	text := m.input.Text()
	if strings.HasPrefix(text, "/") && !strings.Contains(text, " ") {
		m.updateSuggestions(text)
	} else if m.suggestions.Visible() {
		m.suggestions.Hide()
	}

	// Recalculate viewport height if input height changed
	if m.ready {
		m.viewport.Height = m.viewportHeight()
	}

	return m, cmd
}

func (m Model) handleSubmit(msg components.InputSubmitted) (tea.Model, tea.Cmd) {
	text := msg.Text
	m.input.PushHistory(text)
	m.suggestions.Hide()

	// Add user turn to transcript
	m.transcript = append(m.transcript, &blocks.UserTurn{Content: text})

	// For now, add a mock assistant response
	mockAssistant := &blocks.AssistantTurn{
		Text: "(mock response to: " + text + ")",
	}
	m.transcript = append(m.transcript, mockAssistant)

	m.transcriptDirty = true
	m.viewport.SetContent(m.renderTranscript())
	m.transcriptDirty = false
	m.viewport.GotoBottom()

	return m, nil
}

func (m Model) handleHistoryNav(msg components.HistoryNav) (tea.Model, tea.Cmd) {
	var text string
	if msg.Delta < 0 {
		text = m.input.HistoryPrev()
	} else {
		text = m.input.HistoryNext()
	}
	if text != "" {
		m.input.SetText(text)
	} else if msg.Delta > 0 {
		m.input.SetText("")
	}
	return m, nil
}

func (m Model) handleCompletion() (tea.Model, tea.Cmd) {
	if m.suggestions.Visible() {
		sel := m.suggestions.Current()
		if sel != "" {
			m.input.SetText(sel + " ")
			m.suggestions.Hide()
		}
	} else {
		// Trigger suggestions from current input
		text := m.input.Text()
		if strings.HasPrefix(text, "/") {
			m.updateSuggestions(text)
		}
	}
	return m, nil
}

func (m *Model) updateSuggestions(prefix string) {
	var matches []string
	for _, cmd := range m.commands {
		if strings.HasPrefix(cmd, prefix) {
			matches = append(matches, cmd)
		}
	}
	m.suggestions.Populate(matches)
}

func (m *Model) toggleNearestCollapsible() {
	// Toggle the last collapsible block in the transcript
	for i := len(m.transcript) - 1; i >= 0; i-- {
		b := m.transcript[i]
		// Check children of AssistantTurn first
		if at, ok := b.(*blocks.AssistantTurn); ok {
			// Check thinking
			if at.Thinking != nil {
				at.Thinking.SetCollapsed(!at.Thinking.Collapsed())
				return
			}
			// Check tools
			for j := len(at.Tools) - 1; j >= 0; j-- {
				at.Tools[j].SetCollapsed(!at.Tools[j].Collapsed())
				return
			}
			// Check approvals
			for j := len(at.Approvals) - 1; j >= 0; j-- {
				at.Approvals[j].SetCollapsed(!at.Approvals[j].Collapsed())
				return
			}
		}
	}
}

func (m *Model) jumpTurn(delta int) {
	if len(m.blockLineOffsets) == 0 {
		return
	}

	// Collect turn positions from blockLineOffsets
	var turnPositions []int
	for i, b := range m.transcript {
		if b.Kind() == "user" || b.Kind() == "assistant" {
			if i < len(m.blockLineOffsets) {
				turnPositions = append(turnPositions, m.blockLineOffsets[i])
			}
		}
	}

	if len(turnPositions) == 0 {
		return
	}

	current := m.viewport.YOffset
	if delta < 0 {
		for i := len(turnPositions) - 1; i >= 0; i-- {
			if turnPositions[i] < current {
				m.viewport.SetYOffset(turnPositions[i])
				return
			}
		}
		m.viewport.GotoTop()
	} else {
		for _, pos := range turnPositions {
			if pos > current {
				m.viewport.SetYOffset(pos)
				return
			}
		}
		m.viewport.GotoBottom()
	}
}

func (m Model) viewportHeight() int {
	sugHeight := 0
	if m.suggestions.Visible() {
		sugHeight = m.suggestions.Count()
		if sugHeight > 5 {
			sugHeight = 5
		}
	}
	inputHeight := m.input.Height()
	h := m.height - statusBarHeight - inputHeight - sugHeight
	if h < 1 {
		h = 1
	}
	return h
}

// View renders the entire TUI.
func (m Model) View() string {
	if !m.ready {
		return "initializing..."
	}

	// 1. Status bar
	statusView := m.status.View(m.width, m.theme)

	// 2. Viewport with transcript (only re-render when dirty)
	if m.transcriptDirty {
		m.viewport.SetContent(m.renderTranscript())
		// cannot clear m.transcriptDirty here because View() is on a value receiver;
		// the tick handler in Update() clears it.
	}
	vpView := m.viewport.View()

	// 3. Toast overlay on viewport
	toastView := m.toasts.View(m.width/3, m.theme)
	if toastView != "" {
		vpView = overlayBottomRight(vpView, toastView, m.width)
	}

	// 4. Suggestions
	sugView := m.suggestions.View(m.width, m.theme)

	// 5. Input
	inputView := m.input.View(m.width, m.theme)

	// 6. Compose vertically
	parts := []string{statusView, vpView}
	if sugView != "" {
		parts = append(parts, sugView)
	}
	parts = append(parts, inputView)

	return lipgloss.JoinVertical(lipgloss.Left, parts...)
}

func (m *Model) renderTranscript() string {
	if len(m.transcript) == 0 {
		m.blockLineOffsets = nil
		return ""
	}
	var sb strings.Builder
	m.blockLineOffsets = make([]int, len(m.transcript))
	linePos := 0
	for i, b := range m.transcript {
		m.blockLineOffsets[i] = linePos
		if i > 0 {
			sb.WriteString("\n")
			linePos++
		}
		rendered := b.Render(m.width, m.theme)
		sb.WriteString(rendered)
		linePos += strings.Count(rendered, "\n") + 1
	}
	return sb.String()
}

// overlayBottomRight composites the overlay string at the bottom-right of base.
func overlayBottomRight(base, overlay string, width int) string {
	baseLines := strings.Split(base, "\n")
	overlayLines := strings.Split(overlay, "\n")

	if len(baseLines) == 0 {
		return base
	}

	// Start placing the overlay from the bottom of base
	startLine := len(baseLines) - len(overlayLines)
	if startLine < 0 {
		startLine = 0
	}

	for i, ol := range overlayLines {
		targetIdx := startLine + i
		if targetIdx >= len(baseLines) {
			break
		}

		baseLine := baseLines[targetIdx]
		olWidth := lipgloss.Width(ol)
		baseWidth := lipgloss.Width(baseLine)

		// Place overlay at the right edge
		padNeeded := width - olWidth
		if padNeeded < 0 {
			padNeeded = 0
		}

		if baseWidth <= padNeeded {
			baseLines[targetIdx] = baseLine + strings.Repeat(" ", padNeeded-baseWidth) + ol
		} else {
			// Truncate base line by rune width and append overlay
			runes := []rune(baseLine)
			truncated := ""
			w := 0
			for _, r := range runes {
				rw := lipgloss.Width(string(r))
				if w+rw > padNeeded {
					break
				}
				truncated += string(r)
				w += rw
			}
			baseLines[targetIdx] = truncated + strings.Repeat(" ", padNeeded-w) + ol
		}
	}

	return strings.Join(baseLines, "\n")
}
