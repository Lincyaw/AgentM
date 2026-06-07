package app

import (
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/AoyangSpace/agentm-terminal/internal/blocks"
	"github.com/AoyangSpace/agentm-terminal/internal/components"
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
	"github.com/AoyangSpace/agentm-terminal/internal/wire"
)

const (
	statusBarHeight   = 1 // single status line
	doubleCtrlCWindow = 1500 * time.Millisecond
)

// wireMsg wraps a decoded wire outbound event body for the bubbletea loop.
type wireMsg struct {
	body map[string]any
}

// wireDisconnected signals that the wire connection was lost or closed.
type wireDisconnected struct {
	err error
}

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
	tools    []string // registered tool names from gateway

	overlay   Overlay
	bookmarks []Bookmark

	// searchQuery is set while the search overlay is active so
	// renderTranscript can highlight matches.
	searchQuery string

	// transcriptMode: when true, all ThinkingBlocks and ToolBlocks
	// render expanded; when false (default), they render collapsed.
	transcriptMode bool

	// focused is the collapsible block currently selected by the cursor.
	// May be nil if no focus has been set yet.
	focused blocks.Focusable

	// Wire integration
	wireClient      *wire.WireClient
	sessionKey      string
	scenario        string
	chatID          string
	senderID        string
	firstSent       bool // scenario sent only on first message
	router          *Router
	activeTurn      *blocks.AssistantTurn
	toolRegistry    map[string]*blocks.ToolBlock
	childRegistry   map[string]*blocks.SubagentBlock
	pendingApproval *blocks.ApprovalBlock
	turnStartTime   time.Time
	glamourStyle    string // "dark" or "light", passed to new AssistantTurns
}

// Config holds initialization parameters for a Model.
type Config struct {
	WireClient *wire.WireClient // nil = mock mode (for testing UI)
	SenderID   string
	ChatID     string
	Scenario   string
	Theme      string // "dark" or "light"
}

// NewModel creates a new Model. When cfg.WireClient is nil the model runs
// in mock mode with sample data; otherwise it connects to the gateway.
func NewModel(cfg Config) Model {
	th := theme.ForName(cfg.Theme)
	chatID := cfg.ChatID
	if chatID == "" {
		chatID = "terminal"
	}
	senderID := cfg.SenderID
	if senderID == "" {
		senderID = "local"
	}

	m := Model{
		theme:           th,
		input:           components.NewInput(),
		wireClient:      cfg.WireClient,
		senderID:        senderID,
		chatID:          chatID,
		sessionKey:      "terminal:" + chatID,
		scenario:        cfg.Scenario,
		router:          &Router{},
		toolRegistry:    make(map[string]*blocks.ToolBlock),
		childRegistry:   make(map[string]*blocks.SubagentBlock),
		commands:        []string{"/help", "/clear", "/status", "/new", "/end", "/compact"},
		glamourStyle:    cfg.Theme,
		transcriptDirty: true,
	}
	if m.glamourStyle == "" {
		m.glamourStyle = "dark"
	}

	if cfg.WireClient == nil {
		// Mock mode: populate with test data
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
		m.transcript = buildMockTranscript(cfg.Theme)
		m.toasts.Push("mock mode -- no gateway connection", "info", 5*time.Second)
	} else {
		m.status.Update(components.StatusModel{
			Phase:      theme.PhaseIdle,
			SessionKey: m.sessionKey,
		})
		m.toasts.Push("connected to gateway", "info", 5*time.Second)
	}

	return m
}

func buildMockTranscript(themeName string) []blocks.Block {
	glamourStyle := "dark"
	if themeName == "light" {
		glamourStyle = "light"
	}
	thinking1 := blocks.NewThinkingBlock()
	thinking1.Text = "The user wants an overview of the codebase. Let me look at the directory structure and key files to understand the architecture..."

	tool1 := blocks.NewToolBlock("Read", map[string]any{
		"file_path": "src/agentm/core/abi/__init__.py",
	})
	tool1.Done = true
	tool1.OK = true
	tool1.Result = "# ABI protocols for the pluggable architecture\nfrom .protocols import ..."

	text1 := &blocks.TextBlock{
		GlamourStyle: glamourStyle,
		Text: "The codebase follows a layered architecture:\n\n" +
			"**Core** (`src/agentm/core/`) -- The runtime substrate. Contains the ABI protocols, " +
			"extension API, session management, and the agent loop. This layer is write-protected " +
			"and should not be modified by extensions.\n\n" +
			"**Extensions** (`src/agentm/extensions/builtin/`) -- Pluggable atoms that implement " +
			"specific behaviors. Each is a single file exporting MANIFEST and install(). They " +
			"communicate with the runtime only through ExtensionAPI services.\n\n" +
			"**Scenarios** (`contrib/scenarios/`) -- YAML-driven configurations that compose atoms " +
			"into complete agent personas. Selected via `--scenario <name>`.",
	}

	child1 := &blocks.SubagentBlock{
		Purpose: "analyze test coverage for core module",
		Done:    true,
	}

	assistant1 := &blocks.AssistantTurn{
		GlamourStyle: glamourStyle,
		Segments:     []blocks.Block{thinking1, tool1, text1},
		Children:     []*blocks.SubagentBlock{child1},
	}
	assistant1.SetComplete()

	text2 := &blocks.TextBlock{
		GlamourStyle: glamourStyle,
		Text: "Tests focus on **fail-stop positions** -- invariants whose violation would cause " +
			"silent corruption rather than visible errors. The key test categories are:\n\n" +
			"1. Constitution boundary enforcement\n" +
			"2. Atom hash determinism for evidence attribution\n" +
			"3. Active-set fingerprint pairing\n" +
			"4. Catalog freeze idempotence\n" +
			"5. Extension contract validation (section 11)\n\n" +
			"Run `uv run pytest --tb=short` for the full suite. Markers `ui` and `slow` are opt-in.",
	}

	assistant2 := &blocks.AssistantTurn{
		GlamourStyle: glamourStyle,
		Segments:     []blocks.Block{text2},
	}
	assistant2.SetComplete()

	return []blocks.Block{
		&blocks.UserTurn{Content: "explain the codebase structure"},
		assistant1,
		&blocks.UserTurn{Content: "what about tests?"},
		assistant2,
	}
}

// listenWire blocks on the wire client's outbound channel and returns
// one event as a wireMsg. When the channel closes, returns wireDisconnected.
func (m Model) listenWire() tea.Msg {
	if m.wireClient == nil {
		// Should not be called in mock mode. Block forever.
		select {}
	}
	select {
	case env, ok := <-m.wireClient.Outbound():
		if !ok {
			return wireDisconnected{err: nil}
		}
		body := env.Body
		if env.Kind == wire.KindError {
			msg := ""
			if body != nil {
				if m, ok := body["message"].(string); ok {
					msg = m
				}
			}
			body = map[string]any{
				"content":  fmt.Sprintf("gateway error: %s", msg),
				"metadata": map[string]any{"kind": "diagnostic_error"},
			}
		}
		return wireMsg{body: body}
	case <-m.wireClient.Done():
		err := m.wireClient.Err()
		if err == nil {
			err = fmt.Errorf("connection closed")
		}
		return wireDisconnected{err: fmt.Errorf("connection lost: %w", err)}
	}
}

// Init returns the initial commands.
func (m Model) Init() tea.Cmd {
	cmds := []tea.Cmd{
		m.input.Focus(),
		tickCmd(),
	}
	if m.wireClient != nil {
		cmds = append(cmds, m.listenWire)
	}
	return tea.Batch(cmds...)
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

	case wireMsg:
		m.router.Dispatch(&m, msg.body)
		m.transcriptDirty = true
		return m, m.listenWire

	case wireDisconnected:
		reason := "disconnected from gateway"
		if msg.err != nil {
			reason = fmt.Sprintf("disconnected: %v", msg.err)
		}
		log.Printf("[app] %s", reason)
		m.toasts.Push(reason, "warn", 10*time.Second)
		m.inFlight = false
		return m, nil

	case tickMsg:
		m.toasts.Tick()
		if m.inFlight {
			elapsed := time.Since(m.turnStartTime)
			sm := m.status.GetModel()
			sm.Elapsed = elapsed
			if elapsed.Seconds() > 0 && sm.TokensOut > 0 {
				sm.TokPerSec = float64(sm.TokensOut) / elapsed.Seconds()
			}
			m.status.Update(sm)
		}
		if m.transcriptDirty && m.ready {
			m.viewport.SetContent(m.renderTranscript())
			m.transcriptDirty = false
			if m.inFlight {
				m.viewport.GotoBottom()
			}
		}
		cmds = append(cmds, tickCmd())
		return m, tea.Batch(cmds...)
	}

	return m, nil
}

func (m Model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// If an overlay is active, delegate to it first.
	if m.overlay != nil {
		return m.handleOverlayKey(msg)
	}

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
		m.clearFocus()
		m.transcriptDirty = true
		m.viewport.SetContent("")
		m.viewport.GotoTop()
		return m, nil

	case keyCtrlE:
		m.toggleFocused()
		m.transcriptDirty = true
		return m, nil

	case keyCtrlO:
		m.transcriptMode = !m.transcriptMode
		for _, b := range m.transcript {
			if at, ok := b.(*blocks.AssistantTurn); ok {
				for _, seg := range at.Segments {
					if tb, ok := seg.(*blocks.ThinkingBlock); ok {
						tb.SetCollapsed(!m.transcriptMode)
					}
					if tb, ok := seg.(*blocks.ToolBlock); ok {
						tb.SetCollapsed(!m.transcriptMode)
					}
				}
			}
		}
		m.transcriptDirty = true
		return m, nil

	case keyEsc:
		if m.suggestions.Visible() {
			m.suggestions.Hide()
			return m, nil
		}
		if m.inFlight {
			m.sendInterrupt()
			m.toasts.Push("interrupt sent", "warn", 2*time.Second)
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

	// Overlay activations
	switch key {
	case keyCtrlF:
		m.overlay = NewSearchOverlay(m.transcript)
		return m, nil

	case keyCtrlB:
		m.addBookmark()
		return m, nil

	case keyCtrlG:
		m.overlay = NewBookmarkOverlay(m.bookmarks)
		return m, nil

	case keyCtrlR:
		m.overlay = NewResendOverlay(m.input.History())
		return m, nil

	case keyCtrlS:
		o := NewCodeSaveOverlay(m.transcript)
		if o != nil {
			m.overlay = o
		} else {
			m.toasts.Push("no code blocks found", "warn", 2*time.Second)
		}
		return m, nil
	}

	// Approval digit keys: when a pending approval exists, 1-9 selects a button
	if m.pendingApproval != nil && m.input.Text() == "" {
		if key >= "1" && key <= "9" {
			idx := int(key[0]-'0') - 1
			if idx < len(m.pendingApproval.Buttons) {
				m.sendApprovalResponse(m.pendingApproval.Buttons[idx].Value)
				m.pendingApproval = nil
			}
			return m, nil
		}
		if key == "?" {
			// Toggle approval detail view
			m.pendingApproval.SetCollapsed(!m.pendingApproval.Collapsed())
			m.transcriptDirty = true
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
		case keyQuestion:
			m.overlay = NewHelpOverlay()
			return m, nil
		case keyV:
			if o := m.viewOverlayForFocused(); o != nil {
				m.overlay = o
			}
			return m, nil
		case keyAltUp:
			m.moveFocus(-1)
			m.transcriptDirty = true
			return m, nil
		case keyAltDown:
			m.moveFocus(1)
			m.transcriptDirty = true
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

func (m Model) handleOverlayKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// Esc always closes any overlay
	if msg.String() == keyEsc {
		m.closeOverlay()
		return m, nil
	}

	updated, cmd, closed := m.overlay.Update(msg)
	m.overlay = updated

	if closed {
		// Handle post-close actions depending on overlay type
		switch o := m.overlay.(type) {
		case *SearchOverlay:
			m.searchQuery = ""
			m.transcriptDirty = true

		case *BookmarkOverlay:
			m.bookmarks = o.Bookmarks()
			if target := o.JumpTarget(); target >= 0 && target < len(m.blockLineOffsets) {
				m.viewport.SetYOffset(m.blockLineOffsets[target])
			}

		case *ResendOverlay:
			if o.WantsEdit() {
				m.input.SetText(o.Chosen())
			}
			// If resend was chosen, the cmd already contains InputSubmitted

		case *CodeSaveOverlay:
			if o.Saved() {
				m.toasts.Push("saved to "+o.SavedPath(), "info", 3*time.Second)
			} else if o.Error() != nil {
				m.toasts.Push("save error: "+o.Error().Error(), "warn", 3*time.Second)
			}
		}
		m.overlay = nil
		return m, cmd
	}

	// While search overlay is active, update the highlight query
	if so, ok := m.overlay.(*SearchOverlay); ok {
		m.searchQuery = so.Query()
		m.transcriptDirty = true
		// Scroll to current match
		if target := so.CurrentMatchBlock(); target >= 0 && target < len(m.blockLineOffsets) {
			m.viewport.SetYOffset(m.blockLineOffsets[target])
		}
	}

	return m, cmd
}

func (m *Model) closeOverlay() {
	if so, ok := m.overlay.(*SearchOverlay); ok {
		_ = so
		m.searchQuery = ""
		m.transcriptDirty = true
	}
	m.overlay = nil
}

func (m *Model) addBookmark() {
	if len(m.transcript) == 0 {
		return
	}
	// Bookmark the block nearest to current viewport position
	blockIdx := 0
	for i, off := range m.blockLineOffsets {
		if off <= m.viewport.YOffset {
			blockIdx = i
		}
	}
	// Build label from block content (first 40 chars)
	label := ""
	if blockIdx < len(m.transcript) {
		label = blockLabel(m.transcript[blockIdx])
	}
	m.bookmarks = append(m.bookmarks, Bookmark{
		BlockIndex: blockIdx,
		Label:      label,
	})
	m.toasts.Push("bookmark added", "info", 2*time.Second)
}

func blockLabel(b blocks.Block) string {
	var text string
	switch v := b.(type) {
	case *blocks.UserTurn:
		text = v.Content
	case *blocks.AssistantTurn:
		// Use the first TextBlock's content as label. A tool-only turn (no
		// TextBlock segment) yields an empty label, matching prior behavior.
		for _, seg := range v.Segments {
			if tb, ok := seg.(*blocks.TextBlock); ok {
				text = tb.Text
				break
			}
		}
	case *blocks.SystemTurn:
		text = v.Content
	default:
		text = b.Kind()
	}
	text = strings.ReplaceAll(text, "\n", " ")
	if len(text) > 40 {
		text = text[:37] + "..."
	}
	return text
}

func (m Model) handleSubmit(msg components.InputSubmitted) (tea.Model, tea.Cmd) {
	text := msg.Text
	m.input.PushHistory(text)
	m.suggestions.Hide()

	// Slash commands
	if strings.HasPrefix(text, "/") && !strings.HasPrefix(text, "//") {
		if text == "/clear" {
			m.transcript = nil
			m.activeTurn = nil
			m.clearFocus()
			m.transcriptDirty = true
			if m.ready {
				m.viewport.SetContent("")
				m.viewport.GotoTop()
			}
			m.sendToGateway("/new")
			return m, nil
		}
		if text == "/dump" || strings.HasPrefix(text, "/dump ") {
			path := strings.TrimSpace(strings.TrimPrefix(text, "/dump"))
			written, err := m.dumpScreen(path)
			if err != nil {
				m.toasts.Push("dump failed: "+err.Error(), "error", 4*time.Second)
			} else {
				m.toasts.Push("screen dumped to "+written, "info", 3*time.Second)
			}
			return m, nil
		}
		// All other slash commands are forwarded to the gateway
	}

	// Add user turn to transcript
	m.transcript = append(m.transcript, &blocks.UserTurn{Content: text})

	if m.wireClient != nil {
		m.inFlight = true
		m.turnStartTime = time.Now()
		// Reset per-turn status fields
		sm := m.status.GetModel()
		sm.TokensOut = 0
		sm.TokPerSec = 0
		sm.CostTurn = 0
		sm.Elapsed = 0
		sm.ToolCount = 0
		m.status.Update(sm)
		m.sendToGateway(text)
	} else {
		// Mock mode: add a mock assistant response
		mockText := &blocks.TextBlock{
			GlamourStyle: m.glamourStyle,
			Text:         "(mock response to: " + text + ")",
		}
		mockAssistant := &blocks.AssistantTurn{
			GlamourStyle: m.glamourStyle,
			Segments:     []blocks.Block{mockText},
		}
		mockAssistant.SetComplete()
		m.transcript = append(m.transcript, mockAssistant)
	}

	m.transcriptDirty = true
	if m.ready {
		m.viewport.SetContent(m.renderTranscript())
		m.transcriptDirty = false
		m.viewport.GotoBottom()
	}

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

// focusableBlocks returns all focusable blocks in render order by walking
// the transcript. For each AssistantTurn it yields *ThinkingBlock and
// *ToolBlock segments (in order) then its Approvals.
func (m *Model) focusableBlocks() []blocks.Focusable {
	var result []blocks.Focusable
	for _, b := range m.transcript {
		at, ok := b.(*blocks.AssistantTurn)
		if !ok {
			continue
		}
		for _, seg := range at.Segments {
			if f, ok := seg.(blocks.Focusable); ok {
				result = append(result, f)
			}
		}
		for _, appr := range at.Approvals {
			result = append(result, appr)
		}
	}
	return result
}

// clearFocus drops the current block focus, releasing the focused flag on the
// underlying block. Used by transcript-clearing paths so a later 'v' doesn't
// open an overlay for a block that no longer exists.
func (m *Model) clearFocus() {
	if m.focused != nil {
		m.focused.SetFocused(false)
		m.focused = nil
	}
}

// toggleFocused toggles the collapse state of the focused block. If no block
// is focused, it sets focus to the last focusable block and expands it.
func (m *Model) toggleFocused() {
	if m.focused == nil {
		// Fall back to the last focusable block.
		all := m.focusableBlocks()
		if len(all) == 0 {
			return
		}
		f := all[len(all)-1]
		f.SetFocused(true)
		m.focused = f
		f.SetCollapsed(false)
		m.scrollToFocused()
		return
	}
	m.focused.SetCollapsed(!m.focused.Collapsed())
}

// moveFocus moves keyboard focus by delta steps (+1 = next, -1 = prev).
func (m *Model) moveFocus(delta int) {
	all := m.focusableBlocks()
	if len(all) == 0 {
		return
	}

	// Find current index.
	cur := -1
	if m.focused != nil {
		for i, f := range all {
			if f == m.focused {
				cur = i
				break
			}
		}
	}

	// Clear old focus.
	if cur >= 0 {
		all[cur].SetFocused(false)
	}

	// Compute next.
	var next int
	if cur < 0 {
		if delta < 0 {
			next = len(all) - 1
		} else {
			next = 0
		}
	} else {
		next = cur + delta
		if next < 0 {
			next = 0
		}
		if next >= len(all) {
			next = len(all) - 1
		}
	}

	all[next].SetFocused(true)
	m.focused = all[next]
	m.scrollToFocused()
}

// scrollToFocused scrolls the viewport so the AssistantTurn containing the
// focused block is visible. We scroll to AssistantTurn granularity since
// blockLineOffsets is indexed by transcript position.
func (m *Model) scrollToFocused() {
	if m.focused == nil || !m.ready {
		return
	}
	for i, b := range m.transcript {
		at, ok := b.(*blocks.AssistantTurn)
		if !ok {
			continue
		}
		for _, seg := range at.Segments {
			if seg == m.focused {
				if i < len(m.blockLineOffsets) {
					m.viewport.SetYOffset(m.blockLineOffsets[i])
				}
				return
			}
		}
		for _, appr := range at.Approvals {
			if appr == m.focused {
				if i < len(m.blockLineOffsets) {
					m.viewport.SetYOffset(m.blockLineOffsets[i])
				}
				return
			}
		}
	}
}

// viewOverlayForFocused returns a ViewOverlay for the focused block, or nil
// if no suitable focused block exists.
func (m *Model) viewOverlayForFocused() *ViewOverlay {
	if m.focused == nil {
		return nil
	}
	switch f := m.focused.(type) {
	case *blocks.ToolBlock:
		return NewViewOverlay(f.Name, f.FullContent())
	case *blocks.ThinkingBlock:
		return NewViewOverlay("Thinking", f.Text)
	}
	return nil
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

// dumpScreen writes the current rendered frame to a file so the gateway
// agent (via its tui_snapshot tool) can see exactly what the user sees.
// An empty path resolves to $AGENTM_TUI_DUMP, then the /tmp default —
// matching the tui_snapshot atom's resolution order.
func (m Model) dumpScreen(path string) (string, error) {
	if path == "" {
		path = os.Getenv("AGENTM_TUI_DUMP")
	}
	if path == "" {
		path = "/tmp/agentm-tui-dump.txt"
	}
	if err := os.WriteFile(path, []byte(m.View()), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func (m *Model) sendToGateway(content string) {
	if m.wireClient == nil {
		return
	}
	scenario := ""
	if !m.firstSent {
		scenario = m.scenario
		m.firstSent = true
	}
	body := map[string]any{
		"channel":   "terminal",
		"sender_id": m.senderID,
		"chat_id":   m.chatID,
		"content":   content,
	}
	// Best-effort send; if it fails the read loop will detect the disconnect
	_ = m.wireClient.SendInbound(body, m.sessionKey, scenario)
}

func (m *Model) sendInterrupt() {
	if m.wireClient == nil {
		return
	}
	body := map[string]any{
		"channel":   "terminal",
		"sender_id": m.senderID,
		"chat_id":   m.chatID,
		"control":   "interrupt",
	}
	_ = m.wireClient.SendInbound(body, m.sessionKey, "")
}

func (m *Model) sendApprovalResponse(value string) {
	if m.wireClient == nil {
		return
	}
	body := map[string]any{
		"channel":      "terminal",
		"sender_id":    m.senderID,
		"chat_id":      m.chatID,
		"button_value": value,
	}
	_ = m.wireClient.SendInbound(body, m.sessionKey, "")
}

func (m *Model) addToolToCatalog(name string) {
	if name == "" {
		return
	}
	for _, t := range m.tools {
		if t == name {
			return
		}
	}
	m.tools = append(m.tools, name)
}

func (m *Model) addCommand(name string) {
	if name == "" {
		return
	}
	if !strings.HasPrefix(name, "/") {
		name = "/" + name
	}
	for _, c := range m.commands {
		if c == name {
			return
		}
	}
	m.commands = append(m.commands, name)
}

func (m Model) viewportHeight() int {
	sugHeight := 0
	if m.suggestions.Visible() {
		sugHeight = m.suggestions.Count()
		if sugHeight > 5 {
			sugHeight = 5
		}
	}
	inlineOverlayHeight := 0
	if m.overlay != nil && m.overlay.Kind() == OverlaySearch {
		inlineOverlayHeight = 1
	}
	inputHeight := m.input.Height()
	h := m.height - statusBarHeight - inputHeight - sugHeight - inlineOverlayHeight
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

	vpView := m.viewport.View()

	// 2. Full overlays replace the viewport content
	if m.overlay != nil {
		switch m.overlay.Kind() {
		case OverlayHelp, OverlayBookmarks, OverlayResend, OverlayCodeSave, OverlayView:
			vpView = m.overlay.View(m.width, m.viewportHeight(), m.theme)
		}
	}

	// 3. Toast overlay on viewport (only when no full overlay)
	if m.overlay == nil || m.overlay.Kind() == OverlaySearch {
		toastView := m.toasts.View(m.width/3, m.theme)
		if toastView != "" {
			vpView = overlayBottomRight(vpView, toastView, m.width)
		}
	}

	// 4. Status line (1 line, between viewport and input)
	statusView := m.status.View(m.width, m.theme)

	// 5. Inline overlays (search bar between viewport and input)
	var inlineView string
	if m.overlay != nil && m.overlay.Kind() == OverlaySearch {
		inlineView = m.overlay.View(m.width, 1, m.theme)
	}

	// 6. Suggestions
	sugView := m.suggestions.View(m.width, m.theme)

	// 7. Input
	inputView := m.input.View(m.width, m.theme)

	// 8. Compose vertically: viewport, status, [inline overlay], [suggestions], input
	parts := []string{vpView, statusView}
	if inlineView != "" {
		parts = append(parts, inlineView)
	}
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
		if m.searchQuery != "" {
			rendered = highlightMatches(rendered, m.searchQuery, m.theme)
		}
		sb.WriteString(rendered)
		linePos += strings.Count(rendered, "\n") + 1
	}
	return sb.String()
}

// highlightMatches wraps case-insensitive occurrences of query in the
// theme's SearchHighlight style within rendered text.
func highlightMatches(text, query string, th *theme.Theme) string {
	if query == "" {
		return text
	}
	lower := strings.ToLower(text)
	q := strings.ToLower(query)

	var sb strings.Builder
	pos := 0
	for {
		idx := strings.Index(lower[pos:], q)
		if idx < 0 {
			sb.WriteString(text[pos:])
			break
		}
		sb.WriteString(text[pos : pos+idx])
		matchEnd := pos + idx + len(query)
		sb.WriteString(th.SearchHighlight.Render(text[pos+idx : matchEnd]))
		pos = matchEnd
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
