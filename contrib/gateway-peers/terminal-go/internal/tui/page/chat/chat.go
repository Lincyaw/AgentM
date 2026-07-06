package chat

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"time"

	"charm.land/bubbles/v2/help"
	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/chat"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/commands"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/notification"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/sidebar"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	msgtypes "github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/service"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

const (
	// minWindowWidth is the threshold below which sidebar switches to horizontal mode
	minWindowWidth = 120
	// dragThreshold is pixels of movement needed to distinguish click from drag
	dragThreshold = 3
	// toggleColumnWidth is the width of the sidebar toggle/resize handle column
	toggleColumnWidth = 1
	// appPaddingHorizontal is total horizontal padding from AppStyle (left + right)
	appPaddingHorizontal    = 2 * styles.AppPadding
	defaultWelcomeModelLine = "Opus 4.8 (1M context) with high effort · API Usage Billing"
)

// sidebarLayoutMode represents how the sidebar is displayed
type sidebarLayoutMode int

const (
	// sidebarVertical: wide window, sidebar on right side
	sidebarVertical sidebarLayoutMode = iota
	// sidebarCollapsed: wide window but user collapsed sidebar, shown at top with toggle
	sidebarCollapsed
	// sidebarCollapsedNarrow: narrow window, shown at top without toggle
	sidebarCollapsedNarrow
)

// sidebarLayout holds computed layout values for the current frame.
// Computing this once per update avoids repeating calculations across View, SetSize, and input handlers.
type sidebarLayout struct {
	mode          sidebarLayoutMode
	innerWidth    int // window width minus app padding
	chatWidth     int // width available for chat/messages
	sidebarWidth  int // actual sidebar width (varies by mode)
	sidebarStartX int // X coordinate where sidebar content starts (relative to innerWidth)
	handleX       int // X coordinate of resize handle column (only valid in vertical mode)
	chatHeight    int // height available for chat area
	sidebarHeight int // height of sidebar
}

// isOnHandle returns true if adjustedX (already adjusted for app padding) is on the resize handle.
func (l sidebarLayout) isOnHandle(adjustedX int) bool {
	return l.mode == sidebarVertical && adjustedX == l.handleX
}

// isInSidebar returns true if adjustedX is within the sidebar area.
func (l sidebarLayout) isInSidebar(adjustedX int) bool {
	if l.mode != sidebarVertical {
		return false
	}
	return adjustedX >= l.sidebarStartX
}

// showToggle returns true if a toggle glyph should be shown.
func (l sidebarLayout) showToggle() bool {
	return l.mode == sidebarVertical || l.mode == sidebarCollapsed
}

// SidebarSettings holds the sidebar display settings that should persist across session changes.
type SidebarSettings struct {
	Collapsed      bool
	PreferredWidth int
}

type queuedInput struct {
	content     string
	attachments []msgtypes.Attachment
}

// Page represents the main chat content area (messages + sidebar).
// The editor and resize handle are owned by the parent (tui.Model).
type Page interface {
	layout.Model
	layout.Sizeable
	layout.Help
	CompactSession(additionalPrompt string) tea.Cmd
	// SetSessionStarred updates the sidebar star indicator
	SetSessionStarred(starred bool)
	// SetTitleRegenerating sets the title regenerating state on the sidebar
	SetTitleRegenerating(regenerating bool) tea.Cmd
	// ScrollToBottom scrolls the messages viewport to the bottom if auto-scroll is active.
	ScrollToBottom() tea.Cmd
	// ForceScrollToBottom scrolls to the bottom even after local panel navigation
	// marked the transcript as manually scrolled.
	ForceScrollToBottom() tea.Cmd
	// AdjustBottomSlack reserves or releases blank transcript rows so newly
	// appended local command rows can stay visible above centered overlays.
	AdjustBottomSlack(delta int)
	// TranscriptHeight returns the current rendered transcript height.
	TranscriptHeight() int
	// TranscriptMessageHeight returns the message-list height without the
	// welcome header that may be visually prepended by chat page rendering.
	TranscriptMessageHeight() int
	// TranscriptViewportHeight returns the message viewport height.
	TranscriptViewportHeight() int
	// SetTranscriptTopContextLines prepends the trailing welcome/context rows
	// above the message list while a Claude-style overlay is open.
	SetTranscriptTopContextLines(lines int)
	// AddLocalUserMessage appends a local transcript user row without
	// starting an agent turn.
	AddLocalUserMessage(content string) tea.Cmd
	// AddLocalNoticeMessage appends a local transcript notice row without
	// starting an agent turn.
	AddLocalNoticeMessage(content string) tea.Cmd
	// AddLocalSystemMessage appends a local system panel without routing
	// through the runtime event bus.
	AddLocalSystemMessage(content string) tea.Cmd
	// RemoveLastSystemMessage removes the currently displayed local system
	// panel when a Claude-style control dialog is dismissed.
	RemoveLastSystemMessage() tea.Cmd
	// SetWelcomeModelLine updates the Claude-style model subtitle shown in the
	// compact welcome header.
	SetWelcomeModelLine(content string)
	// IsWorking returns whether the agent is currently working
	IsWorking() bool
	// IsTranscriptEmpty returns true before any visible conversation content exists.
	IsTranscriptEmpty() bool
	// NthLatestAssistantMessage returns the nth latest visible assistant text.
	NthLatestAssistantMessage(index int) string
	// AssistantMessageCount returns the number of visible assistant responses.
	AssistantMessageCount() int
	// ExportTranscript returns the plain-text visible conversation transcript.
	ExportTranscript() string
	// IsInlineEditing returns true if a past user message is being edited inline
	IsInlineEditing() bool
	// SetCommandParser replaces the slash-command parser for the chat page.
	SetCommandParser(*commands.Parser)
	// FocusMessages gives focus to the messages panel for keyboard scrolling
	FocusMessages() tea.Cmd
	// FocusMessageAt gives focus and selects the message at the given screen coordinates
	FocusMessageAt(x, y int) tea.Cmd
	// BlurMessages removes focus from the messages panel
	BlurMessages()
	// GetSidebarSettings returns the current sidebar display settings
	GetSidebarSettings() SidebarSettings
	// SetSidebarSettings applies sidebar display settings
	SetSidebarSettings(settings SidebarSettings)
	// SetCompletionOpen tells the chat page whether the inline completion surface is visible.
	SetCompletionOpen(open bool)
}

// chatPage implements Page
type chatPage struct {
	width, height int

	// Components
	sidebar  sidebar.Model
	messages messages.Model

	sessionState *service.SessionState

	// State
	working          bool
	hideSidebar      bool
	completionOpen   bool
	welcomeModelLine string

	msgCancel                  context.CancelFunc
	streamCancelled            bool
	streamDepth                int // nesting depth of active streams (incremented on StreamStarted, decremented on StreamStopped)
	streamStartTime            time.Time
	activeBangCommand          string
	activeUserPrompt           string
	activeUserPromptRestorable bool
	transcriptTopContextLines  int

	// Track whether we've received content from an assistant response
	// Used by --exit-after-response to ensure we don't exit before receiving content
	hasReceivedAssistantContent bool

	// queuedInputs mirrors Claude Code's running-input UX: messages submitted
	// while the agent is already working are rendered as queued rows above the
	// composer instead of disappearing into a weak notification.
	queuedInputs []queuedInput

	// Editing state for branching sessions
	editing          bool
	branchAtPosition int
	editAttachments  []msgtypes.Attachment // Preserved attachments from original message

	// Key map
	keyMap KeyMap

	app *app.App

	// Command parser for handling slash commands in the editor
	commandParser *commands.Parser

	// Sidebar drag state
	isDraggingSidebar     bool // True while dragging the sidebar resize handle
	sidebarDragStartX     int  // X position when drag started
	sidebarDragStartWidth int  // Sidebar preferred width when drag started
	sidebarDragMoved      bool // True if mouse moved beyond threshold during drag
}

// sidebarHidden reports whether the sidebar should be omitted entirely from
// layout and rendering.
func (p *chatPage) sidebarHidden() bool {
	return p.hideSidebar
}

// computeSidebarLayout calculates the layout based on current state.
func (p *chatPage) computeSidebarLayout() sidebarLayout {
	innerWidth := p.width - appPaddingHorizontal

	// No sidebar at all: chat fills the area.
	if p.sidebarHidden() {
		return sidebarLayout{
			mode:       sidebarCollapsedNarrow,
			innerWidth: max(1, p.width),
			chatWidth:  max(1, p.width),
			chatHeight: max(1, p.height),
		}
	}

	var mode sidebarLayoutMode
	switch {
	case p.width >= minWindowWidth && !p.sidebar.IsCollapsed():
		mode = sidebarVertical
	case p.width >= minWindowWidth:
		mode = sidebarCollapsed
	default:
		mode = sidebarCollapsedNarrow
	}

	l := sidebarLayout{
		mode:       mode,
		innerWidth: innerWidth,
	}

	switch mode {
	case sidebarVertical:
		l.sidebarWidth = p.sidebar.ClampWidth(p.sidebar.GetPreferredWidth(), innerWidth)
		l.chatWidth = max(1, innerWidth-l.sidebarWidth)
		l.handleX = l.chatWidth
		l.sidebarStartX = l.chatWidth + toggleColumnWidth
		l.chatHeight = max(1, p.height)
		l.sidebarHeight = l.chatHeight

	case sidebarCollapsed:
		l.sidebarWidth = innerWidth - toggleColumnWidth
		l.chatWidth = innerWidth
		l.sidebarHeight = p.sidebar.CollapsedHeight(l.sidebarWidth)
		l.chatHeight = max(1, p.height-l.sidebarHeight)

	case sidebarCollapsedNarrow:
		l.sidebarWidth = innerWidth
		l.chatWidth = innerWidth
		l.sidebarHeight = p.sidebar.CollapsedHeight(l.sidebarWidth)
		l.chatHeight = max(1, p.height-l.sidebarHeight)
	}

	return l
}

// KeyMap defines key bindings for the chat page
type KeyMap struct {
	Cancel          key.Binding
	ToggleSplitDiff key.Binding
	ToggleSidebar   key.Binding
}

// defaultKeyMap returns the default key bindings.
// ctrl+t is reserved by the top-level TUI for task rows when background
// activity exists, and for new tabs otherwise. ToggleSplitDiff is available via
// /split-diff instead.
func defaultKeyMap() KeyMap {
	splitDiff := key.NewBinding(
		key.WithKeys("ctrl+t"),
		key.WithHelp("Ctrl+t", "toggle split diff"),
	)
	splitDiff.SetEnabled(false)

	return KeyMap{
		Cancel: key.NewBinding(
			key.WithKeys("esc"),
			key.WithHelp("Esc", "interrupt"),
		),
		ToggleSplitDiff: splitDiff,
		ToggleSidebar: key.NewBinding(
			key.WithKeys("ctrl+b"),
			key.WithHelp("Ctrl+b", "toggle sidebar"),
		),
	}
}

// New creates a new chat page
func New(a *app.App, sessionState *service.SessionState, opts ...PageOption) Page {
	p := &chatPage{
		sidebar:       sidebar.New(sessionState),
		messages:      messages.New(sessionState),
		app:           a,
		keyMap:        defaultKeyMap(),
		commandParser: commands.NewParser(),
		sessionState:  sessionState,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// PageOption configures a chat page.
type PageOption func(*chatPage)

// WithHideSidebar hides the sidebar.
// The sidebar cannot be re-shown via the TUI.
func WithHideSidebar() PageOption {
	return func(p *chatPage) {
		p.hideSidebar = true
		p.keyMap.ToggleSidebar.SetEnabled(false)
	}
}

// WithCommandParser injects a command parser for handling slash commands in the editor.
func WithCommandParser(p *commands.Parser) PageOption {
	return func(cp *chatPage) {
		cp.commandParser = p
	}
}

// Init initializes the chat page
func (p *chatPage) Init() tea.Cmd {
	var cmds []tea.Cmd

	cmds = append(cmds,
		p.sidebar.Init(),
		p.messages.Init(),
	)

	// Load state from existing session (for session restore and branching)
	if sess := p.app.Session(); sess != nil {
		p.sidebar.LoadFromSession(sess)
		if len(sess.Messages) > 0 {
			cmds = append(cmds, p.messages.LoadFromSession(sess))
		}
	}

	return tea.Batch(cmds...)
}

// Update handles messages and updates the page state
func (p *chatPage) Update(msg tea.Msg) (layout.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		cmd := p.SetSize(msg.Width, msg.Height)
		return p, cmd

	case tea.KeyPressMsg:
		return p.handleKeyPress(msg)

	case tea.MouseClickMsg:
		return p.handleMouseClick(msg)

	case tea.MouseMotionMsg:
		return p.handleMouseMotion(msg)

	case tea.MouseReleaseMsg:
		return p.handleMouseRelease(msg)

	case msgtypes.WheelCoalescedMsg:
		return p.handleWheelCoalesced(msg)

	case msgtypes.StreamCancelledMsg:
		model, cmd := p.messages.Update(msg)
		p.messages = model.(messages.Model)

		// Forward to sidebar to stop its spinners
		sidebarModel, sidebarCmd := p.sidebar.Update(msg)
		p.sidebar = sidebarModel.(sidebar.Model)

		var cmds []tea.Cmd
		cmds = append(cmds, cmd, sidebarCmd)

		if msg.ShowMessage {
			cmds = append(cmds, p.messages.AddCancelledMessage())
		}
		cmds = append(cmds, p.messages.ScrollToBottom())

		return p, tea.Batch(cmds...)

	case msgtypes.EditUserMessageMsg:
		return p.handleEditUserMessage(msg)

	case messages.InlineEditCommittedMsg:
		return p.handleInlineEditCommitted(msg)

	case messages.InlineEditCancelledMsg:
		return p.handleInlineEditCancelled(msg)

	case msgtypes.SendMsg:
		slog.Debug(msg.Content)
		return p.handleSendMsg(msg)

	case msgtypes.PopQueuedInputMsg:
		return p, p.popQueuedInput()

	case msgtypes.CancelStreamPreserveInputMsg:
		return p, p.cancelStreamPreservingInput(true)

	case msgtypes.UnknownCommandMsg:
		return p.handleUnknownCommand(msg)

	case msgtypes.ToggleHideToolResultsMsg:
		// Forward to messages component to invalidate cache and trigger redraw
		model, cmd := p.messages.Update(messages.ToggleHideToolResultsMsg{})
		p.messages = model.(messages.Model)
		return p, cmd

	case msgtypes.SetTranscriptDetailMsg:
		model, cmd := p.messages.Update(messages.SetTranscriptDetailMsg{
			Detailed: msg.Detailed,
			Verbose:  msg.Verbose,
		})
		p.messages = model.(messages.Model)
		return p, cmd

	case msgtypes.ThemeChangedMsg:
		// Theme changed - forward to all child components to invalidate caches
		var cmds []tea.Cmd

		model, cmd := p.messages.Update(msg)
		p.messages = model.(messages.Model)
		cmds = append(cmds, cmd)

		// Forward to sidebar to ensure it picks up new theme colors
		sidebarModel, sidebarCmd := p.sidebar.Update(msg)
		p.sidebar = sidebarModel.(sidebar.Model)
		cmds = append(cmds, sidebarCmd)

		return p, tea.Batch(cmds...)

	default:
		// Try to handle as a runtime event
		if handled, cmd := p.handleRuntimeEvent(msg); handled {
			return p, cmd
		}
	}

	sidebarModel, sidebarCmd := p.sidebar.Update(msg)
	p.sidebar = sidebarModel.(sidebar.Model)

	chatModel, chatCmd := p.messages.Update(msg)
	p.messages = chatModel.(messages.Model)

	return p, tea.Batch(sidebarCmd, chatCmd)
}

func (p *chatPage) setWorking(working bool) tea.Cmd {
	wasWorking := p.working
	p.working = working

	if working != wasWorking {
		return core.CmdHandler(msgtypes.WorkingStateChangedMsg{
			Working:          working,
			QueuedInputCount: len(p.queuedInputs),
		})
	}

	return nil
}

func (p *chatPage) emitQueuedInputState() tea.Cmd {
	return core.CmdHandler(msgtypes.WorkingStateChangedMsg{
		Working:          p.working,
		QueuedInputCount: len(p.queuedInputs),
	})
}

// setPendingResponse adds or removes the pending-response spinner message
// inside the messages component. When starting, it adds a spinner message to
// the scrollable list; when stopping, it explicitly removes any lingering spinner.
func (p *chatPage) setPendingResponse(pending bool) tea.Cmd {
	if pending {
		return p.messages.AddAssistantMessage()
	}
	p.messages.RemoveSpinner()
	return nil
}

// renderCollapsedSidebar renders the sidebar in collapsed mode (at top of screen).
func (p *chatPage) renderCollapsedSidebar(sl sidebarLayout) string {
	// Guard against unset/invalid layout (can happen before WindowSizeMsg is received).
	width := max(0, sl.innerWidth)
	height := max(0, sl.sidebarHeight)
	if width == 0 || height == 0 {
		return ""
	}

	sidebarView := p.sidebar.View()
	sidebarLines := strings.Split(sidebarView, "\n")

	// Place toggle glyph at the far right of the first line
	if sl.showToggle() && sl.mode != sidebarVertical && len(sidebarLines) > 0 {
		toggleGlyph := styles.MutedStyle.Render("«")
		glyphW := lipgloss.Width(toggleGlyph)
		padded := lipgloss.NewStyle().Width(width - glyphW).Render(sidebarLines[0])
		sidebarLines[0] = padded + toggleGlyph
	}

	// Replace the last line with a subtle divider
	divider := styles.FadingStyle.Render(strings.Repeat("─", width))
	if len(sidebarLines) >= height {
		sidebarLines[height-1] = divider
	} else {
		sidebarLines = append(sidebarLines, divider)
	}

	sidebarWithDivider := strings.Join(sidebarLines, "\n")

	return lipgloss.NewStyle().
		Width(width).
		Height(height).
		Align(lipgloss.Left, lipgloss.Top).
		Render(sidebarWithDivider)
}

func (p *chatPage) queuedInputView(width int) string {
	if len(p.queuedInputs) == 0 || width <= 0 {
		return ""
	}
	rowStyle := lipgloss.NewStyle().Width(width).Foreground(styles.TextPrimary)
	prefix := styles.MutedStyle.Render("  ❯ ")
	available := max(1, width-lipgloss.Width(prefix))
	rows := make([]string, 0, len(p.queuedInputs))
	for _, input := range p.queuedInputs {
		text := strings.ReplaceAll(input.content, "\n", " ")
		rows = append(rows, rowStyle.Render(prefix+lipgloss.NewStyle().MaxWidth(available).Render(text)))
	}
	return strings.Join(rows, "\n")
}

func (p *chatPage) popQueuedInput() tea.Cmd {
	if len(p.queuedInputs) == 0 {
		return nil
	}
	parts := make([]string, 0, len(p.queuedInputs))
	for _, input := range p.queuedInputs {
		parts = append(parts, input.content)
	}
	p.queuedInputs = nil
	return tea.Batch(
		core.CmdHandler(msgtypes.RestoreEditorInputMsg{Content: strings.Join(parts, "\n")}),
		p.emitQueuedInputState(),
		p.messages.ScrollToBottom(),
	)
}

func (p *chatPage) joinMessagesWithQueuedInput(messagesView, queuedView string, height int) string {
	if queuedView == "" || height <= 0 {
		return messagesView
	}
	messageLines := strings.Split(messagesView, "\n")
	queuedLines := strings.Split(queuedView, "\n")
	keep := max(0, height-len(queuedLines))
	if len(messageLines) > keep {
		messageLines = messageLines[:keep]
	}
	for len(messageLines) < keep {
		messageLines = append(messageLines, "")
	}
	return strings.Join(append(messageLines, queuedLines...), "\n")
}

func (p *chatPage) emptyWelcomeView(width int) string {
	if width <= 0 {
		return ""
	}
	cwd := compactWorkingDirectory()
	lines := p.claudeWelcomeLines(cwd, width)
	for i, line := range lines {
		lines[i] = ansi.Truncate(line, width, "…")
	}
	return strings.Join(lines, "\n")
}

func (p *chatPage) claudeWelcomeLines(cwd string, width int) []string {
	if shouldRenderWelcomeCard(width) {
		return claudeWelcomeCardLines(cwd, width)
	}
	return compactClaudeWelcomeLines(cwd, p.currentWelcomeModelLine())
}

func (p *chatPage) currentWelcomeModelLine() string {
	if strings.TrimSpace(p.welcomeModelLine) != "" {
		return p.welcomeModelLine
	}
	return defaultWelcomeModelLine
}

func compactClaudeWelcomeLines(cwd, modelLine string) []string {
	logo := lipgloss.NewStyle().Foreground(lipgloss.Color("174"))
	logoFill := logo.Background(lipgloss.Color("16"))
	muted := lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	bold := lipgloss.NewStyle().Bold(true)
	return []string{
		logo.Render(" ▐") + logoFill.Render("▛███▜") + logo.Render("▌") + "   " + bold.Render("Claude") + " " + bold.Render("Code") + " " + muted.Render("v2.1.201"),
		logo.Render("▝▜") + logoFill.Render("█████") + logo.Render("▛▘") + "  " + muted.Render(modelLine),
		logo.Render(" ") + " " + logo.Render("▘▘") + " " + logo.Render("▝▝") + "    " + muted.Render(cwd),
		"",
		"",
	}
}

func shouldRenderWelcomeCard(width int) bool {
	if width < 80 {
		return false
	}
	info, err := os.Stat(".git")
	return err == nil && info.IsDir()
}

func claudeWelcomeCardLines(cwd string, width int) []string {
	leftWidth, rightWidth := welcomeCardColumnWidths(width)
	logo := lipgloss.NewStyle().Foreground(lipgloss.Color("174"))
	logoFill := logo.Background(lipgloss.Color("16"))
	muted := lipgloss.NewStyle().Foreground(lipgloss.Color("246"))
	bold := lipgloss.NewStyle().Bold(true)
	accentBold := lipgloss.NewStyle().Foreground(lipgloss.Color("174")).Bold(true)
	border := lipgloss.NewStyle().Foreground(lipgloss.Color("174"))
	middleBorder := border.Faint(true)
	mutedItalic := muted.Italic(true)

	leftRows := []string{
		"",
		bold.Render("Welcome back!"),
		"",
		logo.Render("▐") + logoFill.Render("▛███▜") + logo.Render("▌"),
		logo.Render("▝▜") + logoFill.Render("█████") + logo.Render("▛▘"),
		logo.Render("▘▘") + " " + logo.Render("▝▝"),
		"",
		muted.Render(defaultWelcomeModelLine),
		" " + muted.Render(compactWelcomeCardDirectory(cwd, leftWidth-2)),
	}
	rightRows := []string{
		"Tips for getting started",
		"Run /init to create a CLAUDE.md file with …",
		border.Render(strings.Repeat("─", max(1, rightWidth-2))),
		accentBold.Render("What's new"),
		"Claude Sonnet 5 sessions no longer use the…",
		"Changed `AskUserQuestion` dialogs to no lo…",
		`Changed the "default" permission mode to "…`,
		mutedItalic.Render("/release-notes for more"),
		"",
	}

	lines := []string{welcomeCardTopBorder(width, border)}
	for i := range leftRows {
		leftAlign := lipgloss.Center
		if i == len(leftRows)-1 {
			leftAlign = lipgloss.Left
		}
		left := welcomeCardCell(leftRows[i], leftWidth, leftAlign)
		if leftAlign == lipgloss.Center {
			left = welcomeCardCenterCell(leftRows[i], leftWidth, i != 7)
		}
		right := welcomeCardCell(" "+rightRows[i], rightWidth, lipgloss.Left)
		lines = append(lines, border.Render("│")+left+middleBorder.Render("│")+right+border.Render("│"))
	}
	lines = append(lines, border.Render("╰"+strings.Repeat("─", max(0, width-2))+"╯"))
	lines = append(lines, "", "")
	return lines
}

func welcomeCardColumnWidths(width int) (left, right int) {
	available := max(1, width-3)
	left = min(52, max(30, available*52/97))
	right = max(1, available-left)
	return left, right
}

func welcomeCardTopBorder(width int, style lipgloss.Style) string {
	prefix := "╭─── Claude Code v2.1.201 "
	suffix := "╮"
	fillWidth := max(0, width-lipgloss.Width(prefix)-lipgloss.Width(suffix))
	return style.Render(prefix + strings.Repeat("─", fillWidth) + suffix)
}

func welcomeCardCell(text string, width int, align lipgloss.Position) string {
	if width <= 0 {
		return ""
	}
	text = ansi.Truncate(text, width, "…")
	return lipgloss.PlaceHorizontal(width, align, text)
}

func welcomeCardCenterCell(text string, width int, leftBias bool) string {
	if width <= 0 {
		return ""
	}
	text = ansi.Truncate(text, width, "…")
	padding := max(0, width-lipgloss.Width(text))
	left := padding / 2
	if leftBias {
		left = (padding + 1) / 2
	}
	right := padding - left
	return strings.Repeat(" ", left) + text + strings.Repeat(" ", right)
}

func compactWelcomeCardDirectory(cwd string, width int) string {
	if width <= 0 {
		return ""
	}
	if lipgloss.Width(cwd) <= width {
		return cwd
	}
	if strings.HasPrefix(cwd, "~/") {
		rest := strings.TrimPrefix(cwd, "~/")
		if idx := strings.Index(rest, "/."); idx >= 0 {
			candidate := "~/…" + rest[idx:]
			if lipgloss.Width(candidate) <= width {
				return candidate
			}
		}
	}
	return leftTruncate(cwd, width)
}

func leftTruncate(text string, width int) string {
	if width <= 0 {
		return ""
	}
	if lipgloss.Width(text) <= width {
		return text
	}
	ellipsis := "…"
	keep := max(1, width-lipgloss.Width(ellipsis))
	runes := []rune(text)
	for len(runes) > 0 && lipgloss.Width(string(runes)) > keep {
		runes = runes[1:]
	}
	return ellipsis + string(runes)
}

func compactWorkingDirectory() string {
	cwd, err := os.Getwd()
	if err != nil || cwd == "" {
		return ""
	}
	home, err := os.UserHomeDir()
	if err == nil && home != "" {
		if cwd == home {
			return "~"
		}
		prefix := home + "/"
		if strings.HasPrefix(cwd, prefix) {
			return "~/" + strings.TrimPrefix(cwd, prefix)
		}
	}
	return cwd
}

// View renders the chat page (messages + sidebar only, no editor or resize handle)
func (p *chatPage) View() string {
	sl := p.computeSidebarLayout()

	messagesView := p.messages.View()
	queuedView := p.queuedInputView(sl.chatWidth)
	compactEmptyWelcome := false
	if strings.TrimSpace(messagesView) == "" && queuedView == "" && !p.working {
		messagesView = p.emptyWelcomeView(sl.chatWidth)
		compactEmptyWelcome = p.hideSidebar
		if compactEmptyWelcome && p.completionOpen && shouldRenderWelcomeCard(sl.chatWidth) {
			messagesView = dropLeadingRenderedLines(messagesView, 2)
		}
	} else if p.transcriptTopContextLines > 0 && p.hideSidebar && strings.TrimSpace(messagesView) != "" {
		messagesView = p.prependTranscriptTopContext(messagesView, sl.chatWidth, p.transcriptTopContextLines)
	} else if p.hideSidebar && strings.TrimSpace(messagesView) != "" {
		welcomeView := p.emptyWelcomeView(sl.chatWidth)
		if renderedLineCount(welcomeView)+renderedLineCount(messagesView) <= sl.chatHeight {
			messagesView = welcomeView + "\n" + messagesView
		}
	}
	if !compactEmptyWelcome &&
		p.hideSidebar &&
		strings.TrimSpace(messagesView) != "" &&
		!strings.HasSuffix(messagesView, "\n") &&
		renderedLineCount(messagesView) < sl.chatHeight {
		messagesView += "\n"
	}
	if queuedView != "" {
		messagesView = p.joinMessagesWithQueuedInput(messagesView, queuedView, sl.chatHeight)
	}

	var bodyContent string

	switch sl.mode {
	case sidebarVertical:
		chatView := styles.ChatStyle.
			Height(sl.chatHeight).
			Width(sl.chatWidth).
			Render(messagesView)

		toggleCol := p.renderSidebarHandle(sl.chatHeight)

		sidebarView := lipgloss.NewStyle().
			Width(sl.sidebarWidth-toggleColumnWidth).
			Height(sl.chatHeight).
			Align(lipgloss.Left, lipgloss.Top).
			Render(p.sidebar.View())

		bodyContent = lipgloss.JoinHorizontal(lipgloss.Left, chatView, toggleCol, sidebarView)

	case sidebarCollapsed, sidebarCollapsedNarrow:
		switch {
		case p.hideSidebar:
			// Sidebar hidden: chat follows transcript height, no sidebar header.
			bodyContent = styles.ChatStyle.Width(sl.innerWidth).Render(messagesView)
		default:
			sidebarRendered := p.renderCollapsedSidebar(sl)
			chatView := styles.ChatStyle.
				Height(sl.chatHeight).
				Width(sl.innerWidth).
				Render(messagesView)
			bodyContent = lipgloss.JoinVertical(lipgloss.Top, sidebarRendered, chatView)
		}
	}

	if p.hideSidebar {
		return bodyContent
	}
	appStyle := styles.AppStyle.Height(p.height)
	return appStyle.Render(bodyContent)
}

func renderedLineCount(view string) int {
	if view == "" {
		return 0
	}
	return len(strings.Split(view, "\n"))
}

func (p *chatPage) prependTranscriptTopContext(messagesView string, width, lines int) string {
	if lines <= 0 {
		return messagesView
	}
	welcomeLines := strings.Split(p.emptyWelcomeView(width), "\n")
	if len(welcomeLines) == 0 {
		return strings.Repeat("\n", lines) + messagesView
	}
	keep := min(lines, len(welcomeLines))
	prefix := strings.Join(welcomeLines[len(welcomeLines)-keep:], "\n")
	if lines > keep {
		prefix = strings.Repeat("\n", lines-keep) + prefix
	}
	return prefix + "\n" + messagesView
}

func (p *chatPage) IsTranscriptEmpty() bool {
	return p.messages.IsEmpty()
}

func (p *chatPage) NthLatestAssistantMessage(index int) string {
	return p.messages.NthLatestAssistantMessage(index)
}

func (p *chatPage) AssistantMessageCount() int {
	return p.messages.AssistantMessageCount()
}

func (p *chatPage) ExportTranscript() string {
	body := strings.TrimSpace(p.messages.ExportTranscript())
	if body == "" {
		return ""
	}
	sl := p.computeSidebarLayout()
	header := plainExportText(p.emptyWelcomeView(sl.chatWidth))
	if header == "" {
		return body
	}
	return trimExportTextBlock(header + "\n\n" + body)
}

func plainExportText(view string) string {
	if view == "" {
		return ""
	}
	lines := strings.Split(strings.TrimSuffix(view, "\n"), "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(ansi.Strip(line), " \t")
	}
	return trimExportTextBlock(strings.Join(lines, "\n"))
}

func trimExportTextBlock(text string) string {
	lines := strings.Split(strings.TrimSuffix(text, "\n"), "\n")
	start := 0
	for start < len(lines) && strings.TrimSpace(lines[start]) == "" {
		start++
	}
	end := len(lines)
	for end > start && strings.TrimSpace(lines[end-1]) == "" {
		end--
	}
	if start >= end {
		return ""
	}
	return strings.Join(lines[start:end], "\n")
}

func dropLeadingRenderedLines(view string, count int) string {
	if count <= 0 || view == "" {
		return view
	}
	lines := strings.Split(view, "\n")
	if count >= len(lines) {
		return ""
	}
	return strings.Join(lines[count:], "\n")
}

// renderSidebarHandle renders the sidebar toggle/resize handle.
// When collapsed: shows just « at top.
// When expanded: shows » at top, rest is empty space (draggable for resize).
func (p *chatPage) renderSidebarHandle(height int) string {
	lines := make([]string, height)

	if p.sidebar.IsCollapsed() {
		// Collapsed: just the toggle glyph, no vertical line
		lines[0] = styles.MutedStyle.Render("«")
		for i := 1; i < height; i++ {
			lines[i] = " "
		}
	} else {
		// Expanded: just the toggle at top, rest is empty space (still draggable)
		lines[0] = styles.MutedStyle.Render("»")
		for i := 1; i < height; i++ {
			lines[i] = " "
		}
	}

	return strings.Join(lines, "\n")
}

func (p *chatPage) SetSize(width, height int) tea.Cmd {
	p.width = width
	p.height = height

	var cmds []tea.Cmd

	// Compute layout once and use it for all sizing
	sl := p.computeSidebarLayout()

	switch sl.mode {
	case sidebarVertical:
		p.sidebar.SetMode(sidebar.ModeVertical)
		cmds = append(cmds,
			p.sidebar.SetSize(sl.sidebarWidth-toggleColumnWidth, sl.chatHeight),
			p.sidebar.SetPosition(styles.AppPadding+sl.sidebarStartX, 0),
			p.messages.SetPosition(styles.AppPadding, 0),
		)
	case sidebarCollapsed, sidebarCollapsedNarrow:
		p.sidebar.SetMode(sidebar.ModeCollapsed)
		cmds = append(cmds,
			p.sidebar.SetSize(sl.sidebarWidth, sl.sidebarHeight),
			p.sidebar.SetPosition(styles.AppPadding, 0),
			p.messages.SetPosition(styles.AppPadding, sl.sidebarHeight),
		)
	}

	cmds = append(cmds, p.messages.SetSize(sl.chatWidth, sl.chatHeight))

	return tea.Batch(cmds...)
}

// GetSize returns the current dimensions
func (p *chatPage) GetSize() (width, height int) {
	return p.width, p.height
}

// Bindings returns key bindings for the chat page
func (p *chatPage) Bindings() []key.Binding {
	return p.messages.Bindings()
}

// Help returns help information
func (p *chatPage) Help() help.KeyMap {
	return core.NewSimpleHelp(p.Bindings())
}

// cancelStream cancels the current stream and cleans up associated state
func (p *chatPage) cancelStream(showCancelMessage bool) tea.Cmd {
	return p.cancelStreamWithOptions(showCancelMessage, true)
}

func (p *chatPage) cancelStreamPreservingInput(showCancelMessage bool) tea.Cmd {
	return p.cancelStreamWithOptions(showCancelMessage, false)
}

func (p *chatPage) cancelStreamWithOptions(showCancelMessage bool, restoreBangInput bool) tea.Cmd {
	if p.msgCancel == nil && !p.working {
		return nil
	}

	if p.app != nil {
		p.app.Interrupt()
	}
	if p.msgCancel != nil {
		p.msgCancel()
		p.msgCancel = nil
	}
	p.streamCancelled = true
	p.streamDepth = 0
	p.setPendingResponse(false)

	showMessage := showCancelMessage
	var restoreInputCmd tea.Cmd
	if restoreBangInput && p.activeBangCommand != "" {
		p.messages.RemoveLastUserMessage("! " + p.activeBangCommand)
		showMessage = false
		restoreInputCmd = core.CmdHandler(msgtypes.RestoreEditorInputMsg{
			Content:   p.activeBangCommand,
			ShellMode: true,
		})
	} else if p.activeUserPromptRestorable && p.activeUserPrompt != "" && !p.hasReceivedAssistantContent {
		p.messages.RemoveLastUserMessage(p.activeUserPrompt)
		showMessage = false
		restoreInputCmd = core.CmdHandler(msgtypes.RestoreEditorInputMsg{
			Content: p.activeUserPrompt,
		})
	}
	p.activeBangCommand = ""
	p.activeUserPrompt = ""
	p.activeUserPromptRestorable = false

	// Send StreamCancelledMsg to all components to handle cleanup
	return tea.Batch(
		core.CmdHandler(msgtypes.StreamCancelledMsg{ShowMessage: showMessage}),
		restoreInputCmd,
		p.setWorking(false),
	)
}

func isBangCommand(content string) bool {
	return strings.HasPrefix(content, "!")
}

func (p *chatPage) parseImmediateCommand(content string) tea.Cmd {
	if p.commandParser == nil {
		return nil
	}
	if cmd := p.commandParser.Parse(content); cmd != nil {
		return cmd
	}
	return p.commandParser.ParseUnknown(content)
}

// SetCommandParser replaces the slash-command parser for the chat page.
func (p *chatPage) SetCommandParser(parser *commands.Parser) {
	p.commandParser = parser
}

// handleSendMsg handles incoming messages from the editor.
func (p *chatPage) handleSendMsg(msg msgtypes.SendMsg) (layout.Model, tea.Cmd) {
	// Handle "exit", "quit", and ":q" as special keywords to quit the session
	// immediately, equivalent to the /exit slash command.
	switch strings.TrimSpace(msg.Content) {
	case "exit", "quit", ":q":
		return p, core.CmdHandler(msgtypes.ExitSessionMsg{})
	}

	// Immediate UI slash commands (e.g. /exit, /compact) run even in read-only
	// mode. A BypassQueue message has already been resolved (e.g. an agent
	// command or fork-mode skill re-dispatching itself) and must skip parsing:
	// re-parsing would match the same command again and loop forever.
	if !msg.BypassQueue {
		if cmd := p.parseImmediateCommand(msg.Content); cmd != nil {
			return p, cmd
		}
	}

	// Everything below hands work to the model, which read-only sessions must
	// reject: normal input, bang commands, and resolved agent/skill commands
	// flagged BypassQueue.
	if p.app != nil && p.app.IsReadOnly() {
		return p, notification.WarningCmd("Session is read-only. No new messages can be sent.")
	}

	if msg.BypassQueue || isBangCommand(msg.Content) {
		cmd := p.processMessage(msg)
		return p, cmd
	}

	// Active run: keep the message in a local Claude Code-style queue and
	// submit it after the current stream stops. This check intentionally lives
	// here, not only in the editor, because runtime events are the source of
	// truth for whether a gateway turn is still active.
	if p.working || p.msgCancel != nil {
		p.queuedInputs = append(p.queuedInputs, queuedInput{
			content:     msg.Content,
			attachments: msg.Attachments,
		})
		return p, tea.Batch(
			p.emitQueuedInputState(),
			p.messages.ScrollToBottom(),
		)
	}

	// Keep the response path live by interrupting the in-flight run and
	// submitting this message immediately.
	cmd := p.processMessage(msg)
	return p, cmd
}

func (p *chatPage) handleUnknownCommand(msg msgtypes.UnknownCommandMsg) (layout.Model, tea.Cmd) {
	command := strings.TrimSpace(msg.Command)
	if command == "" {
		command = "/"
	}
	content := "Unknown command: " + command
	if suggestion := strings.TrimSpace(msg.Suggestion); suggestion != "" {
		content += ". Did you mean " + suggestion + "?"
	}

	agentName := "assistant"
	if p.sessionState != nil && p.sessionState.CurrentAgentName() != "" {
		agentName = p.sessionState.CurrentAgentName()
	}
	p.setPendingResponse(false)
	return p, tea.Batch(
		p.messages.AddAssistantTextMessage(agentName, content),
		p.setWorking(false),
		p.messages.ScrollToBottom(),
	)
}

func (p *chatPage) handleEditUserMessage(msg msgtypes.EditUserMessageMsg) (layout.Model, tea.Cmd) {
	if msg.SessionPosition < 0 || msg.MsgIndex < 0 {
		return p, nil
	}

	p.editing = true
	p.branchAtPosition = msg.SessionPosition

	// Extract any attachments from the original session message
	p.editAttachments = p.extractAttachmentsFromSession(msg.SessionPosition)

	// Start inline editing in the messages component.
	// Request focus switch to messages panel so the parent blurs the editor.
	editCmd := p.messages.StartInlineEdit(msg.MsgIndex, msg.SessionPosition, msg.OriginalContent)
	focusCmd := core.CmdHandler(msgtypes.RequestFocusMsg{Target: msgtypes.PanelMessages})

	return p, tea.Batch(editCmd, focusCmd)
}

// handleInlineEditCommitted handles the commit of an inline edit, triggering a branch.
func (p *chatPage) handleInlineEditCommitted(msg messages.InlineEditCommittedMsg) (layout.Model, tea.Cmd) {
	if !p.editing {
		return p, nil
	}

	p.editing = false
	branchPosition := p.branchAtPosition
	p.branchAtPosition = 0
	attachments := p.editAttachments
	p.editAttachments = nil

	var cancelCmd tea.Cmd
	if p.msgCancel != nil {
		cancelCmd = p.cancelStream(false)
	}

	parentID := ""
	if sess := p.app.Session(); sess != nil {
		parentID = sess.ID
	}

	branchCmd := core.CmdHandler(msgtypes.BranchFromEditMsg{
		ParentSessionID:  parentID,
		BranchAtPosition: branchPosition,
		Content:          msg.Content,
		Attachments:      attachments,
	})

	return p, tea.Batch(cancelCmd, branchCmd)
}

// handleInlineEditCancelled handles cancellation of an inline edit.
func (p *chatPage) handleInlineEditCancelled(msg messages.InlineEditCancelledMsg) (layout.Model, tea.Cmd) {
	p.editing = false
	p.branchAtPosition = 0
	p.editAttachments = nil

	if msg.WasInSelectionMode {
		// We were in keyboard selection mode before editing, stay in the messages panel.
		// The messages component already restored its selection state.
		return p, core.CmdHandler(msgtypes.RequestFocusMsg{Target: msgtypes.PanelMessages})
	}
	// We weren't in selection mode, return focus to the editor.
	return p, core.CmdHandler(msgtypes.RequestFocusMsg{Target: msgtypes.PanelEditor})
}

// extractAttachmentsFromSession extracts attachments from a session message at the given position.
// Attachments are stored as text parts in MultiContent with format "Contents of <filename>: <dataURL>".
// TODO(krisetto): meh we can store and retrieve attachments better in the session itself
func (p *chatPage) extractAttachmentsFromSession(position int) []msgtypes.Attachment {
	sess := p.app.Session()
	if sess == nil || position < 0 || position >= len(sess.Messages) {
		return nil
	}

	item := sess.Messages[position]
	if !item.IsMessage() || item.Message == nil {
		return nil
	}

	msg := item.Message.Message
	if len(msg.MultiContent) <= 1 {
		// No attachments - only the main text content or nothing
		return nil
	}

	var attachments []msgtypes.Attachment
	const prefix = "Contents of "

	// Skip the first part (main text content), look for attachment parts
	for i := 1; i < len(msg.MultiContent); i++ {
		part := msg.MultiContent[i]
		if part.Type != chat.MessagePartTypeText {
			continue
		}
		text := part.Text
		if !strings.HasPrefix(text, prefix) {
			continue
		}
		// Parse "Contents of <filename>: <dataURL>"
		rest := text[len(prefix):]
		before, after, ok := strings.Cut(rest, ": ")
		if !ok {
			continue
		}
		filename := before
		content := after
		if filename != "" && content != "" {
			attachments = append(attachments, msgtypes.Attachment{
				Name:    filename,
				Content: content,
			})
		}
	}

	return attachments
}

// processMessage processes a message with the runtime
func (p *chatPage) processMessage(msg msgtypes.SendMsg) tea.Cmd {
	// Handle slash commands (e.g., /eval, /compact, /exit) BEFORE cancelling any ongoing stream.
	// These are UI commands that shouldn't interrupt the running agent.
	if !msg.BypassQueue {
		if cmd := p.parseImmediateCommand(msg.Content); cmd != nil {
			return cmd
		}
	}

	if isBangCommand(msg.Content) {
		command := strings.TrimSpace(msg.Content[1:])
		if command == "" {
			return nil
		}
		if p.msgCancel != nil {
			p.msgCancel()
		}
		p.streamCancelled = false
		p.streamDepth = 0
		p.sidebar.ResetStreamTracking()
		p.activeUserPrompt = ""
		p.activeUserPromptRestorable = false
		p.hasReceivedAssistantContent = false

		var ctx context.Context
		ctx, p.msgCancel = context.WithCancel(context.Background())
		p.activeBangCommand = command
		spinnerCmd := p.setWorking(true)

		go p.app.RunBangCommand(ctx, p.msgCancel, command)

		return tea.Batch(p.messages.ScrollToBottom(), spinnerCmd)
	}

	if p.msgCancel != nil {
		p.msgCancel()
	}

	p.streamCancelled = false
	p.activeBangCommand = ""
	p.activeUserPrompt = msg.Content
	p.activeUserPromptRestorable = true
	p.streamDepth = 0
	p.hasReceivedAssistantContent = false
	p.sidebar.ResetStreamTracking()

	var ctx context.Context
	ctx, p.msgCancel = context.WithCancel(context.Background())

	// Start working state immediately to show the user something is happening.
	// This provides visual feedback while the runtime loads tools and prepares the stream.
	spinnerCmd := p.setWorking(true)
	// Check if this is an agent command that needs resolution
	// If so, show a loading message with the command description
	var loadingCmd tea.Cmd
	if strings.HasPrefix(msg.Content, "/") {
		cmdName, _, _ := strings.Cut(msg.Content[1:], " ")
		if cmd, found := p.app.CurrentAgentCommands(ctx)[cmdName]; found {
			loadingCmd = p.messages.AddLoadingMessage(cmd.DisplayText())
		}
	}

	// Run command resolution and agent execution in a goroutine
	// so the UI stays responsive while skill/agent commands are resolved.
	go func() {
		if skillName, task, ok := p.app.SkillCommandFork(ctx, msg.Content); ok {
			// Fork-mode skill: run in an isolated sub-session.
			p.app.RunSkillFork(ctx, p.msgCancel, skillName, task, msg.Attachments)
			return
		}
		p.app.Run(ctx, p.msgCancel, p.app.ResolveInput(ctx, msg.Content), msg.Attachments)
	}()

	return tea.Batch(p.messages.ScrollToBottom(), spinnerCmd, loadingCmd)
}

// processCooperativeMessage submits content to the gateway without cancelling
// the active stream. The gateway pushes it into the core SessionInbox; the
// session driver, not the TUI, owns the actual queueing semantics.
func (p *chatPage) processCooperativeMessage(msg msgtypes.SendMsg) tea.Cmd {
	return func() tea.Msg {
		ctx := context.Background()
		p.app.RunCooperative(
			ctx,
			func() {},
			p.app.ResolveInput(ctx, msg.Content),
			msg.Attachments,
		)
		return nil
	}
}

// CompactSession generates a summary and compacts the session history
func (p *chatPage) CompactSession(additionalPrompt string) tea.Cmd {
	// Cancel any active stream without showing cancellation message
	cancelCmd := p.cancelStream(false)

	var ctx context.Context
	ctx, p.msgCancel = context.WithCancel(context.Background())
	p.app.CompactSession(ctx, p.msgCancel, additionalPrompt)

	return tea.Batch(
		cancelCmd,
		p.setWorking(true),
		p.setPendingResponse(true),
		p.messages.ScrollToBottom(),
	)
}

// SetSessionStarred updates the sidebar star indicator
func (p *chatPage) SetSessionStarred(starred bool) {
	p.sidebar.SetSessionStarred(starred)
}

func (p *chatPage) SetTitleRegenerating(regenerating bool) tea.Cmd {
	return p.sidebar.SetTitleRegenerating(regenerating)
}

// GetSidebarSettings returns the current sidebar display settings.
func (p *chatPage) GetSidebarSettings() SidebarSettings {
	return SidebarSettings{
		Collapsed:      p.sidebar.IsCollapsed(),
		PreferredWidth: p.sidebar.GetPreferredWidth(),
	}
}

// SetSidebarSettings applies sidebar display settings.
func (p *chatPage) SetSidebarSettings(settings SidebarSettings) {
	p.sidebar.SetCollapsed(settings.Collapsed)
	p.sidebar.SetPreferredWidth(settings.PreferredWidth)
}

func (p *chatPage) SetCompletionOpen(open bool) {
	p.completionOpen = open
}

// handleSidebarClickType checks what was clicked in the sidebar area.
// Returns the click type and, for ClickAgent, the agent name.
func (p *chatPage) handleSidebarClickType(x, y int) (sidebar.ClickResult, string) {
	adjustedX := x - styles.AppPadding
	sl := p.computeSidebarLayout()

	switch sl.mode {
	case sidebarCollapsedNarrow, sidebarCollapsed:
		return p.sidebar.HandleClickType(adjustedX, y)
	case sidebarVertical:
		if sl.isInSidebar(adjustedX) {
			return p.sidebar.HandleClickType(adjustedX-sl.sidebarStartX, y)
		}
	}

	return sidebar.ClickNone, ""
}

// routeMouseEvent routes mouse events to the appropriate component based on coordinates.
func (p *chatPage) routeMouseEvent(msg tea.Msg, _ int) tea.Cmd {
	sl := p.computeSidebarLayout()

	if sl.mode == sidebarVertical && !p.sidebar.IsCollapsed() {
		var x int
		switch m := msg.(type) {
		case tea.MouseClickMsg:
			x = m.X
		case tea.MouseMotionMsg:
			x = m.X
		case tea.MouseReleaseMsg:
			x = m.X
		}

		adjustedX := x - styles.AppPadding
		if sl.isInSidebar(adjustedX) {
			model, cmd := p.sidebar.Update(msg)
			p.sidebar = model.(sidebar.Model)
			return cmd
		}
	}

	model, cmd := p.messages.Update(msg)
	p.messages = model.(messages.Model)
	return cmd
}

// IsWorking returns whether the agent is currently working
func (p *chatPage) IsWorking() bool {
	return p.working
}

// IsInlineEditing returns true if a past user message is being edited inline.
func (p *chatPage) IsInlineEditing() bool {
	return p.messages.IsInlineEditing()
}

// FocusMessages gives focus to the messages panel
func (p *chatPage) FocusMessages() tea.Cmd {
	return p.messages.Focus()
}

// FocusMessageAt gives focus and selects the message at the given screen coordinates.
func (p *chatPage) FocusMessageAt(x, y int) tea.Cmd {
	return p.messages.FocusAt(x, y)
}

// BlurMessages removes focus from the messages panel
func (p *chatPage) BlurMessages() {
	p.messages.Blur()
}

// ScrollToBottom scrolls the messages viewport to the bottom if auto-scroll is active.
func (p *chatPage) ScrollToBottom() tea.Cmd {
	return p.messages.ScrollToBottom()
}

func (p *chatPage) ForceScrollToBottom() tea.Cmd {
	return p.messages.ForceScrollToBottom()
}

func (p *chatPage) AdjustBottomSlack(delta int) {
	p.messages.AdjustBottomSlack(delta)
}

func (p *chatPage) TranscriptHeight() int {
	sl := p.computeSidebarLayout()
	messageHeight := p.messages.RenderedHeight()
	if messageHeight == 0 {
		return renderedLineCount(p.emptyWelcomeView(sl.chatWidth))
	}
	if p.hideSidebar {
		welcomeHeight := renderedLineCount(p.emptyWelcomeView(sl.chatWidth))
		if welcomeHeight+messageHeight <= sl.chatHeight {
			return welcomeHeight + messageHeight
		}
	}
	return messageHeight
}

func (p *chatPage) TranscriptMessageHeight() int {
	return p.messages.RenderedHeight()
}

func (p *chatPage) TranscriptViewportHeight() int {
	return p.messages.ViewportHeight()
}

func (p *chatPage) SetTranscriptTopContextLines(lines int) {
	p.transcriptTopContextLines = max(0, lines)
}

func (p *chatPage) AddLocalUserMessage(content string) tea.Cmd {
	return p.messages.AddUserMessage(content)
}

func (p *chatPage) AddLocalNoticeMessage(content string) tea.Cmd {
	return p.messages.AddNoticeMessage(content)
}

func (p *chatPage) AddLocalSystemMessage(content string) tea.Cmd {
	return p.messages.AddSystemMessage("", content)
}

func (p *chatPage) RemoveLastSystemMessage() tea.Cmd {
	return p.messages.RemoveLastSystemMessage()
}

func (p *chatPage) SetWelcomeModelLine(content string) {
	p.welcomeModelLine = content
}

func welcomeModelLineForActiveModel(model string, thinkingEnabled bool) string {
	model = strings.TrimSpace(model)
	if model == "" {
		return ""
	}
	if thinkingEnabled {
		return model + " with high effort · API Usage Billing"
	}
	return model + " · API Usage Billing"
}
