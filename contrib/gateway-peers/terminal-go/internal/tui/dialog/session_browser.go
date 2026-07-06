package dialog

import (
	"fmt"
	"strings"
	"time"

	"charm.land/bubbles/v2/key"
	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/clipboardutil"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/scrollview"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

// sessionBrowserKeyMap defines key bindings for the session browser
type sessionBrowserKeyMap struct {
	Up         key.Binding
	Down       key.Binding
	Enter      key.Binding
	Escape     key.Binding
	Star       key.Binding
	FilterStar key.Binding
	CopyID     key.Binding
	Delete     key.Binding
}

const (
	sessionBrowserTopRow       = 5
	sessionBrowserIndent       = 2
	sessionBrowserRowsPerEntry = 3
)

type sessionBrowserDialog struct {
	BaseDialog

	textInput  textinput.Model
	sessions   []session.Summary
	filtered   []session.Summary
	selected   int
	scrollview *scrollview.Model
	keyMap     sessionBrowserKeyMap
	openedAt   time.Time // when dialog was opened, for stable time display
	starFilter int       // 0 = all, 1 = starred only, 2 = unstarred only
	listStartY int

	// Double-click detection
	lastClickTime  time.Time
	lastClickIndex int
}

// NewSessionBrowserDialog creates a new session browser dialog
func NewSessionBrowserDialog(sessions []session.Summary) Dialog {
	ti := textinput.New()
	ti.Placeholder = "Type to search sessions…"
	ti.Focus()
	ti.CharLimit = 100
	ti.SetWidth(50)

	// Filter out empty sessions (sessions without a title)
	nonEmptySessions := make([]session.Summary, 0, len(sessions))
	for _, s := range sessions {
		if s.Title != "" {
			nonEmptySessions = append(nonEmptySessions, s)
		}
	}

	d := &sessionBrowserDialog{
		textInput:  ti,
		sessions:   nonEmptySessions,
		scrollview: scrollview.New(scrollview.WithReserveScrollbarSpace(true)),
		keyMap: sessionBrowserKeyMap{
			Up:         key.NewBinding(key.WithKeys("up", "ctrl+k")),
			Down:       key.NewBinding(key.WithKeys("down", "ctrl+j")),
			Enter:      key.NewBinding(key.WithKeys("enter")),
			Escape:     key.NewBinding(key.WithKeys("esc")),
			Star:       key.NewBinding(key.WithKeys("ctrl+s")),
			FilterStar: key.NewBinding(key.WithKeys("ctrl+f")),
			CopyID:     key.NewBinding(key.WithKeys("ctrl+y")),
			Delete:     key.NewBinding(key.WithKeys("ctrl+d")),
		},
		openedAt: time.Now(),
	}
	// Initialize filtered list
	d.filterSessions(true)
	return d
}

func (d *sessionBrowserDialog) Init() tea.Cmd {
	return textinput.Blink
}

func (d *sessionBrowserDialog) Update(msg tea.Msg) (layout.Model, tea.Cmd) {
	// Scrollview handles mouse click/motion/release, wheel, and pgup/pgdn/home/end
	if handled, cmd := d.scrollview.Update(msg); handled {
		return d, cmd
	}

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		cmd := d.SetSize(msg.Width, msg.Height)
		return d, cmd

	case messages.SessionStarChangedMsg:
		d.setSessionStarred(msg.SessionID, msg.Starred)
		return d, nil

	case messages.SessionDeletedMsg:
		d.removeSession(msg.SessionID)
		return d, nil

	case tea.PasteMsg:
		var cmd tea.Cmd
		d.textInput, cmd = d.textInput.Update(msg)
		d.filterSessions(true)
		return d, cmd

	case tea.MouseClickMsg:
		// Scrollbar clicks already handled above; this handles list item clicks
		if msg.Button == tea.MouseLeft {
			if idx := d.mouseYToSessionIndex(msg.Y); idx >= 0 {
				now := time.Now()
				if idx == d.lastClickIndex && now.Sub(d.lastClickTime) < styles.DoubleClickThreshold {
					d.selected = idx
					d.lastClickTime = time.Time{}
					return d, d.loadSelectedSessionCmd()
				}
				d.selected = idx
				d.lastClickTime = now
				d.lastClickIndex = idx
			}
		}
		return d, nil

	case tea.KeyPressMsg:
		if cmd := HandleQuit(msg); cmd != nil {
			return d, cmd
		}

		switch {
		case key.Matches(msg, d.keyMap.Escape):
			return d, core.CmdHandler(CloseDialogMsg{})

		case key.Matches(msg, d.keyMap.Up):
			if d.selected > 0 {
				d.selected--
				d.scrollview.EnsureLineVisible(d.selected)
			}
			return d, nil

		case key.Matches(msg, d.keyMap.Down):
			if d.selected < len(d.filtered)-1 {
				d.selected++
				d.scrollview.EnsureLineVisible(d.selected)
			}
			return d, nil

		case key.Matches(msg, d.keyMap.Enter):
			return d, d.loadSelectedSessionCmd()

		case key.Matches(msg, d.keyMap.Star):
			if sess, ok := d.selectedSession(); ok {
				return d, core.CmdHandler(messages.ToggleSessionStarMsg{SessionID: sess.ID})
			}
			return d, nil

		case key.Matches(msg, d.keyMap.FilterStar):
			d.starFilter = (d.starFilter + 1) % 3
			d.filterSessions(true)
			return d, nil

		case key.Matches(msg, d.keyMap.CopyID):
			if sess, ok := d.selectedSession(); ok {
				return d, clipboardutil.CopyNative(
					sess.ID,
					clipboardutil.WithSuccess("Session ID copied to clipboard."),
				)
			}
			return d, nil

		case key.Matches(msg, d.keyMap.Delete):
			if sess, ok := d.selectedSession(); ok {
				return d, core.CmdHandler(messages.DeleteSessionMsg{SessionID: sess.ID})
			}
			return d, nil

		default:
			var cmd tea.Cmd
			d.textInput, cmd = d.textInput.Update(msg)
			d.filterSessions(true)
			return d, cmd
		}
	}

	return d, nil
}

func (d *sessionBrowserDialog) selectedSession() (session.Summary, bool) {
	if d.selected < 0 || d.selected >= len(d.filtered) {
		return session.Summary{}, false
	}
	return d.filtered[d.selected], true
}

func (d *sessionBrowserDialog) loadSelectedSessionCmd() tea.Cmd {
	sess, ok := d.selectedSession()
	if !ok {
		return nil
	}
	return tea.Sequence(
		core.CmdHandler(CloseDialogMsg{}),
		core.CmdHandler(messages.LoadSessionMsg{SessionID: sess.ID}),
	)
}

func (d *sessionBrowserDialog) setSessionStarred(sessionID string, starred bool) {
	var found bool
	for i := range d.sessions {
		if d.sessions[i].ID == sessionID {
			d.sessions[i].Starred = starred
			found = true
			break
		}
	}
	if !found {
		return
	}

	if d.starFilter != 0 {
		d.filterSessions(false)
		return
	}
	for i := range d.filtered {
		if d.filtered[i].ID == sessionID {
			d.filtered[i].Starred = starred
			return
		}
	}
}

func (d *sessionBrowserDialog) removeSession(sessionID string) {
	for i := range d.sessions {
		if d.sessions[i].ID == sessionID {
			d.sessions = append(d.sessions[:i], d.sessions[i+1:]...)
			d.filterSessions(false)
			return
		}
	}
}

func (d *sessionBrowserDialog) filterSessions(resetScroll bool) {
	selectedID := ""
	if !resetScroll {
		if sess, ok := d.selectedSession(); ok {
			selectedID = sess.ID
		}
	}

	query := strings.ToLower(strings.TrimSpace(d.textInput.Value()))

	d.filtered = nil
	for _, sess := range d.sessions {
		switch d.starFilter {
		case 1:
			if !sess.Starred {
				continue
			}
		case 2:
			if sess.Starred {
				continue
			}
		}

		if query != "" {
			title := sess.Title
			if title == "" {
				title = "Untitled"
			}
			if !strings.Contains(strings.ToLower(title), query) {
				continue
			}
		}

		d.filtered = append(d.filtered, sess)
	}

	switch {
	case len(d.filtered) == 0:
		d.selected = -1
	case resetScroll:
		d.selected = 0
	case selectedID != "":
		if idx := indexSessionSummary(d.filtered, selectedID); idx >= 0 {
			d.selected = idx
		} else if d.selected >= len(d.filtered) {
			d.selected = len(d.filtered) - 1
		}
	case d.selected < 0:
		d.selected = 0
	case d.selected >= len(d.filtered):
		d.selected = len(d.filtered) - 1
	}

	// Keep the scrollview's totalHeight in sync so EnsureLineVisible and the
	// scrollbar clamp correctly even before View() runs.
	d.scrollview.SetContent(nil, len(d.filtered))
	if resetScroll {
		d.scrollview.SetScrollOffset(0)
	} else {
		d.scrollview.SetScrollOffset(d.scrollview.ScrollOffset())
		if d.selected >= 0 {
			d.scrollview.EnsureLineVisible(d.selected)
		}
	}
}

func indexSessionSummary(sessions []session.Summary, sessionID string) int {
	for i, sess := range sessions {
		if sess.ID == sessionID {
			return i
		}
	}
	return -1
}

// mouseYToSessionIndex converts a mouse Y position to a session index in the filtered list.
// Returns -1 if the position is not on a session.
func (d *sessionBrowserDialog) mouseYToSessionIndex(y int) int {
	dialogRow, _ := d.Position()
	visEntries := d.scrollview.VisibleHeight()
	listStartY := dialogRow + d.listStartY

	if d.listStartY <= 0 || y < listStartY || y >= listStartY+(visEntries*sessionBrowserRowsPerEntry) {
		return -1
	}
	lineInView := y - listStartY
	idx := d.scrollview.ScrollOffset() + (lineInView / sessionBrowserRowsPerEntry)
	if idx < 0 || idx >= len(d.filtered) {
		return -1
	}
	return idx
}

func (d *sessionBrowserDialog) contentWidth() int {
	return max(1, d.Width()-(sessionBrowserIndent*2))
}

func (d *sessionBrowserDialog) View() string {
	width := max(20, d.Width())
	contentWidth := d.contentWidth()

	searchLines := d.renderSearchBox(contentWidth)
	d.listStartY = 1 + 1 + 1 + len(searchLines) + 1 + 1
	visibleEntries := d.visibleEntryCount(len(searchLines))
	d.scrollview.SetSize(width, visibleEntries)
	row, _ := d.Position()
	d.scrollview.SetPosition(0, row+d.listStartY)

	total := len(d.filtered)
	d.scrollview.SetContent(nil, total)
	d.scrollview.SetScrollOffset(d.scrollview.ScrollOffset())

	var listLines []string
	if total == 0 {
		listLines = []string{
			sessionBrowserLine("", contentWidth),
			sessionBrowserLine(styles.SecondaryStyle.Render("No sessions found"), contentWidth),
		}
	} else {
		offset := d.scrollview.ScrollOffset()
		end := min(offset+visibleEntries, total)
		listLines = make([]string, 0, (end-offset)*sessionBrowserRowsPerEntry)
		for i := offset; i < end; i++ {
			listLines = append(listLines, d.renderSessionLines(d.filtered[i], i == d.selected, contentWidth)...)
		}
	}

	lines := make([]string, 0, 10+len(searchLines)+len(listLines))
	lines = append(lines, "")
	lines = append(lines, styles.DialogSeparatorStyle.Render(strings.Repeat("─", width)))
	lines = append(lines, sessionBrowserLine(styles.BaseStyle.Render(d.title()), contentWidth))
	lines = append(lines, searchLines...)
	lines = append(lines, sessionBrowserLine(styles.SecondaryStyle.Render("  AgentM"), contentWidth))
	lines = append(lines, "")
	lines = append(lines, listLines...)
	lines = append(lines, "")
	lines = append(lines, d.renderHelp(contentWidth)...)

	return strings.Join(lines, "\n")
}

// SetSize sets the dialog dimensions and configures the scrollview region.
func (d *sessionBrowserDialog) SetSize(width, height int) tea.Cmd {
	cmd := d.BaseDialog.SetSize(width, height)
	d.scrollview.SetSize(max(20, width), max(1, height-sessionBrowserTopRow-8))
	return cmd
}

func (d *sessionBrowserDialog) title() string {
	position := 0
	if len(d.filtered) > 0 && d.selected >= 0 {
		position = d.selected + 1
	}
	total := len(d.sessions)
	if total == 0 {
		total = len(d.filtered)
	}
	title := fmt.Sprintf("Resume session (%d of %d)", position, total)
	switch d.starFilter {
	case 1:
		title += " " + styles.StarredStyle.Render("★")
	case 2:
		title += " " + styles.UnstarredStyle.Render("☆")
	}
	return title
}

func (d *sessionBrowserDialog) visibleEntryCount(searchBoxLines int) int {
	row, _ := d.Position()
	fixedRows := 8 + searchBoxLines
	availableRows := d.Height() - row - fixedRows
	if availableRows < sessionBrowserRowsPerEntry {
		return 1
	}
	return max(1, availableRows/sessionBrowserRowsPerEntry)
}

func (d *sessionBrowserDialog) renderSearchBox(contentWidth int) []string {
	boxWidth := max(12, contentWidth)
	innerWidth := max(1, boxWidth-4)
	value := d.textInput.Value()
	text := "⌕ Search…"
	if value != "" {
		text = "⌕ " + value
	}
	textStyle := styles.SecondaryStyle
	if value != "" {
		textStyle = styles.BaseStyle
	}
	if lipgloss.Width(text) > innerWidth {
		text = toolcommon.TruncateText(text, innerWidth)
	}
	pad := strings.Repeat(" ", max(0, innerWidth-lipgloss.Width(text)))
	borderStyle := styles.DialogSeparatorStyle
	return []string{
		sessionBrowserLine(borderStyle.Render("╭"+strings.Repeat("─", max(1, boxWidth-2))+"╮"), contentWidth),
		sessionBrowserLine(borderStyle.Render("│ ")+textStyle.Render(text)+pad+borderStyle.Render(" │"), contentWidth),
		sessionBrowserLine(borderStyle.Render("╰"+strings.Repeat("─", max(1, boxWidth-2))+"╯"), contentWidth),
	}
}

func (d *sessionBrowserDialog) renderSessionLines(sess session.Summary, selected bool, maxWidth int) []string {
	titleStyle := styles.BaseStyle
	cursorStyle := styles.SecondaryStyle
	if selected {
		cursorStyle = styles.HighlightWhiteStyle
	}

	title := sess.Title
	if title == "" {
		title = "Untitled"
	}
	if sess.Starred {
		title = "★ " + title
	}
	cursor := " "
	if selected {
		cursor = "❯"
	}
	titleLine := cursorStyle.Render(cursor) + " " + titleStyle.Render(toolcommon.TruncateText(title, max(1, maxWidth-4)))
	meta := fmt.Sprintf("%s · main · %s", d.timeAgo(sess.CreatedAt), d.sessionSizeLabel(sess))
	metaLine := styles.SecondaryStyle.Render("  " + meta)
	return []string{
		sessionBrowserLine(titleLine, maxWidth),
		sessionBrowserLine(metaLine, maxWidth),
		sessionBrowserLine("", maxWidth),
	}
}

func (d *sessionBrowserDialog) sessionSizeLabel(sess session.Summary) string {
	if sess.SizeBytes > 0 {
		return formatSessionBrowserSize(sess.SizeBytes)
	}
	switch sess.NumMessages {
	case 0:
		return "0 messages"
	case 1:
		return "1 message"
	default:
		return fmt.Sprintf("%d messages", sess.NumMessages)
	}
}

func formatSessionBrowserSize(bytes int64) string {
	if bytes < 1024 {
		return fmt.Sprintf("%dB", bytes)
	}
	kb := float64(bytes) / 1024
	if kb < 10 {
		return fmt.Sprintf("%.1fKB", kb)
	}
	return fmt.Sprintf("%.0fKB", kb)
}

func (d *sessionBrowserDialog) renderHelp(contentWidth int) []string {
	lines := wrapModelPickerText(
		"Type to search · Enter to load · Esc to cancel · Ctrl+S to star · Ctrl+F to filter starred",
		contentWidth,
	)
	rendered := make([]string, 0, len(lines))
	for _, line := range lines {
		rendered = append(rendered, sessionBrowserLine(styles.SecondaryStyle.Render(line), contentWidth))
	}
	return rendered
}

func sessionBrowserLine(content string, contentWidth int) string {
	line := strings.Repeat(" ", sessionBrowserIndent) + content
	maxWidth := contentWidth + sessionBrowserIndent
	if lipgloss.Width(line) > maxWidth {
		return toolcommon.TruncateText(line, maxWidth)
	}
	return line
}

func (d *sessionBrowserDialog) timeAgo(t time.Time) string {
	elapsed := d.openedAt.Sub(t)
	switch {
	case elapsed < time.Minute:
		return fmt.Sprintf("%ds ago", int(elapsed.Seconds()))
	case elapsed < time.Hour:
		return fmt.Sprintf("%dm ago", int(elapsed.Minutes()))
	case elapsed < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(elapsed.Hours()))
	case elapsed < 7*24*time.Hour:
		return fmt.Sprintf("%dd ago", int(elapsed.Hours()/24))
	default:
		return t.Format("Jan 2")
	}
}

func (d *sessionBrowserDialog) Position() (row, col int) {
	if d.Height() <= sessionBrowserTopRow+4 {
		return 0, 0
	}
	return sessionBrowserTopRow, 0
}
