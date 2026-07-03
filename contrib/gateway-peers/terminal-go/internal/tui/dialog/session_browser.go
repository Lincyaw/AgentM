package dialog

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"charm.land/bubbles/v2/key"
	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/clipboardutil"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/scrollview"
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

// Session browser dialog dimension constants
const (
	sessionBrowserListOverhead = 12 // title(1) + space(1) + input(1) + separator(1) + separator(1) + id(1) + space(1) + help(1) + borders(2) + extra(2)
	sessionBrowserListStartY   = 6  // border(1) + padding(1) + title(1) + space(1) + input(1) + separator(1)
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
	visLines := d.scrollview.VisibleHeight()
	listStartY := dialogRow + sessionBrowserListStartY

	if y < listStartY || y >= listStartY+visLines {
		return -1
	}
	lineInView := y - listStartY
	idx := d.scrollview.ScrollOffset() + lineInView
	if idx < 0 || idx >= len(d.filtered) {
		return -1
	}
	return idx
}

func (d *sessionBrowserDialog) dialogSize() (dialogWidth, maxHeight, contentWidth int) {
	dialogWidth = d.ComputeDialogWidth(85, 60, 120)
	maxHeight = min(d.Height()*70/100, 30)
	contentWidth = dialogWidth - 6 - d.scrollview.ReservedCols()
	return dialogWidth, maxHeight, contentWidth
}

func (d *sessionBrowserDialog) View() string {
	dialogWidth, _, contentWidth := d.dialogSize()
	d.textInput.SetWidth(contentWidth)

	regionWidth := contentWidth + d.scrollview.ReservedCols()
	visibleLines := d.scrollview.VisibleHeight()

	// Set scrollview position for mouse hit-testing (auto-computed from dialog position)
	dialogRow, dialogCol := d.Position()
	d.scrollview.SetPosition(dialogCol+3, dialogRow+sessionBrowserListStartY)

	// Tell the scrollview the total content height; pass nil for lines
	// because we render only the visible window below. Rendering every row
	// on every keystroke is the dominant cost when there are many sessions.
	// The follow-up SetScrollOffset call re-clamps the offset against the
	// (possibly shrunk) total — it is intentionally not a no-op.
	total := len(d.filtered)
	d.scrollview.SetContent(nil, total)
	d.scrollview.SetScrollOffset(d.scrollview.ScrollOffset())

	var scrollableContent string
	if total == 0 {
		// Empty state: render manually so "No sessions found" is centered
		emptyLines := []string{"", styles.DialogContentStyle.
			Italic(true).Align(lipgloss.Center).Width(contentWidth).
			Render("No sessions found")}
		for len(emptyLines) < visibleLines {
			emptyLines = append(emptyLines, "")
		}
		scrollableContent = d.scrollview.ViewWithLines(emptyLines)
	} else {
		offset := d.scrollview.ScrollOffset()
		end := min(offset+visibleLines, total)
		windowLines := make([]string, 0, end-offset)
		for i := offset; i < end; i++ {
			windowLines = append(windowLines, d.renderSession(d.filtered[i], i == d.selected, contentWidth))
		}
		scrollableContent = d.scrollview.ViewWithLines(windowLines)
	}

	// Build title with session count and optional star-filter indicator.
	// Show "filtered/total" when a search or star filter reduces the list.
	var countLabel string
	if len(d.filtered) == len(d.sessions) {
		countLabel = strconv.Itoa(len(d.sessions))
	} else {
		countLabel = fmt.Sprintf("%d/%d", len(d.filtered), len(d.sessions))
	}
	title := fmt.Sprintf("Sessions (%s)", countLabel)
	switch d.starFilter {
	case 1:
		title += " " + styles.StarredStyle.Render("★")
	case 2:
		title += " " + styles.UnstarredStyle.Render("☆")
	}

	var filterDesc string
	switch d.starFilter {
	case 0:
		filterDesc = "all"
	case 1:
		filterDesc = "★ only"
	case 2:
		filterDesc = "☆ only"
	}

	var idFooter string
	if d.selected >= 0 && d.selected < len(d.filtered) {
		idFooter = styles.MutedStyle.Render("ID: ") + styles.SecondaryStyle.Render(d.filtered[d.selected].ID)
	}

	content := NewContent(regionWidth).
		AddTitle(title).
		AddSpace().
		AddContent(d.textInput.View()).
		AddSeparator().
		AddContent(scrollableContent).
		AddSeparator().
		AddContent(idFooter).
		AddSpace().
		AddHelpKeys("↑/↓", "navigate", "ctrl+s", "star", "ctrl+f", filterDesc, "ctrl+y", "copy id", "ctrl+d", "delete").
		AddHelpKeys("enter", "load", "esc", "close").
		Build()

	return styles.DialogStyle.Width(dialogWidth).Render(content)
}

// SetSize sets the dialog dimensions and configures the scrollview region.
func (d *sessionBrowserDialog) SetSize(width, height int) tea.Cmd {
	cmd := d.BaseDialog.SetSize(width, height)
	_, maxHeight, contentWidth := d.dialogSize()
	regionWidth := contentWidth + d.scrollview.ReservedCols()
	visibleLines := max(1, maxHeight-sessionBrowserListOverhead)
	d.scrollview.SetSize(regionWidth, visibleLines)
	return cmd
}

func (d *sessionBrowserDialog) renderSession(sess session.Summary, selected bool, maxWidth int) string {
	titleStyle, timeStyle := styles.PaletteUnselectedActionStyle, styles.PaletteUnselectedDescStyle
	if selected {
		titleStyle, timeStyle = styles.PaletteSelectedActionStyle, styles.PaletteSelectedDescStyle
	}

	title := sess.Title
	if title == "" {
		title = "Untitled"
	}

	suffix := fmt.Sprintf(" • (%d msg) • %s", sess.NumMessages, d.timeAgo(sess.CreatedAt))

	starWidth := 3
	maxTitleLen := max(1, maxWidth-len(suffix)-starWidth)
	if r := []rune(title); len(r) > maxTitleLen {
		title = string(r[:maxTitleLen-1]) + "…"
	}

	return styles.StarIndicator(sess.Starred) + titleStyle.Render(title) + timeStyle.Render(suffix)
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
	dialogWidth, maxHeight, _ := d.dialogSize()
	return CenterPosition(d.Width(), d.Height(), dialogWidth, maxHeight)
}
