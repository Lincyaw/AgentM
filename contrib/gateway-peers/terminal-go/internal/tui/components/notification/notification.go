package notification

import (
	"slices"
	"strings"
	"sync/atomic"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/mattn/go-runewidth"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

const (
	defaultDuration      = 4 * time.Second
	notificationPadding  = 1
	maxNotificationWidth = 56
	maxVisibleItems      = 3
)

var nextID atomic.Uint64

var timerDuration = defaultDuration

// Type represents the type of notification
type Type int

const (
	TypeSuccess Type = iota
	TypeWarning
	TypeInfo
	TypeError
)

func (t Type) autoHideDuration() time.Duration {
	return timerDuration
}

// style returns the lipgloss style for this notification type.
func (t Type) style() lipgloss.Style {
	switch t {
	case TypeError:
		return styles.NotificationErrorStyle
	case TypeWarning:
		return styles.NotificationWarningStyle
	case TypeInfo:
		return styles.NotificationInfoStyle
	default:
		return styles.NotificationStyle
	}
}

type ShowMsg struct {
	Text string
	Type Type // Defaults to TypeSuccess for backward compatibility
}

type HideMsg struct {
	ID uint64 // If 0, hides all notifications (backward compatibility)
}

// DismissMsg is sent when the user explicitly dismisses a notification.
type DismissMsg struct {
	ID uint64
}

// AutoHideMsg is sent by a notification timer. The generation field prevents
// stale timers from hiding notifications after hover pause/restart.
type AutoHideMsg struct {
	ID         uint64
	Generation uint64
}

func cmd(msg tea.Msg) tea.Cmd {
	return func() tea.Msg { return msg }
}

func SuccessCmd(text string) tea.Cmd {
	return cmd(ShowMsg{Text: text, Type: TypeSuccess})
}

func WarningCmd(text string) tea.Cmd {
	return cmd(ShowMsg{Text: text, Type: TypeWarning})
}

func InfoCmd(text string) tea.Cmd {
	return cmd(ShowMsg{Text: text, Type: TypeInfo})
}

func ErrorCmd(text string) tea.Cmd {
	return cmd(ShowMsg{Text: text, Type: TypeError})
}

// notificationItem represents a single notification
type notificationItem struct {
	ID       uint64
	Text     string
	Type     Type
	timerGen uint64
}

// render returns the styled view string for this notification item.
func (item notificationItem) render(maxWidth int, closeHovered, bodyHovered, copied bool) string {
	_, _ = closeHovered, bodyHovered
	text := item.compactText(maxWidth)
	if copied {
		text = truncateDisplay(text+" · copied", maxWidth)
	}
	style := item.Type.style()
	return style.Render(text)
}

func (item notificationItem) compactText(maxWidth int) string {
	text := strings.Join(strings.Fields(item.Text), " ")
	if text == "" {
		text = "Done"
	}
	text = item.Type.prefix() + text
	return truncateDisplay(text, maxWidth)
}

func (t Type) prefix() string {
	switch t {
	case TypeError:
		return "Error: "
	case TypeWarning:
		return "Warning: "
	case TypeInfo:
		return ""
	default:
		return ""
	}
}

// Manager displays Claude-style compact status notices in the bottom-right
// corner. Notices are intentionally single-line and short-lived so they do not
// compete with the main transcript.
type Manager struct {
	width, height  int
	items          []notificationItem
	hoveredID      uint64
	closeHoveredID uint64
	copiedID       uint64
}

func New() Manager { return Manager{} }

// SetSize records the screen size used to position notifications.
func (n *Manager) SetSize(width, height int) {
	n.width = width
	n.height = height
}

func makeTimerCmd(id, gen uint64, duration time.Duration) tea.Cmd {
	return tea.Tick(duration, func(time.Time) tea.Msg {
		return AutoHideMsg{ID: id, Generation: gen}
	})
}

func (n *Manager) Update(msg tea.Msg) (Manager, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		n.width = msg.Width
		n.height = msg.Height
		return *n, nil

	case ShowMsg:
		return n.handleShow(msg)

	case AutoHideMsg:
		return n.handleAutoHide(msg)

	case HideMsg:
		return n.handleHide(msg)

	case DismissMsg:
		return n.removeByID(msg.ID)
	}

	return *n, nil
}

func (n *Manager) handleShow(msg ShowMsg) (Manager, tea.Cmd) {
	id := nextID.Add(1)
	notifType := msg.Type
	// Auto-detect error type for backward compatibility when Type is not set.
	if notifType == TypeSuccess && msg.Text != "" {
		textLower := strings.ToLower(msg.Text)
		if strings.Contains(textLower, "failed") || strings.Contains(textLower, "error") {
			notifType = TypeError
		}
	}

	item := notificationItem{ID: id, Text: msg.Text, Type: notifType}
	n.items = append([]notificationItem{item}, n.items...)
	if len(n.items) > maxVisibleItems {
		n.items = n.items[:maxVisibleItems]
	}

	return *n, makeTimerCmd(id, item.timerGen, notifType.autoHideDuration())
}

func (n *Manager) handleAutoHide(msg AutoHideMsg) (Manager, tea.Cmd) {
	id := msg.ID
	gen := msg.Generation
	newItems := make([]notificationItem, 0, len(n.items))
	for _, item := range n.items {
		if item.ID == id && item.timerGen == gen {
			n.clearItemState(item.ID)
			continue
		}
		newItems = append(newItems, item)
	}
	n.items = newItems
	return *n, nil
}

func (n *Manager) handleHide(msg HideMsg) (Manager, tea.Cmd) {
	if msg.ID == 0 {
		n.items = nil
		n.clearAllState()
		return *n, nil
	}

	return n.removeByID(msg.ID)
}

func (n *Manager) removeByID(id uint64) (Manager, tea.Cmd) {
	if i := n.findItemIndex(id); i >= 0 {
		n.clearItemState(id)
		n.items = slices.Delete(n.items, i, i+1)
	}
	return *n, nil
}

func (n *Manager) clearAllState() {
	n.hoveredID, n.closeHoveredID, n.copiedID = 0, 0, 0
}

func (n *Manager) clearItemState(id uint64) {
	if n.hoveredID == id {
		n.hoveredID = 0
	}
	if n.closeHoveredID == id {
		n.closeHoveredID = 0
	}
	if n.copiedID == id {
		n.copiedID = 0
	}
}

// MarkCopied records that the given notification was copied so View can show a
// transient copied label. The state is cleared when hover moves away or the item
// is removed.
func (n *Manager) MarkCopied(id uint64) Manager {
	n.copiedID = id
	return *n
}

// maxWidth returns the effective maximum width for notification text.
func (n *Manager) maxWidth() int {
	if n.width > 0 {
		screenLimit := max(1, (n.width*2)/5)
		return max(1, min(maxNotificationWidth, min(screenLimit, n.width-notificationPadding*2)))
	}
	return maxNotificationWidth
}

func (n *Manager) View() string {
	if len(n.items) == 0 {
		return ""
	}

	mw := n.maxWidth()
	views := make([]string, 0, len(n.items))
	for _, item := range slices.Backward(n.items) {
		views = append(views, item.render(
			mw,
			n.closeHoveredID == item.ID,
			n.hoveredID == item.ID,
			n.copiedID == item.ID,
		))
	}
	return lipgloss.JoinVertical(lipgloss.Right, views...)
}

func (n *Manager) GetLayer() *lipgloss.Layer {
	if len(n.items) == 0 {
		return nil
	}

	view := n.View()
	row, col := n.position()

	return lipgloss.NewLayer(view).X(col).Y(row)
}

func (n *Manager) position() (row, col int) {
	bounds := n.itemBounds()
	if len(bounds) == 0 {
		return max(0, n.height-notificationPadding), max(0, n.width-notificationPadding)
	}

	viewWidth := 0
	for _, b := range bounds {
		viewWidth = max(viewWidth, b.width)
	}

	row = bounds[0].row
	col = max(0, n.width-viewWidth-notificationPadding)
	return row, col
}

func (n *Manager) Open() bool {
	return len(n.items) > 0
}

type notifBounds struct {
	id     uint64
	row    int
	col    int
	width  int
	height int
	text   string
}

// itemBounds computes screen-space bounds in the same order notifications render.
func (n *Manager) itemBounds() []notifBounds {
	if len(n.items) == 0 || n.width == 0 {
		return nil
	}

	mw := n.maxWidth()
	totalHeight := 0
	for _, item := range n.items {
		totalHeight += lipgloss.Height(item.render(mw, false, false, false))
	}

	row := max(0, n.height-totalHeight-notificationPadding)
	bounds := make([]notifBounds, 0, len(n.items))
	for _, item := range slices.Backward(n.items) {
		view := item.render(mw, false, false, false)
		w := lipgloss.Width(view)
		bounds = append(bounds, notifBounds{
			id:     item.ID,
			row:    row,
			col:    max(0, n.width-w-notificationPadding),
			width:  w,
			height: lipgloss.Height(view),
			text:   item.Text,
		})
		row += lipgloss.Height(view)
	}
	return bounds
}

// CloseButtonHit is kept for callers that still ask the notification layer
// about dismiss hits. Compact status hints have no inline close control.
func (n *Manager) CloseButtonHit(x, y int) (uint64, bool) {
	return 0, false
}

// BodyHit checks whether the coordinates hit the body of a notification and
// returns its ID and text.
func (n *Manager) BodyHit(x, y int) (uint64, string, bool) {
	for _, b := range n.itemBounds() {
		if x < b.col || x >= b.col+b.width || y < b.row || y >= b.row+b.height {
			continue
		}
		return b.id, b.text, true
	}
	return 0, "", false
}

// CopyHit checks whether the coordinates hit the currently-hovered notification
// body and returns its ID and text. The close button is excluded so dismiss
// priority stays separate from click-to-copy behavior.
func (n *Manager) CopyHit(x, y int) (uint64, string, bool) {
	id, text, ok := n.BodyHit(x, y)
	if !ok || id != n.hoveredID || id == n.closeHoveredID {
		return 0, "", false
	}
	return id, text, true
}

// HandleClick checks if the given screen coordinates hit a notification close
// button and returns a dismiss command when they do. Body clicks do not dismiss;
// callers can use CopyHit for additional behavior such as click-to-copy.
func (n *Manager) HandleClick(x, y int) tea.Cmd {
	return nil
}

func (n *Manager) hitTestNotification(x, y int) uint64 {
	for _, b := range n.itemBounds() {
		if x >= b.col && x < b.col+b.width && y >= b.row && y < b.row+b.height {
			return b.id
		}
	}
	return 0
}

func (n *Manager) findItemIndex(id uint64) int {
	for i := range n.items {
		if n.items[i].ID == id {
			return i
		}
	}
	return -1
}

// HandleMouseMotion updates hover state. Entering an auto-hide notification
// invalidates its pending timer; leaving restarts a generation-safe timer.
func (n *Manager) HandleMouseMotion(x, y int) (Manager, tea.Cmd) {
	newHoveredID := n.hitTestNotification(x, y)
	newCloseHoveredID, _ := n.CloseButtonHit(x, y)

	var cmd tea.Cmd
	if newHoveredID != n.hoveredID {
		// Clear copied state when the pointer leaves or enters another notification.
		n.copiedID = 0

		if n.hoveredID != 0 {
			if idx := n.findItemIndex(n.hoveredID); idx >= 0 {
				cmd = makeTimerCmd(n.items[idx].ID, n.items[idx].timerGen, n.items[idx].Type.autoHideDuration())
			}
		}

		if newHoveredID != 0 {
			if idx := n.findItemIndex(newHoveredID); idx >= 0 {
				n.items[idx].timerGen++
			}
		}

		n.hoveredID = newHoveredID
	}

	n.closeHoveredID = newCloseHoveredID
	return *n, cmd
}

func truncateDisplay(s string, maxWidth int) string {
	if maxWidth <= 0 || runewidth.StringWidth(s) <= maxWidth {
		return s
	}
	if maxWidth <= 1 {
		return "…"
	}
	limit := maxWidth - 1
	width := 0
	var b strings.Builder
	for _, r := range s {
		w := runewidth.RuneWidth(r)
		if width+w > limit {
			break
		}
		b.WriteRune(r)
		width += w
	}
	return strings.TrimRight(b.String(), " ") + "…"
}
