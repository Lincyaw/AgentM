package components

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// ScrollbarWidth is the intrinsic width of the scrollbar in terminal columns.
const ScrollbarWidth = 1

// Scrollbar renders a 1-character-wide vertical scrollbar with a proportionally
// sized thumb. It supports drag, track-click page up/down, and reports the
// resulting scroll offset back to the caller.
type Scrollbar struct {
	totalLines   int
	visibleLines int
	scrollOffset int

	dragging     bool
	dragStartY   int
	dragStartOff int
}

// NewScrollbar returns a zero-value Scrollbar (invisible until SetContent is called).
func NewScrollbar() Scrollbar {
	return Scrollbar{}
}

// SetContent updates the scrollbar state to reflect the current viewport.
func (s *Scrollbar) SetContent(totalLines, visibleLines, scrollOffset int) {
	s.totalLines = totalLines
	s.visibleLines = visibleLines
	s.scrollOffset = clamp(scrollOffset, 0, s.maxOffset(totalLines, visibleLines))
}

// Visible returns true when content exceeds the viewport (scrollbar needed).
func (s *Scrollbar) Visible() bool {
	return s.totalLines > s.visibleLines && s.visibleLines > 0
}

// View renders the scrollbar as a column of `height` characters.
// Returns an empty string when no scrollbar is needed.
func (s *Scrollbar) View(height int, th *theme.Theme) string {
	if !s.Visible() || height <= 0 {
		return ""
	}

	thumbTop, thumbHeight := s.thumbGeometry(height)
	lines := make([]string, height)

	for i := range height {
		if i >= thumbTop && i < thumbTop+thumbHeight {
			lines[i] = th.ScrollThumb.Render("█")
		} else {
			lines[i] = th.ScrollTrack.Render("░")
		}
	}

	return strings.Join(lines, "\n")
}

// HandleMouse processes a mouse event and returns the new scroll offset and
// whether the event was consumed. The height parameter must match the height
// passed to View so geometry calculations agree. The xOffset/yOffset locate
// the scrollbar column in absolute terminal coordinates.
func (s *Scrollbar) HandleMouse(msg tea.MouseMsg, height, xOffset, yOffset int) (scrollTo int, handled bool) {
	me := tea.MouseEvent(msg)

	switch me.Action {
	case tea.MouseActionRelease:
		if me.Button == tea.MouseButtonLeft && s.dragging {
			s.dragging = false
			return s.scrollOffset, true
		}
		return s.scrollOffset, false

	case tea.MouseActionMotion:
		if !s.dragging {
			return s.scrollOffset, false
		}
		s.updateDrag(me.Y, height)
		return s.scrollOffset, true

	case tea.MouseActionPress:
		if me.Button != tea.MouseButtonLeft {
			return s.scrollOffset, false
		}
		// Only handle clicks on the scrollbar column itself.
		if me.X < xOffset || me.X >= xOffset+ScrollbarWidth {
			return s.scrollOffset, false
		}
		relY := me.Y - yOffset
		if relY < 0 || relY >= height {
			return s.scrollOffset, false
		}

		thumbTop, thumbHeight := s.thumbGeometry(height)

		switch {
		case relY >= thumbTop && relY < thumbTop+thumbHeight:
			// Click on thumb -- start drag.
			s.dragging = true
			s.dragStartY = me.Y
			s.dragStartOff = s.scrollOffset
		case relY < thumbTop:
			// Track above thumb -- page up.
			s.scrollOffset = clamp(s.scrollOffset-s.visibleLines, 0, s.maxOffsetCurrent())
		default:
			// Track below thumb -- page down.
			s.scrollOffset = clamp(s.scrollOffset+s.visibleLines, 0, s.maxOffsetCurrent())
		}
		return s.scrollOffset, true
	}

	return s.scrollOffset, false
}

// IsDragging reports whether a thumb drag is in progress.
func (s *Scrollbar) IsDragging() bool {
	return s.dragging
}

// --- internal helpers ---

func (s *Scrollbar) thumbGeometry(height int) (top, size int) {
	if s.totalLines <= s.visibleLines || height <= 0 {
		return 0, 0
	}
	thumbH := max(1, (height*s.visibleLines)/s.totalLines)

	maxScroll := s.maxOffsetCurrent()
	if maxScroll == 0 {
		return 0, thumbH
	}
	scrollableTrack := height - thumbH
	thumbTop := (s.scrollOffset * scrollableTrack) / maxScroll
	return thumbTop, thumbH
}

func (s *Scrollbar) updateDrag(mouseY int, height int) {
	_, thumbH := s.thumbGeometry(height)
	scrollableTrack := height - thumbH
	if scrollableTrack <= 0 {
		return
	}
	maxScroll := s.maxOffsetCurrent()
	deltaY := mouseY - s.dragStartY
	deltaScroll := (deltaY * maxScroll) / scrollableTrack
	s.scrollOffset = clamp(s.dragStartOff+deltaScroll, 0, maxScroll)
}

func (s *Scrollbar) maxOffsetCurrent() int {
	return s.maxOffset(s.totalLines, s.visibleLines)
}

func (s *Scrollbar) maxOffset(total, visible int) int {
	if total <= visible {
		return 0
	}
	return total - visible
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
