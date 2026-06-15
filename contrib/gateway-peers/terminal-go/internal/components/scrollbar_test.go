package components

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

func TestScrollbar_NotVisibleWhenContentFits(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(10, 20, 0) // 10 total, 20 visible
	if s.Visible() {
		t.Error("expected scrollbar to be invisible when content fits viewport")
	}
	view := s.View(20, theme.DarkTheme())
	if view != "" {
		t.Errorf("expected empty view, got %q", view)
	}
}

func TestScrollbar_VisibleWhenContentExceedsViewport(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(100, 20, 0)
	if !s.Visible() {
		t.Error("expected scrollbar to be visible when content exceeds viewport")
	}
}

func TestScrollbar_ViewHeight(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(100, 20, 0)
	th := theme.DarkTheme()
	view := s.View(20, th)
	lines := strings.Split(view, "\n")
	if len(lines) != 20 {
		t.Errorf("expected 20 lines, got %d", len(lines))
	}
}

func TestScrollbar_ThumbSizeProportional(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(100, 50, 0) // half visible
	th := theme.DarkTheme()
	view := s.View(20, th)
	lines := strings.Split(view, "\n")

	thumbCount := 0
	for _, line := range lines {
		if strings.Contains(line, "█") {
			thumbCount++
		}
	}
	// thumb = max(1, 20*50/100) = 10
	if thumbCount != 10 {
		t.Errorf("expected thumb size 10, got %d", thumbCount)
	}
}

func TestScrollbar_MinimumThumbSize(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(10000, 10, 0) // tiny fraction visible
	th := theme.DarkTheme()
	view := s.View(10, th)
	lines := strings.Split(view, "\n")

	thumbCount := 0
	for _, line := range lines {
		if strings.Contains(line, "█") {
			thumbCount++
		}
	}
	if thumbCount < 1 {
		t.Error("thumb size must be at least 1")
	}
}

func TestScrollbar_ThumbPositionAtBottom(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(100, 20, 80) // scrolled to bottom
	th := theme.DarkTheme()
	view := s.View(20, th)
	lines := strings.Split(view, "\n")

	// Thumb should be at the bottom -- last line should be thumb.
	lastLine := lines[len(lines)-1]
	if !strings.Contains(lastLine, "█") {
		t.Error("expected thumb at the bottom when scrolled to end")
	}
}

func TestScrollbar_TrackClickPageUp(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(100, 20, 50) // scrolled to middle

	// Click on track above the thumb (y=0 relative to scrollbar).
	msg := tea.MouseMsg(tea.MouseEvent{
		X:      0,
		Y:      0,
		Action: tea.MouseActionPress,
		Button: tea.MouseButtonLeft,
	})
	scrollTo, handled := s.HandleMouse(msg, 20, 0, 0)
	if !handled {
		t.Error("expected track click above thumb to be handled")
	}
	// Page up: 50 - 20 = 30
	if scrollTo != 30 {
		t.Errorf("expected scrollTo=30 after page up, got %d", scrollTo)
	}
}

func TestScrollbar_TrackClickPageDown(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(100, 20, 0) // at top

	// Click on track below the thumb (y=19, bottom of track).
	msg := tea.MouseMsg(tea.MouseEvent{
		X:      0,
		Y:      19,
		Action: tea.MouseActionPress,
		Button: tea.MouseButtonLeft,
	})
	scrollTo, handled := s.HandleMouse(msg, 20, 0, 0)
	if !handled {
		t.Error("expected track click below thumb to be handled")
	}
	// Page down: 0 + 20 = 20
	if scrollTo != 20 {
		t.Errorf("expected scrollTo=20 after page down, got %d", scrollTo)
	}
}

func TestScrollbar_DragStartAndRelease(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(100, 20, 0) // at top

	// The thumb is at the top. Click on it to start drag.
	msg := tea.MouseMsg(tea.MouseEvent{
		X:      0,
		Y:      0,
		Action: tea.MouseActionPress,
		Button: tea.MouseButtonLeft,
	})
	_, handled := s.HandleMouse(msg, 20, 0, 0)
	if !handled {
		t.Error("expected thumb click to be handled")
	}
	if !s.IsDragging() {
		t.Error("expected dragging to be true after thumb click")
	}

	// Release.
	releaseMsg := tea.MouseMsg(tea.MouseEvent{
		X:      0,
		Y:      5,
		Action: tea.MouseActionRelease,
		Button: tea.MouseButtonLeft,
	})
	_, handled = s.HandleMouse(releaseMsg, 20, 0, 0)
	if !handled {
		t.Error("expected release to be handled during drag")
	}
	if s.IsDragging() {
		t.Error("expected dragging to be false after release")
	}
}

func TestScrollbar_ClickOutsideColumn(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(100, 20, 0)

	// Click at x=5, but scrollbar is at xOffset=10.
	msg := tea.MouseMsg(tea.MouseEvent{
		X:      5,
		Y:      0,
		Action: tea.MouseActionPress,
		Button: tea.MouseButtonLeft,
	})
	_, handled := s.HandleMouse(msg, 20, 10, 0)
	if handled {
		t.Error("expected click outside scrollbar column to be ignored")
	}
}

func TestScrollbar_SetContentClampsOffset(t *testing.T) {
	s := NewScrollbar()
	s.SetContent(100, 20, 200) // offset way beyond max
	if !s.Visible() {
		t.Error("expected visible")
	}
	// After clamping, offset should be 80 (100 - 20).
	th := theme.DarkTheme()
	view := s.View(20, th)
	lines := strings.Split(view, "\n")
	// Thumb should be at the bottom.
	lastLine := lines[len(lines)-1]
	if !strings.Contains(lastLine, "█") {
		t.Error("expected thumb at bottom after clamped offset")
	}
}
