package components

import (
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// mouseMsg is a helper to build tea.MouseMsg without repeating all fields.
func mouseMsg(x, y int, action tea.MouseAction, button tea.MouseButton) tea.MouseMsg {
	return tea.MouseMsg{X: x, Y: y, Action: action, Button: button}
}

func TestNewSelection(t *testing.T) {
	s := NewSelection()
	if s.IsActive() {
		t.Error("new selection should not be active")
	}
	if s.SelectedText() != "" {
		t.Error("new selection should have empty text")
	}
}

func TestSingleClickStartsSelection(t *testing.T) {
	s := NewSelection()
	getText := func(line int) string { return "hello world" }

	changed := s.HandleMouse(mouseMsg(2, 0, tea.MouseActionPress, tea.MouseButtonLeft), getText)
	if !changed {
		t.Error("press should report a change")
	}
	if !s.IsActive() {
		t.Error("selection should be active after press")
	}
}

func TestDragSelection(t *testing.T) {
	s := NewSelection()
	lines := []string{"hello world"}
	getText := func(line int) string {
		if line >= 0 && line < len(lines) {
			return lines[line]
		}
		return ""
	}

	s.HandleMouse(mouseMsg(0, 0, tea.MouseActionPress, tea.MouseButtonLeft), getText)
	s.HandleMouse(mouseMsg(5, 0, tea.MouseActionMotion, tea.MouseButtonLeft), getText)
	s.HandleMouse(mouseMsg(5, 0, tea.MouseActionRelease, tea.MouseButtonNone), getText)

	if !s.IsActive() {
		t.Error("selection should be active after drag")
	}
	if s.SelectedText() != "hello" {
		t.Errorf("expected %q, got %q", "hello", s.SelectedText())
	}
}

func TestDoubleClickSelectsWord(t *testing.T) {
	s := NewSelection()
	getText := func(line int) string { return "hello world" }

	// First click
	s.HandleMouse(mouseMsg(2, 0, tea.MouseActionPress, tea.MouseButtonLeft), getText)
	s.HandleMouse(mouseMsg(2, 0, tea.MouseActionRelease, tea.MouseButtonNone), getText)

	// Second click within window at same position
	s.lastClickTime = time.Now()
	s.HandleMouse(mouseMsg(2, 0, tea.MouseActionPress, tea.MouseButtonLeft), getText)

	if s.SelectedText() != "hello" {
		t.Errorf("double-click: expected %q, got %q", "hello", s.SelectedText())
	}
}

func TestTripleClickSelectsLine(t *testing.T) {
	s := NewSelection()
	line := "hello world"
	getText := func(y int) string { return line }

	for i := 0; i < 3; i++ {
		s.HandleMouse(mouseMsg(3, 0, tea.MouseActionPress, tea.MouseButtonLeft), getText)
		s.HandleMouse(mouseMsg(3, 0, tea.MouseActionRelease, tea.MouseButtonNone), getText)
		s.lastClickTime = time.Now()
	}

	if s.SelectedText() != line {
		t.Errorf("triple-click: expected %q, got %q", line, s.SelectedText())
	}
}

func TestClickTypeDetection(t *testing.T) {
	s := NewSelection()

	ct := s.detectClickType(5, 5)
	if ct != SingleClick {
		t.Errorf("expected SingleClick, got %d", ct)
	}

	s.lastClickTime = time.Now()
	ct = s.detectClickType(5, 5)
	if ct != DoubleClick {
		t.Errorf("expected DoubleClick, got %d", ct)
	}

	s.lastClickTime = time.Now()
	ct = s.detectClickType(5, 5)
	if ct != TripleClick {
		t.Errorf("expected TripleClick, got %d", ct)
	}

	s.lastClickTime = time.Now()
	ct = s.detectClickType(5, 5)
	if ct != SingleClick {
		t.Errorf("expected SingleClick after triple, got %d", ct)
	}
}

func TestClickTypeResetOnMove(t *testing.T) {
	s := NewSelection()

	s.detectClickType(5, 5)
	s.lastClickTime = time.Now()

	ct := s.detectClickType(10, 5)
	if ct != SingleClick {
		t.Errorf("expected SingleClick after position change, got %d", ct)
	}
}

func TestClickTypeResetOnTimeout(t *testing.T) {
	s := NewSelection()

	s.detectClickType(5, 5)
	s.lastClickTime = time.Now().Add(-multiClickWindow - time.Millisecond)

	ct := s.detectClickType(5, 5)
	if ct != SingleClick {
		t.Errorf("expected SingleClick after timeout, got %d", ct)
	}
}

func TestContains(t *testing.T) {
	s := NewSelection()
	s.active = true
	s.startX = 2
	s.startY = 1
	s.endX = 8
	s.endY = 1

	tests := []struct {
		x, y int
		want bool
	}{
		{2, 1, true},
		{5, 1, true},
		{7, 1, true},
		{8, 1, false},
		{1, 1, false},
		{5, 0, false},
		{5, 2, false},
	}
	for _, tt := range tests {
		if got := s.Contains(tt.x, tt.y); got != tt.want {
			t.Errorf("Contains(%d,%d) = %v, want %v", tt.x, tt.y, got, tt.want)
		}
	}
}

func TestContainsMultiLine(t *testing.T) {
	s := NewSelection()
	s.active = true
	s.startX = 5
	s.startY = 1
	s.endX = 3
	s.endY = 3

	if !s.Contains(5, 1) {
		t.Error("expected (5,1) in selection")
	}
	if !s.Contains(100, 1) {
		t.Error("expected (100,1) in selection (rest of start line)")
	}
	if s.Contains(4, 1) {
		t.Error("expected (4,1) NOT in selection")
	}
	if !s.Contains(0, 2) {
		t.Error("expected (0,2) in selection")
	}
	if !s.Contains(2, 3) {
		t.Error("expected (2,3) in selection")
	}
	if s.Contains(3, 3) {
		t.Error("expected (3,3) NOT in selection (exclusive end)")
	}
}

func TestHighlightLine(t *testing.T) {
	s := NewSelection()
	s.active = true
	s.startX = 6
	s.startY = 0
	s.endX = 11
	s.endY = 0

	result := s.HighlightLine("hello world", 0)
	expected := "hello \033[7mworld\033[27m"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestHighlightLineNoSelection(t *testing.T) {
	s := NewSelection()
	line := "hello world"
	result := s.HighlightLine(line, 0)
	if result != line {
		t.Errorf("expected unchanged line, got %q", result)
	}
}

func TestClear(t *testing.T) {
	s := NewSelection()
	s.active = true
	s.text = "test"
	s.startX = 1
	s.endX = 5

	s.Clear()
	if s.IsActive() {
		t.Error("expected inactive after Clear")
	}
	if s.SelectedText() != "" {
		t.Error("expected empty text after Clear")
	}
}

func TestWordBounds(t *testing.T) {
	tests := []struct {
		line      string
		col       int
		wantStart int
		wantEnd   int
	}{
		{"hello world", 0, 0, 5},
		{"hello world", 4, 0, 5},
		{"hello world", 5, 5, 6},
		{"hello world", 6, 6, 11},
		{"foo_bar baz", 3, 0, 7},
		{"abc 123 xyz", 5, 4, 7},
		{"", 0, 0, 0},
	}
	for _, tt := range tests {
		start, end := wordBounds(tt.line, tt.col)
		if start != tt.wantStart || end != tt.wantEnd {
			t.Errorf("wordBounds(%q, %d) = (%d,%d), want (%d,%d)",
				tt.line, tt.col, start, end, tt.wantStart, tt.wantEnd)
		}
	}
}

func TestRightClickIgnored(t *testing.T) {
	s := NewSelection()
	getText := func(line int) string { return "test" }

	changed := s.HandleMouse(mouseMsg(0, 0, tea.MouseActionPress, tea.MouseButtonRight), getText)
	if changed {
		t.Error("right click should not change selection")
	}
	if s.IsActive() {
		t.Error("right click should not activate selection")
	}
}

func TestMotionWithoutDragIgnored(t *testing.T) {
	s := NewSelection()
	getText := func(line int) string { return "test" }

	changed := s.HandleMouse(mouseMsg(5, 0, tea.MouseActionMotion, tea.MouseButtonNone), getText)
	if changed {
		t.Error("motion without drag should not change selection")
	}
}

func TestEmptySelectionDeactivates(t *testing.T) {
	s := NewSelection()
	getText := func(line int) string { return "hello" }

	s.HandleMouse(mouseMsg(2, 0, tea.MouseActionPress, tea.MouseButtonLeft), getText)
	s.HandleMouse(mouseMsg(2, 0, tea.MouseActionRelease, tea.MouseButtonNone), getText)

	if s.IsActive() {
		t.Error("empty selection (click without drag) should deactivate")
	}
}
