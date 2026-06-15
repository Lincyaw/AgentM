package app

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

// Standard test layout: 80x24 terminal
// Viewport: (0,0) to (78, 19) — 79 cols wide, 20 rows tall
// Scrollbar: column 79, rows 0-19
// Status bar: row 20
// Input: rows 21-23 (3 lines)
func testLayout() LayoutInfo {
	return LayoutInfo{
		ViewportX:      0,
		ViewportY:      0,
		ViewportWidth:  79,
		ViewportHeight: 20,
		ScrollbarX:     79,
		InputY:         21,
		InputHeight:    3,
		StatusBarY:     20,
	}
}

func TestHitTestViewport(t *testing.T) {
	layout := testLayout()

	tests := []struct {
		name string
		x, y int
	}{
		{"top-left", 0, 0},
		{"center", 40, 10},
		{"bottom-right of viewport", 78, 19},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HitTest(tt.x, tt.y, layout)
			if result.Target != TargetViewport {
				t.Errorf("expected TargetViewport, got %d", result.Target)
			}
			if result.X != tt.x-layout.ViewportX {
				t.Errorf("relX = %d, want %d", result.X, tt.x-layout.ViewportX)
			}
			if result.Y != tt.y-layout.ViewportY {
				t.Errorf("relY = %d, want %d", result.Y, tt.y-layout.ViewportY)
			}
		})
	}
}

func TestHitTestScrollbar(t *testing.T) {
	layout := testLayout()

	result := HitTest(79, 5, layout)
	if result.Target != TargetScrollbar {
		t.Errorf("expected TargetScrollbar, got %d", result.Target)
	}
	if result.X != 0 {
		t.Errorf("scrollbar relX = %d, want 0", result.X)
	}
	if result.Y != 5 {
		t.Errorf("scrollbar relY = %d, want 5", result.Y)
	}
}

func TestHitTestScrollbarNotVisible(t *testing.T) {
	layout := testLayout()
	layout.ScrollbarX = -1

	result := HitTest(79, 5, layout)
	if result.Target == TargetScrollbar {
		t.Error("scrollbar should not be a target when ScrollbarX is -1")
	}
}

func TestHitTestStatusBar(t *testing.T) {
	layout := testLayout()

	result := HitTest(40, 20, layout)
	if result.Target != TargetStatusBar {
		t.Errorf("expected TargetStatusBar, got %d", result.Target)
	}
	if result.Y != 0 {
		t.Errorf("statusbar relY = %d, want 0", result.Y)
	}
}

func TestHitTestInput(t *testing.T) {
	layout := testLayout()

	tests := []struct {
		name string
		x, y int
		relY int
	}{
		{"first line", 10, 21, 0},
		{"second line", 5, 22, 1},
		{"third line", 0, 23, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HitTest(tt.x, tt.y, layout)
			if result.Target != TargetInput {
				t.Errorf("expected TargetInput, got %d", result.Target)
			}
			if result.Y != tt.relY {
				t.Errorf("relY = %d, want %d", result.Y, tt.relY)
			}
		})
	}
}

func TestHitTestNone(t *testing.T) {
	layout := testLayout()

	// Beyond the terminal bounds
	result := HitTest(100, 100, layout)
	if result.Target != TargetNone {
		t.Errorf("expected TargetNone for out-of-bounds, got %d", result.Target)
	}
}

func TestHitTestPriority(t *testing.T) {
	// Verify that scrollbar takes priority over viewport at the scrollbar column.
	layout := testLayout()

	result := HitTest(79, 10, layout)
	if result.Target != TargetScrollbar {
		t.Errorf("scrollbar should win over viewport at scrollbar column, got %d", result.Target)
	}
}

func TestExtractMouseCoords(t *testing.T) {
	msg := tea.MouseMsg{
		X:      42,
		Y:      17,
		Action: tea.MouseActionPress,
		Button: tea.MouseButtonLeft,
	}
	x, y, ok := ExtractMouseCoords(msg)
	if !ok {
		t.Fatal("expected ok=true for MouseMsg")
	}
	if x != 42 || y != 17 {
		t.Errorf("coords = (%d, %d), want (42, 17)", x, y)
	}
}

func TestExtractMouseCoordsNonMouse(t *testing.T) {
	msg := tea.KeyMsg{Type: tea.KeyEnter}
	_, _, ok := ExtractMouseCoords(msg)
	if ok {
		t.Error("expected ok=false for non-mouse message")
	}
}
