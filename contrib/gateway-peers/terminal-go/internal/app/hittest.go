package app

import tea "github.com/charmbracelet/bubbletea"

// MouseTarget identifies which UI region a mouse event landed on.
type MouseTarget int

const (
	TargetNone      MouseTarget = iota
	TargetViewport              // main message area
	TargetScrollbar             // scrollbar column (1 char wide)
	TargetInput                 // input area
	TargetStatusBar             // bottom status line
)

// HitTestResult carries the target region and coordinates relative to
// that region's origin.
type HitTestResult struct {
	Target MouseTarget
	X, Y   int // coordinates relative to the target area
}

// LayoutInfo captures the current layout dimensions needed for hit
// testing. All coordinates are absolute (screen-relative).
type LayoutInfo struct {
	ViewportX, ViewportY          int
	ViewportWidth, ViewportHeight int
	ScrollbarX                    int // -1 if scrollbar not visible
	InputY, InputHeight           int
	StatusBarY                    int
}

// HitTest determines which UI region contains the point (x, y) and
// returns the target with coordinates relative to that region.
func HitTest(x, y int, layout LayoutInfo) HitTestResult {
	// Status bar: single line at StatusBarY
	if y == layout.StatusBarY {
		return HitTestResult{
			Target: TargetStatusBar,
			X:      x,
			Y:      0,
		}
	}

	// Input area: starts at InputY, spans InputHeight lines
	if y >= layout.InputY && y < layout.InputY+layout.InputHeight {
		return HitTestResult{
			Target: TargetInput,
			X:      x,
			Y:      y - layout.InputY,
		}
	}

	// Scrollbar column (1 char wide)
	if layout.ScrollbarX >= 0 && x == layout.ScrollbarX &&
		y >= layout.ViewportY && y < layout.ViewportY+layout.ViewportHeight {
		return HitTestResult{
			Target: TargetScrollbar,
			X:      0,
			Y:      y - layout.ViewportY,
		}
	}

	// Viewport area
	if x >= layout.ViewportX && x < layout.ViewportX+layout.ViewportWidth &&
		y >= layout.ViewportY && y < layout.ViewportY+layout.ViewportHeight {
		return HitTestResult{
			Target: TargetViewport,
			X:      x - layout.ViewportX,
			Y:      y - layout.ViewportY,
		}
	}

	return HitTestResult{Target: TargetNone, X: x, Y: y}
}

// ExtractMouseCoords extracts x, y coordinates from a bubbletea mouse message.
func ExtractMouseCoords(msg tea.Msg) (x, y int, ok bool) {
	if m, isMouseMsg := msg.(tea.MouseMsg); isMouseMsg {
		return m.X, m.Y, true
	}
	return 0, 0, false
}
