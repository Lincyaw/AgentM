package components

import (
	"strings"
	"time"
	"unicode"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/util"
)

const (
	// multiClickWindow is the maximum interval between consecutive clicks
	// for them to count as double/triple clicks.
	multiClickWindow = 400 * time.Millisecond
)

// ClickType classifies a mouse click for selection purposes.
type ClickType int

const (
	SingleClick ClickType = iota
	DoubleClick
	TripleClick
)

// Selection tracks text selection state driven by mouse events.
type Selection struct {
	active bool
	// Start and end are in terminal coordinates. When dragging backwards
	// start may be after end; use Normalized to get ordered bounds.
	startX, startY int
	endX, endY     int

	clickType ClickType
	dragging  bool

	lastClickTime      time.Time
	lastClickX, lastClickY int

	text string // selected text, populated on finalize
}

// NewSelection returns a zero-value Selection ready for use.
func NewSelection() *Selection {
	return &Selection{}
}

// HandleMouse processes a mouse event and returns true if the selection state
// changed (the caller should re-render). getText is called with a terminal
// line number and must return the plain text content of that line (no ANSI).
func (s *Selection) HandleMouse(msg tea.MouseMsg, getText func(line int) string) bool {
	switch msg.Action {
	case tea.MouseActionPress:
		if msg.Button != tea.MouseButtonLeft {
			return false
		}
		ct := s.detectClickType(msg.X, msg.Y)
		s.clickType = ct

		switch ct {
		case TripleClick:
			line := getText(msg.Y)
			s.active = true
			s.startX = 0
			s.startY = msg.Y
			s.endX = len([]rune(line))
			s.endY = msg.Y
			s.dragging = false
			s.text = line
			util.CopyToClipboard(s.text)
			return true

		case DoubleClick:
			line := getText(msg.Y)
			start, end := wordBounds(line, msg.X)
			s.active = true
			s.startX = start
			s.startY = msg.Y
			s.endX = end
			s.endY = msg.Y
			s.dragging = false
			if start < end && start < len([]rune(line)) {
				runes := []rune(line)
				if end > len(runes) {
					end = len(runes)
				}
				s.text = string(runes[start:end])
			} else {
				s.text = ""
			}
			util.CopyToClipboard(s.text)
			return true

		default: // SingleClick
			s.active = true
			s.startX = msg.X
			s.startY = msg.Y
			s.endX = msg.X
			s.endY = msg.Y
			s.dragging = true
			s.text = ""
			return true
		}

	case tea.MouseActionMotion:
		if !s.dragging {
			return false
		}
		if msg.X == s.endX && msg.Y == s.endY {
			return false
		}
		s.endX = msg.X
		s.endY = msg.Y
		return true

	case tea.MouseActionRelease:
		if !s.dragging {
			return false
		}
		s.dragging = false
		s.endX = msg.X
		s.endY = msg.Y

		// Build selected text
		s.text = s.extractText(getText)
		if s.text == "" {
			s.active = false
			return true
		}
		util.CopyToClipboard(s.text)
		return true
	}

	return false
}

// extractText reads the selected region from getText.
func (s *Selection) extractText(getText func(line int) string) string {
	startX, startY, endX, endY := s.Normalized()
	if startY == endY && startX == endX {
		return ""
	}

	var sb strings.Builder
	for y := startY; y <= endY; y++ {
		line := getText(y)
		runes := []rune(line)
		lineLen := len(runes)

		colStart := 0
		colEnd := lineLen
		if y == startY {
			colStart = startX
		}
		if y == endY {
			colEnd = endX
		}
		if colStart > lineLen {
			colStart = lineLen
		}
		if colEnd > lineLen {
			colEnd = lineLen
		}
		if colStart > colEnd {
			colStart = colEnd
		}

		if y > startY {
			sb.WriteByte('\n')
		}
		sb.WriteString(string(runes[colStart:colEnd]))
	}
	return sb.String()
}

// detectClickType determines whether this click is a single, double, or
// triple click based on timing and proximity to the previous click.
func (s *Selection) detectClickType(x, y int) ClickType {
	now := time.Now()
	sameSpot := x == s.lastClickX && y == s.lastClickY
	withinWindow := !s.lastClickTime.IsZero() && now.Sub(s.lastClickTime) < multiClickWindow

	s.lastClickTime = now
	s.lastClickX = x
	s.lastClickY = y

	var ct ClickType
	if sameSpot && withinWindow {
		switch s.clickType {
		case SingleClick:
			ct = DoubleClick
		case DoubleClick:
			ct = TripleClick
		default:
			ct = SingleClick
		}
	} else {
		ct = SingleClick
	}
	s.clickType = ct
	return ct
}

// IsActive returns true when a selection is visible.
func (s *Selection) IsActive() bool {
	return s.active
}

// SelectedText returns the text captured by the last completed selection.
func (s *Selection) SelectedText() string {
	return s.text
}

// Clear removes the current selection.
func (s *Selection) Clear() {
	s.active = false
	s.dragging = false
	s.text = ""
	s.startX = 0
	s.startY = 0
	s.endX = 0
	s.endY = 0
}

// Normalized returns the selection bounds with start <= end.
func (s *Selection) Normalized() (startX, startY, endX, endY int) {
	sx, sy, ex, ey := s.startX, s.startY, s.endX, s.endY
	if sy > ey || (sy == ey && sx > ex) {
		sx, ex = ex, sx
		sy, ey = ey, sy
	}
	return sx, sy, ex, ey
}

// Contains returns true if the terminal position (x, y) falls within the
// selection range.
func (s *Selection) Contains(x, y int) bool {
	if !s.active {
		return false
	}
	sx, sy, ex, ey := s.Normalized()
	if y < sy || y > ey {
		return false
	}
	if sy == ey {
		return x >= sx && x < ex
	}
	if y == sy {
		return x >= sx
	}
	if y == ey {
		return x < ex
	}
	return true
}

// HighlightLine wraps selected characters in the given line with reverse-video
// ANSI escape sequences. lineY is the terminal row of this line. The input
// line should be plain text (no existing ANSI).
func (s *Selection) HighlightLine(line string, lineY int) string {
	if !s.active {
		return line
	}
	sx, sy, ex, ey := s.Normalized()
	if lineY < sy || lineY > ey {
		return line
	}

	runes := []rune(line)
	lineLen := len(runes)

	colStart := 0
	colEnd := lineLen
	if lineY == sy {
		colStart = sx
	}
	if lineY == ey {
		colEnd = ex
	}
	if colStart >= lineLen || colEnd <= 0 || colStart >= colEnd {
		return line
	}
	if colStart < 0 {
		colStart = 0
	}
	if colEnd > lineLen {
		colEnd = lineLen
	}

	const reverseOn = "\033[7m"
	const reverseOff = "\033[27m"

	var sb strings.Builder
	sb.Grow(len(line) + len(reverseOn) + len(reverseOff))
	sb.WriteString(string(runes[:colStart]))
	sb.WriteString(reverseOn)
	sb.WriteString(string(runes[colStart:colEnd]))
	sb.WriteString(reverseOff)
	sb.WriteString(string(runes[colEnd:]))
	return sb.String()
}

// wordBounds returns the [start, end) rune indices of the word at column col
// in the given line. A word is a contiguous run of letters, digits, or
// underscores.
func wordBounds(line string, col int) (int, int) {
	runes := []rune(line)
	if len(runes) == 0 {
		return 0, 0
	}
	if col < 0 {
		col = 0
	}
	if col >= len(runes) {
		col = len(runes) - 1
	}

	isWord := isWordRune(runes[col])

	start := col
	for start > 0 && isWordRune(runes[start-1]) == isWord {
		start--
	}
	end := col
	for end < len(runes)-1 && isWordRune(runes[end+1]) == isWord {
		end++
	}
	return start, end + 1
}

// isWordRune returns true for runes that form a "word" for double-click
// selection: letters, digits, and underscores.
func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_'
}
