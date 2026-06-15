package components

import (
	"regexp"
	"strings"
)

// urlPattern matches http:// and https:// URLs, stopping at whitespace or
// common delimiters. The backtick is included as a stop character because
// it is commonly used as a markdown delimiter.
var urlPattern = regexp.MustCompile("https?://[^\\s)\\]}>'\"`]+")

// URLSpan describes a detected URL and its position within a line.
type URLSpan struct {
	URL    string
	StartX int // inclusive display column
	EndX   int // exclusive display column
	LineY  int
}

// URLDetector scans rendered lines for URLs and supports hover tracking.
type URLDetector struct {
	spans   []URLSpan
	hovered *URLSpan
}

// NewURLDetector returns a ready-to-use detector.
func NewURLDetector() *URLDetector {
	return &URLDetector{}
}

// Detect scans the given lines for URLs and returns all found spans. The
// spans are stored internally for subsequent HitTest / highlight calls.
func (d *URLDetector) Detect(lines []string) []URLSpan {
	d.spans = d.spans[:0]
	for y, line := range lines {
		locs := urlPattern.FindAllStringIndex(line, -1)
		for _, loc := range locs {
			// loc offsets are byte positions; convert to rune (column) positions
			// so hit-testing works with terminal coordinates.
			startCol := runeCount(line[:loc[0]])
			endCol := runeCount(line[:loc[1]])
			d.spans = append(d.spans, URLSpan{
				URL:    line[loc[0]:loc[1]],
				StartX: startCol,
				EndX:   endCol,
				LineY:  y,
			})
		}
	}
	return d.spans
}

// HitTest returns the URLSpan at terminal position (x, y), or nil if no
// URL is at that position.
func (d *URLDetector) HitTest(x, y int) *URLSpan {
	for i := range d.spans {
		s := &d.spans[i]
		if s.LineY == y && x >= s.StartX && x < s.EndX {
			return s
		}
	}
	return nil
}

// SetHovered sets the currently hovered span (may be nil).
func (d *URLDetector) SetHovered(span *URLSpan) {
	d.hovered = span
}

// HoveredURL returns the URL string of the hovered span, or "".
func (d *URLDetector) HoveredURL() string {
	if d.hovered == nil {
		return ""
	}
	return d.hovered.URL
}

// HighlightURLs applies ANSI underline to detected URLs in the given line.
// The hovered URL (if any) gets bold+underline instead. lineY identifies
// which line is being rendered so only matching spans are styled.
func (d *URLDetector) HighlightURLs(line string, lineY int) string {
	// Collect spans for this line, sorted by StartX (Detect already yields
	// them in order).
	var lineSpans []*URLSpan
	for i := range d.spans {
		if d.spans[i].LineY == lineY {
			lineSpans = append(lineSpans, &d.spans[i])
		}
	}
	if len(lineSpans) == 0 {
		return line
	}

	runes := []rune(line)
	var sb strings.Builder
	sb.Grow(len(line) + len(lineSpans)*20)

	pos := 0
	for _, span := range lineSpans {
		if span.StartX > pos {
			sb.WriteString(string(runes[pos:span.StartX]))
		}
		end := span.EndX
		if end > len(runes) {
			end = len(runes)
		}
		start := span.StartX
		if start < pos {
			start = pos
		}

		isHovered := d.hovered != nil &&
			d.hovered.LineY == span.LineY &&
			d.hovered.StartX == span.StartX &&
			d.hovered.EndX == span.EndX

		if isHovered {
			// bold + underline
			sb.WriteString("\033[1;4m")
		} else {
			// underline only
			sb.WriteString("\033[4m")
		}
		sb.WriteString(string(runes[start:end]))
		sb.WriteString("\033[0m")
		pos = end
	}
	if pos < len(runes) {
		sb.WriteString(string(runes[pos:]))
	}
	return sb.String()
}

// runeCount returns the number of runes in s. It is a thin wrapper so the
// package avoids importing unicode/utf8 for a single call.
func runeCount(s string) int {
	n := 0
	for range s {
		n++
	}
	return n
}
