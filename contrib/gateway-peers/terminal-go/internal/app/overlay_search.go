package app

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/blocks"
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// searchMatch records one substring match location in the transcript.
type searchMatch struct {
	BlockIdx   int
	CharOffset int
}

// SearchOverlay provides incremental substring search over the transcript.
// It renders as an inline bar between viewport and input.
type SearchOverlay struct {
	query      string
	matches    []searchMatch
	cursor     int // current match index (0-based)
	transcript []blocks.Block
}

// NewSearchOverlay creates a search overlay bound to the current transcript.
func NewSearchOverlay(transcript []blocks.Block) *SearchOverlay {
	return &SearchOverlay{
		transcript: transcript,
	}
}

func (s *SearchOverlay) Kind() OverlayKind { return OverlaySearch }

func (s *SearchOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		key := msg.String()
		switch key {
		case "esc":
			return s, nil, true

		case "enter", "ctrl+n":
			// Next match
			if len(s.matches) > 0 {
				s.cursor = (s.cursor + 1) % len(s.matches)
			}
			return s, nil, false

		case "shift+enter", "ctrl+p":
			// Prev match
			if len(s.matches) > 0 {
				s.cursor = (s.cursor - 1 + len(s.matches)) % len(s.matches)
			}
			return s, nil, false

		case "backspace":
			if len(s.query) > 0 {
				s.query = s.query[:len(s.query)-1]
				s.recompute()
			}
			return s, nil, false

		default:
			// Single printable character input
			if len(key) == 1 && key[0] >= 32 && key[0] < 127 {
				s.query += key
				s.recompute()
			} else if len([]rune(key)) == 1 {
				s.query += key
				s.recompute()
			}
			return s, nil, false
		}
	}
	return s, nil, false
}

// recompute scans the transcript for matches against the current query.
func (s *SearchOverlay) recompute() {
	s.matches = nil
	s.cursor = 0
	if s.query == "" {
		return
	}
	q := strings.ToLower(s.query)
	for i, b := range s.transcript {
		text := blockPlainText(b)
		lower := strings.ToLower(text)
		off := 0
		for {
			idx := strings.Index(lower[off:], q)
			if idx < 0 {
				break
			}
			s.matches = append(s.matches, searchMatch{
				BlockIdx:   i,
				CharOffset: off + idx,
			})
			off += idx + len(q)
		}
	}
	if s.cursor >= len(s.matches) {
		s.cursor = 0
	}
}

// CurrentMatchBlock returns the block index of the currently selected match,
// or -1 if there are no matches.
func (s *SearchOverlay) CurrentMatchBlock() int {
	if len(s.matches) == 0 {
		return -1
	}
	return s.matches[s.cursor].BlockIdx
}

// Query returns the current search query (used by renderTranscript for highlighting).
func (s *SearchOverlay) Query() string {
	return s.query
}

func (s *SearchOverlay) View(width, _ int, th *theme.Theme) string {
	var sb strings.Builder
	sb.WriteString(th.OverlayInput.Render(" search: "))
	sb.WriteString(th.OverlayText.Render(s.query))

	// Match count indicator
	if s.query != "" {
		status := fmt.Sprintf("  %d/%d matches", 0, len(s.matches))
		if len(s.matches) > 0 {
			status = fmt.Sprintf("  %d/%d matches", s.cursor+1, len(s.matches))
		}
		sb.WriteString(th.OverlayDim.Render(status))
	}

	// Pad to full width
	rendered := sb.String()
	if w := len([]rune(rendered)); w < width {
		rendered += strings.Repeat(" ", width-w)
	}
	return rendered
}

// blockPlainText extracts the searchable plain text from a block.
func blockPlainText(b blocks.Block) string {
	switch v := b.(type) {
	case *blocks.UserTurn:
		return v.Content
	case *blocks.AssistantTurn:
		var sb strings.Builder
		if v.Thinking != nil {
			sb.WriteString(v.Thinking.Text)
			sb.WriteByte('\n')
		}
		sb.WriteString(v.Text)
		for _, t := range v.Tools {
			sb.WriteString("\n")
			sb.WriteString(t.Name)
			if t.Result != "" {
				sb.WriteString(" ")
				sb.WriteString(t.Result)
			}
		}
		return sb.String()
	case *blocks.SystemTurn:
		return v.Content
	default:
		return ""
	}
}
