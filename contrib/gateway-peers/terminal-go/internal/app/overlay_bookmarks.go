package app

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// Bookmark records a saved position in the transcript.
type Bookmark struct {
	BlockIndex int
	Label      string
}

// BookmarkOverlay shows a navigable list of bookmarks.
type BookmarkOverlay struct {
	bookmarks []Bookmark
	cursor    int
	jumped    int // block index to jump to, or -1
}

// NewBookmarkOverlay creates a bookmark list overlay.
func NewBookmarkOverlay(bookmarks []Bookmark) *BookmarkOverlay {
	return &BookmarkOverlay{
		bookmarks: bookmarks,
		jumped:    -1,
	}
}

func (b *BookmarkOverlay) Kind() OverlayKind { return OverlayBookmarks }

func (b *BookmarkOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			return b, nil, true

		case "up", "k":
			if b.cursor > 0 {
				b.cursor--
			}
			return b, nil, false

		case "down", "j":
			if b.cursor < len(b.bookmarks)-1 {
				b.cursor++
			}
			return b, nil, false

		case "enter":
			if len(b.bookmarks) > 0 {
				b.jumped = b.bookmarks[b.cursor].BlockIndex
			}
			return b, nil, true

		case "d":
			if len(b.bookmarks) > 0 {
				b.bookmarks = append(b.bookmarks[:b.cursor], b.bookmarks[b.cursor+1:]...)
				if b.cursor >= len(b.bookmarks) && b.cursor > 0 {
					b.cursor--
				}
				if len(b.bookmarks) == 0 {
					return b, nil, true
				}
			}
			return b, nil, false
		}
	}
	return b, nil, false
}

// JumpTarget returns the block index to jump to, or -1 if no jump was requested.
func (b *BookmarkOverlay) JumpTarget() int { return b.jumped }

// Bookmarks returns the (possibly modified) bookmark list.
func (b *BookmarkOverlay) Bookmarks() []Bookmark { return b.bookmarks }

func (b *BookmarkOverlay) View(width, height int, th *theme.Theme) string {
	var sb strings.Builder
	sb.WriteString(th.OverlayTitle.Render("Bookmarks"))
	sb.WriteByte('\n')
	sb.WriteByte('\n')

	if len(b.bookmarks) == 0 {
		sb.WriteString(th.OverlayDim.Render("  (no bookmarks)"))
		sb.WriteByte('\n')
	} else {
		for i, bm := range b.bookmarks {
			prefix := "  "
			style := th.OverlayText
			if i == b.cursor {
				prefix = "> "
				style = th.OverlayActive
			}
			line := fmt.Sprintf("%s[%d] %s", prefix, bm.BlockIndex, bm.Label)
			sb.WriteString(style.Render(line))
			sb.WriteByte('\n')
		}
	}

	sb.WriteByte('\n')
	sb.WriteString(th.OverlayDim.Render("  enter=jump  d=delete  esc=close"))

	content := th.OverlayBorder.Render(sb.String())
	return centerOverlay(content, width, height)
}
