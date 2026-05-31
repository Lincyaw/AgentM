package app

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// HelpOverlay shows a centered keybinding reference card.
// Any keypress closes it.
type HelpOverlay struct{}

func NewHelpOverlay() *HelpOverlay {
	return &HelpOverlay{}
}

func (h *HelpOverlay) Kind() OverlayKind { return OverlayHelp }

func (h *HelpOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	if _, ok := msg.(tea.KeyMsg); ok {
		return h, nil, true
	}
	return h, nil, false
}

func (h *HelpOverlay) View(width, height int, th *theme.Theme) string {
	title := th.OverlayTitle.Render("AgentM Terminal")

	sections := []struct {
		header string
		keys   [][2]string // {key, description}
	}{
		{
			header: "Navigation",
			keys: [][2]string{
				{"[ / ]", "prev / next turn"},
				{"PgUp/PgDn", "scroll viewport"},
				{"Ctrl+F", "search transcript"},
			},
		},
		{
			header: "Editing",
			keys: [][2]string{
				{"Enter", "send message"},
				{"Ctrl+J", "insert newline"},
				{"Tab", "complete command"},
				{"Up/Down", "history / suggestions"},
			},
		},
		{
			header: "Control",
			keys: [][2]string{
				{"Ctrl+C x2", "quit (1.5s window)"},
				{"Ctrl+D", "quit immediately"},
				{"Ctrl+L", "clear transcript"},
				{"Esc", "cancel / close overlay"},
			},
		},
		{
			header: "Utilities",
			keys: [][2]string{
				{"Ctrl+E", "toggle collapse"},
				{"Ctrl+O", "expand/collapse all"},
				{"Ctrl+Y", "copy last reply"},
				{"Ctrl+B", "bookmark position"},
				{"Ctrl+G", "bookmark list"},
				{"Ctrl+R", "re-send from history"},
				{"Ctrl+S", "save code block"},
				{"?", "this help"},
			},
		},
	}

	var sb strings.Builder
	sb.WriteString(title)
	sb.WriteByte('\n')
	sb.WriteByte('\n')

	for i, sec := range sections {
		sb.WriteString(th.OverlayTitle.Render(sec.header))
		sb.WriteByte('\n')
		for _, kv := range sec.keys {
			// Fixed-width key column
			key := kv[0]
			desc := kv[1]
			padded := key + strings.Repeat(" ", 14-len(key))
			sb.WriteString(th.OverlayActive.Render(padded))
			sb.WriteString(th.OverlayText.Render(desc))
			sb.WriteByte('\n')
		}
		if i < len(sections)-1 {
			sb.WriteByte('\n')
		}
	}

	sb.WriteByte('\n')
	sb.WriteString(th.OverlayDim.Render("      press any key to close"))

	content := th.OverlayBorder.Render(sb.String())
	return centerOverlay(content, width, height)
}
