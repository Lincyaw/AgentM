package app

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

const paletteMaxVisible = 15

// PaletteItem represents one entry in the command palette.
type PaletteItem struct {
	Label       string // display name (e.g. "/help" or "toggle sidebar")
	Description string // short description
	Shortcut    string // key shortcut if any (e.g. "Ctrl+]")
	Action      string // slash command text, or internal action ID
	IsCommand   bool   // true for slash commands
}

// PaletteOverlay provides a searchable command palette triggered by Ctrl+K.
type PaletteOverlay struct {
	items    []PaletteItem
	filtered []PaletteItem
	query    string
	cursor   int
	chosen   *PaletteItem
}

// NewPaletteOverlay creates a palette populated from slash commands and
// built-in UI actions.
func NewPaletteOverlay(commands []string, tools []string) *PaletteOverlay {
	var items []PaletteItem

	// Slash commands.
	cmdDescriptions := map[string]string{
		"/help":    "Show help",
		"/clear":   "Clear transcript",
		"/status":  "Show status",
		"/new":     "New session",
		"/end":     "End session",
		"/compact": "Compact context",
		"/dump":    "Dump screen",
	}
	for _, cmd := range commands {
		desc := cmdDescriptions[cmd]
		if desc == "" {
			desc = "Run " + cmd
		}
		items = append(items, PaletteItem{
			Label:     cmd,
			Description: desc,
			Action:    cmd,
			IsCommand: true,
		})
	}

	// Built-in UI actions.
	uiActions := []PaletteItem{
		{Label: "Toggle sidebar", Description: "Show/hide sidebar panel", Shortcut: "Ctrl+]", Action: "toggle_sidebar"},
		{Label: "Search", Description: "Search transcript", Shortcut: "Ctrl+F", Action: "search"},
		{Label: "Bookmarks", Description: "View saved bookmarks", Shortcut: "Ctrl+G", Action: "bookmarks"},
		{Label: "Expand/collapse all", Description: "Toggle all blocks", Shortcut: "Ctrl+O", Action: "toggle_expand"},
		{Label: "Resend history", Description: "Re-send from history", Shortcut: "Ctrl+R", Action: "resend"},
		{Label: "Save code block", Description: "Save code to file", Shortcut: "Ctrl+S", Action: "save_code"},
		{Label: "Clear transcript", Description: "Clear and reset", Shortcut: "Ctrl+L", Action: "clear"},
		{Label: "Help", Description: "Show keybinding reference", Shortcut: "?", Action: "help"},
	}
	items = append(items, uiActions...)

	o := &PaletteOverlay{items: items}
	o.refilter()
	return o
}

func (o *PaletteOverlay) Kind() OverlayKind { return OverlayPalette }

func (o *PaletteOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		key := msg.String()

		switch key {
		case "esc":
			return o, nil, true

		case "enter":
			if len(o.filtered) > 0 && o.cursor < len(o.filtered) {
				chosen := o.filtered[o.cursor]
				o.chosen = &chosen
			}
			return o, nil, true

		case "up":
			if o.cursor > 0 {
				o.cursor--
			}
			return o, nil, false

		case "down":
			if o.cursor < len(o.filtered)-1 {
				o.cursor++
			}
			return o, nil, false

		case "backspace":
			if len(o.query) > 0 {
				o.query = o.query[:len(o.query)-1]
				o.refilter()
			}
			return o, nil, false

		default:
			// Accept printable characters.
			if len(key) == 1 && key[0] >= 32 && key[0] < 127 {
				o.query += key
				o.refilter()
			} else if len([]rune(key)) == 1 {
				o.query += key
				o.refilter()
			}
			return o, nil, false
		}
	}
	return o, nil, false
}

// refilter applies the current query to the item list. Prefix matches sort
// before substring matches.
func (o *PaletteOverlay) refilter() {
	o.cursor = 0
	if o.query == "" {
		o.filtered = make([]PaletteItem, len(o.items))
		copy(o.filtered, o.items)
		return
	}

	q := strings.ToLower(o.query)
	var prefix, substr []PaletteItem
	for _, item := range o.items {
		label := strings.ToLower(item.Label)
		desc := strings.ToLower(item.Description)
		if strings.HasPrefix(label, q) || strings.HasPrefix(desc, q) {
			prefix = append(prefix, item)
		} else if strings.Contains(label, q) || strings.Contains(desc, q) {
			substr = append(substr, item)
		}
	}
	o.filtered = append(prefix, substr...)
}

// Chosen returns the selected palette item, or nil if nothing was chosen.
func (o *PaletteOverlay) Chosen() *PaletteItem { return o.chosen }

func (o *PaletteOverlay) View(width, height int, th *theme.Theme) string {
	innerW := width - 8
	if innerW < 30 {
		innerW = 30
	}
	if innerW > 60 {
		innerW = 60
	}

	var sb strings.Builder
	sb.WriteString(th.OverlayTitle.Render("Command Palette"))
	sb.WriteByte('\n')
	sb.WriteByte('\n')

	// Query input.
	sb.WriteString(th.OverlayInput.Render(" > "))
	sb.WriteString(th.OverlayText.Render(o.query))
	sb.WriteByte('\n')
	sb.WriteString(th.OverlayDim.Render(strings.Repeat("~", 20)))
	sb.WriteByte('\n')

	// Filtered items.
	if len(o.filtered) == 0 {
		sb.WriteString(th.OverlayDim.Render("  (no matches)"))
		sb.WriteByte('\n')
	} else {
		visible := len(o.filtered)
		if visible > paletteMaxVisible {
			visible = paletteMaxVisible
		}

		// Scroll window around cursor.
		start := 0
		if o.cursor >= visible {
			start = o.cursor - visible + 1
		}
		end := start + visible
		if end > len(o.filtered) {
			end = len(o.filtered)
			start = end - visible
			if start < 0 {
				start = 0
			}
		}

		for i := start; i < end; i++ {
			item := o.filtered[i]
			prefix := "  "
			style := th.OverlayText
			if i == o.cursor {
				prefix = "> "
				style = th.OverlayActive
			}

			label := item.Label
			desc := item.Description
			shortcut := item.Shortcut

			// Format: "  /help          Show help             Ctrl+F"
			line := fmt.Sprintf("%s%-16s %s", prefix, label, desc)
			if shortcut != "" {
				pad := innerW - len([]rune(line)) - len([]rune(shortcut))
				if pad < 2 {
					pad = 2
				}
				line += strings.Repeat(" ", pad) + shortcut
			}

			sb.WriteString(style.Render(line))
			sb.WriteByte('\n')
		}

		if len(o.filtered) > paletteMaxVisible {
			sb.WriteString(th.OverlayDim.Render(fmt.Sprintf("  ... %d more", len(o.filtered)-paletteMaxVisible)))
			sb.WriteByte('\n')
		}
	}

	sb.WriteByte('\n')
	sb.WriteString(th.OverlayDim.Render("  enter=select  esc=close"))

	content := th.OverlayBorder.Render(sb.String())
	return centerOverlay(content, width, height)
}
