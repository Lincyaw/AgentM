package dialog

import (
	"fmt"
	"slices"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

// ThemeChoice represents a selectable theme option.
type ThemeChoice struct {
	Ref        string // Theme reference ("default" for built-in default)
	Name       string // Display name
	SyntaxName string // Optional syntax-theme label shown in the preview
	IsCurrent  bool   // Currently active theme
	IsDefault  bool   // Built-in default theme ("default")
	IsBuiltin  bool   // Built-in theme shipped with AgentM Terminal
	HasOrder   bool   // Preserve a Claude-compatible display order
	Order      int    // Display order when HasOrder is true
}

// themePickerDialog is a dialog for selecting a theme.
type themePickerDialog struct {
	pickerCore

	themes   []ThemeChoice
	filtered []ThemeChoice

	// originalThemeRef is the theme ref active when the dialog opened. It is
	// used to restore on cancel.
	originalThemeRef string
	// lastPreviewRef avoids re-applying the same preview repeatedly (e.g.,
	// during filtering).
	lastPreviewRef     string
	listStartOffset    int
	syntaxHighlighting bool
}

// customThemesSeparatorLabel labels the separator above the custom themes group.
const customThemesSeparatorLabel = "Custom themes"

// themePickerLayout is the layout used by the theme picker. It uses the
// shared sectioned-picker overhead so it can host the same group separators
// as the model picker.
var themePickerLayout = pickerLayout{
	WidthPercent:    pickerWidthPercent,
	MinWidth:        pickerMinWidth,
	MaxWidth:        pickerMaxWidth,
	HeightPercent:   pickerHeightPercent,
	MaxHeight:       pickerMaxHeight,
	ListOverhead:    pickerListVerticalOverhead,
	ListStartOffset: pickerListStartOffset,
}

const (
	themePickerTopRow       = 5
	themePickerIndent       = 2
	themePickerPreviewLines = 7
)

// NewThemePickerDialog creates a new theme picker dialog.
// originalThemeRef is the currently active theme ref (for restoration on cancel).
func NewThemePickerDialog(themes []ThemeChoice, originalThemeRef string) Dialog {
	d := &themePickerDialog{
		pickerCore:         newPickerCore(themePickerLayout, "Type to search themes…"),
		originalThemeRef:   originalThemeRef,
		syntaxHighlighting: false,
	}
	d.textInput.CharLimit = 100

	// Sort themes: built-in first, then custom. Within each section: default
	// first, then alphabetically; the current theme is marked and selected
	// without being moved to the top.
	sortedThemes := slices.Clone(themes)
	slices.SortFunc(sortedThemes, func(a, b ThemeChoice) int {
		return comparePickerSortKeys(themeSortKeys(a), themeSortKeys(b))
	})
	d.themes = sortedThemes
	d.filterThemes()

	// Find current theme and select it (if multiple are marked current, pick first).
	for i, t := range d.filtered {
		if t.IsCurrent {
			d.selected = i
			break
		}
	}
	// The current theme is already applied; avoid emitting a duplicate preview
	// for it on the first navigation.
	if d.selected >= 0 && d.selected < len(d.filtered) {
		d.lastPreviewRef = d.filtered[d.selected].Ref
	}

	return d
}

// themeSortKeys derives the sort key tuple from a ThemeChoice.
func themeSortKeys(t ThemeChoice) pickerSortKeys {
	if t.HasOrder {
		return pickerSortKeys{
			Section:  t.Order,
			Tiebreak: t.Ref,
		}
	}
	section := 1
	if t.IsBuiltin {
		section = 0
	}
	return pickerSortKeys{
		Section:   section,
		IsDefault: t.IsDefault,
		Name:      t.Name,
		Tiebreak:  t.Ref,
	}
}

func (d *themePickerDialog) Init() tea.Cmd { return nil }

func (d *themePickerDialog) Update(msg tea.Msg) (layout.Model, tea.Cmd) {
	// Scrollview handles mouse scrollbar, wheel, and pgup/pgdn/home/end.
	if handled, cmd := d.scrollview.Update(msg); handled {
		return d, cmd
	}

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		cmd := d.SetSize(msg.Width, msg.Height)
		return d, cmd

	case messages.ThemeChangedMsg:
		return d, nil

	case tea.PasteMsg:
		return d, nil

	case tea.MouseClickMsg:
		if dbl, handled := d.handleClaudeListClick(msg); dbl {
			cmd := d.handleSelection()
			return d, cmd
		} else if handled {
			cmd := d.emitPreview()
			return d, cmd
		}
		return d, nil

	case tea.KeyPressMsg:
		if cmd := HandleQuit(msg); cmd != nil {
			return d, cmd
		}
		switch {
		case key.Matches(msg, d.keyMap.Escape):
			return d, tea.Sequence(
				closeDialogCmd(),
				core.CmdHandler(messages.ThemeCancelPreviewMsg{OriginalRef: d.originalThemeRef}),
			)
		case key.Matches(msg, d.keyMap.Up):
			cmd := d.navigateAndPreview(-1)
			return d, cmd
		case key.Matches(msg, d.keyMap.Down):
			cmd := d.navigateAndPreview(+1)
			return d, cmd
		case key.Matches(msg, d.keyMap.Enter):
			cmd := d.handleSelection()
			return d, cmd
		default:
			if msg.String() == "ctrl+t" {
				d.syntaxHighlighting = !d.syntaxHighlighting
				return d, nil
			}
			if idx, ok := digitSelectionIndex(msg.String(), len(d.filtered)); ok {
				d.selected = idx
				d.scrollview.EnsureLineVisible(d.findSelectedLine())
				return d, d.emitPreview()
			}
			return d, nil
		}
	}

	return d, nil
}

// navigateAndPreview moves the selection by delta and emits a preview when
// the selection actually moved.
func (d *themePickerDialog) navigateAndPreview(delta int) tea.Cmd {
	if d.navigate(delta, len(d.filtered), d.findSelectedLine) {
		return d.emitPreview()
	}
	return nil
}

// handleSelection applies the selected theme and closes the picker.
func (d *themePickerDialog) handleSelection() tea.Cmd {
	if d.selected < 0 || d.selected >= len(d.filtered) {
		return nil
	}
	return tea.Sequence(
		closeDialogCmd(),
		core.CmdHandler(messages.ChangeThemeMsg{ThemeRef: d.filtered[d.selected].Ref}),
	)
}

// emitPreview requests a theme preview via an app-level message, skipping
// re-emission for the same theme.
func (d *themePickerDialog) emitPreview() tea.Cmd {
	if d.selected < 0 || d.selected >= len(d.filtered) {
		return nil
	}
	selected := d.filtered[d.selected]
	if selected.Ref == d.lastPreviewRef {
		return nil
	}
	d.lastPreviewRef = selected.Ref
	return core.CmdHandler(messages.ThemePreviewMsg{
		ThemeRef:    selected.Ref,
		OriginalRef: d.originalThemeRef,
	})
}

// buildList constructs the list of themes with a "Custom themes" separator
// before the first custom entry (when built-in themes precede it). Pass
// contentWidth=0 to compute the layout without rendering items (used by
// mouse hit-testing and findSelectedLine).
func (d *themePickerDialog) buildList(contentWidth int) *groupedList {
	gl := newGroupedList()
	hasBuiltin := slices.ContainsFunc(d.filtered, func(t ThemeChoice) bool { return t.IsBuiltin })

	customSepShown := false
	for i, theme := range d.filtered {
		if !theme.IsBuiltin && !customSepShown {
			if hasBuiltin {
				gl.AddNonItem(RenderGroupSeparator(customThemesSeparatorLabel, contentWidth))
			}
			customSepShown = true
		}
		gl.AddItem(d.renderTheme(theme, i == d.selected, contentWidth))
	}
	return gl
}

func (d *themePickerDialog) lineToThemeIndex(line int) int {
	return d.buildList(0).ItemForLine(line)
}

func (d *themePickerDialog) findSelectedLine() int {
	return d.buildList(0).LineForItem(d.selected)
}

func (d *themePickerDialog) View() string {
	width := max(20, d.Width())
	contentWidth := max(1, width-(themePickerIndent*2))

	description := wrapModelPickerText(
		"Choose the text style that looks best with your terminal",
		contentWidth,
	)

	rows := d.renderClaudeThemeRows(contentWidth)
	if len(rows) == 0 {
		rows = []string{themePickerLine("No themes found", contentWidth)}
	}

	d.listStartOffset = 1 + 1 + 1 + len(description) + 1
	visibleRows := d.visibleListRows(len(description), len(rows))
	d.scrollview.SetSize(width, visibleRows)
	row, _ := d.Position()
	d.scrollview.SetPosition(0, row+d.listStartOffset)
	d.scrollview.SetContent(rows, len(rows))
	offset := d.scrollview.ScrollOffset()
	end := min(offset+visibleRows, len(rows))
	listView := ""
	if offset < end {
		listView = strings.Join(rows[offset:end], "\n")
	}

	lines := make([]string, 0, 10+len(description)+visibleRows+themePickerPreviewLines)
	lines = append(lines, "")
	lines = append(lines, styles.DialogSeparatorStyle.Render(strings.Repeat("─", width)))
	lines = append(lines, themePickerLine(styles.BaseStyle.Render("Theme"), contentWidth))
	lines = append(lines, "")
	for _, line := range description {
		lines = append(lines, themePickerLine(styles.SecondaryStyle.Render(line), contentWidth))
	}
	lines = append(lines, "")
	lines = append(lines, listView)
	lines = append(lines, "")
	lines = append(lines, d.renderPreviewBlock(contentWidth)...)
	lines = append(lines, "")
	lines = append(lines, themePickerLine(styles.SecondaryStyle.Render("Enter to select · Esc to cancel"), contentWidth))

	return strings.Join(lines, "\n")
}

func (d *themePickerDialog) Position() (row, col int) {
	if d.Height() <= themePickerTopRow+4 {
		return 0, 0
	}
	return themePickerTopRow, 0
}

func (d *themePickerDialog) SetSize(width, height int) tea.Cmd {
	cmd := d.BaseDialog.SetSize(width, height)
	d.scrollview.SetSize(max(20, width), max(1, height-themePickerTopRow-8))
	return cmd
}

func (d *themePickerDialog) visibleListRows(descriptionLines, rowCount int) int {
	row, _ := d.Position()
	fixedRows := 7 + descriptionLines + themePickerPreviewLines
	available := d.Height() - row - fixedRows
	if available < 1 {
		return 1
	}
	if rowCount > 0 {
		return min(available, rowCount)
	}
	return available
}

func (d *themePickerDialog) handleClaudeListClick(msg tea.MouseClickMsg) (doubleClicked, handled bool) {
	if msg.Button != tea.MouseLeft || d.listStartOffset <= 0 {
		return false, false
	}
	row, _ := d.Position()
	listY := row + d.listStartOffset
	if msg.Y < listY || msg.Y >= listY+d.scrollview.VisibleHeight() {
		return false, false
	}
	idx := d.lineToThemeIndex(d.scrollview.ScrollOffset() + (msg.Y - listY))
	if idx < 0 || idx >= len(d.filtered) {
		return false, false
	}
	d.selected = idx
	return d.recordClick(idx), true
}

func (d *themePickerDialog) renderClaudeThemeRows(contentWidth int) []string {
	return d.buildList(contentWidth).Lines()
}

func (d *themePickerDialog) renderPreviewBlock(contentWidth int) []string {
	rule := styles.DialogSeparatorStyle.Render(strings.Repeat("╌", max(1, contentWidth)))
	selectedName := "Default"
	if d.selected >= 0 && d.selected < len(d.filtered) {
		selected := d.filtered[d.selected]
		selectedName = selected.SyntaxName
		if selectedName == "" {
			selectedName = selected.Name
		}
	}
	if selectedName == "" {
		selectedName = "current"
	}

	if !d.syntaxHighlighting {
		return []string{
			themePickerLine(rule, contentWidth),
			themePickerLine(styles.SecondaryStyle.Render(" 1 function greet() {"), contentWidth),
			themePickerLine("", contentWidth),
			themePickerLine(styles.SecondaryStyle.Render(" 2   console.log(\"Hello, World!\");"), contentWidth),
			themePickerLine(styles.SecondaryStyle.Render("-"), contentWidth),
			themePickerLine(styles.SecondaryStyle.Render(" 2   console.log(\"Hello, Claude!\");"), contentWidth),
			themePickerLine(styles.SecondaryStyle.Render("+"), contentWidth),
			themePickerLine(styles.SecondaryStyle.Render(" 3 }"), contentWidth),
			themePickerLine("", contentWidth),
			themePickerLine(rule, contentWidth),
			themePickerLine(styles.SecondaryStyle.Render(" Syntax highlighting disabled (ctrl+t to enable)"), contentWidth),
		}
	}

	lines := []string{
		themePickerLine(rule, contentWidth),
		themePickerLine(styles.SecondaryStyle.Render(" 1  function greet() {"), contentWidth),
		themePickerLine(styles.DiffRemoveStyle.Render(" 2 -  console.log(\"Hello, World!\");"), contentWidth),
		themePickerLine(styles.DiffAddStyle.Render(" 2 +  console.log(\"Hello, Claude!\");"), contentWidth),
		themePickerLine(styles.SecondaryStyle.Render(" 3  }"), contentWidth),
		themePickerLine(rule, contentWidth),
		themePickerLine(styles.SecondaryStyle.Render(" Syntax theme: "+selectedName), contentWidth),
	}
	return lines
}

func themePickerLine(content string, contentWidth int) string {
	line := strings.Repeat(" ", themePickerIndent) + content
	maxWidth := contentWidth + themePickerIndent
	if lipgloss.Width(line) > maxWidth {
		return toolcommon.TruncateText(line, maxWidth)
	}
	return line
}

func (d *themePickerDialog) renderTheme(theme ThemeChoice, selected bool, maxWidth int) string {
	if maxWidth <= 0 {
		return ""
	}

	cursor := " "
	cursorStyle := styles.SecondaryStyle
	if selected {
		cursor = "❯"
		cursorStyle = styles.HighlightWhiteStyle
	}

	label := theme.Name
	if label == "" {
		label = strings.TrimPrefix(theme.Ref, styles.UserThemePrefix)
	}
	if theme.IsDefault && !strings.Contains(strings.ToLower(label), "default") {
		label += " (default)"
	}
	if theme.IsCurrent {
		label += " ✔"
	}

	row := fmt.Sprintf("%s %d. %s", cursorStyle.Render(cursor), d.themeIndex(theme)+1, label)
	if !theme.IsBuiltin {
		desc := strings.TrimPrefix(theme.Ref, styles.UserThemePrefix)
		if desc != "" && desc != label {
			row += styles.SecondaryStyle.Render("  " + desc)
		}
	}

	if lipgloss.Width(row) > maxWidth {
		row = toolcommon.TruncateText(row, maxWidth)
	}
	return themePickerLine(row, maxWidth)
}

func (d *themePickerDialog) themeIndex(theme ThemeChoice) int {
	for i, candidate := range d.filtered {
		if candidate.Ref == theme.Ref {
			return i
		}
	}
	return 0
}

// filterThemes derives the visible theme list from the currently known themes.
// The Claude-style theme picker is not a search UI, so the text input remains
// empty in normal use; this still keeps the pickerCore state internally
// consistent if the component is reused.
func (d *themePickerDialog) filterThemes() (selectionChanged bool) {
	query := strings.ToLower(strings.TrimSpace(d.textInput.Value()))

	prevRef := ""
	if d.selected >= 0 && d.selected < len(d.filtered) {
		prevRef = d.filtered[d.selected].Ref
	}

	d.filtered = d.filtered[:0]
	for _, theme := range d.themes {
		if query == "" || strings.Contains(strings.ToLower(theme.Name+" "+theme.Ref), query) {
			d.filtered = append(d.filtered, theme)
		}
	}

	d.selected = 0
	if prevRef != "" {
		for i, t := range d.filtered {
			if t.Ref == prevRef {
				d.selected = i
				break
			}
		}
	}
	d.scrollview.SetScrollOffset(0)

	newRef := ""
	if d.selected >= 0 && d.selected < len(d.filtered) {
		newRef = d.filtered[d.selected].Ref
	}
	return newRef != prevRef
}
