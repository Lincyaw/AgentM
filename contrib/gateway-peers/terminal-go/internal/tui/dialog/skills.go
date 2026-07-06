package dialog

import (
	"fmt"
	"slices"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/skills"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/scrollview"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

const (
	skillsPanelTopRow       = 5
	skillsPanelIndent       = 2
	skillsListStartOffset   = 7
	skillsSearchBoxMinWidth = 20
)

type skillsDialog struct {
	BaseDialog

	skills          []skills.Skill
	filtered        []skills.Skill
	enabledBySkill  map[string]bool
	selected        int
	query           string
	searchActive    bool
	sortBySource    bool
	listStartOffset int
	scrollview      *scrollview.Model
	keyMap          pickerKeyMap
}

func NewSkillsDialog(skillList []skills.Skill) Dialog {
	d := &skillsDialog{
		enabledBySkill: make(map[string]bool, len(skillList)),
		scrollview:     scrollview.New(scrollview.WithShowScrollbar(false)),
		keyMap:         defaultPickerKeyMap(),
	}
	d.skills = slices.Clone(skillList)
	for _, skill := range d.skills {
		d.enabledBySkill[skillKey(skill)] = true
	}
	d.sortSkills()
	d.filterSkills()
	return d
}

func (d *skillsDialog) Init() tea.Cmd { return nil }

func (d *skillsDialog) Update(msg tea.Msg) (layout.Model, tea.Cmd) {
	if handled, cmd := d.scrollview.Update(msg); handled {
		return d, cmd
	}

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		cmd := d.SetSize(msg.Width, msg.Height)
		return d, cmd

	case tea.PasteMsg:
		if d.searchActive {
			d.query += msg.Content
			d.filterSkills()
		}
		return d, nil

	case tea.MouseClickMsg:
		if msg.Button != tea.MouseLeft {
			return d, nil
		}
		if d.handleSearchClick(msg) {
			return d, nil
		}
		if d.handleSkillClick(msg) {
			return d, nil
		}
		return d, nil

	case tea.KeyPressMsg:
		if cmd := HandleQuit(msg); cmd != nil {
			return d, cmd
		}
		switch {
		case key.Matches(msg, d.keyMap.Escape):
			if d.searchActive || d.query != "" {
				d.searchActive = false
				d.query = ""
				d.filterSkills()
				return d, nil
			}
			return d, closeDialogCmd()
		case key.Matches(msg, d.keyMap.Up):
			d.navigate(-1)
			return d, nil
		case key.Matches(msg, d.keyMap.Down):
			d.navigate(+1)
			return d, nil
		case key.Matches(msg, d.keyMap.Enter):
			if d.searchActive {
				d.searchActive = false
				return d, nil
			}
			d.toggleSelected()
			return d, nil
		case isSpaceKey(msg):
			if d.searchActive {
				d.updateQuery(msg)
				return d, nil
			}
			d.toggleSelected()
			return d, nil
		case msg.String() == "/":
			d.searchActive = true
			return d, nil
		case msg.String() == "t" && !d.searchActive:
			d.sortBySource = !d.sortBySource
			d.sortSkills()
			d.filterSkills()
			return d, nil
		default:
			if d.searchActive {
				d.updateQuery(msg)
				return d, nil
			}
		}
	}

	return d, nil
}

func (d *skillsDialog) SetSize(width, height int) tea.Cmd {
	cmd := d.BaseDialog.SetSize(width, height)
	d.scrollview.SetSize(max(20, width), max(1, height-skillsPanelTopRow-skillsListStartOffset-1))
	return cmd
}

func (d *skillsDialog) Position() (row, col int) {
	if d.Height() <= skillsPanelTopRow+4 {
		return 0, 0
	}
	return skillsPanelTopRow, 0
}

func (d *skillsDialog) View() string {
	width := max(20, d.Width())
	contentWidth := max(1, width-(skillsPanelIndent*2))

	rows := d.renderRows(contentWidth)
	if len(rows) == 0 {
		rows = []string{skillsPanelLine(styles.SecondaryStyle.Render("No skills found"), contentWidth)}
	}

	d.listStartOffset = skillsListStartOffset
	visibleRows := d.visibleRows(len(rows))
	d.scrollview.SetSize(width, visibleRows)
	row, _ := d.Position()
	d.scrollview.SetPosition(0, row+d.listStartOffset)
	d.scrollview.SetContent(rows, len(rows))

	lines := make([]string, 0, d.listStartOffset+visibleRows+3)
	lines = append(lines, styles.DialogSeparatorStyle.Render(strings.Repeat("─", width)))
	lines = append(lines, skillsPanelLine(styles.BaseStyle.Render("Skills"), contentWidth))
	lines = append(lines, skillsPanelLine(styles.SecondaryStyle.Render(d.summaryLine()), contentWidth))
	lines = append(lines, "")
	lines = append(lines, d.renderSearchBox(contentWidth)...)
	lines = append(lines, d.scrollview.View())
	if more := d.moreBelowLine(); more != "" {
		lines = append(lines, skillsPanelLine(styles.SecondaryStyle.Render(more), contentWidth))
	} else {
		lines = append(lines, "")
	}
	lines = append(lines, "")
	lines = append(lines, skillsPanelLine(styles.SecondaryStyle.Render("Plugin skills are managed via /plugin"), contentWidth))
	return strings.Join(lines, "\n")
}

func (d *skillsDialog) summaryLine() string {
	count := fmt.Sprintf("%d skills", len(d.skills))
	if d.query != "" && len(d.filtered) != len(d.skills) {
		count = fmt.Sprintf("%d/%d skills", len(d.filtered), len(d.skills))
	}
	if d.searchActive {
		return count + " · type to filter · ↓/enter to select · esc to clear"
	}
	return count + " · enter/space to cycle, / to search, t to sort, Esc to close"
}

func (d *skillsDialog) renderSearchBox(contentWidth int) []string {
	boxWidth := max(skillsSearchBoxMinWidth, contentWidth)
	innerWidth := max(1, boxWidth-2)
	searchText := "⌕ Search skills…"
	style := styles.MutedStyle
	if d.query != "" {
		searchText = "⌕ " + d.query
		style = styles.BaseStyle
	}
	if d.searchActive && d.query == "" {
		style = styles.SecondaryStyle
	}

	searchText = toolcommon.TruncateText(searchText, max(1, innerWidth-2))
	searchContent := " " + style.Render(searchText)
	searchWidth := lipgloss.Width(searchContent)
	if searchWidth < innerWidth {
		searchContent += strings.Repeat(" ", innerWidth-searchWidth)
	}
	if lipgloss.Width(searchContent) > innerWidth {
		searchContent = ansi.Truncate(searchContent, innerWidth, "")
	}

	return []string{
		skillsPanelLine(styles.DialogSeparatorStyle.Render("╭"+strings.Repeat("─", innerWidth)+"╮"), contentWidth),
		skillsPanelLine(styles.DialogSeparatorStyle.Render("│")+searchContent+styles.DialogSeparatorStyle.Render("│"), contentWidth),
		skillsPanelLine(styles.DialogSeparatorStyle.Render("╰"+strings.Repeat("─", innerWidth)+"╯"), contentWidth),
	}
}

func (d *skillsDialog) renderRows(contentWidth int) []string {
	rows := make([]string, 0, len(d.filtered))
	statusWidth := d.statusColumnWidth()
	for i, skill := range d.filtered {
		rows = append(rows, d.renderRow(skill, i == d.selected, statusWidth, contentWidth))
	}
	return rows
}

func (d *skillsDialog) renderRow(skill skills.Skill, selected bool, statusWidth, contentWidth int) string {
	cursor := " "
	cursorStyle := styles.SecondaryStyle
	if selected {
		cursor = "❯"
		cursorStyle = styles.HighlightWhiteStyle
	}

	status := d.statusLabel(skill)
	statusCell := padSkillColumn(status, statusWidth)
	name := skill.Name
	if name == "" {
		name = "(unnamed skill)"
	}

	meta := []string{skillSourceLabel(skill), skillTokenEstimate(skill)}
	if skillLocked(skill) {
		meta = append(meta, "locked by plugin")
	}
	detail := name + " · " + strings.Join(meta, " · ")
	detailWidth := max(1, contentWidth-2-statusWidth)
	if lipgloss.Width(detail) > detailWidth {
		detail = toolcommon.TruncateText(detail, detailWidth)
	}

	line := cursorStyle.Render(cursor) + " " + styles.BaseStyle.Render(statusCell) + styles.SecondaryStyle.Render(detail)
	return skillsPanelLine(line, contentWidth)
}

func (d *skillsDialog) statusColumnWidth() int {
	width := lipgloss.Width("🔒 off        ")
	for _, skill := range d.filtered {
		width = max(width, lipgloss.Width(d.statusLabel(skill))+8)
	}
	return width
}

func (d *skillsDialog) statusLabel(skill skills.Skill) string {
	state := "on"
	if !d.enabledBySkill[skillKey(skill)] {
		state = "off"
	}
	if skillLocked(skill) {
		return "🔒 " + state
	}
	return "✔ " + state
}

func (d *skillsDialog) visibleRows(rowCount int) int {
	row, _ := d.Position()
	available := d.Height() - row - d.listStartOffset - 3
	if available < 1 {
		return 1
	}
	if rowCount > 0 {
		return min(available, rowCount)
	}
	return available
}

func (d *skillsDialog) moreBelowLine() string {
	below := len(d.filtered) - d.scrollview.ScrollOffset() - d.scrollview.VisibleHeight()
	if below <= 0 {
		return ""
	}
	return fmt.Sprintf("↓ %d more below", below)
}

func (d *skillsDialog) handleSearchClick(msg tea.MouseClickMsg) bool {
	row, _ := d.Position()
	if msg.Y >= row+4 && msg.Y <= row+6 {
		d.searchActive = true
		return true
	}
	return false
}

func (d *skillsDialog) handleSkillClick(msg tea.MouseClickMsg) bool {
	if d.listStartOffset <= 0 {
		return false
	}
	row, _ := d.Position()
	listY := row + d.listStartOffset
	if msg.Y < listY || msg.Y >= listY+d.scrollview.VisibleHeight() {
		return false
	}
	idx := d.scrollview.ScrollOffset() + (msg.Y - listY)
	if idx < 0 || idx >= len(d.filtered) {
		return false
	}
	d.selected = idx
	return true
}

func (d *skillsDialog) navigate(delta int) {
	if len(d.filtered) == 0 {
		d.selected = 0
		return
	}
	next := d.selected + delta
	if next < 0 || next >= len(d.filtered) {
		return
	}
	d.selected = next
	d.scrollview.EnsureLineVisible(next)
}

func (d *skillsDialog) toggleSelected() {
	if d.selected < 0 || d.selected >= len(d.filtered) {
		return
	}
	key := skillKey(d.filtered[d.selected])
	d.enabledBySkill[key] = !d.enabledBySkill[key]
}

func (d *skillsDialog) updateQuery(msg tea.KeyPressMsg) {
	switch msg.String() {
	case "backspace", "ctrl+h":
		d.query = dropLastRune(d.query)
	case "delete", "ctrl+d":
		d.query = ""
	case "enter", "esc", "up", "down", "left", "right", "pgup", "pgdown", "home", "end", "tab":
		return
	default:
		value := msg.String()
		if value == "space" {
			value = " "
		}
		if isPrintableInput(value) {
			d.query += value
		}
	}
	d.filterSkills()
}

func (d *skillsDialog) filterSkills() {
	query := strings.ToLower(strings.TrimSpace(d.query))
	d.filtered = d.filtered[:0]
	for _, skill := range d.skills {
		if query == "" || skillMatchesQuery(skill, query) {
			d.filtered = append(d.filtered, skill)
		}
	}
	if d.selected >= len(d.filtered) {
		d.selected = max(0, len(d.filtered)-1)
	}
	d.scrollview.SetScrollOffset(0)
}

func (d *skillsDialog) sortSkills() {
	slices.SortStableFunc(d.skills, func(a, b skills.Skill) int {
		if d.sortBySource {
			if cmp := strings.Compare(skillSourceLabel(a), skillSourceLabel(b)); cmp != 0 {
				return cmp
			}
		}
		return strings.Compare(strings.ToLower(a.Name), strings.ToLower(b.Name))
	})
}

func skillMatchesQuery(skill skills.Skill, query string) bool {
	fields := []string{
		skill.Name,
		skill.Description,
		skillSourceLabel(skill),
		skill.FilePath,
		skill.BaseDir,
	}
	return strings.Contains(strings.ToLower(strings.Join(fields, " ")), query)
}

func skillSourceLabel(skill skills.Skill) string {
	switch {
	case skill.IsInline():
		return "inline"
	case skillUserManaged(skill):
		return "user"
	default:
		return "plugin"
	}
}

func skillLocked(skill skills.Skill) bool {
	return !skillUserManaged(skill) && !skill.IsInline()
}

func skillUserManaged(skill skills.Skill) bool {
	if strings.HasPrefix(skill.Name, "lark-") {
		return true
	}
	path := strings.ToLower(strings.TrimSpace(skill.BaseDir + " " + skill.FilePath))
	if path == "" {
		return false
	}
	userPathMarkers := []string{
		"/.agentm/skills/",
		"/.agents/skills/",
		"/.claude/skills/",
		"/.codex/skills/",
	}
	for _, marker := range userPathMarkers {
		if strings.Contains(path, marker) {
			return true
		}
	}
	return skill.Local && !strings.Contains(path, "/plugins/cache/")
}

func skillTokenEstimate(skill skills.Skill) string {
	if skillUserManaged(skill) {
		text := strings.TrimSpace(skill.Description)
		if text == "" {
			return "< 20 tok"
		}
		tokens := len([]rune(text)) / 3
		if tokens < 20 {
			return "< 20 tok"
		}
		tokens = ((tokens + 9) / 10) * 10
		return fmt.Sprintf("~%d tok", tokens)
	}

	text := strings.TrimSpace(skill.InlineContent + " " + skill.Context + " " + skill.Description)
	if text == "" {
		return "~200 tok"
	}
	tokens := max(200, len(strings.Fields(text))*4/3)
	tokens = ((tokens + 49) / 50) * 50
	return fmt.Sprintf("~%d tok", tokens)
}

func skillKey(skill skills.Skill) string {
	if skill.FilePath != "" {
		return skill.FilePath
	}
	if skill.BaseDir != "" {
		return skill.BaseDir + "/" + skill.Name
	}
	return skill.Name
}

func padSkillColumn(value string, width int) string {
	if lipgloss.Width(value) >= width {
		return value
	}
	return value + strings.Repeat(" ", width-lipgloss.Width(value))
}

func skillsPanelLine(content string, contentWidth int) string {
	line := strings.Repeat(" ", skillsPanelIndent) + content
	maxWidth := contentWidth + skillsPanelIndent
	if lipgloss.Width(line) > maxWidth {
		return ansi.Truncate(line, maxWidth, "")
	}
	return line
}

func isSpaceKey(msg tea.KeyPressMsg) bool {
	return msg.String() == " " || msg.String() == "space"
}

func isPrintableInput(value string) bool {
	if value == "" || strings.Contains(value, "+") || strings.HasPrefix(value, "ctrl+") || strings.HasPrefix(value, "alt+") {
		return false
	}
	return lipgloss.Width(value) > 0
}

func dropLastRune(value string) string {
	if value == "" {
		return ""
	}
	runes := []rune(value)
	return string(runes[:len(runes)-1])
}
