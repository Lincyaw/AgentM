package dialog

import (
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/model/provider"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

// modelPickerDialog is a dialog for selecting a model for the current agent.
type modelPickerDialog struct {
	pickerCore

	models   []runtime.ModelChoice
	filtered []runtime.ModelChoice
	errMsg   string // validation error message

	effortIndex     int
	effortTouched   bool
	listStartOffset int
	contentWidth    int
	topRow          int
	showTranscript  bool
}

// Model picker dialog dimension constants
const (
	// Column widths for the per-row stats. Values are right-aligned in their
	// own column so the list reads like a table.
	pickerInputColWidth   = 10
	pickerOutputColWidth  = 10
	pickerContextColWidth = 8

	// pickerDetailsLines is the number of lines reserved for the model
	// details panel rendered below the model list.
	pickerDetailsLines = 4

	// pickerListVerticalOverhead is the number of rows used by dialog chrome:
	// title(1) + space(1) + input(1) + separator(1) + column header(1) +
	// details separator(1) + details (pickerDetailsLines) + space at bottom(1) +
	// help keys(1) + borders/padding(2) = 10 + pickerDetailsLines
	pickerListVerticalOverhead = 10 + pickerDetailsLines

	// pickerListStartOffset is the Y offset from dialog top to where the model list starts:
	// border(1) + padding(1) + title(1) + space(1) + input(1) + separator(1) +
	// column header(1) = 7
	pickerListStartOffset = 7

	// pickerDetailsLabelWidth is the column width for the labels in the
	// details panel ("Reference", "Pricing", "Limits", "Modalities").
	pickerDetailsLabelWidth = 12

	// catalogSeparatorLabel labels the separator above the catalog group.
	catalogSeparatorLabel = "Other models"
	// customSeparatorLabel labels the separator above the custom-models group.
	customSeparatorLabel = "Custom models"

	modelPickerTopRow = 5
	modelPickerIndent = 2
)

var modelPickerEffortLabels = []string{"Medium effort", "High effort (default)", "xHigh effort"}

const modelPickerDefaultEffortIndex = 1

type modelPickerApplyMode int

const (
	modelPickerApplyDefault modelPickerApplyMode = iota
	modelPickerApplySession
)

// modelPickerLayout is the layout used by the model picker.
var modelPickerLayout = pickerLayout{
	WidthPercent:    pickerWidthPercent,
	MinWidth:        pickerMinWidth,
	MaxWidth:        pickerMaxWidth,
	HeightPercent:   pickerHeightPercent,
	MaxHeight:       pickerMaxHeight,
	ListOverhead:    pickerListVerticalOverhead,
	ListStartOffset: pickerListStartOffset,
}

// NewModelPickerDialog creates a new model picker dialog.
func NewModelPickerDialog(models []runtime.ModelChoice) Dialog {
	return newModelPickerDialog(models, modelPickerTopRow, false)
}

// NewModelPickerDialogAtTop creates a model picker whose top edge follows the
// transcript row used by Claude Code's slash-command picker presentation.
func NewModelPickerDialogAtTop(models []runtime.ModelChoice, topRow int) Dialog {
	return newModelPickerDialog(models, topRow, true)
}

func newModelPickerDialog(models []runtime.ModelChoice, topRow int, showTranscript bool) Dialog {
	d := &modelPickerDialog{
		pickerCore:     newPickerCore(modelPickerLayout, "Type to search or enter custom model (provider/model)…"),
		effortIndex:    modelPickerDefaultEffortIndex,
		topRow:         max(0, topRow),
		showTranscript: showTranscript,
	}
	d.textInput.CharLimit = 100

	// Claude Code's model picker is a small fixed menu, independent of the
	// provider profile names advertised by the runtime. Keep custom
	// provider/model entry support through the filter path below.
	sortedModels := claudeCodeModelChoices(models)
	slices.SortFunc(sortedModels, func(a, b runtime.ModelChoice) int {
		return comparePickerSortKeys(modelSortKeys(a), modelSortKeys(b))
	})
	d.models = sortedModels
	d.filterModels()
	d.selectCurrentModel()
	return d
}

func claudeCodeModelChoices(advertised []runtime.ModelChoice) []runtime.ModelChoice {
	currentRef := ""
	for _, model := range advertised {
		if model.IsCurrent {
			currentRef = normalizedModelPickerRef(model)
			break
		}
	}

	choices := make([]runtime.ModelChoice, 0, 5)
	if ref, ok := switchRefForClaudeChoice(advertised, "opus", "opus-1m", "claude-opus-4-8", "claude-opus-4-8-1m"); ok {
		choices = append(choices, runtime.ModelChoice{Name: "Opus", Ref: "opus", SwitchRef: ref})
	}
	if ref, ok := switchRefForClaudeChoice(advertised, "sonnet", "claude-sonnet-5"); ok {
		choices = append(choices, runtime.ModelChoice{Name: "Sonnet", Ref: "sonnet", SwitchRef: ref})
	}
	if ref, ok := switchRefForClaudeChoice(advertised, "sonnet-1m", "sonnet-5-1m", "claude-sonnet-5-1m"); ok {
		choices = append(choices, runtime.ModelChoice{Name: "Sonnet 5 (1M context)", Ref: "sonnet-1m", SwitchRef: ref})
	}
	if ref, ok := switchRefForClaudeChoice(advertised, "haiku", "claude-haiku-4-5"); ok {
		choices = append(choices, runtime.ModelChoice{Name: "Haiku", Ref: "haiku", SwitchRef: ref})
	}
	if len(choices) == 0 {
		return slices.Clone(advertised)
	}
	choices = append([]runtime.ModelChoice{{Name: "Default (recommended)", Ref: "default", IsDefault: true}}, choices...)
	if currentRef == "" {
		return choices
	}
	for i := range choices {
		if normalizedModelPickerRef(choices[i]) == currentRef {
			choices[i].IsCurrent = true
			if i != 0 {
				choices[0].IsDefault = false
			}
		}
	}
	return choices
}

func switchRefForClaudeChoice(advertised []runtime.ModelChoice, aliases ...string) (string, bool) {
	aliasSet := make(map[string]struct{}, len(aliases))
	for _, alias := range aliases {
		aliasSet[alias] = struct{}{}
	}
	for _, model := range advertised {
		if _, ok := aliasSet[normalizedModelPickerRef(model)]; ok {
			return modelReference(model), true
		}
	}
	return "", false
}

// modelSortKeys derives the sort key tuple from a runtime.ModelChoice.
func modelSortKeys(m runtime.ModelChoice) pickerSortKeys {
	section := 0
	switch {
	case m.IsCustom:
		section = 2
	case m.IsCatalog:
		section = 1
	}
	name := m.Name
	if rank, ok := claudeModelSortRank(m); ok {
		name = fmt.Sprintf("%03d-%s", rank, name)
	}
	return pickerSortKeys{
		Section:   section,
		IsCurrent: false,
		IsDefault: m.IsDefault,
		Name:      name,
	}
}

func claudeModelSortRank(model runtime.ModelChoice) (int, bool) {
	switch normalizedModelPickerRef(model) {
	case "default":
		return 0, true
	case "opus", "opus-1m", "claude-opus-4-8", "claude-opus-4-8-1m":
		return 1, true
	case "sonnet", "claude-sonnet-5":
		return 2, true
	case "sonnet-1m", "sonnet-5-1m", "claude-sonnet-5-1m":
		return 3, true
	case "haiku", "claude-haiku-4-5":
		return 4, true
	default:
		return 0, false
	}
}

func (d *modelPickerDialog) Init() tea.Cmd { return nil }

func (d *modelPickerDialog) Update(msg tea.Msg) (layout.Model, tea.Cmd) {
	// Scrollview handles mouse scrollbar, wheel, and pgup/pgdn/home/end.
	if handled, cmd := d.scrollview.Update(msg); handled {
		return d, cmd
	}

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		cmd := d.SetSize(msg.Width, msg.Height)
		return d, cmd

	case tea.PasteMsg:
		cmd := d.handleInputChange(msg)
		return d, cmd

	case tea.MouseClickMsg:
		if dbl, handled := d.handleClaudeListClick(msg); dbl {
			cmd := d.handleSelection(modelPickerApplyDefault)
			return d, cmd
		} else if handled {
			return d, nil
		}
		return d, nil

	case tea.KeyPressMsg:
		if cmd := HandleQuit(msg); cmd != nil {
			return d, cmd
		}
		switch {
		case key.Matches(msg, d.keyMap.Escape):
			return d, tea.Sequence(closeDialogCmd(), core.CmdHandler(messages.ModelPickerCanceledMsg{
				ShowTranscript: d.showTranscript,
			}))
		case key.Matches(msg, d.keyMap.Up):
			d.navigate(-1, len(d.filtered), d.findSelectedLine)
			return d, nil
		case key.Matches(msg, d.keyMap.Down):
			d.navigate(+1, len(d.filtered), d.findSelectedLine)
			return d, nil
		case key.Matches(msg, d.keyMap.Enter):
			cmd := d.handleSelection(modelPickerApplyDefault)
			return d, cmd
		case msg.String() == "s":
			cmd := d.handleSelection(modelPickerApplySession)
			return d, cmd
		case msg.String() == "left":
			if d.effortIndex > 0 {
				d.effortIndex--
				d.effortTouched = true
			}
			return d, nil
		case msg.String() == "right":
			if d.effortIndex < len(modelPickerEffortLabels)-1 {
				d.effortIndex++
				d.effortTouched = true
			}
			return d, nil
		default:
			if idx, ok := digitSelectionIndex(msg.String(), len(d.filtered)); ok {
				d.selected = idx
				d.scrollview.EnsureLineVisible(idx)
			}
			return d, nil
		}
	}

	return d, nil
}

// handleInputChange forwards msg to the text input, re-runs the filter, and
// clears any validation error from a previous submission.
func (d *modelPickerDialog) handleInputChange(msg tea.Msg) tea.Cmd {
	return d.updateInput(msg, func() {
		d.filterModels()
		d.errMsg = ""
	})
}

// buildList constructs the rendered model list. Each selectable model can span
// multiple visual rows when its description wraps.
func (d *modelPickerDialog) buildList(contentWidth int) *groupedList {
	gl := newGroupedList()
	labelWidth := d.modelLabelColumnWidth(contentWidth)
	for i, model := range d.filtered {
		gl.AddItemLines(d.renderClaudeModelRowLines(model, i, i == d.selected, labelWidth, contentWidth))
	}

	return gl
}

func (d *modelPickerDialog) lineToModelIndex(line int) int {
	return d.buildList(0).ItemForLine(line)
}

func (d *modelPickerDialog) findSelectedLine() int {
	contentWidth := d.contentWidth
	if contentWidth <= 0 {
		contentWidth = max(1, d.Width()-(modelPickerIndent*2))
	}
	return d.buildList(contentWidth).LineForItem(d.selected)
}

func (d *modelPickerDialog) handleSelection(mode modelPickerApplyMode) tea.Cmd {
	query := strings.TrimSpace(d.textInput.Value())

	// If user typed something that looks like a custom model (contains /), validate and use it
	if strings.Contains(query, "/") {
		if err := validateCustomModelSpec(query); err != nil {
			d.errMsg = err.Error()
			return nil
		}
		notice, welcomeLine := d.modelSelectionFeedback(runtime.ModelChoice{Name: query, Ref: query}, mode)
		return tea.Sequence(
			closeDialogCmd(),
			core.CmdHandler(messages.ChangeModelMsg{
				ModelRef:           query,
				TranscriptNotice:   notice,
				WelcomeModelLine:   welcomeLine,
				RevealFocusWarning: d.effortTouched,
			}),
		)
	}

	// Otherwise, use the selected item from the filtered list
	if d.selected >= 0 && d.selected < len(d.filtered) {
		selected := d.filtered[d.selected]
		// If selecting the default model, send empty ref to clear the override
		modelRef := selected.Ref
		if selected.SwitchRef != "" {
			modelRef = selected.SwitchRef
		}
		if selected.IsDefault {
			modelRef = ""
		}
		notice, welcomeLine := d.modelSelectionFeedback(selected, mode)
		return tea.Sequence(
			closeDialogCmd(),
			core.CmdHandler(messages.ChangeModelMsg{
				ModelRef:           modelRef,
				TranscriptNotice:   notice,
				WelcomeModelLine:   welcomeLine,
				RevealFocusWarning: d.effortTouched,
				ShowTranscript:     d.showTranscript,
			}),
		)
	}

	return nil
}

func (d *modelPickerDialog) modelSelectionFeedback(model runtime.ModelChoice, mode modelPickerApplyMode) (string, string) {
	name := claudeModelNoticeName(model)
	effort := d.effortNoticeLabel()
	effortSuffix := ""
	if d.effortTouched {
		effortSuffix = " with " + effort
	}
	if mode == modelPickerApplySession {
		return fmt.Sprintf("Set model to %s for this session only%s", name, effortSuffix),
			fmt.Sprintf("%s with %s · API Usage Billing", name, effort)
	}
	return fmt.Sprintf("Set model to %s as default%s", name, effortSuffix),
		fmt.Sprintf("%s with %s · API Usage Billing", name, effort)
}

func (d *modelPickerDialog) effortNoticeLabel() string {
	switch d.effortIndex {
	case 0:
		return "medium effort"
	case 2:
		return "xHigh effort"
	default:
		return "high effort"
	}
}

func claudeModelNoticeName(model runtime.ModelChoice) string {
	switch normalizedModelPickerRef(model) {
	case "default", "opus", "opus-1m", "claude-opus-4-8", "claude-opus-4-8-1m":
		return "Opus 4.8 (1M context)"
	case "sonnet", "claude-sonnet-5":
		return "Sonnet 5"
	case "sonnet-1m", "sonnet-5-1m", "claude-sonnet-5-1m":
		return "Sonnet 5 (1M context)"
	case "haiku", "claude-haiku-4-5":
		return "Haiku 4.5"
	default:
		if ref := modelReference(model); ref != "" {
			return ref
		}
		return model.Name
	}
}

// validateCustomModelSpec validates a custom model specification entered by the user.
// It checks that each provider/model pair is properly formatted and uses a supported provider.
func validateCustomModelSpec(spec string) error {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil
	}

	// Handle alloy specs (comma-separated)
	parts := strings.SplitSeq(spec, ",")
	for part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		providerName, modelName, ok := strings.Cut(part, "/")
		if !ok {
			return errors.New("invalid format: expected 'provider/model'")
		}

		providerName = strings.TrimSpace(providerName)
		modelName = strings.TrimSpace(modelName)

		if providerName == "" {
			return fmt.Errorf("provider name cannot be empty (got '/%s')", modelName)
		}
		if modelName == "" {
			return fmt.Errorf("model name cannot be empty (got '%s/')", providerName)
		}

		if !provider.IsKnownProvider(providerName) {
			return fmt.Errorf("unknown provider '%s'. Supported: %s",
				providerName, strings.Join(provider.AllProviders(), ", "))
		}
	}

	return nil
}

func (d *modelPickerDialog) filterModels() {
	query := strings.ToLower(strings.TrimSpace(d.textInput.Value()))

	// If query contains "/", show "Custom" option as well as matches
	isCustomQuery := strings.Contains(query, "/")

	d.filtered = d.filtered[:0]
	for _, model := range d.models {
		if query == "" {
			d.filtered = append(d.filtered, model)
			continue
		}

		// Match against name, provider, and model
		searchText := strings.ToLower(strings.Join([]string{
			model.Name,
			model.Provider,
			model.Model,
			d.modelPickerDisplaySearchText(model),
		}, " "))
		if strings.Contains(searchText, query) {
			d.filtered = append(d.filtered, model)
		}
	}

	// If query looks like a custom model spec and we have no exact match, show it as an option
	if isCustomQuery && len(d.filtered) == 0 {
		d.filtered = append(d.filtered, runtime.ModelChoice{
			Name: "Custom: " + query,
			Ref:  query,
		})
	}

	if d.selected >= len(d.filtered) {
		d.selected = max(0, len(d.filtered)-1)
	}
	d.scrollview.SetScrollOffset(0)
}

func (d *modelPickerDialog) selectCurrentModel() {
	for i, model := range d.filtered {
		if model.IsCurrent {
			d.selected = i
			return
		}
	}
}

func (d *modelPickerDialog) View() string {
	width := max(20, d.Width())
	contentWidth := max(1, width-(modelPickerIndent*2))
	d.contentWidth = contentWidth

	description := wrapModelPickerText(
		"Switch between Claude models. Your pick becomes the default for new sessions. For other/previous model names, specify with --model.",
		contentWidth,
	)
	rows := d.renderClaudeModelRows(contentWidth)
	if len(rows) == 0 {
		rows = []string{modelPickerLine("No models found", contentWidth)}
	}

	d.listStartOffset = 1 + 1 + 1 + len(description) + 1
	visibleRows := d.visibleListRows(len(description), len(rows))
	d.scrollview.SetSize(width, visibleRows)
	row, _ := d.Position()
	d.scrollview.SetPosition(0, row+d.listStartOffset)
	d.scrollview.SetContent(rows, len(rows))

	lines := make([]string, 0, 8+len(description)+visibleRows)
	lines = append(lines, "")
	lines = append(lines, styles.DialogSeparatorStyle.Render(strings.Repeat("─", width)))
	lines = append(lines, modelPickerLine(styles.BaseStyle.Render("Select model"), contentWidth))
	for _, line := range description {
		lines = append(lines, modelPickerLine(styles.SecondaryStyle.Render(line), contentWidth))
	}
	lines = append(lines, "")
	lines = append(lines, d.scrollview.View())
	lines = append(lines, "")
	lines = append(lines, modelPickerLine(d.renderEffortLine(), contentWidth))
	lines = append(lines, "")
	lines = append(lines, modelPickerLine(styles.SecondaryStyle.Render("Enter to set as default · s to use this session only · Esc to cancel"), contentWidth))
	lines = append(lines, strings.Repeat(" ", width))

	return strings.Join(lines, "\n")
}

func (d *modelPickerDialog) Position() (row, col int) {
	topRow := max(0, d.topRow)
	if d.Height() <= topRow+4 {
		return 0, 0
	}
	return topRow, 0
}

func (d *modelPickerDialog) SetSize(width, height int) tea.Cmd {
	cmd := d.BaseDialog.SetSize(width, height)
	topRow := max(0, d.topRow)
	d.scrollview.SetSize(max(20, width), max(1, height-topRow-8))
	return cmd
}

func (d *modelPickerDialog) visibleListRows(descriptionLines, rowCount int) int {
	row, _ := d.Position()
	fixedRows := 8 + descriptionLines
	available := d.Height() - row - fixedRows
	if available < 1 {
		return 1
	}
	if rowCount > 0 {
		return min(available, rowCount)
	}
	return available
}

func (d *modelPickerDialog) handleClaudeListClick(msg tea.MouseClickMsg) (doubleClicked, handled bool) {
	if msg.Button != tea.MouseLeft || d.listStartOffset <= 0 {
		return false, false
	}
	row, _ := d.Position()
	listY := row + d.listStartOffset
	if msg.Y < listY || msg.Y >= listY+d.scrollview.VisibleHeight() {
		return false, false
	}
	idx := d.lineToModelIndex(d.scrollview.ScrollOffset() + (msg.Y - listY))
	if idx < 0 || idx >= len(d.filtered) {
		return false, false
	}
	d.selected = idx
	return d.recordClick(idx), true
}

func (d *modelPickerDialog) renderClaudeModelRows(contentWidth int) []string {
	return d.buildList(contentWidth).Lines()
}

func (d *modelPickerDialog) modelLabelColumnWidth(contentWidth int) int {
	maxLabel := 0
	for i, model := range d.filtered {
		maxLabel = max(maxLabel, lipgloss.Width(d.modelPickerLeftLabel(model, i)))
	}
	preferred := max(28, maxLabel+2)
	maxAllowed := max(12, contentWidth/2)
	return min(preferred, maxAllowed)
}

func (d *modelPickerDialog) renderClaudeModelRowLines(model runtime.ModelChoice, index int, selected bool, labelWidth, contentWidth int) []string {
	cursor := " "
	cursorStyle := styles.SecondaryStyle
	if selected {
		cursor = "❯"
		cursorStyle = styles.HighlightWhiteStyle
	}

	left := d.modelPickerLeftLabel(model, index)
	leftWidth := lipgloss.Width(left)
	if leftWidth > labelWidth {
		left = toolcommon.TruncateText(left, labelWidth)
		leftWidth = lipgloss.Width(left)
	}

	padding := strings.Repeat(" ", max(1, labelWidth-leftWidth))
	descWidth := max(1, contentWidth-2-labelWidth)
	descLines := wrapModelPickerText(d.modelPickerDescription(model), descWidth)
	if len(descLines) == 0 {
		descLines = []string{""}
	}

	lines := make([]string, 0, len(descLines))
	first := cursorStyle.Render(cursor) + " " + styles.BaseStyle.Render(left) + styles.SecondaryStyle.Render(padding+descLines[0])
	lines = append(lines, modelPickerLine(first, contentWidth))
	continuationPrefix := strings.Repeat(" ", 2+labelWidth)
	for _, descLine := range descLines[1:] {
		lines = append(lines, modelPickerLine(styles.SecondaryStyle.Render(continuationPrefix+descLine), contentWidth))
	}
	return lines
}

func (d *modelPickerDialog) modelPickerLeftLabel(model runtime.ModelChoice, index int) string {
	display, ok := d.claudeModelDisplay(model)
	label := model.Name
	checked := model.IsCurrent || model.IsDefault
	if ok {
		label = display.Name
		checked = display.Checked
	} else if model.IsDefault {
		label = "Default (recommended)"
	}
	if checked {
		label += " ✔"
	}
	return fmt.Sprintf("%d. %s", index+1, label)
}

func (d *modelPickerDialog) modelPickerDescription(model runtime.ModelChoice) string {
	if display, ok := d.claudeModelDisplay(model); ok {
		return display.Description
	}

	var parts []string
	switch {
	case model.IsDefault:
		parts = append(parts, "Use the default model")
	case model.IsCurrent:
		parts = append(parts, "Current model")
	default:
		parts = append(parts, "Use "+model.Name)
	}

	ref := modelReference(model)
	if ref != "" && ref != model.Name {
		parts = append(parts, ref)
	}
	if model.ContextLimit > 0 {
		parts = append(parts, formatTokenCount(int64(model.ContextLimit))+" context")
	}
	if model.InputCost > 0 || model.OutputCost > 0 {
		parts = append(parts, formatCostPerMillion(model.InputCost)+"/"+formatCostPerMillion(model.OutputCost)+" per Mtok")
	}
	return strings.Join(parts, " · ")
}

func (d *modelPickerDialog) modelPickerDisplaySearchText(model runtime.ModelChoice) string {
	display, ok := d.claudeModelDisplay(model)
	if !ok {
		return ""
	}
	return display.Name + " " + display.Description
}

type claudeModelPickerDisplay struct {
	Name        string
	Description string
	Checked     bool
}

func (d *modelPickerDialog) claudeModelDisplay(model runtime.ModelChoice) (claudeModelPickerDisplay, bool) {
	ref := normalizedModelPickerRef(model)
	checked := model.IsCurrent || model.IsDefault
	switch ref {
	case "default":
		return claudeModelPickerDisplay{
			Name:        "Default (recommended)",
			Description: "Use the default model (currently Opus 4.8 (1M context)) · $5/$25 per Mtok",
			Checked:     true,
		}, true
	case "opus", "opus-1m", "claude-opus-4-8", "claude-opus-4-8-1m":
		return claudeModelPickerDisplay{
			Name:        "Opus",
			Description: "Opus 4.8 with 1M context · Best for everyday, complex tasks · $5/$25 per Mtok",
			Checked:     checked,
		}, true
	case "sonnet", "claude-sonnet-5":
		return claudeModelPickerDisplay{
			Name:        "Sonnet",
			Description: "Sonnet 5 · Efficient for routine tasks · $3/$15 per Mtok",
			Checked:     checked,
		}, true
	case "sonnet-1m", "sonnet-5-1m", "claude-sonnet-5-1m":
		return claudeModelPickerDisplay{
			Name:        "Sonnet 5 (1M context)",
			Description: "Sonnet 5 for long sessions · $3/$15 per Mtok",
			Checked:     checked,
		}, true
	case "haiku", "claude-haiku-4-5":
		return claudeModelPickerDisplay{
			Name:        "Haiku",
			Description: "Haiku 4.5 · Fastest for quick answers · $1/$5 per Mtok",
			Checked:     checked,
		}, true
	default:
		return claudeModelPickerDisplay{}, false
	}
}

func normalizedModelPickerRef(model runtime.ModelChoice) string {
	ref := model.Ref
	if ref == "" {
		ref = model.Name
	}
	ref = strings.ToLower(strings.TrimSpace(ref))
	ref = strings.TrimPrefix(ref, "anthropic/")
	ref = strings.NewReplacer("_", "-", "[", "-", "]", "", " ", "-").Replace(ref)
	for strings.Contains(ref, "--") {
		ref = strings.ReplaceAll(ref, "--", "-")
	}
	return strings.Trim(ref, "-")
}

func (d *modelPickerDialog) modelOriginalIndex(model runtime.ModelChoice) int {
	for i, candidate := range d.models {
		if candidate.Ref == model.Ref &&
			candidate.Name == model.Name &&
			candidate.Provider == model.Provider &&
			candidate.Model == model.Model {
			return i
		}
	}
	return -1
}

func (d *modelPickerDialog) renderEffortLine() string {
	if d.effortIndex < 0 || d.effortIndex >= len(modelPickerEffortLabels) {
		d.effortIndex = modelPickerDefaultEffortIndex
	}
	icon := "●"
	switch d.effortIndex {
	case 0:
		icon = "◐"
	case 2:
		icon = "◉"
	}
	return styles.BaseStyle.Render(icon+" "+modelPickerEffortLabels[d.effortIndex]) + styles.SecondaryStyle.Render(" ←/→ to adjust")
}

func modelPickerLine(content string, contentWidth int) string {
	line := strings.Repeat(" ", modelPickerIndent) + content
	maxWidth := contentWidth + modelPickerIndent
	if lipgloss.Width(line) > maxWidth {
		return toolcommon.TruncateText(line, maxWidth)
	}
	return line
}

func wrapModelPickerText(text string, width int) []string {
	if width <= 0 {
		return []string{text}
	}
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	var lines []string
	current := words[0]
	for _, word := range words[1:] {
		candidate := current + " " + word
		if lipgloss.Width(candidate) <= width {
			current = candidate
			continue
		}
		lines = append(lines, current)
		current = word
	}
	lines = append(lines, current)
	return lines
}

func digitSelectionIndex(s string, limit int) (int, bool) {
	if len(s) != 1 || s[0] < '1' || s[0] > '9' {
		return 0, false
	}
	idx := int(s[0] - '1')
	if idx >= limit {
		return 0, false
	}
	return idx, true
}

// pickerRowPalette is the set of styles used to render one row of the
// model list. Selection inverts the foreground/background colours of
// every visible element so the row reads as a single highlighted band.
type pickerRowPalette struct {
	name     lipgloss.Style
	desc     lipgloss.Style
	alloy    lipgloss.Style
	defBadge lipgloss.Style
	current  lipgloss.Style
	stats    lipgloss.Style
	missing  lipgloss.Style
}

func pickerRowStyles(selected bool) pickerRowPalette {
	p := pickerRowPalette{
		name:     styles.PaletteUnselectedActionStyle,
		desc:     styles.PaletteUnselectedDescStyle,
		alloy:    styles.BadgeAlloyStyle,
		defBadge: styles.BadgeDefaultStyle,
		current:  styles.BadgeCurrentStyle,
		stats:    styles.SecondaryStyle,
		missing:  styles.MutedStyle,
	}
	if !selected {
		return p
	}
	p.name = styles.PaletteSelectedActionStyle
	p.desc = styles.PaletteSelectedDescStyle
	p.alloy = p.alloy.Background(styles.MobyBlue)
	p.defBadge = p.defBadge.Background(styles.MobyBlue)
	p.current = p.current.Background(styles.MobyBlue)
	// Reuse the description style so the cells share the selection band.
	p.stats = p.desc
	p.missing = p.desc.Italic(true)
	return p
}

func (d *modelPickerDialog) renderModel(model runtime.ModelChoice, selected bool, maxWidth int) string {
	if maxWidth <= 0 {
		return ""
	}
	p := pickerRowStyles(selected)
	nameWidth := pickerNameColWidth(maxWidth)
	return renderRowName(model, nameWidth, p) + renderRowStats(model, p)
}

// pickerNameColWidth returns the width allotted to the name column for
// a given total content width.
func pickerNameColWidth(maxWidth int) int {
	return max(1, maxWidth-pickerInputColWidth-pickerOutputColWidth-pickerContextColWidth)
}

// renderRowName renders the model name and any badges, padded to width.
func renderRowName(model runtime.ModelChoice, width int, p pickerRowPalette) string {
	badges, badgeWidth := renderRowBadges(model, p)

	nameMax := max(1, width-badgeWidth)
	displayName := model.Name
	if lipgloss.Width(displayName) > nameMax {
		displayName = toolcommon.TruncateText(displayName, nameMax)
	}

	name := p.name.Render(displayName) + badges
	padding := max(0, width-lipgloss.Width(name))
	return name + p.desc.Render(strings.Repeat(" ", padding))
}

// renderRowBadges returns the rendered badge segment plus its width.
func renderRowBadges(model runtime.ModelChoice, p pickerRowPalette) (string, int) {
	var (
		text  string
		width int
	)
	add := func(label string, style lipgloss.Style) {
		text += style.Render(label)
		width += lipgloss.Width(label)
	}
	if isAlloyModel(model) {
		add(" (alloy)", p.alloy)
	}
	switch {
	case model.IsCurrent:
		add(" (current)", p.current)
	case model.IsDefault:
		add(" (default)", p.defBadge)
	}
	return text, width
}

// renderRowStats renders the three right-aligned stats columns.
func renderRowStats(model runtime.ModelChoice, p pickerRowPalette) string {
	return renderStatsCell(formatCostPerMillion(model.InputCost), pickerInputColWidth, p, model.InputCost > 0) +
		renderStatsCell(formatCostPerMillion(model.OutputCost), pickerOutputColWidth, p, model.OutputCost > 0) +
		renderStatsCell(formatContextCell(model.ContextLimit), pickerContextColWidth, p, model.ContextLimit > 0)
}

// renderStatsCell right-aligns value in a fixed-width column. Missing
// values fade by using the palette's missing style.
func renderStatsCell(value string, width int, p pickerRowPalette, present bool) string {
	padding := max(0, width-lipgloss.Width(value))
	pad := p.stats.Render(strings.Repeat(" ", padding))
	valueStyle := p.stats
	if !present {
		valueStyle = p.missing
	}
	return pad + valueStyle.Render(value)
}

// isAlloyModel returns true when the model is an alloy spec (no
// provider, comma-separated provider/model list in Model).
func isAlloyModel(model runtime.ModelChoice) bool {
	return model.Provider == "" && strings.Contains(model.Model, ",")
}

// renderColumnHeader renders the static header above the model list,
// labelling the per-row stats columns.
func (d *modelPickerDialog) renderColumnHeader(maxWidth int) string {
	header := strings.Repeat(" ", pickerNameColWidth(maxWidth)) +
		rightAlign("Input/1M", pickerInputColWidth) +
		rightAlign("Output/1M", pickerOutputColWidth) +
		rightAlign("Context", pickerContextColWidth)
	return styles.MutedStyle.Render(header)
}

// rightAlign returns s padded with leading spaces so its rendered width
// equals width. Strings already wider than width are returned unchanged.
func rightAlign(s string, width int) string {
	padding := width - lipgloss.Width(s)
	if padding <= 0 {
		return s
	}
	return strings.Repeat(" ", padding) + s
}

// leftPad returns s padded with trailing spaces to width. Strings already
// wider than width are returned unchanged.
func leftPad(s string, width int) string {
	padding := width - lipgloss.Width(s)
	if padding <= 0 {
		return s
	}
	return s + strings.Repeat(" ", padding)
}

// formatContextCell formats a context window size for the table column.
// Returns an em-dash placeholder when the size is unknown.
func formatContextCell(tokens int) string {
	if tokens <= 0 {
		return "—"
	}
	return formatTokenCount(int64(tokens))
}

// formatCostPerMillion renders a USD-per-million-tokens price using a
// compact representation. Values <= 0 render as an em-dash; sub-cent
// values keep four decimals so they don't collapse to "$0.00";
// sub-dollar values keep two decimals; larger values trim trailing
// zeros (e.g., $3 instead of $3.00).
func formatCostPerMillion(cost float64) string {
	switch {
	case cost <= 0:
		return "—"
	case cost < 0.01:
		return fmt.Sprintf("$%.4f", cost)
	case cost < 1:
		return fmt.Sprintf("$%.2f", cost)
	}
	s := strconv.FormatFloat(cost, 'f', 2, 64)
	s = strings.TrimRight(s, "0")
	s = strings.TrimRight(s, ".")
	return "$" + s
}

// modelReference returns the technical "provider/model" reference for a
// model choice, suitable for the details panel.
func modelReference(model runtime.ModelChoice) string {
	switch {
	case model.IsCustom:
		return model.Ref
	case isAlloyModel(model):
		return model.Model
	case model.Provider != "" && model.Model != "":
		return model.Provider + "/" + model.Model
	default:
		return model.Ref
	}
}

// detailsStyles bundles the styles used by the details panel.
type detailsStyles struct {
	label lipgloss.Style
	value lipgloss.Style
	muted lipgloss.Style
}

func newDetailsStyles() detailsStyles {
	return detailsStyles{
		label: styles.SecondaryStyle.Bold(true),
		value: styles.BaseStyle,
		muted: styles.MutedStyle.Italic(true),
	}
}

// renderDetails returns the details panel for the currently-selected
// model. It always renders pickerDetailsLines lines so the dialog has a
// stable height.
func (d *modelPickerDialog) renderDetails(width int) string {
	s := newDetailsStyles()

	var lines []string
	if d.selected >= 0 && d.selected < len(d.filtered) {
		lines = formatDetailsLines(d.filtered[d.selected], s)
	} else {
		lines = []string{s.muted.Render("No model selected")}
	}

	// Pad to a stable height so the dialog doesn't change size.
	for len(lines) < pickerDetailsLines {
		lines = append(lines, "")
	}
	// Truncate any line that would wrap.
	for i, l := range lines {
		if lipgloss.Width(l) > width {
			lines[i] = toolcommon.TruncateText(l, width)
		}
	}
	return strings.Join(lines[:pickerDetailsLines], "\n")
}

// formatDetailsLines builds the four labelled rows shown for a model.
func formatDetailsLines(model runtime.ModelChoice, s detailsStyles) []string {
	row := func(label, value string) string {
		return s.label.Render(leftPad(label, pickerDetailsLabelWidth)) + value
	}

	ref := s.value.Render(modelReference(model))
	if model.Family != "" && !strings.EqualFold(model.Family, model.Provider) {
		ref += s.muted.Render(" · " + model.Family + " family")
	}

	return []string{
		row("Reference", ref),
		row("Pricing", formatPricingRow(model, s)),
		row("Limits", formatLimitsRow(model, s)),
		row("Modalities", formatModalitiesRow(model, s)),
	}
}

// formatPricingRow renders the pricing line of the details panel.
func formatPricingRow(model runtime.ModelChoice, s detailsStyles) string {
	var parts []string
	if model.InputCost > 0 || model.OutputCost > 0 {
		parts = append(parts,
			s.value.Render(formatCostPerMillion(model.InputCost)+" in"),
			s.value.Render(formatCostPerMillion(model.OutputCost)+" out"),
		)
	}
	if model.CacheReadCost > 0 {
		parts = append(parts, s.value.Render(formatCostPerMillion(model.CacheReadCost)+" cache read"))
	}
	if model.CacheWriteCost > 0 {
		parts = append(parts, s.value.Render(formatCostPerMillion(model.CacheWriteCost)+" cache write"))
	}
	if len(parts) == 0 {
		return s.muted.Render("unavailable")
	}
	parts = append(parts, s.muted.Render("per 1M tokens"))
	return strings.Join(parts, s.muted.Render(" · "))
}

// formatLimitsRow renders the limits line of the details panel.
func formatLimitsRow(model runtime.ModelChoice, s detailsStyles) string {
	var parts []string
	if model.ContextLimit > 0 {
		parts = append(parts, s.value.Render(formatTokenCount(int64(model.ContextLimit))+" context window"))
	}
	if model.OutputLimit > 0 {
		parts = append(parts, s.value.Render(formatTokenCount(model.OutputLimit)+" max output"))
	}
	if len(parts) == 0 {
		return s.muted.Render("unavailable")
	}
	return strings.Join(parts, s.muted.Render(" · "))
}

// formatModalitiesRow renders the modalities line of the details panel.
func formatModalitiesRow(model runtime.ModelChoice, s detailsStyles) string {
	if len(model.InputModalities) == 0 && len(model.OutputModalities) == 0 {
		return s.muted.Render("unavailable")
	}
	in := joinOrDash(model.InputModalities)
	out := joinOrDash(model.OutputModalities)
	return s.value.Render(in) + s.muted.Render(" → ") + s.value.Render(out)
}

// joinOrDash returns the comma-joined list, or an em-dash when empty.
func joinOrDash(parts []string) string {
	if len(parts) == 0 {
		return "—"
	}
	return strings.Join(parts, ", ")
}
