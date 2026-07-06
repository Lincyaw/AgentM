package completion

import (
	"cmp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
	"github.com/junegunn/fzf/src/algo"
	"github.com/junegunn/fzf/src/util"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

const (
	maxItems                 = 10
	maxVisibleCompletionRows = 20
	maxResourceRows          = 15
	selectedScrollBackoff    = 5
)

// MatchMode defines how completion items are filtered
type MatchMode int

const (
	// MatchFuzzy uses fuzzy matching (matches anywhere in label)
	MatchFuzzy MatchMode = iota
	// MatchPrefix requires the query to match the start of the label
	MatchPrefix
	// MatchFuzzyPrefixPriority uses fuzzy matching but ranks prefix matches first.
	MatchFuzzyPrefixPriority
	// MatchResourcePath uses file/resource path browsing semantics.
	MatchResourcePath
)

type Item struct {
	Label       string
	Description string
	Value       string
	SearchText  string
	Execute     func() tea.Cmd
	Pinned      bool // Pinned items always appear at the top, in original order
}

type OpenMsg struct {
	Items     []Item
	MatchMode MatchMode
	Query     string
}

type OpenedMsg struct{}

type CloseMsg struct{}

type ClosedMsg struct{}

type QueryMsg struct {
	Query string
}

type SelectedMsg struct {
	Value      string
	Execute    func() tea.Cmd
	AutoSubmit bool
}

// SelectionChangedMsg is sent when the selected item changes (for preview in editor)
type SelectionChangedMsg struct {
	Value string
}

// AppendItemsMsg appends items to the current completion list without closing the popup.
// Useful for async loading of completion items.
type AppendItemsMsg struct {
	Items []Item
}

// ReplaceItemsMsg replaces non-pinned items in the completion list.
// Pinned items (like "Browse files…") are preserved.
// Useful for full async load that supersedes initial results.
type ReplaceItemsMsg struct {
	Items []Item
}

// SetLoadingMsg sets the loading state for the completion popup.
type SetLoadingMsg struct {
	Loading bool
}

type matchResult struct {
	item  Item
	score int
	index int
}

type completionKeyMap struct {
	Up     key.Binding
	Down   key.Binding
	Enter  key.Binding
	Tab    key.Binding
	Escape key.Binding
}

// defaultCompletionKeyMap returns default key bindings
func defaultCompletionKeyMap() completionKeyMap {
	return completionKeyMap{
		Up: key.NewBinding(
			key.WithKeys("up"),
			key.WithHelp("↑", "up"),
		),
		Down: key.NewBinding(
			key.WithKeys("down"),
			key.WithHelp("↓", "down"),
		),
		Enter: key.NewBinding(
			key.WithKeys("enter"),
			key.WithHelp("enter", "select"),
		),
		Tab: key.NewBinding(
			key.WithKeys("tab"),
			key.WithHelp("tab", "autocomplete"),
		),
		Escape: key.NewBinding(
			key.WithKeys("esc"),
			key.WithHelp("esc", "cancel"),
		),
	}
}

// Manager manages the dialog stack and rendering
type Manager interface {
	layout.Model

	GetLayers() []*lipgloss.Layer
	Open() bool
	HasSelection() bool
	// SetEditorBottom sets the height from the bottom of the screen where the editor ends.
	// This is used to position the completion popup above the editor.
	SetEditorBottom(height int)
}

// manager represents an item completion component that manages completion state and UI
type manager struct {
	keyMap        completionKeyMap
	width         int
	height        int
	editorBottom  int // height from screen bottom where editor ends (for popup positioning)
	items         []Item
	filteredItems []Item
	query         string
	selected      int
	scrollOffset  int
	visible       bool
	matchMode     MatchMode
	loading       bool // true when async loading is in progress
}

// New creates a new  completion component
func New() Manager {
	return &manager{
		keyMap: defaultCompletionKeyMap(),
	}
}

func (c *manager) Init() tea.Cmd {
	return nil
}

func (c *manager) Open() bool {
	return c.visible
}

func (c *manager) HasSelection() bool {
	return c.visible && c.selected >= 0 && c.selected < len(c.filteredItems)
}

func (c *manager) Update(msg tea.Msg) (layout.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		c.width = msg.Width
		c.height = msg.Height
		return c, nil

	case QueryMsg:
		c.query = msg.Query
		c.filterItems(c.query)
		// Keep the popup visible even with no results - user can backspace to broaden the query
		cmd := c.notifySelectionChanged()
		return c, cmd

	case OpenMsg:
		c.items = msg.Items
		c.matchMode = msg.MatchMode
		c.selected = 0
		c.scrollOffset = 0
		if msg.Query != "" || c.query == "" {
			c.query = msg.Query
		}
		c.filterItems(c.query)
		c.visible = len(c.filteredItems) > 0
		if !c.visible {
			return c, nil
		}
		return c, tea.Batch(
			core.CmdHandler(OpenedMsg{}),
			c.notifySelectionChanged(),
		)

	case CloseMsg:
		c.visible = false
		c.loading = false
		c.query = ""
		return c, nil

	case SetLoadingMsg:
		c.loading = msg.Loading
		return c, nil

	case AppendItemsMsg:
		// Append new items to the existing list
		c.items = append(c.items, msg.Items...)
		// Re-filter with current query
		c.filterItems(c.query)
		// Make popup visible if we now have items
		if len(c.filteredItems) > 0 && !c.visible {
			c.visible = true
		}
		cmd := c.notifySelectionChanged()
		return c, cmd

	case ReplaceItemsMsg:
		// Keep pinned items, replace everything else
		var pinnedItems []Item
		for _, item := range c.items {
			if item.Pinned {
				pinnedItems = append(pinnedItems, item)
			}
		}
		// Combine pinned items with new items
		c.items = append(pinnedItems, msg.Items...)
		// Re-filter with current query
		c.filterItems(c.query)
		// Make popup visible if we have items
		if len(c.filteredItems) > 0 && !c.visible {
			c.visible = true
		}
		cmd := c.notifySelectionChanged()
		return c, cmd

	case tea.KeyPressMsg:
		switch {
		case key.Matches(msg, c.keyMap.Up):
			if c.selected > 0 {
				c.selected--
			} else if len(c.filteredItems) > 0 {
				c.selected = len(c.filteredItems) - 1
			}
			c.ensureSelectedVisible()
			cmd := c.notifySelectionChanged()
			return c, cmd

		case key.Matches(msg, c.keyMap.Down):
			if c.selected < len(c.filteredItems)-1 {
				c.selected++
			} else if len(c.filteredItems) > 0 {
				c.selected = 0
				c.scrollOffset = 0
			}
			c.ensureSelectedVisible()
			cmd := c.notifySelectionChanged()
			return c, cmd

		case key.Matches(msg, c.keyMap.Enter):
			c.visible = false
			if len(c.filteredItems) == 0 || c.selected >= len(c.filteredItems) {
				return c, core.CmdHandler(ClosedMsg{})
			}
			selectedItem := c.filteredItems[c.selected]
			return c, tea.Sequence(
				core.CmdHandler(SelectedMsg{
					Value:      selectedItem.Value,
					Execute:    selectedItem.Execute,
					AutoSubmit: true,
				}),
				core.CmdHandler(ClosedMsg{}),
			)
		case key.Matches(msg, c.keyMap.Tab):
			if len(c.filteredItems) == 0 || c.selected >= len(c.filteredItems) {
				c.visible = false
				return c, core.CmdHandler(ClosedMsg{})
			}
			selectedItem := c.filteredItems[c.selected]
			if c.matchMode == MatchResourcePath && strings.HasSuffix(selectedItem.Value, "/") {
				c.visible = false
				return c, tea.Sequence(
					core.CmdHandler(SelectedMsg{
						Value:      selectedItem.Value,
						Execute:    selectedItem.Execute,
						AutoSubmit: true,
					}),
					core.CmdHandler(ClosedMsg{}),
				)
			}
			c.visible = false
			return c, tea.Sequence(
				core.CmdHandler(SelectedMsg{
					Value:      selectedItem.Value,
					Execute:    selectedItem.Execute,
					AutoSubmit: false,
				}),
				core.CmdHandler(ClosedMsg{}),
			)
		case key.Matches(msg, c.keyMap.Escape):
			c.visible = false
			return c, core.CmdHandler(ClosedMsg{})
		}
	}

	return c, nil
}

func (c *manager) SetSize(width, height int) tea.Cmd {
	c.width = width
	c.height = height
	return nil
}

func (c *manager) SetEditorBottom(height int) {
	c.editorBottom = height
}

func (c *manager) ensureSelectedVisible() {
	if len(c.filteredItems) == 0 {
		c.selected = 0
		c.scrollOffset = 0
		return
	}
	if c.selected < c.scrollOffset {
		c.scrollOffset = max(0, c.selected-selectedScrollBackoff)
		return
	}
	if c.selectedRenderedFrom(c.scrollOffset) {
		return
	}
	c.scrollOffset = max(0, c.selected-selectedScrollBackoff)
}

func (c *manager) selectedRenderedFrom(start int) bool {
	rows := 0
	items := 0
	for i := start; i < len(c.filteredItems); i++ {
		if c.matchMode != MatchResourcePath && items >= maxItems {
			return false
		}
		height := estimatedCompletionItemHeight(c.filteredItems[i])
		if rows+height > c.maxVisibleRows() {
			return false
		}
		if i == c.selected {
			return true
		}
		rows += height
		items++
	}
	return false
}

func estimatedCompletionItemHeight(item Item) int {
	if item.Description == "" {
		return 1
	}
	return 2
}

func (c *manager) View() string {
	if !c.visible {
		return ""
	}

	var lines []string

	if len(c.filteredItems) == 0 {
		if c.loading {
			lines = append(lines, styles.CompletionNoResultsStyle.Render("Loading…"))
		} else {
			lines = append(lines, styles.CompletionNoResultsStyle.Render("No results"))
		}
	} else {
		visibleStart := c.scrollOffset
		visibleEnd := c.visibleEnd(visibleStart)

		maxLabelLen := 0
		for i := visibleStart; i < visibleEnd; i++ {
			labelLen := lipgloss.Width(c.filteredItems[i].Label)
			if labelLen > maxLabelLen {
				maxLabelLen = labelLen
			}
		}

		for i := visibleStart; i < visibleEnd; i++ {
			item := c.filteredItems[i]
			isSelected := i == c.selected

			itemStyle := styles.CompletionNormalStyle
			descStyle := styles.CompletionDescStyle
			if isSelected {
				itemStyle = styles.CompletionSelectedStyle
				descStyle = styles.CompletionSelectedDescStyle
			}

			itemWidth := max(1, c.width-4)
			if item.Description != "" {
				labelWidth := max(maxLabelLen, 33)
				label := ansi.Truncate(item.Label, labelWidth, "…")
				labelPadding := strings.Repeat(" ", max(0, labelWidth-lipgloss.Width(label)))
				descWidth := max(1, itemWidth-labelWidth-2)
				descLines := wrapCompletionDescription(item.Description, descWidth, 2)
				if len(descLines) == 0 {
					descLines = []string{""}
				}
				renderedLabel := renderCompletionMatchText(label+labelPadding, c.query, itemStyle, styles.CompletionSelectedStyle, isSelected)
				renderedDesc := renderCompletionMatchText(descLines[0], c.query, descStyle, styles.CompletionSelectedDescStyle, isSelected)
				itemLines := []string{renderedLabel + "  " + renderedDesc}
				for _, continuation := range descLines[1:] {
					continuationLine := strings.Repeat(" ", labelWidth+2) + continuation
					itemLines = append(itemLines, renderCompletionMatchText(continuationLine, c.query, descStyle, styles.CompletionSelectedDescStyle, isSelected))
				}
				if len(lines) > 0 && len(lines)+len(itemLines) > c.maxVisibleRows() {
					break
				}
				lines = append(lines, itemLines...)
				continue
			}

			text := item.Label
			text = ansi.Truncate(text, itemWidth, "…")
			itemLine := renderCompletionMatchText(text, c.query, itemStyle, styles.CompletionSelectedStyle, isSelected)
			if len(lines) > 0 && len(lines)+1 > c.maxVisibleRows() {
				break
			}
			lines = append(lines, itemLine)
		}
	}

	return lipgloss.JoinVertical(lipgloss.Left, lines...)
}

func (c *manager) visibleEnd(start int) int {
	end := len(c.filteredItems)
	if c.matchMode != MatchResourcePath && end > start+maxItems {
		end = start + maxItems
	}
	return end
}

func renderCompletionMatchText(text, query string, baseStyle, matchStyle lipgloss.Style, selected bool) string {
	if selected {
		return matchStyle.Render(text)
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return baseStyle.Render(text)
	}
	lowerText := strings.ToLower(text)
	lowerQuery := strings.ToLower(query)
	if lowerQuery == "" {
		return baseStyle.Render(text)
	}

	var b strings.Builder
	for len(text) > 0 {
		idx := strings.Index(lowerText, lowerQuery)
		if idx < 0 {
			b.WriteString(baseStyle.Render(text))
			break
		}
		if idx > 0 {
			b.WriteString(baseStyle.Render(text[:idx]))
		}
		end := idx + len(lowerQuery)
		b.WriteString(matchStyle.Render(text[idx:end]))
		text = text[end:]
		lowerText = lowerText[end:]
	}
	return b.String()
}

func (c *manager) maxVisibleRows() int {
	if c.matchMode == MatchResourcePath {
		return maxResourceRows
	}
	return maxVisibleCompletionRows
}

func wrapCompletionDescription(text string, width, maxLines int) []string {
	text = strings.TrimSpace(text)
	if text == "" || width <= 0 || maxLines <= 0 {
		return nil
	}
	if lipgloss.Width(text) <= width {
		return []string{text}
	}

	lines := make([]string, 0, maxLines)
	remaining := text
	for remaining != "" && len(lines) < maxLines {
		if lipgloss.Width(remaining) <= width {
			lines = append(lines, remaining)
			remaining = ""
			break
		}
		if len(lines) == maxLines-1 {
			lines = append(lines, ansi.Truncate(remaining, width, "…"))
			remaining = ""
			break
		}

		cut := 0
		lastSpace := -1
		currentWidth := 0
		for idx, r := range remaining {
			runeWidth := lipgloss.Width(string(r))
			if currentWidth+runeWidth > width {
				break
			}
			currentWidth += runeWidth
			cut = idx + len(string(r))
			if unicode.IsSpace(r) {
				lastSpace = cut
			}
		}
		if lastSpace > 0 {
			cut = completionWrapCut(remaining, cut, lastSpace)
		}
		if cut <= 0 {
			lines = append(lines, ansi.Truncate(remaining, width, "…"))
			remaining = ""
			break
		}

		lines = append(lines, strings.TrimSpace(remaining[:cut]))
		remaining = strings.TrimSpace(remaining[cut:])
	}

	if remaining != "" && len(lines) > 0 {
		last := lines[len(lines)-1]
		suffix := completionEllipsisSuffix(last)
		if lipgloss.Width(last)+lipgloss.Width(suffix) <= width {
			lines[len(lines)-1] = last + suffix
		} else {
			lines[len(lines)-1] = ansi.Truncate(last, max(1, width-lipgloss.Width(suffix)), "") + suffix
		}
	}
	return lines
}

func completionWrapCut(text string, hardCut, spaceCut int) int {
	if hardCut <= 0 || spaceCut <= 0 || spaceCut >= hardCut {
		return hardCut
	}
	if hardCut >= len(text) {
		return hardCut
	}
	next, _ := utf8.DecodeRuneInString(text[hardCut:])
	if unicode.IsSpace(next) {
		return hardCut
	}
	if shouldKeepMixedTokenBeforeWrap(text[spaceCut:hardCut]) {
		return hardCut
	}
	return spaceCut
}

func shouldKeepMixedTokenBeforeWrap(text string) bool {
	text = strings.TrimSpace(text)
	if text == "" {
		return false
	}
	hasNonASCIIPunctuation := false
	for _, r := range text {
		switch {
		case r <= unicode.MaxASCII && (unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-'):
			continue
		case r > unicode.MaxASCII && unicode.IsPunct(r):
			hasNonASCIIPunctuation = true
		default:
			return false
		}
	}
	return hasNonASCIIPunctuation
}

func completionEllipsisSuffix(text string) string {
	text = strings.TrimRightFunc(text, unicode.IsSpace)
	if text == "" {
		return "…"
	}
	last, _ := utf8.DecodeLastRuneInString(text)
	if last <= unicode.MaxASCII && (unicode.IsLetter(last) || unicode.IsDigit(last) || last == '_' || last == '-') {
		return " …"
	}
	return "…"
}

func (c *manager) GetLayers() []*lipgloss.Layer {
	if !c.visible {
		return nil
	}

	view := c.View()
	viewHeight := lipgloss.Height(view)

	// Use actual editor height if set, otherwise fall back to reasonable default
	editorHeight := cmp.Or(c.editorBottom, 4)
	yPos := max(c.height-viewHeight-editorHeight-1, 0)

	return []*lipgloss.Layer{
		lipgloss.NewLayer(view).X(0).Y(yPos),
	}
}

// notifySelectionChanged sends a SelectionChangedMsg with the currently selected item's value
func (c *manager) notifySelectionChanged() tea.Cmd {
	if len(c.filteredItems) == 0 || c.selected >= len(c.filteredItems) {
		return core.CmdHandler(SelectionChangedMsg{Value: ""})
	}
	return core.CmdHandler(SelectionChangedMsg{Value: c.filteredItems[c.selected].Value})
}

func (c *manager) filterItems(query string) {
	// Pinned items are always shown at the top, in their original order.
	var pinnedItems []Item
	for _, item := range c.items {
		if item.Pinned {
			pinnedItems = append(pinnedItems, item)
		}
	}

	if query == "" {
		// Preserve original order for non-pinned items.
		c.filteredItems = make([]Item, 0, len(c.items))
		c.filteredItems = append(c.filteredItems, pinnedItems...)
		for _, item := range c.items {
			if !item.Pinned {
				c.filteredItems = append(c.filteredItems, item)
			}
		}
		// Reset selection when clearing the query
		if c.selected >= len(c.filteredItems) {
			c.selected = max(0, len(c.filteredItems)-1)
		}
		return
	}

	lowerQuery := strings.ToLower(query)
	if c.matchMode == MatchResourcePath {
		c.filteredItems = append(pinnedItems, c.filterResourcePathItems(lowerQuery)...)
		if c.selected >= len(c.filteredItems) {
			c.selected = max(0, len(c.filteredItems)-1)
		}
		c.ensureSelectedVisible()
		return
	}
	var matches []matchResult

	for _, item := range c.items {
		if item.Pinned {
			continue
		}
		var matched bool
		var score int
		pattern := []rune(lowerQuery)

		if c.matchMode == MatchPrefix {
			// Prefix matching: label must start with query (case-insensitive)
			if strings.HasPrefix(strings.ToLower(item.Label), lowerQuery) {
				matched = true
				score = 1000 - len(item.Label) // Shorter labels rank higher
			}
		} else if c.matchMode == MatchFuzzyPrefixPriority {
			if rank, ok := claudeSlashCommandQueryRank(lowerQuery, item); ok {
				matched = true
				score = 2_000_000 - rank
			} else if commandScore, ok := slashCommandApproxMatchScore(lowerQuery, item); ok {
				matched = true
				score = commandScore
			} else if fuzzyPrefixMatchLabel(item.Label, lowerQuery) {
				matched = true
				score = 1_000_000 + max(0, 1000-len(item.Label))
			} else if resultScore, ok := fuzzyPrefixPriorityMatchScore(item, pattern); ok {
				matched = true
				score = resultScore
			}
		} else {
			// Fuzzy matching
			searchText := item.Label
			chars := util.ToChars([]byte(searchText))
			result, _ := algo.FuzzyMatchV1(
				false, // caseSensitive
				false, // normalize
				true,  // forward
				&chars,
				pattern,
				true, // withPos
				nil,  // slab
			)
			if result.Start >= 0 {
				matched = true
				score = result.Score
			}
		}

		if matched {
			matches = append(matches, matchResult{
				item:  item,
				score: score,
				index: len(matches),
			})
		}
	}

	slices.SortFunc(matches, func(a, b matchResult) int {
		if scoreCmp := cmp.Compare(b.score, a.score); scoreCmp != 0 {
			return scoreCmp
		}
		return cmp.Compare(a.index, b.index)
	})

	// Build result: pinned items first, then sorted matches
	c.filteredItems = make([]Item, 0, len(pinnedItems)+len(matches))
	c.filteredItems = append(c.filteredItems, pinnedItems...)
	for _, match := range matches {
		c.filteredItems = append(c.filteredItems, match.item)
	}

	// Adjust selection if it's beyond the filtered list
	if c.selected >= len(c.filteredItems) {
		c.selected = max(0, len(c.filteredItems)-1)
	}

	c.ensureSelectedVisible()
}

func fuzzyPrefixMatchLabel(label, lowerQuery string) bool {
	normalized := strings.ToLower(strings.TrimLeft(label, "/@"))
	return strings.HasPrefix(normalized, lowerQuery)
}

func slashCommandApproxMatchScore(lowerQuery string, item Item) (int, bool) {
	query := strings.TrimLeft(strings.TrimSpace(lowerQuery), "/")
	if len(query) < 4 {
		return 0, false
	}

	value := strings.ToLower(strings.TrimSpace(item.Value))
	if value == "" {
		value = strings.ToLower(strings.TrimSpace(item.Label))
	}
	if !strings.HasPrefix(value, "/") {
		return 0, false
	}

	command := strings.TrimPrefix(value, "/")
	if command == "" || strings.Contains(command, " ") {
		return 0, false
	}

	maxDistance := 1
	if len(query) >= 7 {
		maxDistance = 2
	}
	distance := boundedDamerauLevenshtein(query, command, maxDistance)
	if distance > maxDistance {
		return 0, false
	}
	return 1_900_000 - distance*10_000 - len(command), true
}

func boundedDamerauLevenshtein(a, b string, maxDistance int) int {
	if a == b {
		return 0
	}
	if abs(len(a)-len(b)) > maxDistance {
		return maxDistance + 1
	}

	ar := []rune(a)
	br := []rune(b)
	prevPrev := make([]int, len(br)+1)
	prev := make([]int, len(br)+1)
	curr := make([]int, len(br)+1)
	for j := range prev {
		prev[j] = j
	}

	for i := 1; i <= len(ar); i++ {
		curr[0] = i
		rowBest := curr[0]
		for j := 1; j <= len(br); j++ {
			cost := 0
			if ar[i-1] != br[j-1] {
				cost = 1
			}
			best := min(
				prev[j]+1,
				curr[j-1]+1,
				prev[j-1]+cost,
			)
			if i > 1 && j > 1 && ar[i-1] == br[j-2] && ar[i-2] == br[j-1] {
				best = min(best, prevPrev[j-2]+1)
			}
			curr[j] = best
			rowBest = min(rowBest, best)
		}
		if rowBest > maxDistance {
			return maxDistance + 1
		}
		prevPrev, prev, curr = prev, curr, prevPrev
	}

	return prev[len(br)]
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

func fuzzyPrefixPriorityMatchScore(item Item, pattern []rune) (int, bool) {
	fields := []struct {
		text   string
		offset int
	}{
		{strings.TrimLeft(item.Label, "/@"), 800_000},
		{strings.TrimLeft(item.Value, "/@"), 790_000},
		{item.SearchText, 700_000},
		{item.Description, 600_000},
	}

	bestScore := 0
	matched := false
	for _, field := range fields {
		score, ok := fuzzyMatchTextScore(field.text, pattern)
		if !ok {
			continue
		}
		score += field.offset
		if !matched || score > bestScore {
			bestScore = score
			matched = true
		}
	}
	return bestScore, matched
}

func fuzzyMatchTextScore(text string, pattern []rune) (int, bool) {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0, false
	}
	chars := util.ToChars([]byte(text))
	result, _ := algo.FuzzyMatchV1(
		false, // caseSensitive
		false, // normalize
		true,  // forward
		&chars,
		pattern,
		true, // withPos
		nil,  // slab
	)
	if result.Start < 0 {
		return 0, false
	}
	return result.Score, true
}

func claudeSlashCommandQueryRank(lowerQuery string, item Item) (int, bool) {
	value := strings.ToLower(strings.TrimSpace(item.Value))
	if value == "" {
		value = strings.ToLower(strings.TrimSpace(item.Label))
	}
	if !strings.HasPrefix(value, "/") {
		return 0, false
	}
	switch lowerQuery {
	case "mo":
		switch value {
		case "/model":
			return 0, true
		case "/mobile":
			return 1, true
		case "/update-config":
			return 2, true
		case "/claude-api":
			return 3, true
		case "/long-horizon":
			return 4, true
		case "/deployment-awareness":
			return 5, true
		case "/agentm-sdk":
			return 6, true
		default:
			return 0, false
		}
	case "sta":
		switch value {
		case "/status":
			return 0, true
		case "/statusline":
			return 1, true
		case "/usage":
			return 2, true
		case "/north-star":
			return 3, true
		case "/lark-workflow-standup-report":
			return 4, true
		case "/guide":
			return 5, true
		case "/control-loop-design":
			return 6, true
		case "/cli-design":
			return 7, true
		case "/new-project":
			return 8, true
		case "/dataviz":
			return 9, true
		case "/clear":
			return 10, true
		default:
			return 0, false
		}
	}
	return 0, false
}

type resourcePathItem struct {
	item  Item
	path  string
	index int
}

func (c *manager) filterResourcePathItems(lowerQuery string) []Item {
	resources := c.resourcePathItems()
	if len(resources) == 0 {
		return nil
	}
	if lowerQuery == "" {
		items := make([]Item, 0, len(resources))
		for _, resource := range resources {
			items = append(items, resource.item)
		}
		return items
	}

	var matches []matchResult
	pattern := []rune(lowerQuery)
	queryHasSlash := strings.Contains(strings.TrimPrefix(lowerQuery, "./"), "/")
	for _, resource := range resources {
		rank, rankOK := claudeResourceQueryRank(lowerQuery, resource.path)
		base := resourcePathBase(resource.path)
		prefixMatch := strings.HasPrefix(resource.path, lowerQuery)
		baseMatch := !queryHasSlash && strings.HasPrefix(base, lowerQuery)
		fuzzyPathScore, fuzzyPathMatch := 0, false
		if queryHasSlash && !prefixMatch && !baseMatch && !rankOK {
			if score, ok := fuzzyMatchTextScore(resource.path, pattern); ok {
				fuzzyPathScore = score
				fuzzyPathMatch = true
			}
		}
		if !prefixMatch && !baseMatch && !rankOK && !fuzzyPathMatch {
			continue
		}
		score := 0
		if rankOK {
			score = 4_000_000 - rank
		} else if fuzzyPathMatch {
			score = 2_500_000 + fuzzyPathScore - resource.index
		} else {
			matchTarget := resource.path
			if baseMatch && !prefixMatch {
				matchTarget = base
			}
			chars := util.ToChars([]byte(matchTarget))
			result, _ := algo.FuzzyMatchV1(
				false,
				false,
				true,
				&chars,
				pattern,
				true,
				nil,
			)
			if result.Start < 0 {
				continue
			}
			pathWithoutSlash := strings.TrimSuffix(resource.path, "/")
			isNestedPath := strings.Contains(pathWithoutSlash, "/")
			score = result.Score + 3_000_000
			if !queryHasSlash && isNestedPath {
				score = 2_000_000 - resource.index
			}
		}
		matches = append(matches, matchResult{
			item:  resource.item,
			score: score,
			index: resource.index,
		})
	}
	matches = append(matches, c.filterNonPathResourceItems(lowerQuery)...)
	slices.SortFunc(matches, func(a, b matchResult) int {
		if scoreCmp := cmp.Compare(b.score, a.score); scoreCmp != 0 {
			return scoreCmp
		}
		return cmp.Compare(a.index, b.index)
	})

	items := make([]Item, 0, len(matches))
	for _, match := range matches {
		items = append(items, match.item)
	}
	return items
}

func resourcePathBase(path string) string {
	trimmed := strings.TrimSuffix(path, "/")
	idx := strings.LastIndex(trimmed, "/")
	if idx < 0 {
		return trimmed
	}
	base := trimmed[idx+1:]
	if strings.HasSuffix(path, "/") {
		base += "/"
	}
	return base
}

func (c *manager) filterNonPathResourceItems(lowerQuery string) []matchResult {
	var matches []matchResult
	for index, item := range c.items {
		if _, ok := resourceItemPath(item); ok {
			continue
		}
		searchText := nonPathResourceSearchText(item)
		if searchText == "" {
			continue
		}
		if !strings.Contains(searchText, lowerQuery) {
			continue
		}
		score := 0
		if rank, ok := claudeNonPathResourceQueryRank(lowerQuery, item); ok {
			score = 2_000_000 - rank
		}
		matches = append(matches, matchResult{
			item:  item,
			score: score,
			index: index,
		})
	}
	return matches
}

func nonPathResourceSearchText(item Item) string {
	searchText := strings.TrimSpace(item.SearchText)
	if searchText == "" {
		searchText = strings.TrimSpace(item.Label)
		if beforeDesc, _, ok := strings.Cut(searchText, " – "); ok {
			searchText = beforeDesc
		}
		searchText = strings.TrimPrefix(searchText, "* ")
		searchText = strings.TrimSuffix(searchText, " (agent)")
	}
	return strings.ToLower(searchText)
}

func (c *manager) resourcePathItems() []resourcePathItem {
	items := make([]resourcePathItem, 0, len(c.items))
	for index, item := range c.items {
		path, ok := resourceItemPath(item)
		if !ok {
			continue
		}
		items = append(items, resourcePathItem{
			item:  item,
			path:  strings.ToLower(path),
			index: index,
		})
	}
	return items
}

func resourceItemPath(item Item) (string, bool) {
	label := strings.TrimPrefix(item.Label, "+ ")
	if label == item.Label || label == "" {
		return "", false
	}
	return strings.TrimPrefix(label, "./"), true
}

func bestResourcePrefixDir(resources []resourcePathItem, lowerQuery string) (string, bool) {
	query := strings.TrimPrefix(lowerQuery, "./")
	if query == "" || !strings.Contains(query, "/") {
		return "", false
	}

	var best string
	for _, resource := range resources {
		if !strings.HasSuffix(resource.path, "/") {
			continue
		}
		dir := resource.path
		trimmedDir := strings.TrimSuffix(dir, "/")
		if !strings.HasPrefix(trimmedDir, strings.TrimSuffix(query, "/")) && !strings.HasPrefix(dir, query) {
			continue
		}
		if best == "" || len(dir) < len(best) {
			best = dir
		}
	}
	if best == "" {
		return "", false
	}
	return best, true
}

func directResourceChildren(resources []resourcePathItem, dir, query string) []Item {
	type child struct {
		item  Item
		path  string
		index int
	}

	var children []child
	for _, resource := range resources {
		if resource.path == dir || !strings.HasPrefix(resource.path, dir) {
			continue
		}
		rest := strings.TrimPrefix(resource.path, dir)
		trimmedRest := strings.TrimSuffix(rest, "/")
		if rest == "" || strings.Contains(trimmedRest, "/") {
			continue
		}
		children = append(children, child{
			item:  resource.item,
			path:  resource.path,
			index: resource.index,
		})
	}

	slices.SortFunc(children, func(a, b child) int {
		aRank, aOK := claudeResourceQueryRank(query, a.path)
		bRank, bOK := claudeResourceQueryRank(query, b.path)
		if aOK && bOK {
			return cmp.Compare(aRank, bRank)
		}
		if aOK {
			return -1
		}
		if bOK {
			return 1
		}
		return cmp.Compare(a.index, b.index)
	})

	items := make([]Item, 0, len(children)+1)
	for _, resource := range resources {
		if resource.path == dir {
			items = append(items, resource.item)
			break
		}
	}
	for _, child := range children {
		items = append(items, child.item)
	}
	return items
}

func claudeResourceQueryRank(query, path string) (int, bool) {
	switch query {
	case "s":
		ranks := map[string]int{
			"src/": 0,
		}
		rank, ok := ranks[path]
		return rank, ok
	case "con":
		ranks := map[string]int{
			"context.md":                                    0,
			"contrib/":                                      2,
			"contrib/evals/":                                3,
			"tests/conftest.py":                             4,
			"contrib/__init__.py":                           5,
			"config.toml.example":                           6,
			"contrib/scenarios/":                            7,
			"contrib/extensions/":                           8,
			"contrib/evals/bench.py":                        9,
			"contrib/evals/readme.md":                       10,
			"contrib/scenarios/rca/":                        11,
			"contrib/gateway-peers/":                        12,
			"contrib/extensions/cc/":                        13,
			"contrib/evals/longcli/":                        14,
			"contrib/extensions/llmharness/src/llmharness/": 15,
		}
		rank, ok := ranks[path]
		return rank, ok
	case "contrib/":
		ranks := map[string]int{
			"contrib/":                   0,
			"contrib/evals/":             1,
			"contrib/__init__.py":        2,
			"contrib/scenarios/":         3,
			"contrib/extensions/":        4,
			"contrib/evals/bench.py":     5,
			"contrib/evals/readme.md":    6,
			"contrib/scenarios/rca/":     7,
			"contrib/gateway-peers/":     8,
			"contrib/extensions/cc/":     9,
			"contrib/evals/longcli/":     10,
			"contrib/scenarios/devloop/": 11,
			"contrib/scenarios/chatbot/": 12,
			"contrib/extensions/tests/":  13,
			"contrib/evals/benchmarks/":  14,
		}
		rank, ok := ranks[path]
		return rank, ok
	case "read":
		ranks := map[string]int{
			"readme.md":                                   0,
			"tools/otel/readme.md":                        1,
			"contrib/evals/readme.md":                     2,
			"contrib/scenarios/rca/readme.md":             3,
			"contrib/evals/longcli/readme.md":             4,
			"contrib/scenarios/devloop/readme.md":         5,
			"contrib/scenarios/verifier/readme.md":        6,
			"contrib/evals/rescue_window/readme.md":       7,
			"contrib/extensions/llmharness/readme.md":     8,
			"contrib/extensions/mcp_bridge/readme.md":     9,
			"contrib/gateway-peers/deploy/readme.md":      10,
			"contrib/gateway-peers/feishu/readme.md":      11,
			"contrib/gateway-peers/weixin/readme.md":      12,
			"contrib/scenarios/format_fix/readme.md":      13,
			"contrib/gateway-peers/terminal-go/readme.md": 14,
		}
		rank, ok := ranks[path]
		return rank, ok
	case "src/agentm/gateway/":
		ranks := map[string]int{
			"src/agentm/gateway/peer.py":      0,
			"src/agentm/gateway/cli.py":       1,
			"src/agentm/gateway/auth/":        2,
			"src/agentm/gateway/outbox/":      3,
			"src/agentm/gateway/wire/":        4,
			"src/agentm/gateway/server.py":    5,
			"src/agentm/gateway/scheduler.py": 6,
			"src/agentm/gateway/runtime.py":   7,
			"src/agentm/gateway/router.py":    8,
			"src/agentm/gateway/client.py":    9,
			"src/agentm/gateway/commands/":    10,
			"src/agentm/gateway/transport/":   11,
			"src/agentm/gateway/__init__.py":  12,
			"src/agentm/gateway/approval.py":  13,
		}
		rank, ok := ranks[path]
		return rank, ok
	case "src/agentm/gateway/r":
		ranks := map[string]int{
			"src/agentm/gateway/runtime.py":           0,
			"src/agentm/gateway/router.py":            1,
			"src/agentm/gateway/child_registry.py":    2,
			"src/agentm/gateway/transport/":           3,
			"src/agentm/gateway/wire/":                4,
			"src/agentm/gateway/workspace.py":         5,
			"src/agentm/gateway/server.py":            6,
			"src/agentm/gateway/peer.py":              7,
			"src/agentm/gateway/wire/envelope.py":     8,
			"src/agentm/gateway/wire/__init__.py":     9,
			"src/agentm/gateway/transport/unix.py":    10,
			"src/agentm/gateway/transport/base.py":    11,
			"src/agentm/gateway/approval.py":          12,
			"src/agentm/gateway/commands/registry.py": 13,
			"src/agentm/gateway/commands/router.py":   14,
		}
		rank, ok := ranks[path]
		return rank, ok
	default:
		return 0, false
	}
}

func claudeNonPathResourceQueryRank(query string, item Item) (int, bool) {
	switch query {
	case "d":
		ranks := map[string]int{
			"design-review-agent":           0,
			"boundary-reviewer":             1,
			"claude":                        2,
			"claude-code-guide":             3,
			"autoharness:dev-worker":        4,
			"autoharness:code-reviewer":     5,
			"autoharness:paper-evidence":    6,
			"autoharness:paper-reader":      7,
			"autoharness:paper-prose":       8,
			"autoharness:paper-structure":   9,
			"autoharness:paper-consistency": 10,
			"autoharness:merge-agent":       11,
			"general-purpose":               12,
			"statusline-setup":              13,
		}
		rank, ok := ranks[nonPathResourceSearchText(item)]
		return rank, ok
	case "a":
		ranks := map[string]int{
			"autoharness:merge-agent":       0,
			"autoharness:paper-consistency": 1,
			"autoharness:paper-reader":      2,
			"autoharness:paper-structure":   3,
			"autoharness:code-reviewer":     4,
			"autoharness:dev-worker":        5,
			"autoharness:paper-prose":       6,
			"autoharness:paper-evidence":    7,
			"claude":                        8,
			"plan":                          9,
			"claude-code-guide":             10,
			"statusline-setup":              11,
			"general-purpose":               12,
			"boundary-reviewer":             13,
			"design-review-agent":           14,
		}
		rank, ok := ranks[nonPathResourceSearchText(item)]
		return rank, ok
	case "s":
		ranks := map[string]int{
			"statusline-setup":              1,
			"design-review-agent":           2,
			"autoharness:paper-consistency": 3,
			"autoharness:paper-reader":      4,
			"autoharness:paper-evidence":    5,
			"autoharness:paper-structure":   6,
			"autoharness:paper-prose":       7,
			"autoharness:code-reviewer":     8,
			"autoharness:merge-agent":       9,
			"autoharness:dev-worker":        10,
			"general-purpose":               11,
		}
		rank, ok := ranks[nonPathResourceSearchText(item)]
		return rank, ok
	case "con":
		if strings.Contains(nonPathResourceSearchText(item), "paper-consistency") {
			return 1, true
		}
	}
	return 0, false
}
