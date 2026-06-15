package app

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/blocks"
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

func testTheme() *theme.Theme { return theme.DarkTheme() }

// assistantWithText is a helper that creates an AssistantTurn with a single
// TextBlock segment, mirroring the new Segments model.
func assistantWithText(text string) *blocks.AssistantTurn {
	tb := &blocks.TextBlock{Text: text}
	return &blocks.AssistantTurn{Segments: []blocks.Block{tb}}
}

func keyMsg(s string) tea.KeyMsg {
	// Map simple key strings to KeyMsg
	switch s {
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEsc}
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "up":
		return tea.KeyMsg{Type: tea.KeyUp}
	case "down":
		return tea.KeyMsg{Type: tea.KeyDown}
	case "backspace":
		return tea.KeyMsg{Type: tea.KeyBackspace}
	case "tab":
		return tea.KeyMsg{Type: tea.KeyTab}
	default:
		if len(s) == 1 {
			return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
		}
		return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
	}
}

// -- Overlay interface compliance --

func TestOverlayInterfaceCompliance(t *testing.T) {
	var _ Overlay = &HelpOverlay{}
	var _ Overlay = &SearchOverlay{}
	var _ Overlay = &BookmarkOverlay{}
	var _ Overlay = &ResendOverlay{}
	var _ Overlay = &CodeSaveOverlay{}
	var _ Overlay = &ApprovalOverlay{}
	var _ Overlay = &PaletteOverlay{}
	var _ Overlay = &RejectionOverlay{}
	var _ Overlay = &ElicitationOverlay{}
}

// -- HelpOverlay --

func TestHelpOverlayKind(t *testing.T) {
	h := NewHelpOverlay()
	if h.Kind() != OverlayHelp {
		t.Errorf("expected OverlayHelp, got %d", h.Kind())
	}
}

func TestHelpOverlayClosesOnAnyKey(t *testing.T) {
	h := NewHelpOverlay()
	_, _, closed := h.Update(keyMsg("a"))
	if !closed {
		t.Error("HelpOverlay should close on any keypress")
	}
}

func TestHelpOverlayViewNonEmpty(t *testing.T) {
	h := NewHelpOverlay()
	view := h.View(80, 40, testTheme())
	if view == "" {
		t.Error("HelpOverlay.View should produce non-empty output")
	}
	if !strings.Contains(view, "AgentM Terminal") {
		t.Error("HelpOverlay should contain title")
	}
	if !strings.Contains(view, "Ctrl+F") {
		t.Error("HelpOverlay should list Ctrl+F keybinding")
	}
}

// -- SearchOverlay --

func TestSearchOverlayKind(t *testing.T) {
	s := NewSearchOverlay(nil)
	if s.Kind() != OverlaySearch {
		t.Errorf("expected OverlaySearch, got %d", s.Kind())
	}
}

func TestSearchOverlayFindsMatches(t *testing.T) {
	transcript := []blocks.Block{
		&blocks.UserTurn{Content: "hello world"},
		assistantWithText("Hello back to you"),
	}
	s := NewSearchOverlay(transcript)

	// Type "hello"
	for _, ch := range "hello" {
		s.Update(keyMsg(string(ch)))
	}

	if s.Query() != "hello" {
		t.Errorf("expected query 'hello', got %q", s.Query())
	}
	if len(s.matches) != 2 {
		t.Errorf("expected 2 matches for 'hello', got %d", len(s.matches))
	}
}

func TestSearchOverlayEscCloses(t *testing.T) {
	s := NewSearchOverlay(nil)
	_, _, closed := s.Update(keyMsg("esc"))
	if !closed {
		t.Error("Esc should close SearchOverlay")
	}
}

func TestSearchOverlayNavigateMatches(t *testing.T) {
	transcript := []blocks.Block{
		&blocks.UserTurn{Content: "test one"},
		&blocks.UserTurn{Content: "test two"},
		&blocks.UserTurn{Content: "test three"},
	}
	s := NewSearchOverlay(transcript)
	for _, ch := range "test" {
		s.Update(keyMsg(string(ch)))
	}

	if len(s.matches) != 3 {
		t.Fatalf("expected 3 matches, got %d", len(s.matches))
	}
	if s.cursor != 0 {
		t.Errorf("expected cursor at 0, got %d", s.cursor)
	}

	// Next match
	s.Update(keyMsg("enter"))
	if s.cursor != 1 {
		t.Errorf("expected cursor at 1, got %d", s.cursor)
	}

	// Wrap around
	s.Update(keyMsg("enter"))
	s.Update(keyMsg("enter"))
	if s.cursor != 0 {
		t.Errorf("expected cursor wrapped to 0, got %d", s.cursor)
	}
}

func TestSearchOverlayBackspace(t *testing.T) {
	s := NewSearchOverlay(nil)
	for _, ch := range "abc" {
		s.Update(keyMsg(string(ch)))
	}
	if s.Query() != "abc" {
		t.Fatalf("expected 'abc', got %q", s.Query())
	}
	s.Update(keyMsg("backspace"))
	if s.Query() != "ab" {
		t.Errorf("expected 'ab' after backspace, got %q", s.Query())
	}
}

// -- BookmarkOverlay --

func TestBookmarkOverlayKind(t *testing.T) {
	b := NewBookmarkOverlay(nil)
	if b.Kind() != OverlayBookmarks {
		t.Errorf("expected OverlayBookmarks, got %d", b.Kind())
	}
}

func TestBookmarkOverlayNavigation(t *testing.T) {
	bms := []Bookmark{
		{BlockIndex: 0, Label: "first"},
		{BlockIndex: 2, Label: "second"},
		{BlockIndex: 4, Label: "third"},
	}
	b := NewBookmarkOverlay(bms)

	b.Update(keyMsg("down"))
	if b.cursor != 1 {
		t.Errorf("expected cursor 1, got %d", b.cursor)
	}

	b.Update(keyMsg("up"))
	if b.cursor != 0 {
		t.Errorf("expected cursor 0, got %d", b.cursor)
	}
}

func TestBookmarkOverlayDelete(t *testing.T) {
	bms := []Bookmark{
		{BlockIndex: 0, Label: "first"},
		{BlockIndex: 2, Label: "second"},
	}
	b := NewBookmarkOverlay(bms)

	b.Update(keyMsg("d"))
	remaining := b.Bookmarks()
	if len(remaining) != 1 {
		t.Fatalf("expected 1 bookmark after delete, got %d", len(remaining))
	}
	if remaining[0].Label != "second" {
		t.Errorf("expected 'second' to remain, got %q", remaining[0].Label)
	}
}

func TestBookmarkOverlayJump(t *testing.T) {
	bms := []Bookmark{
		{BlockIndex: 5, Label: "target"},
	}
	b := NewBookmarkOverlay(bms)

	_, _, closed := b.Update(keyMsg("enter"))
	if !closed {
		t.Error("Enter should close BookmarkOverlay")
	}
	if b.JumpTarget() != 5 {
		t.Errorf("expected jump target 5, got %d", b.JumpTarget())
	}
}

func TestBookmarkOverlayEscCloses(t *testing.T) {
	b := NewBookmarkOverlay(nil)
	_, _, closed := b.Update(keyMsg("esc"))
	if !closed {
		t.Error("Esc should close BookmarkOverlay")
	}
}

// -- ResendOverlay --

func TestResendOverlayKind(t *testing.T) {
	r := NewResendOverlay(nil)
	if r.Kind() != OverlayResend {
		t.Errorf("expected OverlayResend, got %d", r.Kind())
	}
}

func TestResendOverlayFiltering(t *testing.T) {
	history := []string{
		"explain the code",
		"run tests",
		"explain architecture",
	}
	r := NewResendOverlay(history)

	// All history visible initially (most recent first)
	if len(r.candidates) != 3 {
		t.Fatalf("expected 3 candidates, got %d", len(r.candidates))
	}
	if r.candidates[0] != "explain architecture" {
		t.Errorf("expected most recent first, got %q", r.candidates[0])
	}

	// Filter for "explain"
	for _, ch := range "explain" {
		r.Update(keyMsg(string(ch)))
	}
	if len(r.candidates) != 2 {
		t.Errorf("expected 2 candidates for 'explain', got %d", len(r.candidates))
	}
}

func TestResendOverlayTabEdits(t *testing.T) {
	history := []string{"test command"}
	r := NewResendOverlay(history)

	_, _, closed := r.Update(keyMsg("tab"))
	if !closed {
		t.Error("Tab should close ResendOverlay")
	}
	if !r.WantsEdit() {
		t.Error("Tab should set WantsEdit")
	}
	if r.Chosen() != "test command" {
		t.Errorf("expected chosen 'test command', got %q", r.Chosen())
	}
}

// -- CodeSaveOverlay --

func TestCodeSaveOverlayKind(t *testing.T) {
	transcript := []blocks.Block{
		assistantWithText("```go\npackage main\n```"),
	}
	c := NewCodeSaveOverlay(transcript)
	if c == nil {
		t.Fatal("expected non-nil CodeSaveOverlay")
	}
	if c.Kind() != OverlayCodeSave {
		t.Errorf("expected OverlayCodeSave, got %d", c.Kind())
	}
}

func TestCodeSaveOverlayNilWithoutCodeBlocks(t *testing.T) {
	transcript := []blocks.Block{
		assistantWithText("just plain text"),
	}
	c := NewCodeSaveOverlay(transcript)
	if c != nil {
		t.Error("expected nil CodeSaveOverlay when no code blocks")
	}
}

func TestCodeSaveOverlayDetectsBlocks(t *testing.T) {
	text := "Here:\n```go\nfunc main() {}\n```\nAnd:\n```python\nprint('hi')\n```"
	transcript := []blocks.Block{
		assistantWithText(text),
	}
	c := NewCodeSaveOverlay(transcript)
	if c == nil {
		t.Fatal("expected non-nil CodeSaveOverlay")
	}
	if len(c.codeBlocks) != 2 {
		t.Errorf("expected 2 code blocks, got %d", len(c.codeBlocks))
	}
	if c.codeBlocks[0].Lang != "go" {
		t.Errorf("expected first block lang 'go', got %q", c.codeBlocks[0].Lang)
	}
	if c.codeBlocks[1].Lang != "python" {
		t.Errorf("expected second block lang 'python', got %q", c.codeBlocks[1].Lang)
	}
	// Multiple blocks should start at phase 0 (selection)
	if c.phase != 0 {
		t.Errorf("expected phase 0, got %d", c.phase)
	}
}

func TestCodeSaveOverlaySingleBlockSkipsSelection(t *testing.T) {
	text := "```bash\necho hello\n```"
	transcript := []blocks.Block{
		assistantWithText(text),
	}
	c := NewCodeSaveOverlay(transcript)
	if c == nil {
		t.Fatal("expected non-nil CodeSaveOverlay")
	}
	if c.phase != 1 {
		t.Errorf("single block should skip to phase 1, got %d", c.phase)
	}
}

func TestCodeSaveOverlayEscCloses(t *testing.T) {
	text := "```go\nfunc main() {}\n```"
	transcript := []blocks.Block{
		assistantWithText(text),
	}
	c := NewCodeSaveOverlay(transcript)
	_, _, closed := c.Update(keyMsg("esc"))
	if !closed {
		t.Error("Esc should close CodeSaveOverlay")
	}
}

// -- suggestFilename --

func TestSuggestFilename(t *testing.T) {
	tests := []struct {
		lang     string
		idx      int
		expected string
	}{
		{"go", 0, "snippet.go"},
		{"python", 0, "snippet.py"},
		{"js", 1, "snippet_2.js"},
		{"", 0, "snippet.txt"},
		{"rust", 0, "snippet.rs"},
		{"yaml", 0, "snippet.yaml"},
	}
	for _, tt := range tests {
		got := suggestFilename(tt.lang, tt.idx)
		if got != tt.expected {
			t.Errorf("suggestFilename(%q, %d) = %q, want %q", tt.lang, tt.idx, got, tt.expected)
		}
	}
}

// -- centerOverlay --

func TestCenterOverlayProducesOutput(t *testing.T) {
	content := "hello\nworld"
	result := centerOverlay(content, 40, 10)
	if result == "" {
		t.Error("centerOverlay should produce non-empty output")
	}
	lines := strings.Split(result, "\n")
	// Should have approximately height lines
	if len(lines) < 5 {
		t.Errorf("expected at least 5 lines for height=10, got %d", len(lines))
	}
}

// -- highlightMatches --

func TestHighlightMatchesCaseInsensitive(t *testing.T) {
	th := testTheme()
	result := highlightMatches("Hello World hello", "hello", th)
	// In a non-TTY test environment, lipgloss may not emit ANSI codes,
	// so we just verify the original text content is preserved.
	if !strings.Contains(result, "Hello") || !strings.Contains(result, "hello") {
		t.Errorf("highlightMatches should preserve matched text, got %q", result)
	}
	if !strings.Contains(result, " World ") {
		t.Errorf("highlightMatches should preserve non-matched text, got %q", result)
	}
}

func TestHighlightMatchesEmptyQuery(t *testing.T) {
	th := testTheme()
	input := "Hello World"
	result := highlightMatches(input, "", th)
	if result != input {
		t.Error("empty query should return input unchanged")
	}
}

// -- blockPlainText --

func TestBlockPlainText(t *testing.T) {
	u := &blocks.UserTurn{Content: "user text"}
	if blockPlainText(u) != "user text" {
		t.Errorf("unexpected plain text for UserTurn")
	}

	a := assistantWithText("assistant text")
	pt := blockPlainText(a)
	if !strings.Contains(pt, "assistant text") {
		t.Errorf("expected 'assistant text' in plain text, got %q", pt)
	}

	s := &blocks.SystemTurn{Content: "system text"}
	if blockPlainText(s) != "system text" {
		t.Errorf("unexpected plain text for SystemTurn")
	}
}

// -- blockLabel --

func TestBlockLabel(t *testing.T) {
	u := &blocks.UserTurn{Content: "short"}
	if blockLabel(u) != "short" {
		t.Errorf("expected 'short', got %q", blockLabel(u))
	}

	long := &blocks.UserTurn{Content: strings.Repeat("a", 50)}
	label := blockLabel(long)
	if len(label) > 40 {
		t.Errorf("label should be truncated to 40 chars, got %d", len(label))
	}
	if !strings.HasSuffix(label, "...") {
		t.Error("truncated label should end with ...")
	}
}

// -- ApprovalOverlay --

func TestApprovalOverlayKind(t *testing.T) {
	o := NewApprovalOverlay("test", "Bash", nil, []blocks.Button{
		{Label: "Allow", Value: "allow"},
	})
	if o.Kind() != OverlayApproval {
		t.Errorf("expected OverlayApproval, got %d", o.Kind())
	}
}

func TestApprovalOverlayDigitSelectsButton(t *testing.T) {
	buttons := []blocks.Button{
		{Label: "Allow", Value: "allow"},
		{Label: "Deny", Value: "deny"},
		{Label: "Always", Value: "always"},
	}
	o := NewApprovalOverlay("test", "Bash", nil, buttons)

	_, _, closed := o.Update(keyMsg("1"))
	if !closed {
		t.Error("digit 1 should close the overlay")
	}
	if !o.Resolved() {
		t.Error("should be resolved after digit selection")
	}
	if o.Chosen() != "allow" {
		t.Errorf("expected 'allow', got %q", o.Chosen())
	}
}

func TestApprovalOverlayLetterShortcuts(t *testing.T) {
	buttons := []blocks.Button{
		{Label: "Allow", Value: "allow"},
		{Label: "Deny", Value: "deny"},
		{Label: "Always", Value: "always"},
	}

	// y = first button
	o := NewApprovalOverlay("test", "Bash", nil, buttons)
	_, _, closed := o.Update(keyMsg("y"))
	if !closed || o.Chosen() != "allow" {
		t.Errorf("y should select first button, got closed=%v chosen=%q", closed, o.Chosen())
	}

	// n = second button
	o = NewApprovalOverlay("test", "Bash", nil, buttons)
	_, _, closed = o.Update(keyMsg("n"))
	if !closed || o.Chosen() != "deny" {
		t.Errorf("n should select second button, got closed=%v chosen=%q", closed, o.Chosen())
	}

	// a = third button
	o = NewApprovalOverlay("test", "Bash", nil, buttons)
	_, _, closed = o.Update(keyMsg("a"))
	if !closed || o.Chosen() != "always" {
		t.Errorf("a should select third button, got closed=%v chosen=%q", closed, o.Chosen())
	}
}

func TestApprovalOverlayTabCyclesButtons(t *testing.T) {
	buttons := []blocks.Button{
		{Label: "Allow", Value: "allow"},
		{Label: "Deny", Value: "deny"},
	}
	o := NewApprovalOverlay("test", "Bash", nil, buttons)

	if o.selected != 0 {
		t.Errorf("initial selection should be 0, got %d", o.selected)
	}

	o.Update(keyMsg("tab"))
	if o.selected != 1 {
		t.Errorf("after tab, selection should be 1, got %d", o.selected)
	}

	o.Update(keyMsg("tab"))
	if o.selected != 0 {
		t.Errorf("after second tab, selection should wrap to 0, got %d", o.selected)
	}
}

func TestApprovalOverlayEnterConfirms(t *testing.T) {
	buttons := []blocks.Button{
		{Label: "Allow", Value: "allow"},
		{Label: "Deny", Value: "deny"},
	}
	o := NewApprovalOverlay("test", "Bash", nil, buttons)

	// Tab to "Deny", then Enter.
	o.Update(keyMsg("tab"))
	_, _, closed := o.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter should close the overlay")
	}
	if o.Chosen() != "deny" {
		t.Errorf("expected 'deny', got %q", o.Chosen())
	}
}

func TestApprovalOverlayEscDenies(t *testing.T) {
	buttons := []blocks.Button{
		{Label: "Allow", Value: "allow"},
		{Label: "Deny", Value: "deny"},
	}
	o := NewApprovalOverlay("test", "Bash", nil, buttons)

	_, _, closed := o.Update(keyMsg("esc"))
	if !closed {
		t.Error("esc should close the overlay")
	}
	if o.Chosen() != "deny" {
		t.Errorf("esc should default to deny, got %q", o.Chosen())
	}
}

func TestApprovalOverlayToggleDetails(t *testing.T) {
	args := map[string]any{"command": "ls -la"}
	o := NewApprovalOverlay("test", "Bash", args, []blocks.Button{
		{Label: "Allow", Value: "allow"},
	})

	if o.expanded {
		t.Error("should start collapsed")
	}

	o.Update(keyMsg("?"))
	if !o.expanded {
		t.Error("? should toggle expanded")
	}

	o.Update(keyMsg("?"))
	if o.expanded {
		t.Error("second ? should collapse")
	}
}

func TestApprovalOverlayViewNonEmpty(t *testing.T) {
	buttons := []blocks.Button{
		{Label: "Allow", Value: "allow"},
		{Label: "Deny", Value: "deny"},
	}
	o := NewApprovalOverlay("Run dangerous command", "Bash",
		map[string]any{"command": "rm -rf /tmp/test"}, buttons)

	view := o.View(80, 40, testTheme())
	if view == "" {
		t.Error("View should produce non-empty output")
	}
	if !strings.Contains(view, "Tool Approval") {
		t.Error("View should contain title")
	}
	if !strings.Contains(view, "Bash") {
		t.Error("View should contain tool name")
	}
	if !strings.Contains(view, "Allow") {
		t.Error("View should contain button labels")
	}
}

// -- PaletteOverlay --

func TestPaletteOverlayKind(t *testing.T) {
	o := NewPaletteOverlay([]string{"/help"}, nil)
	if o.Kind() != OverlayPalette {
		t.Errorf("expected OverlayPalette, got %d", o.Kind())
	}
}

func TestPaletteOverlayPopulatesItems(t *testing.T) {
	o := NewPaletteOverlay([]string{"/help", "/clear", "/status"}, nil)

	// Should have slash commands + built-in UI actions.
	if len(o.items) < 3 {
		t.Errorf("expected at least 3 items, got %d", len(o.items))
	}

	// All items should be in filtered list initially.
	if len(o.filtered) != len(o.items) {
		t.Errorf("filtered = %d, items = %d; expected equal", len(o.filtered), len(o.items))
	}
}

func TestPaletteOverlayFiltering(t *testing.T) {
	o := NewPaletteOverlay([]string{"/help", "/clear", "/status"}, nil)
	total := len(o.filtered)

	// Type "help" to filter.
	for _, ch := range "help" {
		o.Update(keyMsg(string(ch)))
	}

	if o.query != "help" {
		t.Errorf("expected query 'help', got %q", o.query)
	}
	if len(o.filtered) >= total {
		t.Error("filtering should reduce the list")
	}
	// Should match /help and possibly "Help" UI action.
	found := false
	for _, item := range o.filtered {
		if item.Label == "/help" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected /help in filtered results")
	}
}

func TestPaletteOverlayNavigation(t *testing.T) {
	o := NewPaletteOverlay([]string{"/help", "/clear"}, nil)

	if o.cursor != 0 {
		t.Errorf("initial cursor should be 0, got %d", o.cursor)
	}

	o.Update(keyMsg("down"))
	if o.cursor != 1 {
		t.Errorf("expected cursor 1 after down, got %d", o.cursor)
	}

	o.Update(keyMsg("up"))
	if o.cursor != 0 {
		t.Errorf("expected cursor 0 after up, got %d", o.cursor)
	}
}

func TestPaletteOverlayEnterChooses(t *testing.T) {
	o := NewPaletteOverlay([]string{"/help", "/clear"}, nil)

	_, _, closed := o.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter should close the palette")
	}
	if o.Chosen() == nil {
		t.Fatal("expected a chosen item")
	}
	if o.Chosen().Label != "/help" {
		t.Errorf("expected first item /help, got %q", o.Chosen().Label)
	}
}

func TestPaletteOverlayEscClosesWithoutChoice(t *testing.T) {
	o := NewPaletteOverlay([]string{"/help"}, nil)

	_, _, closed := o.Update(keyMsg("esc"))
	if !closed {
		t.Error("esc should close the palette")
	}
	if o.Chosen() != nil {
		t.Error("esc should not set a chosen item")
	}
}

func TestPaletteOverlayBackspace(t *testing.T) {
	o := NewPaletteOverlay([]string{"/help"}, nil)

	for _, ch := range "xyz" {
		o.Update(keyMsg(string(ch)))
	}
	if o.query != "xyz" {
		t.Fatalf("expected 'xyz', got %q", o.query)
	}

	o.Update(keyMsg("backspace"))
	if o.query != "xy" {
		t.Errorf("expected 'xy' after backspace, got %q", o.query)
	}
}

func TestPaletteOverlayViewNonEmpty(t *testing.T) {
	o := NewPaletteOverlay([]string{"/help", "/clear"}, nil)

	view := o.View(80, 40, testTheme())
	if view == "" {
		t.Error("View should produce non-empty output")
	}
	if !strings.Contains(view, "Command Palette") {
		t.Error("View should contain title")
	}
}

func TestPaletteOverlayCommandMarkedAsCommand(t *testing.T) {
	o := NewPaletteOverlay([]string{"/help"}, nil)

	// Find the /help item.
	for _, item := range o.items {
		if item.Label == "/help" {
			if !item.IsCommand {
				t.Error("/help should be marked IsCommand")
			}
			if item.Action != "/help" {
				t.Errorf("action should be '/help', got %q", item.Action)
			}
			return
		}
	}
	t.Error("/help item not found")
}

func TestPaletteOverlayUIActionNotCommand(t *testing.T) {
	o := NewPaletteOverlay(nil, nil)

	for _, item := range o.items {
		if item.Action == "toggle_sidebar" {
			if item.IsCommand {
				t.Error("toggle_sidebar should not be IsCommand")
			}
			return
		}
	}
	t.Error("toggle_sidebar action not found")
}

// -- RejectionOverlay --

func TestRejectionOverlayKind(t *testing.T) {
	r := NewRejectionOverlay("bash", nil)
	if r.Kind() != OverlayRejection {
		t.Errorf("expected OverlayRejection, got %d", r.Kind())
	}
}

func TestRejectionOverlayNavigation(t *testing.T) {
	r := NewRejectionOverlay("bash", nil)
	if r.cursor != 0 {
		t.Fatalf("expected cursor 0, got %d", r.cursor)
	}

	r.Update(keyMsg("down"))
	if r.cursor != 1 {
		t.Errorf("expected cursor 1 after down, got %d", r.cursor)
	}

	r.Update(keyMsg("up"))
	if r.cursor != 0 {
		t.Errorf("expected cursor 0 after up, got %d", r.cursor)
	}

	// Can't go above 0
	r.Update(keyMsg("up"))
	if r.cursor != 0 {
		t.Errorf("expected cursor 0 after up at top, got %d", r.cursor)
	}
}

func TestRejectionOverlayPresetSelect(t *testing.T) {
	var got string
	r := NewRejectionOverlay("bash", func(reason string) { got = reason })

	// Move to second preset and confirm
	r.Update(keyMsg("down"))
	_, _, closed := r.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter on preset should close overlay")
	}
	if got != "Try a different approach" {
		t.Errorf("expected 'Try a different approach', got %q", got)
	}
}

func TestRejectionOverlayDigitShortcut(t *testing.T) {
	var got string
	r := NewRejectionOverlay("bash", func(reason string) { got = reason })

	_, _, closed := r.Update(keyMsg("3"))
	if !closed {
		t.Error("digit shortcut should close overlay")
	}
	if got != "The command is dangerous" {
		t.Errorf("expected 'The command is dangerous', got %q", got)
	}
}

func TestRejectionOverlayCustomInput(t *testing.T) {
	var got string
	r := NewRejectionOverlay("bash", func(reason string) { got = reason })

	// Tab switches to custom mode
	r.Update(keyMsg("tab"))
	if !r.inCustom {
		t.Fatal("tab should switch to custom mode")
	}

	// Type custom reason
	for _, ch := range "bad idea" {
		r.Update(keyMsg(string(ch)))
	}
	if r.custom != "bad idea" {
		t.Errorf("expected custom 'bad idea', got %q", r.custom)
	}

	// Enter confirms
	_, _, closed := r.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter should close overlay")
	}
	if got != "bad idea" {
		t.Errorf("expected 'bad idea', got %q", got)
	}
}

func TestRejectionOverlayCustomBackspace(t *testing.T) {
	r := NewRejectionOverlay("bash", nil)
	r.Update(keyMsg("tab"))

	for _, ch := range "abc" {
		r.Update(keyMsg(string(ch)))
	}
	r.Update(keyMsg("backspace"))
	if r.custom != "ab" {
		t.Errorf("expected 'ab' after backspace, got %q", r.custom)
	}
}

func TestRejectionOverlayCustomEscReturnsToList(t *testing.T) {
	r := NewRejectionOverlay("bash", nil)
	r.Update(keyMsg("tab"))
	if !r.inCustom {
		t.Fatal("expected custom mode")
	}

	_, _, closed := r.Update(keyMsg("esc"))
	if closed {
		t.Error("esc in custom mode should return to list, not close")
	}
	if r.inCustom {
		t.Error("expected return to list mode")
	}
}

func TestRejectionOverlayEscClosesFromList(t *testing.T) {
	r := NewRejectionOverlay("bash", nil)
	_, _, closed := r.Update(keyMsg("esc"))
	if !closed {
		t.Error("esc from list should close overlay")
	}
}

func TestRejectionOverlayViewNonEmpty(t *testing.T) {
	r := NewRejectionOverlay("bash", nil)
	view := r.View(80, 40, testTheme())
	if view == "" {
		t.Error("RejectionOverlay.View should produce non-empty output")
	}
	if !strings.Contains(view, "Why reject") {
		t.Error("view should contain title")
	}
	if !strings.Contains(view, "bash") {
		t.Error("view should contain tool name")
	}
}

func TestRejectionOverlayEmptyCustomDefaultsToRejected(t *testing.T) {
	var got string
	r := NewRejectionOverlay("bash", func(reason string) { got = reason })

	r.Update(keyMsg("tab"))
	_, _, closed := r.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter should close overlay")
	}
	if got != "Rejected" {
		t.Errorf("expected 'Rejected' for empty custom, got %q", got)
	}
}

// -- ElicitationOverlay --

func TestElicitationOverlayKind(t *testing.T) {
	e := NewElicitationOverlay("Q", "", ElicitText, nil, nil)
	if e.Kind() != OverlayElicitation {
		t.Errorf("expected OverlayElicitation, got %d", e.Kind())
	}
}

func TestElicitationTextInput(t *testing.T) {
	var got string
	e := NewElicitationOverlay("Name?", "Enter your name", ElicitText, nil,
		func(result string) { got = result })

	for _, ch := range "Alice" {
		e.Update(keyMsg(string(ch)))
	}
	if e.textInput != "Alice" {
		t.Fatalf("expected textInput 'Alice', got %q", e.textInput)
	}

	_, _, closed := e.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter should close text overlay")
	}
	if got != "Alice" {
		t.Errorf("expected result 'Alice', got %q", got)
	}
}

func TestElicitationTextBackspace(t *testing.T) {
	e := NewElicitationOverlay("Q", "", ElicitText, nil, nil)
	for _, ch := range "abc" {
		e.Update(keyMsg(string(ch)))
	}
	e.Update(keyMsg("backspace"))
	if e.textInput != "ab" {
		t.Errorf("expected 'ab' after backspace, got %q", e.textInput)
	}
}

func TestElicitationTextEscCancels(t *testing.T) {
	var got string
	e := NewElicitationOverlay("Q", "", ElicitText, nil,
		func(result string) { got = result })

	_, _, closed := e.Update(keyMsg("esc"))
	if !closed {
		t.Error("esc should close text overlay")
	}
	if got != "" {
		t.Errorf("esc should yield empty result, got %q", got)
	}
}

func TestElicitationSelectNavigation(t *testing.T) {
	opts := []ElicitationOption{
		{Label: "Red", Value: "red"},
		{Label: "Green", Value: "green"},
		{Label: "Blue", Value: "blue"},
	}
	e := NewElicitationOverlay("Color?", "", ElicitSelect, opts, nil)

	if e.cursor != 0 {
		t.Fatalf("expected cursor 0, got %d", e.cursor)
	}

	e.Update(keyMsg("down"))
	if e.cursor != 1 {
		t.Errorf("expected cursor 1, got %d", e.cursor)
	}

	e.Update(keyMsg("down"))
	if e.cursor != 2 {
		t.Errorf("expected cursor 2, got %d", e.cursor)
	}

	// Can't go past end
	e.Update(keyMsg("down"))
	if e.cursor != 2 {
		t.Errorf("expected cursor 2 at bottom, got %d", e.cursor)
	}

	e.Update(keyMsg("up"))
	if e.cursor != 1 {
		t.Errorf("expected cursor 1 after up, got %d", e.cursor)
	}
}

func TestElicitationSelectConfirm(t *testing.T) {
	opts := []ElicitationOption{
		{Label: "Red", Value: "red"},
		{Label: "Green", Value: "green"},
	}
	var got string
	e := NewElicitationOverlay("Color?", "", ElicitSelect, opts,
		func(result string) { got = result })

	e.Update(keyMsg("down"))
	_, _, closed := e.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter should close select overlay")
	}
	if got != "green" {
		t.Errorf("expected 'green', got %q", got)
	}
}

func TestElicitationMultiSelectToggle(t *testing.T) {
	opts := []ElicitationOption{
		{Label: "A", Value: "a"},
		{Label: "B", Value: "b"},
		{Label: "C", Value: "c"},
	}
	e := NewElicitationOverlay("Pick", "", ElicitMultiSelect, opts, nil)

	// Toggle first
	e.Update(keyMsg(" "))
	if !e.selected[0] {
		t.Error("space should toggle selection on")
	}

	// Toggle off
	e.Update(keyMsg(" "))
	if e.selected[0] {
		t.Error("second space should toggle selection off")
	}
}

func TestElicitationMultiSelectConfirm(t *testing.T) {
	opts := []ElicitationOption{
		{Label: "A", Value: "a"},
		{Label: "B", Value: "b"},
		{Label: "C", Value: "c"},
	}
	var got string
	e := NewElicitationOverlay("Pick", "", ElicitMultiSelect, opts,
		func(result string) { got = result })

	// Select A and C
	e.Update(keyMsg(" "))       // toggle A
	e.Update(keyMsg("down"))    // move to B
	e.Update(keyMsg("down"))    // move to C
	e.Update(keyMsg(" "))       // toggle C

	_, _, closed := e.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter should close multi-select overlay")
	}
	if got != "a,c" {
		t.Errorf("expected 'a,c', got %q", got)
	}
}

func TestElicitationMultiSelectEmptyConfirm(t *testing.T) {
	opts := []ElicitationOption{
		{Label: "A", Value: "a"},
	}
	var got string
	e := NewElicitationOverlay("Pick", "", ElicitMultiSelect, opts,
		func(result string) { got = result })

	_, _, closed := e.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter should close overlay")
	}
	if got != "" {
		t.Errorf("expected empty result for no selection, got %q", got)
	}
}

func TestElicitationViewNonEmpty(t *testing.T) {
	th := testTheme()

	// Text mode
	e1 := NewElicitationOverlay("Name?", "What is your name?", ElicitText, nil, nil)
	v1 := e1.View(80, 40, th)
	if v1 == "" {
		t.Error("text view should be non-empty")
	}
	if !strings.Contains(v1, "Name?") {
		t.Error("text view should contain title")
	}

	// Select mode
	opts := []ElicitationOption{{Label: "Yes", Value: "y"}, {Label: "No", Value: "n"}}
	e2 := NewElicitationOverlay("Confirm?", "", ElicitSelect, opts, nil)
	v2 := e2.View(80, 40, th)
	if v2 == "" {
		t.Error("select view should be non-empty")
	}

	// MultiSelect mode
	e3 := NewElicitationOverlay("Features", "", ElicitMultiSelect, opts, nil)
	v3 := e3.View(80, 40, th)
	if v3 == "" {
		t.Error("multi-select view should be non-empty")
	}
	if !strings.Contains(v3, "[ ]") && !strings.Contains(v3, "[x]") {
		t.Error("multi-select view should contain checkboxes")
	}
}
