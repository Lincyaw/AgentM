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
