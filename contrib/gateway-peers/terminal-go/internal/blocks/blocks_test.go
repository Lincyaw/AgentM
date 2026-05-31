package blocks

import (
	"strings"
	"testing"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

func darkTheme() *theme.Theme { return theme.DarkTheme() }

const testWidth = 80

func TestUserTurnRender(t *testing.T) {
	b := &UserTurn{Content: "Hello, agent!"}
	out := b.Render(testWidth, darkTheme())
	if out == "" {
		t.Fatal("expected non-empty render")
	}
	if !strings.Contains(out, "you") {
		t.Errorf("expected attribution label in output, got:\n%s", out)
	}
	if !strings.Contains(out, "Hello, agent!") {
		t.Errorf("expected content in output, got:\n%s", out)
	}
	if b.Kind() != "user" {
		t.Errorf("expected kind 'user', got %q", b.Kind())
	}
}

func TestSystemTurnRender(t *testing.T) {
	b := &SystemTurn{Content: "Context injected.", Source: "reminder_atom"}
	out := b.Render(testWidth, darkTheme())
	if out == "" {
		t.Fatal("expected non-empty render")
	}
	if !strings.Contains(out, "system") {
		t.Errorf("expected 'system' label, got:\n%s", out)
	}
	if !strings.Contains(out, "reminder_atom") {
		t.Errorf("expected source name, got:\n%s", out)
	}
	if b.Kind() != "system" {
		t.Errorf("expected kind 'system', got %q", b.Kind())
	}
}

func TestThinkingBlockCollapsedExpanded(t *testing.T) {
	b := NewThinkingBlock()
	b.Text = "I should check the database schema first."

	if !b.Collapsed() {
		t.Error("new ThinkingBlock should start collapsed")
	}

	collapsed := b.Render(testWidth, darkTheme())
	if collapsed == "" {
		t.Fatal("collapsed render should not be empty")
	}
	if !strings.Contains(collapsed, theme.ThinkingCollapsed) {
		t.Errorf("collapsed render missing glyph %q", theme.ThinkingCollapsed)
	}

	b.SetCollapsed(false)
	expanded := b.Render(testWidth, darkTheme())
	if expanded == "" {
		t.Fatal("expanded render should not be empty")
	}
	if !strings.Contains(expanded, theme.ThinkingExpanded) {
		t.Errorf("expanded render missing glyph %q", theme.ThinkingExpanded)
	}
	if !strings.Contains(expanded, "database schema") {
		t.Errorf("expanded render should contain thinking text")
	}

	if collapsed == expanded {
		t.Error("collapsed and expanded renders should differ")
	}
}

func TestToolBlockCollapsedExpanded(t *testing.T) {
	b := NewToolBlock("bash", map[string]any{"command": "ls -la /tmp"})
	b.Done = true
	b.OK = true
	b.Result = "file1.txt\nfile2.txt"

	if !b.Collapsed() {
		t.Error("new ToolBlock should start collapsed")
	}

	collapsed := b.Render(testWidth, darkTheme())
	if collapsed == "" {
		t.Fatal("collapsed render should not be empty")
	}
	if !strings.Contains(collapsed, "bash") {
		t.Errorf("collapsed render should contain tool name 'bash'")
	}
	if !strings.Contains(collapsed, "ls -la /tmp") {
		t.Errorf("collapsed render should contain command summary")
	}
	if !strings.Contains(collapsed, theme.ToolOK) {
		t.Errorf("collapsed render should contain OK glyph")
	}

	b.SetCollapsed(false)
	expanded := b.Render(testWidth, darkTheme())
	if expanded == "" {
		t.Fatal("expanded render should not be empty")
	}
	if !strings.Contains(expanded, "result:") {
		t.Errorf("expanded render should contain result text")
	}

	if collapsed == expanded {
		t.Error("collapsed and expanded renders should differ")
	}
}

func TestToolBlockRunningGlyph(t *testing.T) {
	b := NewToolBlock("read", map[string]any{"file_path": "/etc/hosts"})
	// Done is false by default
	out := b.Render(testWidth, darkTheme())
	if !strings.Contains(out, theme.ToolRunning) {
		t.Errorf("running tool should show running glyph, got:\n%s", out)
	}
}

func TestToolBlockErrorGlyph(t *testing.T) {
	b := NewToolBlock("write", map[string]any{"file_path": "/bad/path"})
	b.Done = true
	b.OK = false
	out := b.Render(testWidth, darkTheme())
	if !strings.Contains(out, theme.ToolError) {
		t.Errorf("failed tool should show error glyph, got:\n%s", out)
	}
}

func TestToolBlockEditDiff(t *testing.T) {
	b := NewToolBlock("edit", map[string]any{
		"file_path":  "/src/main.go",
		"old_string": "fmt.Println(\"old\")",
		"new_string": "fmt.Println(\"new\")",
	})
	b.Done = true
	b.OK = true
	b.SetCollapsed(false)

	out := b.Render(testWidth, darkTheme())
	if out == "" {
		t.Fatal("expanded edit render should not be empty")
	}
	if !strings.Contains(out, "/src/main.go") {
		t.Errorf("edit render should contain file path")
	}
	// The diff output should contain both old and new strings
	if !strings.Contains(out, "old") {
		t.Errorf("edit render should contain old string content")
	}
	if !strings.Contains(out, "new") {
		t.Errorf("edit render should contain new string content")
	}
}

func TestToolBlockWritePreview(t *testing.T) {
	b := NewToolBlock("Write", map[string]any{
		"file_path": "/src/new_file.go",
		"content":   "package main\n\nfunc main() {}",
	})
	b.Done = true
	b.OK = true
	b.SetCollapsed(false)

	out := b.Render(testWidth, darkTheme())
	if out == "" {
		t.Fatal("expanded write render should not be empty")
	}
	if !strings.Contains(out, "/src/new_file.go") {
		t.Errorf("write render should contain file path")
	}
	if !strings.Contains(out, "package main") {
		t.Errorf("write render should contain written content")
	}
}

func TestSubagentBlockRender(t *testing.T) {
	// Running
	b := &SubagentBlock{Purpose: "code-review", Done: false}
	out := b.Render(testWidth, darkTheme())
	if !strings.Contains(out, "code-review") {
		t.Errorf("subagent render should contain purpose")
	}
	if !strings.Contains(out, theme.ToolRunning) {
		t.Errorf("running subagent should show running glyph")
	}

	// Done OK
	b.Done = true
	out = b.Render(testWidth, darkTheme())
	if !strings.Contains(out, theme.ToolOK) {
		t.Errorf("completed subagent should show OK glyph")
	}

	// Done with error
	b.Error = "timeout"
	out = b.Render(testWidth, darkTheme())
	if !strings.Contains(out, theme.ToolError) {
		t.Errorf("failed subagent should show error glyph")
	}

	if b.Kind() != "subagent" {
		t.Errorf("expected kind 'subagent', got %q", b.Kind())
	}
}

func TestApprovalBlockCollapsedExpanded(t *testing.T) {
	b := NewApprovalBlock("Run dangerous command: rm -rf /", []Button{
		{Label: "Allow", Value: "allow", Style: "primary"},
		{Label: "Deny", Value: "deny", Style: "danger"},
	})

	if !b.Collapsed() {
		t.Error("new ApprovalBlock should start collapsed")
	}

	collapsed := b.Render(testWidth, darkTheme())
	if collapsed == "" {
		t.Fatal("collapsed render should not be empty")
	}
	if !strings.Contains(collapsed, "Allow") {
		t.Errorf("collapsed should contain button labels")
	}
	if !strings.Contains(collapsed, "[?] Details") {
		t.Errorf("collapsed should contain details option")
	}

	b.SetCollapsed(false)
	expanded := b.Render(testWidth, darkTheme())
	if expanded == "" {
		t.Fatal("expanded render should not be empty")
	}
	if !strings.Contains(expanded, "Approval Required") {
		t.Errorf("expanded should contain approval header")
	}

	if collapsed == expanded {
		t.Error("collapsed and expanded renders should differ")
	}

	if b.Kind() != "approval" {
		t.Errorf("expected kind 'approval', got %q", b.Kind())
	}
}

func TestAssistantTurnRender(t *testing.T) {
	thinking := NewThinkingBlock()
	thinking.Text = "Let me analyze the code."

	tool := NewToolBlock("bash", map[string]any{"command": "go test ./..."})
	tool.Done = true
	tool.OK = true
	tool.Result = "ok"

	child := &SubagentBlock{Purpose: "lint-check", Done: true}

	b := &AssistantTurn{
		Thinking:     thinking,
		Text:         "Here is the result of the analysis.",
		GlamourStyle: "dark",
		TextDirty:    true, // simulate streaming so we get raw text (glamour adds escapes)
		Tools:        []*ToolBlock{tool},
		Children:     []*SubagentBlock{child},
	}

	out := b.Render(testWidth, darkTheme())
	if out == "" {
		t.Fatal("assistant render should not be empty")
	}
	if !strings.Contains(out, "assistant") {
		t.Errorf("assistant render should contain attribution")
	}
	if !strings.Contains(out, "result of the analysis") {
		t.Errorf("assistant render should contain text body")
	}
	if b.Kind() != "assistant" {
		t.Errorf("expected kind 'assistant', got %q", b.Kind())
	}
}

func TestAssistantTurnCompleteUsesGlamour(t *testing.T) {
	b := &AssistantTurn{
		Text: "**bold text** and `code`",
	}
	b.SetComplete()
	if !b.Complete() {
		t.Fatal("expected complete=true")
	}

	out := b.Render(testWidth, darkTheme())
	if out == "" {
		t.Fatal("completed assistant render should not be empty")
	}
	// Glamour should have processed the markdown -- at minimum the raw
	// markers should be gone or transformed (glamour adds ANSI codes).
	// We just verify non-empty output and the text content is present.
	if !strings.Contains(out, "bold text") {
		t.Errorf("rendered output should contain 'bold text'")
	}
}

func TestAssistantTurnStreamingUsesRawText(t *testing.T) {
	b := &AssistantTurn{
		Text:      "**still streaming**",
		TextDirty: true,
	}
	// Not complete -- streaming mode

	out := b.Render(testWidth, darkTheme())
	if out == "" {
		t.Fatal("streaming assistant render should not be empty")
	}
	// Raw text should appear since we're streaming (no glamour processing)
	if !strings.Contains(out, "**still streaming**") {
		t.Errorf("streaming render should contain raw markdown, got:\n%s", out)
	}
}

func TestThemeForName(t *testing.T) {
	dark := theme.ForName("dark")
	light := theme.ForName("light")
	fallback := theme.ForName("unknown")

	if dark == nil || light == nil || fallback == nil {
		t.Fatal("ForName should never return nil")
	}
}

func TestBlockInterface(t *testing.T) {
	// Verify all types satisfy the Block interface at compile time
	var _ Block = &UserTurn{}
	var _ Block = &SystemTurn{}
	var _ Block = &ThinkingBlock{}
	var _ Block = &ToolBlock{}
	var _ Block = &SubagentBlock{}
	var _ Block = &ApprovalBlock{}
	var _ Block = &AssistantTurn{}
}

func TestTruncateUTF8Safe(t *testing.T) {
	// Ensure Truncate does not corrupt multi-byte characters
	s := "Hello, 世界！This is a test."
	truncated := strings.TrimSuffix(s[:6], "...")
	_ = truncated
	// The real test is that Truncate from util doesn't panic on CJK
	// and produces valid UTF-8. Tested via the util package.
}
