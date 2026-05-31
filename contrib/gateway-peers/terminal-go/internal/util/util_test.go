package util

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCopyToClipboard(t *testing.T) {
	// Just verify it doesn't panic. We can't test the actual clipboard.
	CopyToClipboard("test text")
}

func TestLinkify(t *testing.T) {
	input := "check https://example.com and http://foo.bar/path"
	result := Linkify(input)
	// Verify OSC 8 wrapping is present
	if !strings.Contains(result, "\033]8;;https://example.com\033\\") {
		t.Error("missing OSC 8 for https URL")
	}
	if !strings.Contains(result, "\033]8;;http://foo.bar/path\033\\") {
		t.Error("missing OSC 8 for http URL")
	}
}

func TestLinkifyNoURLs(t *testing.T) {
	input := "no urls here"
	result := Linkify(input)
	if result != input {
		t.Errorf("expected unchanged text, got %q", result)
	}
}

func TestExtractCodeBlocks(t *testing.T) {
	md := "text\n```go\npackage main\n```\nmore\n```python\nprint('hi')\n```"
	blocks := ExtractCodeBlocks(md)
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}
	if blocks[0].Lang != "go" {
		t.Errorf("expected go, got %s", blocks[0].Lang)
	}
	if blocks[1].Lang != "python" {
		t.Errorf("expected python, got %s", blocks[1].Lang)
	}
	if !strings.Contains(blocks[0].Content, "package main") {
		t.Error("missing content in go block")
	}
	if !strings.Contains(blocks[1].Content, "print('hi')") {
		t.Error("missing content in python block")
	}
}

func TestExtractCodeBlocksEmpty(t *testing.T) {
	blocks := ExtractCodeBlocks("no code blocks here")
	if len(blocks) != 0 {
		t.Errorf("expected 0 blocks, got %d", len(blocks))
	}
}

func TestSuggestFilename(t *testing.T) {
	cases := map[string]string{
		"go":         "snippet.go",
		"python":     "snippet.py",
		"":           "snippet.txt",
		"rust":       "snippet.rs",
		"dockerfile": "Dockerfile",
		"makefile":   "Makefile",
		"java":       "Snippet.java",
		"json":       "data.json",
		"yaml":       "config.yaml",
		"unknown":    "snippet.unknown",
	}
	for lang, want := range cases {
		got := SuggestFilename(lang)
		if got != want {
			t.Errorf("SuggestFilename(%q) = %q, want %q", lang, got, want)
		}
	}
}

func TestEstimateCost(t *testing.T) {
	cost := EstimateCost("doubao", 1_000_000, 1_000_000)
	if cost != 2.0 {
		t.Errorf("expected 2.0, got %f", cost)
	}

	// Unknown model uses default
	cost = EstimateCost("unknown-model", 1_000_000, 0)
	if cost != 1.0 {
		t.Errorf("expected 1.0 default input, got %f", cost)
	}
}

func TestEstimateCostZeroTokens(t *testing.T) {
	cost := EstimateCost("doubao", 0, 0)
	if cost != 0.0 {
		t.Errorf("expected 0.0, got %f", cost)
	}
}

func TestExportTranscript(t *testing.T) {
	blocks := []ExportBlock{
		{Kind: "user", Content: "hello"},
		{Kind: "assistant", Content: "world"},
		{Kind: "tool", Content: "result", Meta: map[string]string{"name": "bash", "status": "ok"}},
		{Kind: "thinking", Content: "let me think"},
	}
	tmpFile := filepath.Join(t.TempDir(), "export.md")
	err := ExportTranscript(tmpFile, blocks, ExportOptions{
		Model: "test", TokensIn: 100, TokensOut: 50, SessionKey: "test:1",
	})
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	if !strings.Contains(content, "model: test") {
		t.Error("missing frontmatter model")
	}
	if !strings.Contains(content, "## User") {
		t.Error("missing user section")
	}
	if !strings.Contains(content, "hello") {
		t.Error("missing user content")
	}
	if !strings.Contains(content, "## Assistant") {
		t.Error("missing assistant section")
	}
	if !strings.Contains(content, "### Tool: bash (ok)") {
		t.Error("missing tool section")
	}
	if !strings.Contains(content, "### Thinking") {
		t.Error("missing thinking section")
	}
	if !strings.Contains(content, "> let me think") {
		t.Error("missing thinking content as blockquote")
	}
}

func TestNotify(t *testing.T) {
	// Just verify it doesn't panic.
	Notify("title", "body")
}
