package util

import (
	"strings"
	"testing"
)

func TestTokenizeEmpty(t *testing.T) {
	tokens := tokenize("")
	if len(tokens) != 0 {
		t.Errorf("expected 0 tokens, got %d", len(tokens))
	}
}

func TestTokenizeIdentifiers(t *testing.T) {
	tokens := tokenize("foo bar_baz")
	// "foo" (ident), " " (ws), "bar_baz" (ident)
	if len(tokens) != 3 {
		t.Fatalf("expected 3 tokens, got %d: %v", len(tokens), tokens)
	}
	assertToken(t, tokens[0], "foo", TokenIdentifier)
	assertToken(t, tokens[1], " ", TokenWhitespace)
	assertToken(t, tokens[2], "bar_baz", TokenIdentifier)
}

func TestTokenizePunctuation(t *testing.T) {
	tokens := tokenize("a(b,c)")
	// "a" "(" "b" "," "c" ")"
	if len(tokens) != 6 {
		t.Fatalf("expected 6 tokens, got %d: %v", len(tokens), tokens)
	}
	assertToken(t, tokens[0], "a", TokenIdentifier)
	assertToken(t, tokens[1], "(", TokenPunctuation)
	assertToken(t, tokens[2], "b", TokenIdentifier)
	assertToken(t, tokens[3], ",", TokenPunctuation)
	assertToken(t, tokens[4], "c", TokenIdentifier)
	assertToken(t, tokens[5], ")", TokenPunctuation)
}

func TestTokenizePreservesInput(t *testing.T) {
	inputs := []string{
		"func main() { fmt.Println(42) }",
		"  \ttabs and  spaces  ",
		"a+b*c/d",
		"",
		"hello",
	}
	for _, input := range inputs {
		tokens := tokenize(input)
		var sb strings.Builder
		for _, tok := range tokens {
			sb.WriteString(tok.Text)
		}
		if sb.String() != input {
			t.Errorf("tokenize roundtrip failed: input=%q, got=%q", input, sb.String())
		}
	}
}

func TestWordDiffIdentical(t *testing.T) {
	oldSegs, newSegs := WordDiff("hello world", "hello world")
	if len(oldSegs) != 1 || oldSegs[0].Changed {
		t.Errorf("expected single unchanged segment for old, got %v", oldSegs)
	}
	if len(newSegs) != 1 || newSegs[0].Changed {
		t.Errorf("expected single unchanged segment for new, got %v", newSegs)
	}
}

func TestWordDiffSingleWordChange(t *testing.T) {
	oldSegs, newSegs := WordDiff("foo bar baz", "foo qux baz")

	// old should have: "foo " (unchanged), "bar" (changed), " baz" (unchanged)
	assertSegmentText(t, "old", oldSegs, "foo bar baz")
	assertSegmentText(t, "new", newSegs, "foo qux baz")

	// "bar" should be marked changed in old
	assertContainsChanged(t, "old", oldSegs, "bar")
	// "qux" should be marked changed in new
	assertContainsChanged(t, "new", newSegs, "qux")

	// "foo" should be unchanged in both
	assertContainsUnchanged(t, "old", oldSegs, "foo")
	assertContainsUnchanged(t, "new", newSegs, "foo")
}

func TestWordDiffPunctuationChange(t *testing.T) {
	oldSegs, newSegs := WordDiff("a(b)", "a(c)")
	assertSegmentText(t, "old", oldSegs, "a(b)")
	assertSegmentText(t, "new", newSegs, "a(c)")
	assertContainsChanged(t, "old", oldSegs, "b")
	assertContainsChanged(t, "new", newSegs, "c")
	assertContainsUnchanged(t, "old", oldSegs, "a")
}

func TestWordDiffEmptyOld(t *testing.T) {
	oldSegs, newSegs := WordDiff("", "added")
	if len(oldSegs) != 1 || oldSegs[0].Text != "" {
		t.Errorf("expected empty old segment, got %v", oldSegs)
	}
	if len(newSegs) != 1 || !newSegs[0].Changed || newSegs[0].Text != "added" {
		t.Errorf("expected changed new segment 'added', got %v", newSegs)
	}
}

func TestWordDiffEmptyNew(t *testing.T) {
	oldSegs, newSegs := WordDiff("removed", "")
	if len(oldSegs) != 1 || !oldSegs[0].Changed || oldSegs[0].Text != "removed" {
		t.Errorf("expected changed old segment 'removed', got %v", oldSegs)
	}
	if len(newSegs) != 1 || newSegs[0].Text != "" {
		t.Errorf("expected empty new segment, got %v", newSegs)
	}
}

func TestWordDiffComplexCode(t *testing.T) {
	old := `fmt.Println("hello")`
	new := `fmt.Println("world")`
	oldSegs, newSegs := WordDiff(old, new)
	assertSegmentText(t, "old", oldSegs, old)
	assertSegmentText(t, "new", newSegs, new)
	// "hello" changed to "world"
	assertContainsChanged(t, "old", oldSegs, "hello")
	assertContainsChanged(t, "new", newSegs, "world")
	// "fmt" and "Println" unchanged
	assertContainsUnchanged(t, "old", oldSegs, "fmt")
	assertContainsUnchanged(t, "new", newSegs, "Println")
}

func TestLCSTokensEmpty(t *testing.T) {
	result := lcsTokens(nil, nil)
	if len(result) != 0 {
		t.Errorf("expected empty LCS, got %v", result)
	}
}

func TestLCSTokensNoCommon(t *testing.T) {
	a := []Token{{Text: "a", Kind: TokenIdentifier}}
	b := []Token{{Text: "b", Kind: TokenIdentifier}}
	result := lcsTokens(a, b)
	if len(result) != 0 {
		t.Errorf("expected empty LCS, got %v", result)
	}
}

func TestLCSTokensFullMatch(t *testing.T) {
	toks := []Token{
		{Text: "hello", Kind: TokenIdentifier},
		{Text: " ", Kind: TokenWhitespace},
		{Text: "world", Kind: TokenIdentifier},
	}
	result := lcsTokens(toks, toks)
	if len(result) != 3 {
		t.Fatalf("expected 3 LCS tokens, got %d", len(result))
	}
	for i, tok := range result {
		if tok.Text != toks[i].Text {
			t.Errorf("LCS[%d] = %q, want %q", i, tok.Text, toks[i].Text)
		}
	}
}

// --- helpers ---

func assertToken(t *testing.T, tok Token, text string, kind TokenKind) {
	t.Helper()
	if tok.Text != text || tok.Kind != kind {
		t.Errorf("expected Token{%q, %d}, got Token{%q, %d}", text, kind, tok.Text, tok.Kind)
	}
}

func assertSegmentText(t *testing.T, side string, segs []DiffSegment, want string) {
	t.Helper()
	var sb strings.Builder
	for _, s := range segs {
		sb.WriteString(s.Text)
	}
	if sb.String() != want {
		t.Errorf("%s segments concatenate to %q, want %q", side, sb.String(), want)
	}
}

func assertContainsChanged(t *testing.T, side string, segs []DiffSegment, substr string) {
	t.Helper()
	for _, s := range segs {
		if s.Changed && strings.Contains(s.Text, substr) {
			return
		}
	}
	t.Errorf("%s segments: no changed segment containing %q; segs=%v", side, substr, segs)
}

func assertContainsUnchanged(t *testing.T, side string, segs []DiffSegment, substr string) {
	t.Helper()
	for _, s := range segs {
		if !s.Changed && strings.Contains(s.Text, substr) {
			return
		}
	}
	t.Errorf("%s segments: no unchanged segment containing %q; segs=%v", side, substr, segs)
}
