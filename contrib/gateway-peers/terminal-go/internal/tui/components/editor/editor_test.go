package editor

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
)

func TestGraphemeBackspaceDeletesAtLogicalCursorInLongWrappedWideLine(t *testing.T) {
	e := New(nil).(*editor)
	e.textarea.SetWidth(10)
	e.SetValue("你好世界abcdefghijXYZ")

	// Place the cursor after "你好世界ab". This line is soft-wrapped because
	// the CJK runes are double-width, but textarea.Column remains a logical
	// rune index. Backspace should delete the preceding "b", not a visually
	// offset rune from the middle of the line.
	e.textarea.SetCursorColumn(len([]rune("你好世界ab")))

	e.Update(tea.KeyPressMsg{Code: tea.KeyBackspace})

	want := "你好世界acdefghijXYZ"
	if got := e.Value(); got != want {
		t.Fatalf("backspace deleted wrong text: got %q, want %q", got, want)
	}
	if got, want := e.textarea.Column(), len([]rune("你好世界a")); got != want {
		t.Fatalf("cursor column after backspace = %d, want %d", got, want)
	}
}

func TestGraphemeBackspaceDeletesEntireEmojiGrapheme(t *testing.T) {
	e := New(nil).(*editor)
	e.SetValue("abc⚠️def")
	e.textarea.SetCursorColumn(len([]rune("abc⚠️")))

	e.Update(tea.KeyPressMsg{Code: tea.KeyBackspace})

	want := "abcdef"
	if got := e.Value(); got != want {
		t.Fatalf("backspace deleted wrong grapheme: got %q, want %q", got, want)
	}
	if got, want := e.textarea.Column(), len([]rune("abc")); got != want {
		t.Fatalf("cursor column after grapheme backspace = %d, want %d", got, want)
	}
}

func TestGraphemeBackspacePreservesSuffixAcrossFollowingLines(t *testing.T) {
	e := New(nil).(*editor)
	e.SetValue("abc⚠️def\nnext")
	e.textarea.MoveToBegin()
	e.textarea.SetCursorColumn(len([]rune("abc⚠️")))

	e.Update(tea.KeyPressMsg{Code: tea.KeyBackspace})

	want := "abcdef\nnext"
	if got := e.Value(); got != want {
		t.Fatalf("backspace before multiline suffix: got %q, want %q", got, want)
	}
	if got, want := e.textarea.Line(), 0; got != want {
		t.Fatalf("cursor line after multiline suffix backspace = %d, want %d", got, want)
	}
	if got, want := e.textarea.Column(), len([]rune("abc")); got != want {
		t.Fatalf("cursor column after multiline suffix backspace = %d, want %d", got, want)
	}
}

func TestGraphemeBackspaceAtEndOfLongWrappedLine(t *testing.T) {
	e := New(nil).(*editor)
	e.textarea.SetWidth(10)
	e.SetValue(strings.Repeat("a", 36))

	e.Update(tea.KeyPressMsg{Code: tea.KeyBackspace})

	want := strings.Repeat("a", 35)
	if got := e.Value(); got != want {
		t.Fatalf("backspace at end of wrapped line: got %q, want %q", got, want)
	}
	if got, want := e.textarea.Column(), len([]rune(want)); got != want {
		t.Fatalf("cursor column after end backspace = %d, want %d", got, want)
	}
}

func TestGraphemeBackspaceMergesLogicalLinesAtLineStart(t *testing.T) {
	e := New(nil).(*editor)
	e.SetValue("first\nsecond")
	e.textarea.MoveToBegin()
	e.textarea.CursorDown()
	e.textarea.CursorStart()

	e.Update(tea.KeyPressMsg{Code: tea.KeyBackspace})

	want := "firstsecond"
	if got := e.Value(); got != want {
		t.Fatalf("backspace at logical line start: got %q, want %q", got, want)
	}
	if got, want := e.textarea.Line(), 0; got != want {
		t.Fatalf("cursor line after merge = %d, want %d", got, want)
	}
	if got, want := e.textarea.Column(), len([]rune("first")); got != want {
		t.Fatalf("cursor column after merge = %d, want %d", got, want)
	}
}
