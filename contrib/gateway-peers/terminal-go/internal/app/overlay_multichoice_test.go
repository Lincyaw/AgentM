package app

import (
	"strings"
	"testing"
)

func sampleOptions() []MultiChoiceOption {
	return []MultiChoiceOption{
		{Label: "Alpha", Description: "First option", Key: "alpha"},
		{Label: "Beta", Description: "Second option", Key: "beta"},
		{Label: "Gamma", Key: "gamma"},
	}
}

// -- Interface compliance --

func TestMultiChoiceOverlayImplementsOverlay(t *testing.T) {
	var _ Overlay = &MultiChoiceOverlay{}
}

// -- Kind --

func TestMultiChoiceOverlayKind(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(string) {})
	if m.Kind() != OverlayMultiChoice {
		t.Errorf("expected OverlayMultiChoice, got %d", m.Kind())
	}
}

// -- Navigation --

func TestMultiChoiceNavigateDown(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(string) {})
	if m.selected != 0 {
		t.Fatalf("expected initial selected=0, got %d", m.selected)
	}

	m.Update(keyMsg("down"))
	if m.selected != 1 {
		t.Errorf("expected selected=1 after down, got %d", m.selected)
	}

	m.Update(keyMsg("j"))
	if m.selected != 2 {
		t.Errorf("expected selected=2 after j, got %d", m.selected)
	}

	// Should clamp at the last option.
	m.Update(keyMsg("down"))
	if m.selected != 2 {
		t.Errorf("expected selected=2 (clamped), got %d", m.selected)
	}
}

func TestMultiChoiceNavigateUp(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(string) {})
	m.selected = 2

	m.Update(keyMsg("up"))
	if m.selected != 1 {
		t.Errorf("expected selected=1 after up, got %d", m.selected)
	}

	m.Update(keyMsg("k"))
	if m.selected != 0 {
		t.Errorf("expected selected=0 after k, got %d", m.selected)
	}

	// Should clamp at 0.
	m.Update(keyMsg("up"))
	if m.selected != 0 {
		t.Errorf("expected selected=0 (clamped), got %d", m.selected)
	}
}

// -- Enter selects --

func TestMultiChoiceEnterConfirms(t *testing.T) {
	var result string
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(key string) {
		result = key
	})
	m.selected = 1

	_, _, closed := m.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter should close the overlay")
	}
	if result != "beta" {
		t.Errorf("expected key 'beta', got %q", result)
	}
}

// -- Esc cancels --

func TestMultiChoiceEscCancels(t *testing.T) {
	var result string
	called := false
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(key string) {
		result = key
		called = true
	})

	_, _, closed := m.Update(keyMsg("esc"))
	if !closed {
		t.Error("esc should close the overlay")
	}
	if !called {
		t.Error("onSelect should be called on esc")
	}
	if result != "" {
		t.Errorf("expected empty key on cancel, got %q", result)
	}
}

// -- Digit shortcuts --

func TestMultiChoiceDigitShortcuts(t *testing.T) {
	var result string
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(key string) {
		result = key
	})

	_, _, closed := m.Update(keyMsg("2"))
	if !closed {
		t.Error("digit shortcut should close the overlay")
	}
	if result != "beta" {
		t.Errorf("expected key 'beta' for digit 2, got %q", result)
	}
}

func TestMultiChoiceDigitOutOfRange(t *testing.T) {
	called := false
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(string) {
		called = true
	})

	_, _, closed := m.Update(keyMsg("9"))
	if closed {
		t.Error("out-of-range digit should not close")
	}
	if called {
		t.Error("out-of-range digit should not trigger callback")
	}
}

// -- Custom input --

func TestMultiChoiceTabSwitchesToCustom(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), true, func(string) {})

	m.Update(keyMsg("tab"))
	if !m.inCustom {
		t.Error("tab should switch to custom input mode")
	}
}

func TestMultiChoiceTabWithoutAllowCustom(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(string) {})

	m.Update(keyMsg("tab"))
	if m.inCustom {
		t.Error("tab should not switch to custom when allowCustom=false")
	}
}

func TestMultiChoiceCustomTextInput(t *testing.T) {
	var result string
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), true, func(key string) {
		result = key
	})

	// Switch to custom mode.
	m.Update(keyMsg("tab"))

	// Type "hello".
	for _, ch := range "hello" {
		m.Update(keyMsg(string(ch)))
	}
	if m.customInput != "hello" {
		t.Fatalf("expected customInput='hello', got %q", m.customInput)
	}

	// Submit.
	_, _, closed := m.Update(keyMsg("enter"))
	if !closed {
		t.Error("enter should close custom input")
	}
	if result != "hello" {
		t.Errorf("expected 'hello', got %q", result)
	}
}

func TestMultiChoiceCustomBackspace(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), true, func(string) {})

	m.Update(keyMsg("tab"))
	for _, ch := range "abc" {
		m.Update(keyMsg(string(ch)))
	}

	m.Update(keyMsg("backspace"))
	if m.customInput != "ab" {
		t.Errorf("expected 'ab' after backspace, got %q", m.customInput)
	}
}

func TestMultiChoiceCustomTabBack(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), true, func(string) {})

	m.Update(keyMsg("tab"))
	if !m.inCustom {
		t.Fatal("should be in custom mode")
	}

	m.Update(keyMsg("tab"))
	if m.inCustom {
		t.Error("second tab should exit custom mode")
	}
	if m.selected < 0 {
		t.Error("should return to an option after exiting custom mode")
	}
}

func TestMultiChoiceCustomEscCancels(t *testing.T) {
	var result string
	called := false
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), true, func(key string) {
		result = key
		called = true
	})

	m.Update(keyMsg("tab"))
	for _, ch := range "some text" {
		m.Update(keyMsg(string(ch)))
	}

	_, _, closed := m.Update(keyMsg("esc"))
	if !closed {
		t.Error("esc should close in custom mode")
	}
	if !called || result != "" {
		t.Errorf("esc in custom mode should cancel (empty key), got called=%v key=%q", called, result)
	}
}

// -- View rendering --

func TestMultiChoiceViewContainsTitle(t *testing.T) {
	m := NewMultiChoiceOverlay("Choose Wisely", sampleOptions(), false, func(string) {})
	view := m.View(80, 40, testTheme())
	if !strings.Contains(view, "Choose Wisely") {
		t.Error("view should contain the title")
	}
}

func TestMultiChoiceViewContainsOptionLabels(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(string) {})
	view := m.View(80, 40, testTheme())

	for _, opt := range sampleOptions() {
		if !strings.Contains(view, opt.Label) {
			t.Errorf("view should contain option label %q", opt.Label)
		}
	}
}

func TestMultiChoiceViewContainsDescriptions(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(string) {})
	view := m.View(80, 40, testTheme())

	if !strings.Contains(view, "First option") {
		t.Error("view should contain option description 'First option'")
	}
}

func TestMultiChoiceViewShowsCustomOption(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), true, func(string) {})
	view := m.View(80, 40, testTheme())

	if !strings.Contains(view, "Custom") {
		t.Error("view should show Custom option when allowCustom=true")
	}
	if !strings.Contains(view, "tab=custom") {
		t.Error("view should show tab=custom hint when allowCustom=true")
	}
}

func TestMultiChoiceViewHidesCustomWhenNotAllowed(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(string) {})
	view := m.View(80, 40, testTheme())

	if strings.Contains(view, "tab=custom") {
		t.Error("view should not show tab=custom hint when allowCustom=false")
	}
}

func TestMultiChoiceViewNonEmpty(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(string) {})
	view := m.View(80, 40, testTheme())
	if view == "" {
		t.Error("view should produce non-empty output")
	}
}

// -- Empty options --

func TestMultiChoiceEmptyOptions(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", nil, false, func(string) {})

	// Enter with no options should not close (nothing to select).
	_, _, closed := m.Update(keyMsg("enter"))
	if closed {
		t.Error("enter with no options should not close")
	}

	// Navigation should not panic.
	m.Update(keyMsg("down"))
	m.Update(keyMsg("up"))
}

// -- Non-key messages are ignored --

func TestMultiChoiceIgnoresNonKeyMsg(t *testing.T) {
	m := NewMultiChoiceOverlay("Pick", sampleOptions(), false, func(string) {})
	// Sending a non-KeyMsg (e.g., a WindowSizeMsg analog) should not crash.
	type otherMsg struct{}
	_, _, closed := m.Update(otherMsg{})
	if closed {
		t.Error("non-key message should not close the overlay")
	}
}
