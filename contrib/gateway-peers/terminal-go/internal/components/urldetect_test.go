package components

import (
	"strings"
	"testing"
)

func TestDetectHTTPS(t *testing.T) {
	d := NewURLDetector()
	lines := []string{"visit https://example.com for details"}
	spans := d.Detect(lines)
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if spans[0].URL != "https://example.com" {
		t.Errorf("expected %q, got %q", "https://example.com", spans[0].URL)
	}
	if spans[0].LineY != 0 {
		t.Errorf("expected lineY=0, got %d", spans[0].LineY)
	}
}

func TestDetectHTTP(t *testing.T) {
	d := NewURLDetector()
	lines := []string{"http://foo.bar/path?q=1"}
	spans := d.Detect(lines)
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if spans[0].URL != "http://foo.bar/path?q=1" {
		t.Errorf("got %q", spans[0].URL)
	}
}

func TestDetectMultipleURLs(t *testing.T) {
	d := NewURLDetector()
	lines := []string{"see https://a.com and https://b.com/path"}
	spans := d.Detect(lines)
	if len(spans) != 2 {
		t.Fatalf("expected 2 spans, got %d", len(spans))
	}
	if spans[0].URL != "https://a.com" {
		t.Errorf("span[0] = %q", spans[0].URL)
	}
	if spans[1].URL != "https://b.com/path" {
		t.Errorf("span[1] = %q", spans[1].URL)
	}
}

func TestDetectMultipleLines(t *testing.T) {
	d := NewURLDetector()
	lines := []string{
		"line one",
		"https://first.com",
		"line three https://second.com done",
	}
	spans := d.Detect(lines)
	if len(spans) != 2 {
		t.Fatalf("expected 2 spans, got %d", len(spans))
	}
	if spans[0].LineY != 1 {
		t.Errorf("first span lineY = %d, want 1", spans[0].LineY)
	}
	if spans[1].LineY != 2 {
		t.Errorf("second span lineY = %d, want 2", spans[1].LineY)
	}
}

func TestDetectNoURLs(t *testing.T) {
	d := NewURLDetector()
	lines := []string{"no urls here", "just text"}
	spans := d.Detect(lines)
	if len(spans) != 0 {
		t.Errorf("expected 0 spans, got %d", len(spans))
	}
}

func TestDetectStopsAtDelimiters(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"(https://example.com)", "https://example.com"},
		{"[https://example.com]", "https://example.com"},
		{`"https://example.com"`, "https://example.com"},
		{"https://example.com>", "https://example.com"},
		{"https://example.com'done", "https://example.com"},
	}
	for _, tt := range tests {
		d := NewURLDetector()
		spans := d.Detect([]string{tt.input})
		if len(spans) != 1 {
			t.Errorf("input %q: expected 1 span, got %d", tt.input, len(spans))
			continue
		}
		if spans[0].URL != tt.want {
			t.Errorf("input %q: got %q, want %q", tt.input, spans[0].URL, tt.want)
		}
	}
}

func TestDetectColumnPositions(t *testing.T) {
	d := NewURLDetector()
	// "abc https://x.co def"
	//  0123456789...
	lines := []string{"abc https://x.co def"}
	spans := d.Detect(lines)
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if spans[0].StartX != 4 {
		t.Errorf("StartX = %d, want 4", spans[0].StartX)
	}
	// "https://x.co" is 12 runes, so EndX = 4+12 = 16
	if spans[0].EndX != 16 {
		t.Errorf("EndX = %d, want 16", spans[0].EndX)
	}
}

func TestHitTest(t *testing.T) {
	d := NewURLDetector()
	lines := []string{"abc https://x.co def"}
	d.Detect(lines)

	// Inside the URL
	span := d.HitTest(5, 0)
	if span == nil {
		t.Fatal("expected a span at (5,0)")
	}
	if span.URL != "https://x.co" {
		t.Errorf("URL = %q", span.URL)
	}

	// Outside the URL
	span = d.HitTest(0, 0)
	if span != nil {
		t.Error("expected nil at (0,0)")
	}

	// After the URL
	span = d.HitTest(17, 0)
	if span != nil {
		t.Error("expected nil at (17,0)")
	}

	// Wrong line
	span = d.HitTest(5, 1)
	if span != nil {
		t.Error("expected nil on wrong line")
	}
}

func TestHoveredURL(t *testing.T) {
	d := NewURLDetector()
	if d.HoveredURL() != "" {
		t.Error("expected empty hovered URL initially")
	}

	d.Detect([]string{"https://test.com"})
	span := d.HitTest(0, 0)
	d.SetHovered(span)
	if d.HoveredURL() != "https://test.com" {
		t.Errorf("hovered = %q", d.HoveredURL())
	}

	d.SetHovered(nil)
	if d.HoveredURL() != "" {
		t.Error("expected empty after clearing hover")
	}
}

func TestHighlightURLsUnderline(t *testing.T) {
	d := NewURLDetector()
	lines := []string{"go to https://x.co now"}
	d.Detect(lines)

	result := d.HighlightURLs(lines[0], 0)
	if !strings.Contains(result, "\033[4m") {
		t.Error("expected underline ANSI in result")
	}
	if !strings.Contains(result, "https://x.co") {
		t.Error("expected URL text in result")
	}
	if !strings.Contains(result, "\033[0m") {
		t.Error("expected reset ANSI in result")
	}
	// "go to " and " now" should remain plain
	if !strings.HasPrefix(result, "go to ") {
		t.Error("prefix should be unchanged")
	}
	if !strings.HasSuffix(result, " now") {
		t.Error("suffix should be unchanged")
	}
}

func TestHighlightURLsHovered(t *testing.T) {
	d := NewURLDetector()
	lines := []string{"https://hov.er"}
	d.Detect(lines)
	span := d.HitTest(0, 0)
	d.SetHovered(span)

	result := d.HighlightURLs(lines[0], 0)
	// Hovered URL gets bold+underline
	if !strings.Contains(result, "\033[1;4m") {
		t.Error("expected bold+underline ANSI for hovered URL")
	}
}

func TestHighlightURLsNoMatch(t *testing.T) {
	d := NewURLDetector()
	d.Detect([]string{"no urls"})

	result := d.HighlightURLs("no urls", 0)
	if result != "no urls" {
		t.Errorf("expected unchanged line, got %q", result)
	}
}

func TestHighlightURLsWrongLine(t *testing.T) {
	d := NewURLDetector()
	d.Detect([]string{"https://x.co"})

	// Line 1 has no spans
	result := d.HighlightURLs("plain text", 1)
	if result != "plain text" {
		t.Errorf("expected unchanged line for non-matching lineY, got %q", result)
	}
}

func TestDetectReuseClearsOldSpans(t *testing.T) {
	d := NewURLDetector()
	d.Detect([]string{"https://first.com"})
	if len(d.spans) != 1 {
		t.Fatal("setup failed")
	}

	d.Detect([]string{"no urls"})
	if len(d.spans) != 0 {
		t.Errorf("expected 0 spans after re-detect, got %d", len(d.spans))
	}
}
