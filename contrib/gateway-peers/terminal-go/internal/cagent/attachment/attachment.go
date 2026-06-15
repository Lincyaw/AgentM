// Package attachment provides MIME-aware routing for document attachments.
//
// It defines how a chat.Document should be sent to a model: either dropped
// (unsupported), wrapped in a plain-text envelope (StrategyTXT), or encoded
// as inline base64 data (StrategyB64).
package attachment

import (
	"fmt"
	"strings"
	"unicode"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/chat"
)

// ModelCapabilities describes what MIME types a given model can accept as
// document attachments. The full models.dev-backed loader runs server-side and
// never reaches this client on the wire path, so only the two booleans the
// routing decision reads are modelled locally.
type ModelCapabilities struct {
	SupportsImage bool
	SupportsPDF   bool
}

// Supports reports whether the model can accept an attachment with the given
// MIME type. image/* requires image support, application/pdf requires PDF
// support, text/* is always accepted, and everything else is rejected.
func (mc ModelCapabilities) Supports(mimeType string) bool {
	mt := strings.ToLower(mimeType)
	switch {
	case strings.HasPrefix(mt, "image/"):
		return mc.SupportsImage
	case mt == "application/pdf":
		return mc.SupportsPDF
	case strings.HasPrefix(mt, "text/"):
		return true
	default:
		return false
	}
}

// Strategy describes how an attachment should be handled before sending to the
// provider.
type Strategy int

const (
	// StrategyDrop means the attachment is not supported by the model or has no
	// inline content, and should be silently skipped (with a log warning).
	StrategyDrop Strategy = iota

	// StrategyTXT means the attachment should be wrapped in a TXTEnvelope and
	// sent as plain text.  Used for text/* MIME types whose content is already
	// in Source.InlineText.
	StrategyTXT

	// StrategyB64 means the attachment content (Source.InlineData) should be
	// base64-encoded and sent as a native provider image/document block.
	StrategyB64
)

// Decide returns the routing Strategy for a document given the current model's
// capabilities.
//
// Algorithm:
//  1. If the model does not support the document's MIME type → (Drop, reason).
//  2. If Source.InlineData is non-empty → (B64, "").
//  3. If Source.InlineText is non-empty → (TXT, "").
//  4. Otherwise → (Drop, "no inline content").
func Decide(doc chat.Document, mc ModelCapabilities) (Strategy, string) {
	if !mc.Supports(doc.MimeType) {
		return StrategyDrop, fmt.Sprintf("model does not support MIME type %q", doc.MimeType)
	}
	if len(doc.Source.InlineData) > 0 {
		return StrategyB64, ""
	}
	if doc.Source.InlineText != "" {
		return StrategyTXT, ""
	}
	return StrategyDrop, "no inline content"
}

// TXTEnvelope wraps text content in a unique XML-like tag derived from the
// document name and MIME type. The tag name is a slug of both, making
// accidental tag break-out in the content practically impossible without
// escaping the body.
//
// Example: a document named "report.md" with MIME "text/markdown" produces:
//
//	<document-report-md-text-markdown>
//	…body…
//	</document-report-md-text-markdown>
func TXTEnvelope(name, mimeType, body string) string {
	slug := slugify(name + "-" + mimeType)
	tag := "document-" + slug
	return fmt.Sprintf("<%s>\n%s\n</%s>", tag, body, tag)
}

// slugify converts s to a lowercase, alphanumeric-and-hyphens-only string.
// Non-alphanumeric runes are replaced with hyphens; consecutive hyphens are
// collapsed to one; leading and trailing hyphens are trimmed.
// If the result is empty, "doc" is returned as a safe fallback.
func slugify(s string) string {
	var b strings.Builder
	prevHyphen := false
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			prevHyphen = false
		} else if !prevHyphen {
			b.WriteRune('-')
			prevHyphen = true
		}
	}
	result := strings.Trim(b.String(), "-")
	if result == "" {
		return "doc"
	}
	return result
}
