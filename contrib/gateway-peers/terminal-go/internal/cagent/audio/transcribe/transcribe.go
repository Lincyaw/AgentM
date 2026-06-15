// Package transcribe provides real-time audio transcription. This is a minimal
// shim using cagent's non-macOS stub variant (no transitive deps); the adapter
// owner may later supply a platform-specific implementation. The build tag from
// the original is dropped so the stub applies on every platform.
package transcribe

import (
	"context"
	"errors"
)

// ErrNotSupported is returned when transcription is not supported.
var ErrNotSupported = errors.New("speech-to-text is only supported on macOS")

// TranscriptHandler is called when new transcription text is received.
type TranscriptHandler func(delta string)

// Transcriber provides real-time audio transcription. This shim is a stub
// that returns errors.
type Transcriber struct{}

// New creates a new Transcriber with the given OpenAI API key.
func New(apiKey string) *Transcriber {
	return &Transcriber{}
}

// Start returns ErrNotSupported.
func (t *Transcriber) Start(ctx context.Context, handler TranscriptHandler) error {
	return ErrNotSupported
}

// Stop is a no-op.
func (t *Transcriber) Stop() {}

// IsRunning always returns false.
func (t *Transcriber) IsRunning() bool {
	return false
}

// IsSupported returns false.
func (t *Transcriber) IsSupported() bool {
	return false
}
