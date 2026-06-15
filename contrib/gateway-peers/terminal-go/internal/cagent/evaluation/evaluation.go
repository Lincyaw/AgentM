// Package evaluation is a minimal shim exposing only the Save entry point the
// TUI consumes. The real cagent evaluation package pulls eval-run scoring and
// session serialization machinery; this shim keeps the faithful signature with
// a stub body. The adapter owner may later supply real eval persistence.
package evaluation

import (
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
)

// Save persists an evaluation for the given session under the supplied
// filename and returns the written path. This shim is a stub.
func Save(sess *session.Session, filename string) (string, error) {
	panic("stub: evaluation.Save not implemented")
}
