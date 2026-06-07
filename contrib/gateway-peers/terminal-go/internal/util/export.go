package util

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// ExportOptions holds metadata for the exported transcript.
type ExportOptions struct {
	Model       string
	TokensIn    int
	TokensOut   int
	CostSession float64
	SessionKey  string
	Duration    time.Duration
}

// ExportBlock represents one block in the transcript for export.
type ExportBlock struct {
	Kind    string // "user", "assistant", "system", "tool", "subagent", "thinking"
	Content string
	Meta    map[string]string // extra metadata (tool name, source, purpose, etc.)
}

// ExportTranscript writes the session transcript as a markdown file.
func ExportTranscript(path string, blocks []ExportBlock, opts ExportOptions) error {
	var sb strings.Builder

	// YAML frontmatter
	sb.WriteString("---\n")
	sb.WriteString(fmt.Sprintf("model: %s\n", opts.Model))
	sb.WriteString(fmt.Sprintf("tokens_in: %d\n", opts.TokensIn))
	sb.WriteString(fmt.Sprintf("tokens_out: %d\n", opts.TokensOut))
	sb.WriteString(fmt.Sprintf("cost: $%.4f\n", opts.CostSession))
	sb.WriteString(fmt.Sprintf("session: %s\n", opts.SessionKey))
	sb.WriteString(fmt.Sprintf("duration: %s\n", opts.Duration.Round(time.Second)))
	sb.WriteString(fmt.Sprintf("exported: %s\n", time.Now().Format(time.RFC3339)))
	sb.WriteString("---\n\n")

	sb.WriteString("# Session Transcript\n\n")

	for _, b := range blocks {
		switch b.Kind {
		case "user":
			sb.WriteString("## User\n\n")
			sb.WriteString(b.Content + "\n\n")
		case "assistant":
			sb.WriteString("## Assistant\n\n")
			sb.WriteString(b.Content + "\n\n")
		case "system":
			source := b.Meta["source"]
			sb.WriteString(fmt.Sprintf("## System (%s)\n\n", source))
			sb.WriteString(b.Content + "\n\n")
		case "tool":
			name := b.Meta["name"]
			status := b.Meta["status"]
			sb.WriteString(fmt.Sprintf("### Tool: %s (%s)\n\n", name, status))
			sb.WriteString("```\n" + b.Content + "\n```\n\n")
		case "subagent":
			purpose := b.Meta["purpose"]
			status := b.Meta["status"]
			sb.WriteString(fmt.Sprintf("### Subagent: %s (%s)\n\n", purpose, status))
		case "thinking":
			sb.WriteString("### Thinking\n\n")
			sb.WriteString("> " + strings.ReplaceAll(b.Content, "\n", "\n> ") + "\n\n")
		}
	}

	return os.WriteFile(path, []byte(sb.String()), 0644)
}
