package tui

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

func styledIDEContextStatusText() string {
	if text := ideContextStatusText(); text != "" {
		return styles.InfoStyle.Render(text)
	}
	return ""
}

func ideContextStatusText() string {
	if file := ideContextFileFromEnv(); file != "" {
		return "⧉ In " + file
	}
	return ""
}

func ideContextFileFromEnv() string {
	for _, key := range []string{
		"AGENTM_IDE_ACTIVE_FILE",
		"AGENTM_ACTIVE_FILE",
		"CLAUDE_CODE_ACTIVE_FILE",
		"CLAUDE_CODE_IDE_ACTIVE_FILE",
	} {
		if file := basenameForIDEContext(os.Getenv(key)); file != "" {
			return file
		}
	}
	return ""
}

func basenameForIDEContext(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	base := filepath.Base(path)
	if base == "." || base == string(filepath.Separator) || base == "" {
		return ""
	}
	return base
}
