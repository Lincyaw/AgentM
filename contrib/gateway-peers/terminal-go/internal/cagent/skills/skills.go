// Package skills is a minimal shim exposing only the Skill type and the
// IsHomeSkillPath helper the TUI consumes. The real cagent skills package
// pulls skill loading/expansion/remote-cache machinery; this shim keeps the
// faithful Skill shape plus the home-path check, both backed by the vendored
// paths/path leaf utils.
package skills

import (
	"path/filepath"

	pathx "github.com/AoyangSpace/agentm-terminal/internal/cagent/path"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/paths"
)

// Skill represents a loaded skill with its metadata and content location.
type Skill struct {
	Name          string
	Description   string
	FilePath      string
	BaseDir       string
	Files         []string
	Local         bool // true for filesystem-loaded skills, false for remote
	License       string
	Compatibility string
	Metadata      map[string]string
	AllowedTools  []string
	Context       string // "fork" to run the skill as an isolated sub-agent
	Model         string
	InlineContent string
}

// IsInline reports whether the skill is defined inline in the agent config.
func (s Skill) IsInline() bool {
	return s.InlineContent != ""
}

// IsFork returns true when the skill should be executed in an isolated
// sub-agent context rather than inline in the current conversation.
func (s Skill) IsFork() bool {
	return s.Context == "fork"
}

// localSearchPath describes one directory to scan for local skills.
type localSearchPath struct {
	dir       string
	recursive bool
}

// IsHomeSkillPath reports whether path is under one of the global skill
// directories in the user's home directory.
func IsHomeSkillPath(path string) bool {
	home := paths.GetHomeDir()
	if home == "" {
		return false
	}

	for _, p := range homeSkillSearchPaths(home) {
		if pathx.IsWithin(path, p.dir) {
			return true
		}
	}
	return false
}

func homeSkillSearchPaths(home string) []localSearchPath {
	return []localSearchPath{
		{filepath.Join(home, ".codex", "skills"), true},
		{filepath.Join(home, ".claude", "skills"), false},
		{filepath.Join(home, ".agents", "skills"), true},
	}
}
