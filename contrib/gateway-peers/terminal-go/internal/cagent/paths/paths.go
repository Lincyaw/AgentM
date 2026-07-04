package paths

import (
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"

	pathx "github.com/AoyangSpace/agentm-terminal/internal/cagent/path"
)

const terminalStateDir = "terminal-go"

// overridable holds an optional directory override backed by an atomic pointer.
// A nil pointer (the zero value) means "use the default".
type overridable struct{ p atomic.Pointer[string] }

// Set stores an override directory. An empty value clears the override.
func (o *overridable) Set(dir string) {
	if dir == "" {
		o.p.Store(nil)
	} else {
		dir = pathx.ExpandPath(dir)
		o.p.Store(&dir)
	}
}

// get returns the override if set, or falls back to the result of defaultFn.
func (o *overridable) get(defaultFn func() string) string {
	if p := o.p.Load(); p != nil {
		return filepath.Clean(*p)
	}
	return defaultFn()
}

var (
	cacheDirOverride  overridable
	configDirOverride overridable
	dataDirOverride   overridable
)

// SetCacheDir overrides the default cache directory returned by [GetCacheDir].
// A leading "~" and environment variables are expanded. An empty value restores
// the default behaviour.
// This should be called early (e.g. during CLI flag processing) before any
// goroutine calls the corresponding getter.
func SetCacheDir(dir string) { cacheDirOverride.Set(dir) }

// SetConfigDir overrides the default config directory returned by [GetConfigDir].
// A leading "~" and environment variables are expanded. An empty value restores
// the default behaviour.
func SetConfigDir(dir string) { configDirOverride.Set(dir) }

// SetDataDir overrides the default data directory returned by [GetDataDir].
// A leading "~" and environment variables are expanded. An empty value restores
// the default behaviour.
func SetDataDir(dir string) { dataDirOverride.Set(dir) }

// SetRoot re-homes all AgentM Terminal state under one directory: data,
// config, and cache land in the "data", "config", and "cache"
// subdirectories of root. It is the one-call override for embedders that
// must keep their embedded agent's state isolated from another installation.
// A leading "~" and environment variables are expanded. An empty root restores
// the per-directory defaults.
func SetRoot(root string) {
	if root == "" {
		SetDataDir("")
		SetConfigDir("")
		SetCacheDir("")
		return
	}
	SetDataDir(filepath.Join(root, "data"))
	SetConfigDir(filepath.Join(root, "config"))
	SetCacheDir(filepath.Join(root, "cache"))
}

// GetCacheDir returns the user's cache directory for AgentM Terminal.
//
// If an override has been set via [SetCacheDir] it is returned instead.
//
// By default this is $AGENTM_HOME/terminal-go/cache, with $AGENTM_HOME
// defaulting to ~/.agentm.
func GetCacheDir() string {
	return cacheDirOverride.get(func() string {
		return filepath.Join(GetAgentMHome(), terminalStateDir, "cache")
	})
}

// GetConfigDir returns the user's config directory for AgentM Terminal.
//
// If an override has been set via [SetConfigDir] it is returned instead.
//
// By default this is $AGENTM_HOME/terminal-go/config, with $AGENTM_HOME
// defaulting to ~/.agentm.
func GetConfigDir() string {
	return configDirOverride.get(func() string {
		return filepath.Join(GetAgentMHome(), terminalStateDir, "config")
	})
}

// GetDataDir returns the user's data directory for AgentM Terminal.
//
// If an override has been set via [SetDataDir] it is returned instead.
//
// By default this is $AGENTM_HOME/terminal-go/data, with $AGENTM_HOME
// defaulting to ~/.agentm.
func GetDataDir() string {
	return dataDirOverride.get(func() string {
		return filepath.Join(GetAgentMHome(), terminalStateDir, "data")
	})
}

// GetAgentMHome returns the shared AgentM home directory.
//
// It honors $AGENTM_HOME, including a leading "~", and otherwise defaults to
// ~/.agentm. When the OS home directory is unavailable, it falls back to a
// per-user directory under the system temporary directory.
func GetAgentMHome() string {
	home := os.Getenv("AGENTM_HOME")
	if home == "" {
		if userHome, err := os.UserHomeDir(); err == nil && userHome != "" {
			return filepath.Clean(filepath.Join(userHome, ".agentm"))
		}
		return filepath.Clean(filepath.Join(os.TempDir(), fmt.Sprintf("agentm-home-%d", os.Getuid())))
	}
	expanded, err := pathx.ExpandHomeDir(home)
	if err != nil {
		return filepath.Clean(home)
	}
	return filepath.Clean(expanded)
}

// GetHomeDir returns the user's home directory.
//
// Returns an empty string if the home directory cannot be determined.
func GetHomeDir() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Clean(homeDir)
}
