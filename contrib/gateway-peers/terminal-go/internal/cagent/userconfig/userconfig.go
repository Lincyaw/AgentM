// Package userconfig is a minimal shim exposing only the config surface the
// TUI consumes: the Settings type (with its getter methods), the Config type
// (Settings field), and the Get/Load entry points. The real cagent userconfig
// package reads ~/.config/cagent/config.yaml and pulls go-yaml + config/latest;
// this shim keeps faithful types with defaulting getters and stub persistence.
// The adapter owner can later back Get/Load with real loading.
package userconfig

// DefaultTabTitleMaxLength is the default maximum tab title length when not configured.
const DefaultTabTitleMaxLength = 20

// DefaultSoundThreshold is the default duration threshold for sound notifications.
const DefaultSoundThreshold = 10

// Settings represents global user settings.
type Settings struct {
	// HideToolResults hides verbose tool parameters and results in the TUI by default.
	// Historical config name kept for compatibility.
	HideToolResults bool `yaml:"hide_tool_results,omitempty"`
	// ExpandThinking expands reasoning/tool blocks in the TUI by default.
	ExpandThinking *bool `yaml:"expand_thinking,omitempty"`
	// SplitDiffView enables side-by-side split diff rendering for file edits.
	SplitDiffView *bool `yaml:"split_diff_view,omitempty"`
	// Theme is the default theme reference (e.g., "dark", "light").
	Theme string `yaml:"theme,omitempty"`
	// YOLO enables auto-approve mode for all tool calls globally.
	YOLO bool `yaml:"YOLO,omitempty"`
	// TabTitleMaxLength is the maximum display length for tab titles in the TUI.
	TabTitleMaxLength int `yaml:"tab_title_max_length,omitempty"`
	// RestoreTabs restores previously open tabs when launching the TUI.
	RestoreTabs *bool `yaml:"restore_tabs,omitempty"`
	// Sound enables playing notification sounds on task success or failure.
	Sound bool `yaml:"sound,omitempty"`
	// SoundThreshold is the minimum duration in seconds a task must run
	// before a success sound is played.
	SoundThreshold int `yaml:"sound_threshold,omitempty"`
}

// GetTabTitleMaxLength returns the configured tab title max length, falling back to the default.
func (s *Settings) GetTabTitleMaxLength() int {
	if s == nil || s.TabTitleMaxLength <= 0 {
		return DefaultTabTitleMaxLength
	}
	return s.TabTitleMaxLength
}

// GetSound returns whether sound notifications are enabled, defaulting to false.
func (s *Settings) GetSound() bool {
	if s == nil {
		return false
	}
	return s.Sound
}

// GetSoundThreshold returns the minimum duration for sound notifications, defaulting to 10s.
func (s *Settings) GetSoundThreshold() int {
	if s == nil || s.SoundThreshold <= 0 {
		return DefaultSoundThreshold
	}
	return s.SoundThreshold
}

// GetExpandThinking returns whether reasoning/tool blocks are expanded by default.
func (s *Settings) GetExpandThinking() bool {
	if s == nil || s.ExpandThinking == nil {
		return false
	}
	return *s.ExpandThinking
}

// GetSplitDiffView returns whether split diff view is enabled, defaulting to true.
func (s *Settings) GetSplitDiffView() bool {
	if s == nil || s.SplitDiffView == nil {
		return true
	}
	return *s.SplitDiffView
}

// Config represents the user-level configuration.
type Config struct {
	// Settings contains global user settings.
	Settings *Settings `yaml:"settings,omitempty"`
}

// Save persists the configuration to disk. This shim is a no-op; the adapter
// owner may replace it with real persistence.
func (c *Config) Save() error {
	return nil
}

// Load loads the user configuration. This shim returns an empty config; the
// adapter owner may replace it with real disk loading.
func Load() (*Config, error) {
	return &Config{Settings: &Settings{}}, nil
}

// Get returns the global user settings. This shim returns defaults; the
// adapter owner may replace it with real loading.
//
// Pointer-typed fields the TUI dereferences directly (without a nil guard) MUST
// be populated here. In particular tabs.go reads `*userconfig.Get().RestoreTabs`
// with no nil check — cagent's real loader always materialises the pointer, so
// the shim mirrors that by defaulting RestoreTabs to a non-nil false. Leaving it
// nil panics the TUI at launch.
func Get() *Settings {
	restoreTabs := false
	return &Settings{
		RestoreTabs: &restoreTabs,
	}
}
