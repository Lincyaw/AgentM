package messages

// UI toggle messages control various UI state toggles.
type (
	// ToggleYoloMsg toggles YOLO mode (auto-approve tools).
	ToggleYoloMsg struct{}

	// TogglePauseMsg toggles whether the runtime loop is paused at
	// iteration boundaries. The pause takes effect as soon as the
	// in-flight LLM request and its tool calls complete.
	TogglePauseMsg struct{}

	// ToggleHideToolResultsMsg toggles hiding of verbose tool details.
	ToggleHideToolResultsMsg struct{}

	// SetTranscriptDetailMsg applies Claude-style transcript disclosure state:
	// compact, detailed, or detailed+verbose.
	SetTranscriptDetailMsg struct {
		Detailed bool
		Verbose  bool
	}

	// ToggleSidebarMsg toggles sidebar visibility.
	// The top-level model also handles this to persist the collapsed state.
	ToggleSidebarMsg struct{}

	// SessionToggleChangedMsg is sent after any session toggle (YOLO, split diff, etc.)
	// changes so that components like the sidebar can invalidate their caches.
	SessionToggleChangedMsg struct{}

	// SetThinkingModeMsg sets whether extended thinking is enabled for this
	// terminal session.
	SetThinkingModeMsg struct{ Enabled bool }

	// SetThinkingLevelMsg sets the concrete thinking-effort level for this
	// terminal session. Supported backend levels are off, low, medium, high.
	SetThinkingLevelMsg struct {
		Level          string
		ShowTranscript bool
	}

	// ShowCostDialogMsg shows the cost/usage dialog.
	ShowCostDialogMsg struct{}

	// CycleSessionColorMsg cycles the Claude-style prompt bar color for this session.
	CycleSessionColorMsg struct{}

	// ShowConfigDialogMsg opens the settings dialog on the Config tab.
	ShowConfigDialogMsg struct{}

	// ShowSettingsDialogMsg opens the settings dialog on the Config tab.
	ShowSettingsDialogMsg struct{}

	// ShowContextDialogMsg renders the Claude-style context usage report.
	ShowContextDialogMsg struct{}

	// ShowUsageDialogMsg shows the Claude-style usage dialog.
	ShowUsageDialogMsg struct{}

	// ShowPermissionsDialogMsg shows the permissions dialog.
	ShowPermissionsDialogMsg struct{}

	// ShowHelpMsg renders the local Claude-style help panel.
	ShowHelpMsg struct{}

	// ShowToolsDialogMsg shows the gateway-advertised tool catalogue.
	ShowToolsDialogMsg struct{}

	// ShowSkillsDialogMsg shows the skills dialog: the list of skills
	// available to the current agent.
	ShowSkillsDialogMsg struct{}

	// ShowStatusMsg renders the local Claude-style status panel.
	ShowStatusMsg struct{}
)
