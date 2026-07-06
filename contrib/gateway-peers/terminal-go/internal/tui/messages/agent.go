package messages

// Agent messages control agent switching, commands, and model selection.
type (
	// SwitchAgentMsg switches to a different agent.
	SwitchAgentMsg struct{ AgentName string }

	// AgentCommandMsg sends a command to the agent.
	AgentCommandMsg struct{ Command string }

	// UnknownCommandMsg renders a Claude-style unknown slash command notice
	// in the transcript without starting a model turn.
	UnknownCommandMsg struct {
		Command    string
		Suggestion string
	}

	// OpenModelPickerMsg opens the model picker dialog.
	OpenModelPickerMsg struct {
		ShowTranscript bool
	}

	// OpenEffortPickerMsg opens the thinking-effort picker dialog.
	OpenEffortPickerMsg struct {
		ShowTranscript bool
	}

	// ModelPickerCanceledMsg closes the model picker. Slash-command model
	// pickers render Claude-style transcript feedback; keyboard pickers do not.
	ModelPickerCanceledMsg struct {
		ShowTranscript bool
	}

	// EffortPickerCanceledMsg closes the effort picker and restores transcript
	// layout state. Slash-command pickers may render transcript feedback.
	EffortPickerCanceledMsg struct {
		ShowTranscript bool
	}

	// ChangeModelMsg changes the model for the current agent.
	ChangeModelMsg struct {
		ModelRef           string
		TranscriptNotice   string
		WelcomeModelLine   string
		RevealFocusWarning bool
		ShowTranscript     bool
	}
)
