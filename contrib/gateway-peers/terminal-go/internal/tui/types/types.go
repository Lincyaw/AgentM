package types

import (
	"strings"
	"time"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/tools"
)

// MessageType represents different types of messages
type MessageType int

const (
	MessageTypeUser MessageType = iota
	MessageTypeAssistant
	MessageTypeAssistantReasoningBlock // Collapsed reasoning + tool calls block
	MessageTypeSpinner
	MessageTypeError
	MessageTypeShellOutput
	MessageTypeCancelled
	MessageTypeToolCall
	MessageTypeToolResult
	MessageTypeWelcome
	MessageTypeLoading
	// MessageTypeSystem is a system notice — the output of a control command
	// (/status, /help, ...). Rendered neutrally, with no agent author.
	MessageTypeSystem
	// MessageTypeNotice is a local, single-line transcript notice such as
	// Claude Code's `⎿  Kept ...` feedback after canceling a dialog.
	MessageTypeNotice
)

const (
	UserMessageEditLabel      = "✎"
	AssistantMessageCopyLabel = "⎘"
)

const (
	maxLiveToolOutputSize        = 100_000
	liveToolOutputTruncatedLabel = "[earlier live output truncated]\n"
)

// ToolStatus represents the status of a tool call
type ToolStatus int

const (
	ToolStatusPending ToolStatus = iota
	ToolStatusConfirmation
	ToolStatusRunning
	ToolStatusCompleted
	ToolStatusError
)

// Message represents a single message in the chat
type Message struct {
	Type           MessageType
	Content        string
	Sender         string                // Agent name for assistant messages
	ToolCall       tools.ToolCall        // Associated tool call for tool messages
	ToolDefinition tools.Tool            // Definition of the tool being called
	ToolStatus     ToolStatus            // Status for tool calls
	ToolResult     *tools.ToolCallResult // Result of tool call (when completed)
	// DirectShell is true for tools launched through Claude-style `!` shell
	// mode, where the user transcript already carries the command line.
	DirectShell bool
	// StartedAt records when a tool call entered ToolStatusRunning.
	// Used to display elapsed time for long-running tool calls.
	StartedAt *time.Time
	// SessionPosition is the index of this message in session.Messages (when known).
	// Used for operations like branching on edits.
	SessionPosition *int
}

func Agent(typ MessageType, agentName, content string) *Message {
	return &Message{
		Type:    typ,
		Sender:  agentName,
		Content: strings.ReplaceAll(content, "\t", "    "),
	}
}

func ShellOutput(content string) *Message {
	return &Message{
		Type:    MessageTypeShellOutput,
		Content: strings.ReplaceAll(content, "\t", "    "),
	}
}

func Spinner() *Message {
	return &Message{
		Type: MessageTypeSpinner,
	}
}

func Error(content string) *Message {
	return &Message{
		Type:    MessageTypeError,
		Content: strings.ReplaceAll(content, "\t", "    "),
	}
}

func User(content string) *Message {
	return &Message{
		Type:    MessageTypeUser,
		Content: strings.ReplaceAll(content, "\t", "    "),
	}
}

func Cancelled() *Message {
	return &Message{
		Type: MessageTypeCancelled,
	}
}

func Welcome(content string) *Message {
	return &Message{
		Type:    MessageTypeWelcome,
		Content: strings.ReplaceAll(content, "\t", "    "),
	}
}

// System builds a system-notice message (control-command output). An optional
// title is carried in Sender so the renderer can show it as a header without a
// dedicated field.
func System(title, content string) *Message {
	return &Message{
		Type:    MessageTypeSystem,
		Sender:  title,
		Content: strings.ReplaceAll(content, "\t", "    "),
	}
}

func Notice(content string) *Message {
	return &Message{
		Type:    MessageTypeNotice,
		Content: strings.ReplaceAll(content, "\t", "    "),
	}
}

func ToolCallMessage(agentName string, toolCall tools.ToolCall, toolDef tools.Tool, status ToolStatus) *Message {
	msg := &Message{
		Type:           MessageTypeToolCall,
		Sender:         agentName,
		ToolCall:       toolCall,
		ToolDefinition: toolDef,
		ToolStatus:     status,
	}
	if status == ToolStatusRunning {
		now := time.Now()
		msg.StartedAt = &now
	}
	return msg
}

func (m *Message) AppendToolOutput(output string) {
	if output == "" {
		return
	}
	combined := m.Content + strings.ReplaceAll(output, "\t", "    ")
	if len(combined) <= maxLiveToolOutputSize {
		m.Content = combined
		return
	}

	tailSize := maxLiveToolOutputSize - len(liveToolOutputTruncatedLabel)
	if tailSize <= 0 {
		m.Content = combined[len(combined)-maxLiveToolOutputSize:]
		return
	}
	m.Content = liveToolOutputTruncatedLabel + combined[len(combined)-tailSize:]
}

func Loading(description string) *Message {
	return &Message{
		Type:    MessageTypeLoading,
		Content: strings.ReplaceAll(description, "\t", "    "),
	}
}
