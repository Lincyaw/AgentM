package runtime

// ResumeType identifies the user's response to a confirmation request.
//
// The runtime emits a TOOL_PERMISSION_REQUEST event whenever a tool call
// requires user approval, then blocks until the embedder calls Resume(...)
// with one of the values below.
type ResumeType string

const (
	// ResumeTypeApprove approves the single pending tool call.
	ResumeTypeApprove ResumeType = "approve"
	// ResumeTypeApproveSession approves the pending tool call and every
	// subsequent permission-gated call for the rest of the session.
	ResumeTypeApproveSession ResumeType = "approve-session"
	// ResumeTypeApproveTool approves the pending call and every future
	// call to the same tool name within the session.
	ResumeTypeApproveTool ResumeType = "approve-tool"
	// ResumeTypeReject rejects the pending tool call.
	ResumeTypeReject ResumeType = "reject"
)

// ResumeRequest carries the user's confirmation decision along with an optional
// reason (used when rejecting a tool call to help the model understand why).
type ResumeRequest struct {
	Type     ResumeType
	Reason   string // Optional; primarily used with [ResumeTypeReject]
	ToolName string // Optional; used with [ResumeTypeApproveTool]
}

// ResumeApprove creates a ResumeRequest to approve a single tool call.
func ResumeApprove() ResumeRequest {
	return ResumeRequest{Type: ResumeTypeApprove}
}

// ResumeApproveSession creates a ResumeRequest to approve all tool calls for the session.
func ResumeApproveSession() ResumeRequest {
	return ResumeRequest{Type: ResumeTypeApproveSession}
}

// ResumeApproveTool creates a ResumeRequest to always approve a specific tool for the session.
func ResumeApproveTool(toolName string) ResumeRequest {
	return ResumeRequest{Type: ResumeTypeApproveTool, ToolName: toolName}
}

// ResumeReject creates a ResumeRequest to reject a tool call with an optional reason.
func ResumeReject(reason string) ResumeRequest {
	return ResumeRequest{Type: ResumeTypeReject, Reason: reason}
}

// IsValidResumeType validates confirmation values coming from /resume.
//
// The runtime may be resumed by multiple entry points (API, CLI, TUI, tests).
// Even if upstream layers perform validation, the runtime must never assume
// the ResumeType is valid; accepting invalid values leads to confusing
// downstream behaviour where tool execution fails without a clear cause.
func IsValidResumeType(t ResumeType) bool {
	switch t {
	case ResumeTypeApprove,
		ResumeTypeApproveSession,
		ResumeTypeApproveTool,
		ResumeTypeReject:
		return true
	default:
		return false
	}
}

// ValidResumeTypes returns all allowed confirmation values, in declaration order.
func ValidResumeTypes() []ResumeType {
	return []ResumeType{
		ResumeTypeApprove,
		ResumeTypeApproveSession,
		ResumeTypeApproveTool,
		ResumeTypeReject,
	}
}
