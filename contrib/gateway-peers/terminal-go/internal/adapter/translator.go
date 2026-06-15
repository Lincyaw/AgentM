// Package adapter bridges the AgentM gateway wire protocol (internal/wire) to
// the vendored cagent TUI data layer (internal/cagent). It has two halves:
//
//   - translator: incoming wire `outbound` envelopes -> runtime.*Event values,
//     pushed into *app.App so the TUI's supervisor renders them. It also keeps a
//     session.Session view-model in sync (token counts, title) as events arrive.
//   - controller: an app.Controller implementation that turns the TUI's user
//     actions (Run / Resume / CompactSession / ...) into `inbound` envelopes the
//     WireClient sends to the gateway.
//
// The gateway's wire vocabulary is the metadata.kind discriminator from
// .claude/designs/single-process-gateway.md §2.5 (assistant_text, stream_text,
// stream_thinking, tool_call, tool_result, usage, turn_start, agent_end,
// approval_request, diagnostic_*, session_ready, ...). Where a kind has no
// cagent equivalent we log and drop — fidelity of the TUI is never compromised
// by an unmapped event.
package adapter

import (
	"encoding/json"
	"log"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/tools"
	"github.com/AoyangSpace/agentm-terminal/internal/wire"
)

// defaultAgentName labels events that carry no agent attribution on the wire.
// The gateway protocol is single-agent from the client's point of view, so a
// stable label is enough for the sidebar / transcript header.
const defaultAgentName = "agent"

// Translator converts wire envelopes into runtime events and feeds them into an
// *app.App, keeping the App's session view-model current as a side effect.
type Translator struct {
	app       *app.App
	sess      *session.Session
	agentName string

	// streaming tracks whether a stream is currently open so we can synthesise
	// StreamStarted / StreamStopped around the gateway's turn boundaries.
	streaming bool

	// pendingApprovalID is the approval_id of the most recent approval_request
	// card. The controller reads it to compose the "<approval_id>:approve|deny"
	// button_value an inbound carries when the user resolves the tool dialog.
	// The gateway runs one approval at a time per session (a blocking Future),
	// so a single slot suffices.
	pendingApprovalID string
}

// PendingApprovalID returns the approval_id awaiting resolution, or "".
func (t *Translator) PendingApprovalID() string {
	return t.pendingApprovalID
}

// NewTranslator builds a Translator bound to an App and its session.
func NewTranslator(a *app.App, sess *session.Session) *Translator {
	return &Translator{
		app:       a,
		sess:      sess,
		agentName: defaultAgentName,
	}
}

// emit pushes an event into the App's fan-out (and thus to the TUI).
func (t *Translator) emit(ev runtime.Event) {
	if ev == nil {
		return
	}
	t.app.EmitEvent(ev)
}

func (t *Translator) sessionID() string {
	if t.sess == nil {
		return ""
	}
	return t.sess.ID
}

// HandleEnvelope dispatches one envelope received from the gateway. Non-outbound
// kinds (error) are mapped to an ErrorEvent; everything else is ignored at the
// wire layer (ping/pong/ack are handled inside WireClient).
func (t *Translator) HandleEnvelope(env *wire.Envelope) {
	switch env.Kind {
	case wire.KindOutbound:
		t.handleOutbound(env.Body)
	case wire.KindError:
		msg, _ := env.Body["message"].(string)
		if msg == "" {
			msg = "gateway error"
		}
		t.emit(runtime.Error(msg))
	default:
		log.Printf("[adapter] dropping envelope kind=%s", env.Kind)
	}
}

// handleOutbound maps one outbound body (§2.5) to zero or more runtime events.
//
// The discriminator lives in body.metadata.kind; body.content is the primary
// text and the rest of metadata carries the kind-specific fields the
// wire_driver atom projected (turn_id, tool_call_id, name, args, usage counts,
// ...). See src/agentm/extensions/builtin/wire_driver.py for the producer side.
func (t *Translator) handleOutbound(body map[string]any) {
	meta, _ := body["metadata"].(map[string]any)
	kind, _ := meta["kind"].(string)
	content, _ := body["content"].(string)

	switch kind {
	case "turn_start":
		// Open a stream so the TUI shows the working spinner and groups the
		// upcoming deltas under one assistant turn.
		if !t.streaming {
			t.streaming = true
			t.emit(runtime.StreamStarted(t.sessionID(), t.agentName))
		}

	case "stream_text":
		t.ensureStreaming()
		if content != "" {
			t.emit(runtime.AgentChoice(t.agentName, t.sessionID(), content))
		}

	case "stream_thinking":
		t.ensureStreaming()
		if content != "" {
			t.emit(runtime.AgentChoiceReasoning(t.agentName, t.sessionID(), content))
		}

	case "assistant_text":
		// Durable final assistant text. The streamed deltas already painted the
		// transcript, but the assistant_text frame is the reliability floor (it
		// is replayed on reconnect) and is the only assistant signal when the
		// gateway dropped ephemeral stream frames under backpressure. Emitting an
		// AgentChoice here would double-paint after streaming, so we only surface
		// it when no stream produced text — approximated by "not currently
		// streaming", i.e. the turn already closed without deltas.
		if !t.streaming && content != "" {
			t.emit(runtime.AgentChoice(t.agentName, t.sessionID(), content))
		}

	case "tool_call":
		t.emit(t.toolCallEvent(meta))

	case "tool_result":
		t.emit(t.toolResultEvent(meta, content))

	case "usage":
		t.applyUsage(meta)

	case "approval_request":
		if id, _ := meta["approval_id"].(string); id != "" {
			t.pendingApprovalID = id
		}
		t.emit(t.confirmationEvent(meta, content, body))

	case "approval_resolved":
		// The dialog is closed client-side already; clear the pending slot.
		t.pendingApprovalID = ""

	case "agent_end":
		if t.streaming {
			t.streaming = false
			reason, _ := meta["cause"].(string)
			t.emit(runtime.StreamStopped(t.sessionID(), t.agentName, reason))
		}

	case "diagnostic_warning":
		t.emit(runtime.Warning(content, t.agentName))

	case "diagnostic_error":
		t.emit(runtime.Error(content))

	case "session_ready":
		t.emit(t.agentInfoEvent(meta))

	case "child_start":
		// Surface as a warning-styled note; the TUI has no nested sub-session
		// panel on the wire path.
		if purpose, _ := meta["purpose"].(string); purpose != "" {
			t.emit(runtime.Warning("sub-agent started: "+purpose, t.agentName))
		}

	case "child_end":
		if errStr, _ := meta["error"].(string); errStr != "" {
			t.emit(runtime.Warning("sub-agent failed: "+errStr, t.agentName))
		}

	default:
		// extension_install/reload/unload, api_register, resource_write,
		// plan_submitted, after_compact, cost_budget_exceeded,
		// command_dispatched — runtime-control/observability kinds with no
		// cagent render. Log and drop per the migration rule.
		if kind != "" {
			log.Printf("[adapter] no cagent mapping for outbound kind=%q; dropping", kind)
		}
	}
}

// ensureStreaming opens a stream if one is not already open, so a stray
// stream_text without a preceding turn_start still renders.
func (t *Translator) ensureStreaming() {
	if !t.streaming {
		t.streaming = true
		t.emit(runtime.StreamStarted(t.sessionID(), t.agentName))
	}
}

// toolCallEvent builds a ToolCallEvent from a tool_call body.
func (t *Translator) toolCallEvent(meta map[string]any) runtime.Event {
	name, _ := meta["name"].(string)
	id, _ := meta["tool_call_id"].(string)
	args := jsonString(meta["args"])
	call := tools.ToolCall{
		ID:   id,
		Type: tools.ToolType("function"),
		Function: tools.FunctionCall{
			Name:      name,
			Arguments: args,
		},
	}
	def := tools.Tool{Name: name}
	return runtime.ToolCall(call, def, t.agentName)
}

// toolResultEvent builds a ToolCallResponseEvent from a tool_result body.
func (t *Translator) toolResultEvent(meta map[string]any, content string) runtime.Event {
	name, _ := meta["name"].(string)
	id, _ := meta["tool_call_id"].(string)
	ok, hasOK := meta["ok"].(bool)
	result := &tools.ToolCallResult{
		Output:  content,
		IsError: hasOK && !ok,
	}
	def := tools.Tool{Name: name}
	return runtime.ToolCallResponse(id, def, result, content, t.agentName)
}

// confirmationEvent builds a ToolCallConfirmationEvent from an approval_request
// body. The gateway sends approval requests as buttons + content; cagent's
// confirmation dialog only needs the tool call shape, which the gateway carries
// in metadata (name / args / tool_call_id) on the same frame.
func (t *Translator) confirmationEvent(meta map[string]any, content string, body map[string]any) runtime.Event {
	name, _ := meta["name"].(string)
	if name == "" {
		name, _ = meta["tool_name"].(string)
	}
	id, _ := meta["tool_call_id"].(string)
	args := jsonString(meta["args"])
	if args == "" {
		if ta, ok := meta["tool_args"]; ok {
			args = jsonString(ta)
		}
	}
	_ = content
	_ = body
	call := tools.ToolCall{
		ID:   id,
		Type: tools.ToolType("function"),
		Function: tools.FunctionCall{
			Name:      name,
			Arguments: args,
		},
	}
	def := tools.Tool{Name: name}
	return runtime.ToolCallConfirmation(call, def, t.agentName)
}

// agentInfoEvent builds an AgentInfoEvent from a session_ready body.
func (t *Translator) agentInfoEvent(meta map[string]any) runtime.Event {
	model, _ := meta["model"].(string)
	return runtime.AgentInfo(t.agentName, model, "", "")
}

// applyUsage updates the session token counters and emits a TokenUsageEvent.
func (t *Translator) applyUsage(meta map[string]any) {
	in := intField(meta, "input_tokens")
	out := intField(meta, "output_tokens")
	if t.sess != nil {
		t.sess.InputTokens = in
		t.sess.OutputTokens = out
	}
	usage := &runtime.Usage{
		InputTokens:   in,
		OutputTokens:  out,
		ContextLength: in + out,
	}
	if t.sess != nil {
		usage.Cost = t.sess.OwnCost()
	}
	t.emit(runtime.NewTokenUsageEvent(t.sessionID(), t.agentName, usage))
}

// --- small JSON helpers ----------------------------------------------------

// jsonString renders an arbitrary JSON-decoded value back to a compact JSON
// string. Tool arguments arrive on the wire as a decoded map/slice; the cagent
// FunctionCall.Arguments field is a raw JSON string.
func jsonString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(b)
}

// intField reads a numeric metadata field, tolerating the float64 that
// encoding/json produces for all JSON numbers.
func intField(meta map[string]any, key string) int64 {
	switch n := meta[key].(type) {
	case float64:
		return int64(n)
	case int64:
		return n
	case int:
		return int64(n)
	case json.Number:
		i, _ := n.Int64()
		return i
	default:
		return 0
	}
}
