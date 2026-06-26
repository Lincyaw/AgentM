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
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/tools"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
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

	// children routes child-session bodies (those carrying metadata.child_id)
	// into per-child sub-sessions/tabs. Set only on the ROOT translator; nil on
	// a child translator (a child does not spawn grandchild tabs of its own —
	// the gateway flattens nested children onto the same parent wire).
	children *ChildManager
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

	// Child-session routing (root translator only). Every body that originates
	// from a spawned sub-agent carries metadata.child_id; parent bodies carry
	// none. child_start/child_end are control frames handled here on the parent
	// so they open / finalize the child tab; all other child-stamped bodies are
	// the child's live trajectory and must paint the child tab, not the parent
	// transcript.
	if t.children != nil {
		childID, _ := meta["child_id"].(string)
		switch kind {
		case "child_start":
			purpose, _ := meta["purpose"].(string)
			t.children.Start(childID, purpose)
			// Parent-side breadcrumb: a concise note that delegation happened,
			// now that the real surface is a dedicated tab.
			if purpose != "" {
				t.emit(runtime.AgentChoice(t.agentName, t.sessionID(), "→ delegated to sub-agent: "+purpose))
			} else {
				t.emit(runtime.AgentChoice(t.agentName, t.sessionID(), "→ delegated to sub-agent"))
			}
			return
		case "child_end":
			errStr, _ := meta["error"].(string)
			final := intField(meta, "final_message_count")
			t.children.End(childID, errStr, final)
			return
		default:
			if childID != "" && t.children.Route(childID, body) {
				return
			}
		}
	}

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

	case "command_result":
		// Output of a gateway control command (/status, /help, /context, ...).
		// Not agent speech — render as a system notice with no author. The chat
		// page also settles the working spinner the slash command triggered,
		// since a control command runs no turn (no agent_end is coming).
		title, _ := meta["title"].(string)
		t.emit(runtime.SystemNote(content, t.sessionID(), title))

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
		t.settleIdle()

	case "diagnostic_error":
		t.emit(runtime.Error(content))
		t.settleIdle()

	case "session_ready":
		t.handleSessionReady(meta)

	case "child_start":
		// Fallback only: reached when this translator has no child manager
		// (i.e. it is itself a child translator). The root translator handles
		// child_start above by opening a dedicated sub-session tab, so it never
		// falls through to here. Defensive note in case the gateway ever forwards
		// a nested child frame onto a child wire.
		if purpose, _ := meta["purpose"].(string); purpose != "" {
			t.emit(runtime.Warning("sub-agent started: "+purpose, t.agentName))
		}

	case "child_end":
		// Fallback only (see child_start above).
		if errStr, _ := meta["error"].(string); errStr != "" {
			t.emit(runtime.Warning("sub-agent failed: "+errStr, t.agentName))
		}

	// --- runtime-control / observability signals -------------------------
	// These have no bespoke cagent surface, so they render onto the two
	// existing generic surfaces: ephemeral notices become Warning toasts;
	// content-bearing ones become a system-styled assistant note (AgentChoice
	// with a clear prefix). Nothing is silently dropped.
	case "extension_install":
		t.emit(t.extensionInstallEvent(meta))

	case "extension_reload":
		t.emit(t.extensionReloadEvent(meta))

	case "extension_unload":
		if name, _ := meta["name"].(string); name != "" {
			t.emit(runtime.Warning("extension unloaded: "+name, t.agentName))
		}

	case "resource_write":
		t.emit(t.resourceWriteNote(meta))

	case "plan_submitted":
		t.emit(t.planSubmittedNote(meta, content))

	case "after_compact":
		t.emit(t.afterCompactNote(meta, content))

	case "cost_budget_exceeded":
		t.emit(t.costBudgetEvent(meta))

	case "command_dispatched":
		t.emit(t.commandDispatchedEvent(meta))

	case "api_register":
		t.emit(t.apiRegisterEvent(meta))

	case "api_send_user_message":
		t.emit(t.apiUserMessageNote(meta, content))

	case "session_list":
		t.handleSessionList(meta, content)
		t.settleIdle()

	case "session_snapshot":
		// Acknowledged but no visual surface yet — the cagent session model
		// does not carry a Phase field. The snapshot is still useful for
		// debugging (logged at debug level by the default path otherwise).

	default:
		// Truly unmapped kinds (ping/ack-adjacent noise, future additions):
		// log and drop. Reaching here is now the exception, not the rule.
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

// settleIdle clears the TUI's optimistic working spinner for replies that
// arrive outside any turn. Gateway control commands (/status, /help, ...) reply
// with a single frame and run no agent turn, so they never emit the agent_end
// the chat page relies on to stop working. A StreamStopped at depth zero is the
// chat page's settle path (clears working + pending, drains the queue); the
// "command" reason keeps it out of the success-chime branch. No-op when a real
// stream is in flight — that turn's own agent_end will settle it.
func (t *Translator) settleIdle() {
	if t.streaming {
		return
	}
	t.emit(runtime.StreamStopped(t.sessionID(), t.agentName, "command"))
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

// handleSessionReady projects a session_ready frame onto the cagent sidebar and
// capability surfaces. The frame carries tool_names / command_names (so /tools
// and /skills populate), model (the active profile) and, per the shared wire
// contract, models (the selectable profile names for the model picker). The
// names are stashed on the App so CurrentAgentTools / CurrentAgentCommands /
// AvailableModels return them; the sidebar gets a ToolsetInfo (tool count) and a
// richer AgentInfo (active model).
func (t *Translator) handleSessionReady(meta map[string]any) {
	toolNames := stringSlice(meta["tool_names"])
	commandNames := stringSlice(meta["command_names"])
	modelNames := stringSlice(meta["models"])
	model, _ := meta["model"].(string)

	if t.app != nil {
		t.app.SetAgentInfo(toolNames, commandNames, modelNames, model)
	}

	desc := ""
	if len(modelNames) > 1 {
		desc = strconv.Itoa(len(modelNames)) + " models available"
	}
	t.emit(runtime.AgentInfo(t.agentName, model, desc, ""))
	t.emit(runtime.ToolsetInfo(len(toolNames), false, t.agentName))
}

// handleSessionList projects a session_list outbound (from bare /resume) onto the
// TUI's session browser. The metadata carries a "sessions" array of
// {session_id, title, created_at, scenario} maps.
func (t *Translator) handleSessionList(meta map[string]any, textFallback string) {
	entries, ok := meta["sessions"].([]any)
	if !ok || len(entries) == 0 {
		t.emit(runtime.SystemNote(textFallback, t.sessionID(), ""))
		return
	}
	summaries := make([]session.Summary, 0, len(entries))
	for _, e := range entries {
		m, ok := e.(map[string]any)
		if !ok {
			continue
		}
		sid, _ := m["session_id"].(string)
		if sid == "" {
			continue
		}
		title, _ := m["title"].(string)
		if title == "" {
			if len(sid) > 12 {
				title = sid[:12] + "…"
			} else {
				title = sid
			}
		}
		var createdAt time.Time
		switch v := m["created_at"].(type) {
		case float64:
			createdAt = time.Unix(int64(v), 0)
		case string:
			if parsed, err := time.Parse(time.RFC3339, v); err == nil {
				createdAt = parsed
			}
		}
		summaries = append(summaries, session.Summary{
			ID:        sid,
			Title:     title,
			CreatedAt: createdAt,
		})
	}
	if len(summaries) == 0 {
		t.emit(runtime.SystemNote(textFallback, t.sessionID(), ""))
		return
	}
	if t.app != nil {
		t.app.EmitEvent(messages.OpenSessionBrowserWithDataMsg{Sessions: summaries})
	}
}

// extensionInstallEvent renders an extension_install frame. A non-empty error is
// a warning; a successful install/registration is a quiet warning-styled notice.
func (t *Translator) extensionInstallEvent(meta map[string]any) runtime.Event {
	path, _ := meta["module_path"].(string)
	phase, _ := meta["phase"].(string)
	if errStr, _ := meta["error"].(string); errStr != "" {
		return runtime.Warning("extension install failed ("+path+"): "+errStr, t.agentName)
	}
	msg := "extension installed: " + path
	if phase != "" {
		msg += " [" + phase + "]"
	}
	return runtime.Warning(msg, t.agentName)
}

// extensionReloadEvent renders an extension_reload frame, making a self-modify
// reload prominent (the agent rewrote one of its own atoms).
func (t *Translator) extensionReloadEvent(meta map[string]any) runtime.Event {
	name, _ := meta["name"].(string)
	selfMod, _ := meta["is_self_modify"].(bool)
	if errStr, _ := meta["error"].(string); errStr != "" {
		return runtime.Warning("extension reload failed ("+name+"): "+errStr, t.agentName)
	}
	if selfMod {
		// Surface self-modification as a durable system note, not an ephemeral
		// toast: it is a load-bearing identity event worth keeping in transcript.
		return runtime.AgentChoice(t.agentName, t.sessionID(), "⟳ self-modified: "+name)
	}
	return runtime.Warning("extension reloaded: "+name, t.agentName)
}

// resourceWriteNote renders a resource_write frame as a system-styled note: the
// path, the author and (when present) the rationale and post-write hash.
func (t *Translator) resourceWriteNote(meta map[string]any) runtime.Event {
	path, _ := meta["path"].(string)
	author, _ := meta["author"].(string)
	rationale, _ := meta["rationale"].(string)
	post, _ := meta["post_sha"].(string)
	msg := "📝 resource write: " + path
	if author != "" {
		msg += " (by " + author + ")"
	}
	if rationale != "" {
		msg += " — " + rationale
	}
	if post != "" {
		msg += " [" + shortSHA(post) + "]"
	}
	return runtime.AgentChoice(t.agentName, t.sessionID(), msg)
}

// planSubmittedNote renders a plan_submitted frame as a system-styled note
// carrying the plan id and its content.
func (t *Translator) planSubmittedNote(meta map[string]any, content string) runtime.Event {
	planID, _ := meta["plan_id"].(string)
	header := "🧭 plan submitted"
	if planID != "" {
		header += " (" + planID + ")"
	}
	if content != "" {
		header += ":\n" + content
	}
	return runtime.AgentChoice(t.agentName, t.sessionID(), header)
}

// afterCompactNote renders an after_compact frame: how many messages were kept
// vs discarded, plus the summary content.
func (t *Translator) afterCompactNote(meta map[string]any, content string) runtime.Event {
	kept := intField(meta, "kept")
	discarded := intField(meta, "discarded")
	msg := "🗜 history compacted: kept " + strconv.FormatInt(kept, 10) +
		", discarded " + strconv.FormatInt(discarded, 10)
	if content != "" {
		msg += "\n" + content
	}
	return runtime.AgentChoice(t.agentName, t.sessionID(), msg)
}

// costBudgetEvent renders a cost_budget_exceeded frame as a warning toast.
func (t *Translator) costBudgetEvent(meta map[string]any) runtime.Event {
	currency, _ := meta["currency"].(string)
	used := floatField(meta, "used")
	limit := floatField(meta, "limit")
	return runtime.Warning(
		"cost budget exceeded: "+formatMoney(used, currency)+" / "+formatMoney(limit, currency),
		t.agentName,
	)
}

// commandDispatchedEvent renders a command_dispatched frame as a warning-styled
// notice naming the command and its owner.
func (t *Translator) commandDispatchedEvent(meta map[string]any) runtime.Event {
	name, _ := meta["name"].(string)
	owner, _ := meta["owner"].(string)
	msg := "command dispatched: /" + name
	if owner != "" {
		msg += " (owner " + owner + ")"
	}
	return runtime.Warning(msg, t.agentName)
}

// apiRegisterEvent renders an api_register frame as a warning-styled notice: an
// atom registered a capability (tool/command/event) at runtime.
func (t *Translator) apiRegisterEvent(meta map[string]any) runtime.Event {
	regKind, _ := meta["reg_kind"].(string)
	name, _ := meta["name"].(string)
	ext, _ := meta["extension"].(string)
	msg := "registered " + regKind + " " + name
	if ext != "" {
		msg += " (by " + ext + ")"
	}
	return runtime.Warning(msg, t.agentName)
}

// apiUserMessageNote renders an api_send_user_message frame (an atom injected a
// user-visible message) as a system-styled note attributing the source.
func (t *Translator) apiUserMessageNote(meta map[string]any, content string) runtime.Event {
	ext, _ := meta["extension"].(string)
	prefix := "✉ message"
	if ext != "" {
		prefix += " from " + ext
	}
	if content != "" {
		prefix += ": " + content
	}
	return runtime.AgentChoice(t.agentName, t.sessionID(), prefix)
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

// stringSlice coerces a JSON-decoded value into a []string, tolerating the
// []any that encoding/json produces for JSON arrays. Non-string elements and
// non-array values yield an empty slice.
func stringSlice(v any) []string {
	switch s := v.(type) {
	case []string:
		return s
	case []any:
		out := make([]string, 0, len(s))
		for _, e := range s {
			if str, ok := e.(string); ok {
				out = append(out, str)
			}
		}
		return out
	default:
		return nil
	}
}

// floatField reads a numeric metadata field as a float64, tolerating the
// float64 / json.Number forms encoding/json produces.
func floatField(meta map[string]any, key string) float64 {
	switch n := meta[key].(type) {
	case float64:
		return n
	case int64:
		return float64(n)
	case int:
		return float64(n)
	case json.Number:
		f, _ := n.Float64()
		return f
	default:
		return 0
	}
}

// shortSHA truncates a hash to a readable prefix for one-line notes.
func shortSHA(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

// formatMoney renders a money amount with its currency for budget notices.
func formatMoney(amount float64, currency string) string {
	s := strconv.FormatFloat(amount, 'f', -1, 64)
	if currency == "" {
		return s
	}
	return fmt.Sprintf("%s %s", s, currency)
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
