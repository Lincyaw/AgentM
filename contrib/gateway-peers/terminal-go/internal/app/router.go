package app

import (
	"fmt"
	"time"

	"github.com/AoyangSpace/agentm-terminal/internal/blocks"
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// Router dispatches wire outbound events to Model state mutations.
type Router struct{}

// Dispatch routes a wire event body to the appropriate handler based on
// metadata.kind. Unknown kinds are silently ignored (forward-compatible).
func (r *Router) Dispatch(m *Model, body map[string]any) {
	if body == nil {
		return
	}
	meta, _ := body["metadata"].(map[string]any)
	kind := "assistant_text"
	if meta != nil {
		if k, ok := meta["kind"].(string); ok && k != "" {
			kind = k
		}
	}

	switch kind {
	case "turn_start":
		r.turnStart(m, body, meta)
	case "stream_text":
		r.streamText(m, body, meta)
	case "stream_thinking":
		r.streamThinking(m, body, meta)
	case "tool_call":
		r.toolCall(m, body, meta)
	case "tool_result":
		r.toolResult(m, body, meta)
	case "assistant_text":
		r.assistantText(m, body, meta)
	case "agent_end":
		r.agentEnd(m)
	case "usage":
		r.usage(m, body, meta)
	case "session_ready":
		r.sessionReady(m, body, meta)
	case "approval_request":
		r.approvalRequest(m, body, meta)
	case "approval_resolved":
		r.approvalResolved(m, body, meta)
	case "diagnostic_warning":
		r.diagnostic(m, body, "warn")
	case "diagnostic_error":
		r.diagnostic(m, body, "warn")
	case "child_start":
		r.childStart(m, body, meta)
	case "child_end":
		r.childEnd(m, body, meta)
	case "cost_budget_exceeded":
		r.budgetExceeded(m, body, meta)
	case "extension_install":
		if phase, _ := meta["phase"].(string); phase == "error" {
			r.diagnostic(m, body, "warn")
		}
	case "extension_reload":
		r.extensionReload(m, body, meta)
	case "api_register":
		r.apiRegister(m, body, meta)
	case "after_compact":
		r.afterCompact(m, body, meta)
	}
}

// ensureActiveTurn guarantees an active assistant turn exists.
func (r *Router) ensureActiveTurn(m *Model) *blocks.AssistantTurn {
	if m.activeTurn == nil {
		m.activeTurn = &blocks.AssistantTurn{GlamourStyle: m.glamourStyle}
		m.transcript = append(m.transcript, m.activeTurn)
	}
	return m.activeTurn
}

func (r *Router) turnStart(m *Model, _ map[string]any, _ map[string]any) {
	turn := &blocks.AssistantTurn{GlamourStyle: m.glamourStyle}
	m.transcript = append(m.transcript, turn)
	m.activeTurn = turn
	m.turnStartTime = time.Now()
	sm := m.status.GetModel()
	sm.Phase = theme.PhaseThinking
	m.status.Update(sm)
}

func (r *Router) streamText(m *Model, body map[string]any, _ map[string]any) {
	turn := r.ensureActiveTurn(m)
	if content, ok := body["content"].(string); ok {
		if turn.OpenText() == nil {
			tb := &blocks.TextBlock{GlamourStyle: m.glamourStyle}
			turn.AppendSegment(tb)
			turn.SetOpenText(tb)
			turn.SetOpenThinking(nil)
		}
		turn.OpenText().Text += content
	}
	sm := m.status.GetModel()
	sm.Phase = theme.PhaseStreaming
	m.status.Update(sm)
}

func (r *Router) streamThinking(m *Model, body map[string]any, _ map[string]any) {
	turn := r.ensureActiveTurn(m)
	if content, ok := body["content"].(string); ok {
		if turn.OpenThinking() == nil {
			tb := blocks.NewThinkingBlock()
			turn.AppendSegment(tb)
			turn.SetOpenThinking(tb)
			turn.SetOpenText(nil)
		}
		turn.OpenThinking().Text += content
	}
	sm := m.status.GetModel()
	sm.Phase = theme.PhaseThinking
	m.status.Update(sm)
}

func (r *Router) toolCall(m *Model, _ map[string]any, meta map[string]any) {
	turn := r.ensureActiveTurn(m)
	name, _ := meta["name"].(string)
	if name == "" {
		name = "unknown"
	}

	var args map[string]any
	if a, ok := meta["args"].(map[string]any); ok {
		args = a
	}

	tb := blocks.NewToolBlock(name, args)
	turn.AppendSegment(tb)
	// Close any open streaming segments so the next think/text starts fresh.
	turn.SetOpenText(nil)
	turn.SetOpenThinking(nil)

	if callID, ok := meta["tool_call_id"].(string); ok && callID != "" {
		m.toolRegistry[callID] = tb
	}

	sm := m.status.GetModel()
	sm.Phase = theme.PhaseTool
	sm.ToolCount++
	m.status.Update(sm)
}

func (r *Router) toolResult(m *Model, body map[string]any, meta map[string]any) {
	callID, _ := meta["tool_call_id"].(string)
	tb := m.toolRegistry[callID]
	if tb == nil {
		return
	}
	if result, ok := body["content"].(string); ok {
		tb.Result = result
	}
	if ok, exists := meta["ok"].(bool); exists {
		tb.OK = ok
	} else {
		tb.OK = true
	}
	tb.Done = true
}

func (r *Router) assistantText(m *Model, body map[string]any, _ map[string]any) {
	turn := r.ensureActiveTurn(m)
	if content, ok := body["content"].(string); ok {
		// Set the content on the last TextBlock, or create one if none exists.
		tb := turn.LastTextBlock()
		if tb == nil {
			tb = &blocks.TextBlock{GlamourStyle: m.glamourStyle}
			turn.AppendSegment(tb)
		}
		tb.Text = content
	}
	turn.SetComplete()
	m.activeTurn = nil
	sm := m.status.GetModel()
	sm.Phase = theme.PhaseIdle
	m.status.Update(sm)
}

func (r *Router) agentEnd(m *Model) {
	m.inFlight = false
	m.activeTurn = nil
	sm := m.status.GetModel()
	sm.Phase = theme.PhaseIdle
	m.status.Update(sm)
}

func (r *Router) usage(m *Model, body map[string]any, _ map[string]any) {
	sm := m.status.GetModel()
	if tokIn, ok := toInt(body["input_tokens"]); ok {
		sm.TokensIn += tokIn
	}
	if tokOut, ok := toInt(body["output_tokens"]); ok {
		sm.TokensOut += tokOut
	}
	if cost, ok := toFloat(body["cost"]); ok {
		sm.CostTurn = cost
	}
	if sessionCost, ok := toFloat(body["session_cost"]); ok {
		sm.CostSession = sessionCost
	}
	elapsed := time.Since(m.turnStartTime)
	if elapsed.Seconds() > 0 && sm.TokensOut > 0 {
		sm.TokPerSec = float64(sm.TokensOut) / elapsed.Seconds()
	}
	m.status.Update(sm)
}

func (r *Router) sessionReady(m *Model, body map[string]any, meta map[string]any) {
	if model, ok := meta["model"].(string); ok {
		sm := m.status.GetModel()
		sm.Model = model
		m.status.Update(sm)
	}
	if ctxWin, ok := toInt(meta["context_window"]); ok {
		sm := m.status.GetModel()
		sm.CtxTotal = ctxWin
		m.status.Update(sm)
	}
	if tools, ok := body["tools"].([]any); ok {
		for _, t := range tools {
			if name, ok := t.(string); ok {
				m.addToolToCatalog(name)
			}
		}
	}
	if cmds, ok := body["commands"].([]any); ok {
		for _, c := range cmds {
			if name, ok := c.(string); ok {
				m.addCommand(name)
			}
		}
	}
}

func (r *Router) approvalRequest(m *Model, body map[string]any, meta map[string]any) {
	content, _ := body["content"].(string)
	if content == "" {
		content = "Approval required"
	}

	var buttons []blocks.Button
	if rawBtns, ok := meta["buttons"].([]any); ok {
		for _, rb := range rawBtns {
			if bm, ok := rb.(map[string]any); ok {
				label, _ := bm["label"].(string)
				value, _ := bm["value"].(string)
				style, _ := bm["style"].(string)
				buttons = append(buttons, blocks.Button{Label: label, Value: value, Style: style})
			}
		}
	}
	if len(buttons) == 0 {
		buttons = []blocks.Button{
			{Label: "Allow", Value: "allow", Style: "primary"},
			{Label: "Deny", Value: "deny", Style: "danger"},
		}
	}

	ab := blocks.NewApprovalBlock(content, buttons)
	ab.SetCollapsed(false)

	if turn := m.activeTurn; turn != nil {
		turn.Approvals = append(turn.Approvals, ab)
	} else {
		// Standalone approval as a system block
		m.transcript = append(m.transcript, &blocks.SystemTurn{
			Content: content,
			Source:  "approval",
		})
	}
	m.pendingApproval = ab
}

func (r *Router) approvalResolved(m *Model, body map[string]any, _ map[string]any) {
	decision, _ := body["decision"].(string)
	if decision == "" {
		decision = "resolved"
	}
	m.toasts.Push("approval: "+decision, "info", 3*time.Second)
	m.pendingApproval = nil
}

func (r *Router) diagnostic(m *Model, body map[string]any, variant string) {
	content, _ := body["content"].(string)
	if content == "" {
		content = "diagnostic event"
	}
	m.toasts.Push(content, variant, 5*time.Second)
}

func (r *Router) childStart(m *Model, _ map[string]any, meta map[string]any) {
	turn := r.ensureActiveTurn(m)
	purpose, _ := meta["purpose"].(string)
	childID, _ := meta["child_id"].(string)

	sb := &blocks.SubagentBlock{Purpose: purpose}
	turn.Children = append(turn.Children, sb)
	if childID != "" {
		m.childRegistry[childID] = sb
	}

	sm := m.status.GetModel()
	sm.Phase = theme.PhaseSubagent
	m.status.Update(sm)
}

func (r *Router) childEnd(m *Model, _ map[string]any, meta map[string]any) {
	childID, _ := meta["child_id"].(string)
	if sb, ok := m.childRegistry[childID]; ok {
		sb.Done = true
		if errMsg, ok := meta["error"].(string); ok {
			sb.Error = errMsg
		}
	}
}

func (r *Router) budgetExceeded(m *Model, body map[string]any, _ map[string]any) {
	cost, _ := toFloat(body["cost"])
	sm := m.status.GetModel()
	sm.CostSession = cost
	sm.BudgetWarn = true
	m.status.Update(sm)
	m.toasts.Push("cost budget exceeded", "warn", 10*time.Second)
}

func (r *Router) extensionReload(m *Model, body map[string]any, meta map[string]any) {
	isSelfMod, _ := meta["is_self_modify"].(bool)
	name, _ := body["extension"].(string)
	variant := "info"
	msg := fmt.Sprintf("extension reloaded: %s", name)
	if isSelfMod {
		variant = "selfmod"
		msg = fmt.Sprintf("self-modification: %s reloaded", name)
	}
	m.toasts.Push(msg, variant, 5*time.Second)
}

func (r *Router) apiRegister(m *Model, body map[string]any, meta map[string]any) {
	regType, _ := meta["type"].(string)
	name, _ := body["name"].(string)
	switch regType {
	case "tool":
		m.addToolToCatalog(name)
	case "command":
		m.addCommand(name)
	}
}

func (r *Router) afterCompact(m *Model, body map[string]any, _ map[string]any) {
	m.toasts.Push("context compacted", "info", 3*time.Second)
	if ctxUsed, ok := toInt(body["ctx_used"]); ok {
		sm := m.status.GetModel()
		sm.CtxUsed = ctxUsed
		m.status.Update(sm)
	}
}

// toInt coerces a JSON number (float64) or int to int.
func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	default:
		return 0, false
	}
}

// toFloat coerces a JSON number to float64.
func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}
