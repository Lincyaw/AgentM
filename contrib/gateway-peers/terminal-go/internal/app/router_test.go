package app

import (
	"testing"

	"github.com/AoyangSpace/agentm-terminal/internal/blocks"
	"github.com/AoyangSpace/agentm-terminal/internal/components"
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

func newTestModel() *Model {
	m := NewModel(Config{Theme: "dark"})
	return &m
}

func TestRouterTurnStartCreatesAssistantTurn(t *testing.T) {
	m := newTestModel()
	m.transcript = nil

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "turn_start"},
	})

	if len(m.transcript) == 0 {
		t.Fatal("expected transcript to have an entry after turn_start")
	}
	if m.activeTurn == nil {
		t.Fatal("expected activeTurn to be set")
	}
	if m.status.GetModel().Phase != theme.PhaseThinking {
		t.Errorf("phase = %q, want %q", m.status.GetModel().Phase, theme.PhaseThinking)
	}
}

func TestRouterStreamTextAppendsContent(t *testing.T) {
	m := newTestModel()
	m.transcript = nil

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "turn_start"},
	})
	r.Dispatch(m, map[string]any{
		"content":  "hello ",
		"metadata": map[string]any{"kind": "stream_text"},
	})
	r.Dispatch(m, map[string]any{
		"content":  "world",
		"metadata": map[string]any{"kind": "stream_text"},
	})

	if m.activeTurn == nil {
		t.Fatal("expected activeTurn")
	}
	tb := m.activeTurn.OpenText()
	if tb == nil {
		t.Fatal("expected open TextBlock")
	}
	if tb.Text != "hello world" {
		t.Errorf("text = %q, want %q", tb.Text, "hello world")
	}
	// During streaming the block is not yet complete.
	if tb.Complete() {
		t.Error("expected TextBlock not complete during streaming")
	}
}

func TestRouterStreamThinkingCreatesBlock(t *testing.T) {
	m := newTestModel()
	m.transcript = nil

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "turn_start"},
	})
	r.Dispatch(m, map[string]any{
		"content":  "reasoning...",
		"metadata": map[string]any{"kind": "stream_thinking"},
	})

	tb := m.activeTurn.OpenThinking()
	if tb == nil {
		t.Fatal("expected thinking block to be created")
	}
	if tb.Text != "reasoning..." {
		t.Errorf("thinking text = %q, want %q", tb.Text, "reasoning...")
	}
	// Should have one segment: the ThinkingBlock.
	if len(m.activeTurn.Segments) != 1 {
		t.Errorf("expected 1 segment, got %d", len(m.activeTurn.Segments))
	}
}

func TestRouterToolCallAndResult(t *testing.T) {
	m := newTestModel()
	m.transcript = nil

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "turn_start"},
	})
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{
			"kind":         "tool_call",
			"name":         "Read",
			"tool_call_id": "tc_123",
			"args":         map[string]any{"file_path": "/tmp/foo.go"},
		},
	})

	// ToolBlock should be in Segments.
	if len(m.activeTurn.Segments) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(m.activeTurn.Segments))
	}
	tb, ok := m.activeTurn.Segments[0].(*blocks.ToolBlock)
	if !ok {
		t.Fatal("expected segment to be *ToolBlock")
	}
	if tb.Name != "Read" {
		t.Errorf("tool name = %q, want Read", tb.Name)
	}
	if tb.Done {
		t.Error("tool should not be done yet")
	}

	r.Dispatch(m, map[string]any{
		"content": "file contents here",
		"metadata": map[string]any{
			"kind":         "tool_result",
			"tool_call_id": "tc_123",
			"ok":           true,
		},
	})

	if !tb.Done {
		t.Error("tool should be done after result")
	}
	if !tb.OK {
		t.Error("tool should be OK")
	}
	if tb.Result != "file contents here" {
		t.Errorf("result = %q, want %q", tb.Result, "file contents here")
	}
}

func TestRouterAssistantTextCompletesTurn(t *testing.T) {
	m := newTestModel()
	m.transcript = nil

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "turn_start"},
	})
	r.Dispatch(m, map[string]any{
		"content":  "final answer",
		"metadata": map[string]any{"kind": "assistant_text"},
	})

	if m.activeTurn != nil {
		t.Error("activeTurn should be nil after assistant_text")
	}
	at, ok := m.transcript[len(m.transcript)-1].(*blocks.AssistantTurn)
	if !ok {
		t.Fatal("last transcript entry should be AssistantTurn")
	}
	// The final text should be in the last TextBlock segment.
	tb := at.LastTextBlock()
	if tb == nil {
		t.Fatal("expected a TextBlock segment")
	}
	if tb.Text != "final answer" {
		t.Errorf("text = %q, want %q", tb.Text, "final answer")
	}
	if !at.Complete() {
		t.Error("turn should be marked complete")
	}
}

func TestRouterAgentEndClearsInFlight(t *testing.T) {
	m := newTestModel()
	m.inFlight = true

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "agent_end"},
	})

	if m.inFlight {
		t.Error("inFlight should be false after agent_end")
	}
}

func TestRouterUsageAccumulates(t *testing.T) {
	m := newTestModel()
	m.status.Update(components.StatusModel{Phase: theme.PhaseIdle})

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"input_tokens":  float64(100),
		"output_tokens": float64(50),
		"cost":          0.05,
		"metadata":      map[string]any{"kind": "usage"},
	})

	sm := m.status.GetModel()
	if sm.TokensIn != 100 {
		t.Errorf("TokensIn = %d, want 100", sm.TokensIn)
	}
	if sm.TokensOut != 50 {
		t.Errorf("TokensOut = %d, want 50", sm.TokensOut)
	}
}

func TestRouterSessionReady(t *testing.T) {
	m := newTestModel()

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"tools":    []any{"Read", "Edit"},
		"commands": []any{"help", "clear"},
		"metadata": map[string]any{
			"kind":           "session_ready",
			"model":          "glm-5.1",
			"context_window": float64(65536),
		},
	})

	sm := m.status.GetModel()
	if sm.Model != "glm-5.1" {
		t.Errorf("Model = %q, want glm-5.1", sm.Model)
	}
	if sm.CtxTotal != 65536 {
		t.Errorf("CtxTotal = %d, want 65536", sm.CtxTotal)
	}
	if len(m.tools) < 2 {
		t.Errorf("tools = %v, expected at least Read and Edit", m.tools)
	}
}

func TestRouterChildStartEnd(t *testing.T) {
	m := newTestModel()
	m.transcript = nil

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "turn_start"},
	})
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{
			"kind":     "child_start",
			"purpose":  "analyze coverage",
			"child_id": "ch_1",
		},
	})

	if len(m.activeTurn.Children) != 1 {
		t.Fatalf("expected 1 child, got %d", len(m.activeTurn.Children))
	}
	child := m.activeTurn.Children[0]
	if child.Done {
		t.Error("child should not be done yet")
	}

	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{
			"kind":     "child_end",
			"child_id": "ch_1",
		},
	})

	if !child.Done {
		t.Error("child should be done after child_end")
	}
}

func TestRouterApprovalRequest(t *testing.T) {
	m := newTestModel()
	m.transcript = nil

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "turn_start"},
	})
	r.Dispatch(m, map[string]any{
		"content": "Run rm -rf?",
		"metadata": map[string]any{
			"kind": "approval_request",
			"buttons": []any{
				map[string]any{"label": "Allow", "value": "allow", "style": "primary"},
				map[string]any{"label": "Deny", "value": "deny", "style": "danger"},
			},
		},
	})

	if m.pendingApproval == nil {
		t.Fatal("expected pendingApproval to be set")
	}
	if len(m.pendingApproval.Buttons) != 2 {
		t.Errorf("buttons = %d, want 2", len(m.pendingApproval.Buttons))
	}
}

func TestRouterApprovalDefaultButtons(t *testing.T) {
	m := newTestModel()
	m.transcript = nil

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "turn_start"},
	})
	r.Dispatch(m, map[string]any{
		"content":  "proceed?",
		"metadata": map[string]any{"kind": "approval_request"},
	})

	if m.pendingApproval == nil {
		t.Fatal("expected pendingApproval")
	}
	if len(m.pendingApproval.Buttons) != 2 {
		t.Errorf("default buttons = %d, want 2", len(m.pendingApproval.Buttons))
	}
}

func TestRouterUnknownKindIgnored(t *testing.T) {
	m := newTestModel()
	before := len(m.transcript)

	r := &Router{}
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "future_unknown_event"},
	})

	if len(m.transcript) != before {
		t.Error("unknown kind should not modify transcript")
	}
}

func TestRouterNilBody(t *testing.T) {
	m := newTestModel()
	r := &Router{}
	r.Dispatch(m, nil)
}

func TestRouterAfterCompactUpdatesCtx(t *testing.T) {
	m := newTestModel()
	r := &Router{}
	r.Dispatch(m, map[string]any{
		"ctx_used": float64(8192),
		"metadata": map[string]any{"kind": "after_compact"},
	})

	sm := m.status.GetModel()
	if sm.CtxUsed != 8192 {
		t.Errorf("CtxUsed = %d, want 8192", sm.CtxUsed)
	}
}

func TestRouterBudgetExceeded(t *testing.T) {
	m := newTestModel()
	r := &Router{}
	r.Dispatch(m, map[string]any{
		"cost":     5.0,
		"metadata": map[string]any{"kind": "cost_budget_exceeded"},
	})

	sm := m.status.GetModel()
	if sm.CostSession != 5.0 {
		t.Errorf("CostSession = %f, want 5.0", sm.CostSession)
	}
	if !sm.BudgetWarn {
		t.Error("BudgetWarn should be true")
	}
}

func TestRouterFullTurnLifecycle(t *testing.T) {
	m := newTestModel()
	m.transcript = nil

	r := &Router{}

	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "turn_start"},
	})
	r.Dispatch(m, map[string]any{
		"content":  "let me think...",
		"metadata": map[string]any{"kind": "stream_thinking"},
	})
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{
			"kind":         "tool_call",
			"name":         "Bash",
			"tool_call_id": "tc_1",
			"args":         map[string]any{"command": "ls"},
		},
	})
	r.Dispatch(m, map[string]any{
		"content": "file1\nfile2",
		"metadata": map[string]any{
			"kind":         "tool_result",
			"tool_call_id": "tc_1",
			"ok":           true,
		},
	})
	r.Dispatch(m, map[string]any{
		"content":  "Here are the files.",
		"metadata": map[string]any{"kind": "stream_text"},
	})
	r.Dispatch(m, map[string]any{
		"content":  "Here are the files.",
		"metadata": map[string]any{"kind": "assistant_text"},
	})
	r.Dispatch(m, map[string]any{
		"metadata": map[string]any{"kind": "agent_end"},
	})

	if len(m.transcript) != 1 {
		t.Fatalf("expected 1 transcript entry, got %d", len(m.transcript))
	}
	at, ok := m.transcript[0].(*blocks.AssistantTurn)
	if !ok {
		t.Fatal("entry should be AssistantTurn")
	}
	if !at.Complete() {
		t.Error("turn should be complete")
	}

	// Verify the segments contain a ThinkingBlock, ToolBlock, and TextBlock.
	var hasThinking, hasTool, hasText bool
	for _, seg := range at.Segments {
		switch seg.(type) {
		case *blocks.ThinkingBlock:
			hasThinking = true
		case *blocks.ToolBlock:
			hasTool = true
		case *blocks.TextBlock:
			hasText = true
		}
	}
	if !hasThinking {
		t.Error("should have thinking segment")
	}
	if !hasTool {
		t.Error("should have tool segment")
	}
	if !hasText {
		t.Error("should have text segment")
	}

	tb := at.LastTextBlock()
	if tb == nil || tb.Text != "Here are the files." {
		t.Errorf("text = %q", func() string {
			if tb == nil {
				return "<nil>"
			}
			return tb.Text
		}())
	}
	if m.inFlight {
		t.Error("inFlight should be false")
	}
}
