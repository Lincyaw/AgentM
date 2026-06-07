package blocks

import (
	"strings"

	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

// AssistantTurn is a composite block representing a full assistant response.
// Segments holds ThinkingBlock, TextBlock, and ToolBlock items in the
// chronological order they arrived during streaming. Children and Approvals
// are always rendered after all Segments.
type AssistantTurn struct {
	Segments  []Block // *ThinkingBlock, *TextBlock, *ToolBlock in arrival order
	Children  []*SubagentBlock
	Approvals []*ApprovalBlock
	complete  bool

	// GlamourStyle is forwarded to each new TextBlock created during streaming.
	GlamourStyle string

	// openText and openThinking track the segment currently being streamed.
	// They are set to nil when a new segment type starts.
	openText     *TextBlock
	openThinking *ThinkingBlock
}

func (b *AssistantTurn) Kind() string        { return "assistant" }
func (b *AssistantTurn) Collapsed() bool     { return false }
func (b *AssistantTurn) SetCollapsed(_ bool) {}

// SetComplete marks the turn as finished and flushes glamour on all text segments.
func (b *AssistantTurn) SetComplete() {
	b.complete = true
	for _, seg := range b.Segments {
		if tb, ok := seg.(*TextBlock); ok {
			tb.SetComplete()
		}
	}
}

// Complete reports whether the assistant turn has finished streaming.
func (b *AssistantTurn) Complete() bool { return b.complete }

// OpenText returns the currently open (streaming) TextBlock, or nil.
func (b *AssistantTurn) OpenText() *TextBlock { return b.openText }

// OpenThinking returns the currently open (streaming) ThinkingBlock, or nil.
func (b *AssistantTurn) OpenThinking() *ThinkingBlock { return b.openThinking }

// SetOpenText replaces the open text pointer.
func (b *AssistantTurn) SetOpenText(tb *TextBlock) { b.openText = tb }

// SetOpenThinking replaces the open thinking pointer.
func (b *AssistantTurn) SetOpenThinking(th *ThinkingBlock) { b.openThinking = th }

// AppendSegment appends a block to the segment list.
func (b *AssistantTurn) AppendSegment(seg Block) { b.Segments = append(b.Segments, seg) }

// LastTextBlock returns the last *TextBlock in Segments, or nil if none exists.
func (b *AssistantTurn) LastTextBlock() *TextBlock {
	for i := len(b.Segments) - 1; i >= 0; i-- {
		if tb, ok := b.Segments[i].(*TextBlock); ok {
			return tb
		}
	}
	return nil
}

func (b *AssistantTurn) Render(width int, th *theme.Theme) string {
	var sb strings.Builder

	// Chronological segments: ThinkingBlock, TextBlock, ToolBlock
	for _, seg := range b.Segments {
		rendered := seg.Render(width, th)
		if rendered != "" {
			sb.WriteString(rendered + "\n")
		}
	}

	// Sub-agent blocks after all segments
	for _, child := range b.Children {
		sb.WriteString(child.Render(width, th) + "\n")
	}

	// Approval blocks after sub-agents
	for _, appr := range b.Approvals {
		sb.WriteString(appr.Render(width, th) + "\n")
	}

	return strings.TrimRight(sb.String(), "\n")
}
