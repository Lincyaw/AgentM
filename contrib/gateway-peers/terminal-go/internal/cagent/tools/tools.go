package tools

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/docker/aijson"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ToolSet defines the interface for a set of tools.
type ToolSet interface {
	Tools(ctx context.Context) ([]Tool, error)
}

// NewHandler creates a type-safe tool handler from a function that accepts
// typed parameters. It unmarshals the tool-call arguments via
// [aijson.Unmarshal], which runs strict [encoding/json.Unmarshal] first and
// only falls back to a narrow set of shape repairs (stringified array,
// bare scalar where an array is expected, single-object placeholder, null
// for primitive) when the strict parse fails. Repaired calls emit a
// tool_input_repaired log entry so per-(model, tool) repair rates can be
// tracked.
func NewHandler[T any](fn func(context.Context, T) (*ToolCallResult, error)) ToolHandler {
	return func(ctx context.Context, toolCall ToolCall) (*ToolCallResult, error) {
		var params T
		args := toolCall.Function.Arguments
		if args == "" {
			args = "{}"
		}

		err := aijson.Unmarshal([]byte(args), &params, aijson.OnRepair(func(kinds []aijson.Kind) {
			slog.InfoContext(ctx, "tool_input_repaired",
				"tool", toolCall.Function.Name,
				"repairs", kinds,
			)
		}))
		if err != nil {
			return nil, err
		}
		return fn(ctx, params)
	}
}

type ToolHandler func(ctx context.Context, toolCall ToolCall) (*ToolCallResult, error)

// ToolOutputEmitter receives incremental output from a running tool. Tool
// handlers can retrieve the emitter from their context and call it whenever
// command output or other long-running progress is available.
type ToolOutputEmitter func(output string)

type toolOutputEmitterKey struct{}

// WithToolOutputEmitter returns a context carrying emit as the callback used
// by tool handlers to stream incremental output. A nil emitter leaves ctx
// unchanged.
func WithToolOutputEmitter(ctx context.Context, emit ToolOutputEmitter) context.Context {
	if emit == nil {
		return ctx
	}
	return context.WithValue(ctx, toolOutputEmitterKey{}, emit)
}

// ToolOutputEmitterFromContext returns the incremental-output emitter attached
// to ctx, if any.
func ToolOutputEmitterFromContext(ctx context.Context) (ToolOutputEmitter, bool) {
	emit, ok := ctx.Value(toolOutputEmitterKey{}).(ToolOutputEmitter)
	return emit, ok
}

// EmitOutput streams output through the emitter in ctx. It returns false when
// no emitter is attached or output is empty.
func EmitOutput(ctx context.Context, output string) bool {
	if output == "" {
		return false
	}
	emit, ok := ToolOutputEmitterFromContext(ctx)
	if !ok {
		return false
	}
	emit(output)
	return true
}

type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     ToolType     `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// MediaContent represents base64-encoded binary data (image, audio, etc.)
// returned by a tool.
type MediaContent struct {
	// Data is the base64-encoded payload.
	Data string `json:"data"`
	// MimeType identifies the content type (e.g. "image/png", "audio/wav").
	MimeType string `json:"mimeType"`
}

// ImageContent is an alias kept for readability at call sites.
type ImageContent = MediaContent

// AudioContent is an alias kept for readability at call sites.
type AudioContent = MediaContent

// DocumentContent represents inline document-like content returned by a tool.
// Exactly one of Data or Text should be set. Data is base64-encoded.
type DocumentContent struct {
	Name     string `json:"name,omitempty"`
	URI      string `json:"uri,omitempty"`
	MimeType string `json:"mimeType"`
	Data     string `json:"data,omitempty"`
	Text     string `json:"text,omitempty"`
}

type ToolCallResult struct {
	Output  string `json:"output"`
	IsError bool   `json:"isError,omitempty"`
	Meta    any    `json:"meta,omitempty"`
	// Images contains optional image attachments returned by the tool.
	Images []MediaContent `json:"images,omitempty"`
	// Audios contains optional audio attachments returned by the tool.
	Audios []MediaContent `json:"audios,omitempty"`
	// Documents contains optional inline document attachments returned by the tool.
	Documents []DocumentContent `json:"documents,omitempty"`
	// StructuredContent holds optional structured output returned by an MCP
	// tool whose definition includes an OutputSchema. When non-nil it is the
	// JSON-decoded structured result from the server.
	StructuredContent any `json:"structuredContent,omitempty"`
}

func (r *ToolCallResult) WithoutPayload() *ToolCallResult {
	if r == nil {
		return nil
	}
	return &ToolCallResult{
		IsError: r.IsError,
		Meta:    r.Meta,
	}
}

func ResultError(output string) *ToolCallResult {
	return &ToolCallResult{
		Output:  output,
		IsError: true,
	}
}

func ResultSuccess(output string) *ToolCallResult {
	return &ToolCallResult{
		Output:  output,
		IsError: false,
	}
}

// ResultJSON marshals v as JSON and returns it as a successful tool result.
// If marshaling fails, it returns an error result.
func ResultJSON(v any) *ToolCallResult {
	data, err := json.Marshal(v)
	if err != nil {
		return ResultError(err.Error())
	}
	return &ToolCallResult{Output: string(data)}
}

type ToolType string

type Tool struct {
	Name                    string          `json:"name"`
	Category                string          `json:"category"`
	Description             string          `json:"description,omitempty"`
	Parameters              any             `json:"parameters"`
	Annotations             ToolAnnotations `json:"annotations"`
	OutputSchema            any             `json:"outputSchema"`
	Handler                 ToolHandler     `json:"-"`
	AddDescriptionParameter bool            `json:"-"`
	// ModelOverride is the per-toolset model for the LLM turn that processes
	// this tool's results. Set automatically from the toolset "model" field.
	ModelOverride string `json:"-"`
}

type ToolAnnotations mcp.ToolAnnotations
