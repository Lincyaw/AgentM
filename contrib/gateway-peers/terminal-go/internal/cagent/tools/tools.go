package tools

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

type ToolType string

type Tool struct {
	Name         string          `json:"name"`
	Category     string          `json:"category"`
	Description  string          `json:"description,omitempty"`
	Parameters   any             `json:"parameters,omitempty"`
	Annotations  ToolAnnotations `json:"annotations,omitempty"`
	OutputSchema any             `json:"outputSchema,omitempty"`
}

type ToolAnnotations struct {
	Title           string `json:"title,omitempty"`
	ReadOnlyHint    bool   `json:"readOnlyHint,omitempty"`
	DestructiveHint *bool  `json:"destructiveHint,omitempty"`
	IdempotentHint  bool   `json:"idempotentHint,omitempty"`
	OpenWorldHint   *bool  `json:"openWorldHint,omitempty"`
}
