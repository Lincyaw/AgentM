package chat

import (
	"cmp"
	"encoding/base64"
	"log/slog"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/tools"
)

// BuildToolResultMultiContent attaches inline media and documents to a tool
// response as a MultiContent payload that providers can convert to native
// multimodal tool-result content where supported.
func BuildToolResultMultiContent(text string, images []tools.MediaContent, documents []tools.DocumentContent) []MessagePart {
	parts := make([]MessagePart, 0, 1+len(documents)+len(images))
	parts = append(parts, MessagePart{Type: MessagePartTypeText, Text: text})
	for _, doc := range documents {
		part, ok := ToolDocumentPart(doc)
		if ok {
			parts = append(parts, part)
		}
	}
	for _, img := range images {
		parts = append(parts, MessagePart{
			Type: MessagePartTypeImageURL,
			ImageURL: &MessageImageURL{
				URL:    "data:" + img.MimeType + ";base64," + img.Data,
				Detail: ImageURLDetailAuto,
			},
		})
	}
	return parts
}

// ToolDocumentPart converts a tool-returned document payload into a chat
// document part. It returns false when the payload is empty or malformed.
func ToolDocumentPart(content tools.DocumentContent) (MessagePart, bool) {
	doc := Document{
		Name:     cmp.Or(content.Name, "document"),
		MimeType: content.MimeType,
	}

	switch {
	case content.Text != "":
		doc.Size = int64(len(content.Text))
		doc.Source = DocumentSource{InlineText: content.Text}
		if doc.MimeType == "" {
			doc.MimeType = "text/plain"
		}
	case content.Data != "":
		data, err := base64.StdEncoding.DecodeString(content.Data)
		if err != nil {
			slog.Warn("Dropping tool document with invalid base64 payload", "name", content.Name, "mime", content.MimeType, "error", err)
			return MessagePart{}, false
		}
		doc.Size = int64(len(data))
		doc.Source = DocumentSource{InlineData: data}
		if doc.MimeType == "" {
			doc.MimeType = DetectMimeTypeByContent(data)
		}
	default:
		return MessagePart{}, false
	}

	return MessagePart{Type: MessagePartTypeDocument, Document: &doc}, true
}
