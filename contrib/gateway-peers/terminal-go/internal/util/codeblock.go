package util

import (
	"regexp"
	"strings"
)

// CodeBlock represents a fenced code block from markdown.
type CodeBlock struct {
	Lang    string // language identifier (e.g., "go", "python")
	Content string
	Start   int // byte offset in source
	End     int
}

var codeBlockPattern = regexp.MustCompile("(?s)```(\\w*)\\n(.*?)```")

// ExtractCodeBlocks finds all fenced code blocks in the given markdown text.
func ExtractCodeBlocks(text string) []CodeBlock {
	matches := codeBlockPattern.FindAllStringSubmatchIndex(text, -1)
	var result []CodeBlock
	for _, m := range matches {
		lang := text[m[2]:m[3]]
		content := text[m[4]:m[5]]
		result = append(result, CodeBlock{
			Lang:    lang,
			Content: content,
			Start:   m[0],
			End:     m[1],
		})
	}
	return result
}

// SuggestFilename guesses a filename from the language identifier.
func SuggestFilename(lang string) string {
	switch strings.ToLower(lang) {
	case "go", "golang":
		return "snippet.go"
	case "python", "py":
		return "snippet.py"
	case "javascript", "js":
		return "snippet.js"
	case "typescript", "ts":
		return "snippet.ts"
	case "rust", "rs":
		return "snippet.rs"
	case "bash", "sh", "shell", "zsh":
		return "snippet.sh"
	case "json":
		return "data.json"
	case "yaml", "yml":
		return "config.yaml"
	case "toml":
		return "config.toml"
	case "sql":
		return "query.sql"
	case "html":
		return "page.html"
	case "css":
		return "style.css"
	case "c":
		return "snippet.c"
	case "cpp", "c++", "cxx":
		return "snippet.cpp"
	case "java":
		return "Snippet.java"
	case "ruby", "rb":
		return "snippet.rb"
	case "swift":
		return "snippet.swift"
	case "kotlin", "kt":
		return "snippet.kt"
	case "dockerfile":
		return "Dockerfile"
	case "makefile", "make":
		return "Makefile"
	case "markdown", "md":
		return "doc.md"
	default:
		if lang != "" {
			return "snippet." + lang
		}
		return "snippet.txt"
	}
}
