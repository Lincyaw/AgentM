package app

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/blocks"
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

var fencedCodeRe = regexp.MustCompile("(?s)```(\\w*)\\n(.*?)```")

// codeBlock is a fenced code block extracted from an assistant turn.
type codeBlock struct {
	Lang          string
	Content       string
	SuggestedPath string
}

// CodeSaveOverlay lets the user pick a code block from the last assistant
// turn and save it to a file.
type CodeSaveOverlay struct {
	codeBlocks []codeBlock
	cursor     int
	phase      int // 0 = selecting block, 1 = entering path
	pathBuf    string
	saved      bool
	savedPath  string
	err        error
}

// NewCodeSaveOverlay scans the last assistant turn for fenced code blocks.
// Returns nil if no code blocks are found.
func NewCodeSaveOverlay(transcript []blocks.Block) *CodeSaveOverlay {
	// Find the last assistant turn
	var lastAssistant *blocks.AssistantTurn
	for i := len(transcript) - 1; i >= 0; i-- {
		if at, ok := transcript[i].(*blocks.AssistantTurn); ok {
			lastAssistant = at
			break
		}
	}
	if lastAssistant == nil || lastAssistant.Text == "" {
		return nil
	}

	matches := fencedCodeRe.FindAllStringSubmatch(lastAssistant.Text, -1)
	if len(matches) == 0 {
		return nil
	}

	var cbs []codeBlock
	for _, m := range matches {
		lang := m[1]
		content := m[2]
		cbs = append(cbs, codeBlock{
			Lang:          lang,
			Content:       content,
			SuggestedPath: suggestFilename(lang, len(cbs)),
		})
	}

	o := &CodeSaveOverlay{
		codeBlocks: cbs,
	}
	// If exactly one block, skip selection and go straight to path entry
	if len(cbs) == 1 {
		o.phase = 1
		o.pathBuf = cbs[0].SuggestedPath
	}
	return o
}

func (c *CodeSaveOverlay) Kind() OverlayKind { return OverlayCodeSave }

func (c *CodeSaveOverlay) Update(msg tea.Msg) (Overlay, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc":
			return c, nil, true

		case "enter":
			if c.phase == 0 {
				// Move to path entry for selected block
				c.phase = 1
				c.pathBuf = c.codeBlocks[c.cursor].SuggestedPath
				return c, nil, false
			}
			// Phase 1: write file
			c.writeFile()
			return c, nil, true

		case "up", "k":
			if c.phase == 0 && c.cursor > 0 {
				c.cursor--
			}
			return c, nil, false

		case "down", "j":
			if c.phase == 0 && c.cursor < len(c.codeBlocks)-1 {
				c.cursor++
			}
			return c, nil, false

		case "backspace":
			if c.phase == 1 && len(c.pathBuf) > 0 {
				c.pathBuf = c.pathBuf[:len(c.pathBuf)-1]
			}
			return c, nil, false

		default:
			if c.phase == 1 {
				key := msg.String()
				if len(key) == 1 && key[0] >= 32 && key[0] < 127 {
					c.pathBuf += key
				}
			}
			return c, nil, false
		}
	}
	return c, nil, false
}

func (c *CodeSaveOverlay) writeFile() {
	if c.pathBuf == "" {
		c.err = fmt.Errorf("empty path")
		return
	}
	content := c.codeBlocks[c.cursor].Content

	dir := filepath.Dir(c.pathBuf)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			c.err = err
			return
		}
	}

	if err := os.WriteFile(c.pathBuf, []byte(content), 0o644); err != nil {
		c.err = err
		return
	}
	c.saved = true
	c.savedPath = c.pathBuf
}

// Saved returns whether a file was successfully written.
func (c *CodeSaveOverlay) Saved() bool { return c.saved }

// SavedPath returns the path of the saved file.
func (c *CodeSaveOverlay) SavedPath() string { return c.savedPath }

// Error returns any error from the save operation.
func (c *CodeSaveOverlay) Error() error { return c.err }

func (c *CodeSaveOverlay) View(width, height int, th *theme.Theme) string {
	var sb strings.Builder

	if c.phase == 0 {
		sb.WriteString(th.OverlayTitle.Render("Save Code Block"))
		sb.WriteByte('\n')
		sb.WriteByte('\n')

		for i, cb := range c.codeBlocks {
			prefix := "  "
			style := th.OverlayText
			if i == c.cursor {
				prefix = "> "
				style = th.OverlayActive
			}
			lang := cb.Lang
			if lang == "" {
				lang = "text"
			}
			// Show first line preview
			preview := firstLine(cb.Content)
			if len(preview) > 50 {
				preview = preview[:47] + "..."
			}
			line := fmt.Sprintf("%s[%d] %s: %s", prefix, i+1, lang, preview)
			sb.WriteString(style.Render(line))
			sb.WriteByte('\n')
		}

		sb.WriteByte('\n')
		sb.WriteString(th.OverlayDim.Render("  enter=select  esc=close"))
	} else {
		sb.WriteString(th.OverlayTitle.Render("Save Code Block"))
		sb.WriteByte('\n')
		sb.WriteByte('\n')

		lang := c.codeBlocks[c.cursor].Lang
		if lang == "" {
			lang = "text"
		}
		lines := strings.Count(c.codeBlocks[c.cursor].Content, "\n") + 1
		sb.WriteString(th.OverlayText.Render(fmt.Sprintf("  %s (%d lines)", lang, lines)))
		sb.WriteByte('\n')
		sb.WriteByte('\n')

		sb.WriteString(th.OverlayInput.Render(" path: "))
		sb.WriteString(th.OverlayText.Render(c.pathBuf))
		sb.WriteByte('\n')
		sb.WriteByte('\n')
		sb.WriteString(th.OverlayDim.Render("  enter=save  esc=close"))
	}

	content := th.OverlayBorder.Render(sb.String())
	return centerOverlay(content, width, height)
}

func suggestFilename(lang string, idx int) string {
	suffix := ""
	if idx > 0 {
		suffix = fmt.Sprintf("_%d", idx+1)
	}
	switch strings.ToLower(lang) {
	case "go":
		return fmt.Sprintf("snippet%s.go", suffix)
	case "python", "py":
		return fmt.Sprintf("snippet%s.py", suffix)
	case "javascript", "js":
		return fmt.Sprintf("snippet%s.js", suffix)
	case "typescript", "ts":
		return fmt.Sprintf("snippet%s.ts", suffix)
	case "rust", "rs":
		return fmt.Sprintf("snippet%s.rs", suffix)
	case "bash", "sh", "shell":
		return fmt.Sprintf("snippet%s.sh", suffix)
	case "yaml", "yml":
		return fmt.Sprintf("snippet%s.yaml", suffix)
	case "json":
		return fmt.Sprintf("snippet%s.json", suffix)
	case "sql":
		return fmt.Sprintf("snippet%s.sql", suffix)
	case "toml":
		return fmt.Sprintf("snippet%s.toml", suffix)
	default:
		return fmt.Sprintf("snippet%s.txt", suffix)
	}
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		return s[:idx]
	}
	return s
}
