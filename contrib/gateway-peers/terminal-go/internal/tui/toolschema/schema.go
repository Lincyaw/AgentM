package toolschema

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
)

const (
	ToolNameFetch              = "fetch"
	ToolNameHandoff            = "handoff"
	ToolNameTransferTask       = "transfer_task"
	ToolNameUserPrompt         = "user_prompt"
	ToolNameShell              = "shell"
	ToolNameBash               = "bash"
	ToolNameCreateTodo         = "create_todo"
	ToolNameCreateTodos        = "create_todos"
	ToolNameUpdateTodos        = "update_todos"
	ToolNameListTodos          = "list_todos"
	ToolNameRead               = "read"
	ToolNameReadFile           = "read_file"
	ToolNameReadMultipleFiles  = "read_multiple_files"
	ToolNameEditFile           = "edit_file"
	ToolNameWriteFile          = "write_file"
	ToolNameDirectoryTree      = "directory_tree"
	ToolNameListDirectory      = "list_directory"
	ToolNameSearchFilesContent = "search_files_content"
)

type FetchArgs struct {
	URLs    []string `json:"urls"`
	Timeout int      `json:"timeout,omitempty"`
	Format  string   `json:"format,omitempty"`
}

type FetchResult struct {
	URL           string `json:"url"`
	StatusCode    int    `json:"statusCode"`
	Status        string `json:"status"`
	ContentType   string `json:"contentType,omitempty"`
	ContentLength int    `json:"contentLength"`
	Body          string `json:"body,omitempty"`
	Error         string `json:"error,omitempty"`
}

type HandoffArgs struct {
	Agent string `json:"agent"`
}

type TransferTaskArgs struct {
	Agent          string `json:"agent"`
	Task           string `json:"task"`
	ExpectedOutput string `json:"expected_output"`
}

type UserPromptArgs struct {
	Message string         `json:"message"`
	Title   string         `json:"title,omitempty"`
	Schema  map[string]any `json:"schema,omitempty"`
}

type UserPromptResponse struct {
	Action  string         `json:"action"`
	Content map[string]any `json:"content,omitempty"`
}

type RunShellArgs struct {
	Cmd     string `json:"cmd"`
	Cwd     string `json:"cwd,omitempty"`
	Timeout int    `json:"timeout,omitempty"`
}

func (a *RunShellArgs) UnmarshalJSON(data []byte) error {
	var raw struct {
		Cmd     string `json:"cmd"`
		Command string `json:"command"`
		Cwd     string `json:"cwd"`
		Timeout int    `json:"timeout"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	a.Cmd = preferNonBlank(raw.Cmd, raw.Command)
	a.Cwd = raw.Cwd
	a.Timeout = raw.Timeout
	return nil
}

type Todo struct {
	ID          string `json:"id"`
	Description string `json:"description"`
	Status      string `json:"status"`
}

type CreateTodoArgs struct {
	Description string `json:"description"`
}

type CreateTodosArgs struct {
	Descriptions []string `json:"descriptions"`
}

type TodoUpdate struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

type UpdateTodosArgs struct {
	Updates []TodoUpdate `json:"updates"`
}

type CreateTodoOutput struct {
	Created  Todo   `json:"created"`
	AllTodos []Todo `json:"all_todos"`
	Reminder string `json:"reminder,omitempty"`
}

type CreateTodosOutput struct {
	Created  []Todo `json:"created"`
	AllTodos []Todo `json:"all_todos"`
	Reminder string `json:"reminder,omitempty"`
}

type UpdateTodosOutput struct {
	Updated  []TodoUpdate `json:"updated,omitempty"`
	NotFound []string     `json:"not_found,omitempty"`
	AllTodos []Todo       `json:"all_todos"`
	Reminder string       `json:"reminder,omitempty"`
}

type ListTodosOutput struct {
	Todos    []Todo `json:"todos"`
	Reminder string `json:"reminder,omitempty"`
}

type Edit struct {
	OldText string `json:"oldText"`
	NewText string `json:"newText"`
}

type ReadFileArgs struct {
	Path string `json:"path"`
}

type ReadFileMeta struct {
	Path      string `json:"path"`
	LineCount int    `json:"lineCount"`
	Error     string `json:"error,omitempty"`
}

type WriteFileArgs struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type DirectoryTreeArgs struct {
	Path string `json:"path"`
}

type DirectoryTreeMeta struct {
	FileCount int  `json:"fileCount"`
	DirCount  int  `json:"dirCount"`
	Truncated bool `json:"truncated"`
}

type ListDirectoryArgs struct {
	Path string `json:"path"`
}

type ListDirectoryMeta struct {
	Files     []string `json:"files"`
	Dirs      []string `json:"dirs"`
	Truncated bool     `json:"truncated"`
}

type SearchFilesContentArgs struct {
	Path            string   `json:"path"`
	Query           string   `json:"query"`
	IsRegex         bool     `json:"is_regex,omitempty"`
	ExcludePatterns []string `json:"excludePatterns,omitempty"`
}

type SearchFilesContentMeta struct {
	MatchCount int `json:"matchCount"`
	FileCount  int `json:"fileCount"`
}

type ReadMultipleFilesArgs struct {
	Paths []string `json:"paths"`
	JSON  bool     `json:"json,omitempty"`
}

type ReadMultipleFilesMeta struct {
	Files []ReadFileMeta `json:"files"`
}

type EditFileArgs struct {
	Path  string `json:"path"`
	Edits []Edit `json:"edits"`
}

func ParseEditFileArgs(data []byte) (EditFileArgs, error) {
	var raw struct {
		Path  string          `json:"path"`
		Edits json.RawMessage `json:"edits"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		repaired, ok := tryRepairEditFileJSON(data)
		if !ok {
			return EditFileArgs{}, fmt.Errorf("failed to parse edit_file arguments: %w", err)
		}
		if err := json.Unmarshal(repaired, &raw); err != nil {
			return EditFileArgs{}, fmt.Errorf("failed to parse edit_file arguments after repair: %w", err)
		}
		slog.Debug("Repaired malformed edit_file JSON arguments")
	}

	args := EditFileArgs{Path: raw.Path}
	if len(raw.Edits) == 0 || string(raw.Edits) == "null" {
		return args, nil
	}
	if err := json.Unmarshal(raw.Edits, &args.Edits); err == nil {
		return args, nil
	}

	var editsStr string
	if err := json.Unmarshal(raw.Edits, &editsStr); err != nil {
		return EditFileArgs{}, fmt.Errorf("edits field is neither an array nor a JSON string: %w", err)
	}
	if err := json.Unmarshal([]byte(editsStr), &args.Edits); err != nil {
		return EditFileArgs{}, fmt.Errorf("failed to parse double-serialized edits string: %w", err)
	}
	return args, nil
}

func tryRepairEditFileJSON(data []byte) ([]byte, bool) {
	current := append([]byte(nil), data...)
	for range 3 {
		var synErr *json.SyntaxError
		if err := json.Unmarshal(current, &json.RawMessage{}); err == nil {
			return current, true
		} else if !errors.As(err, &synErr) {
			return nil, false
		}

		offset := int(synErr.Offset) - 1
		if offset < 0 || offset >= len(current) {
			return nil, false
		}

		ch := current[offset]
		removeCount := 1
		switch ch {
		case '}', ']':
		case '\\':
			if offset+1 < len(current) {
				switch current[offset+1] {
				case 'n', 't', 'r':
					removeCount = 2
				}
			}
		default:
			return nil, false
		}

		repaired := make([]byte, 0, len(current)-removeCount)
		repaired = append(repaired, current[:offset]...)
		repaired = append(repaired, current[offset+removeCount:]...)
		current = repaired
	}

	if json.Valid(current) {
		return current, true
	}
	return nil, false
}

func preferNonBlank(primary, fallback string) string {
	if strings.TrimSpace(primary) != "" {
		return primary
	}
	return fallback
}
