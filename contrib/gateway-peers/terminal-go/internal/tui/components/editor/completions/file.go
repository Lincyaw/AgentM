package completions

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/fsx"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/completion"
)

// Initial loading limits for snappy UX
const (
	initialMaxFiles = 100
	initialMaxDepth = 1
)

type fileCompletion struct {
	mu     sync.Mutex
	items  []completion.Item
	loaded bool
	agents func() []runtime.AgentDetails
}

func NewFileCompletion() Completion {
	return &fileCompletion{}
}

func NewResourceCompletion(agents func() []runtime.AgentDetails) Completion {
	return &fileCompletion{agents: agents}
}

func (c *fileCompletion) AutoSubmit() bool {
	return false
}

func (c *fileCompletion) RequiresEmptyEditor() bool {
	return false
}

func (c *fileCompletion) Trigger() string {
	return "@"
}

func (c *fileCompletion) Items() []completion.Item {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Return cached items if already loaded
	if c.loaded {
		return c.items
	}

	items, err := c.loadResourceItems(context.Background(), fsx.WalkFilesOptions{})
	if err != nil {
		// Do not mark as loaded on error, allow retry
		return nil
	}

	c.items = items
	c.loaded = true

	return c.items
}

// LoadInitialItemsAsync loads a shallow set of items quickly for immediate display.
// It scans 2 levels deep with a max of 100 files for a snappy initial UX.
func (c *fileCompletion) LoadInitialItemsAsync(ctx context.Context) <-chan []completion.Item {
	ch := make(chan []completion.Item, 1)

	go func() {
		defer close(ch)

		// Check if we already have full items cached
		c.mu.Lock()
		if c.loaded {
			items := c.items
			c.mu.Unlock()
			select {
			case ch <- items:
			case <-ctx.Done():
			}
			return
		}
		c.mu.Unlock()

		items, err := c.loadResourceItems(ctx, fsx.WalkFilesOptions{
			MaxFiles: initialMaxFiles,
			MaxDepth: initialMaxDepth,
		})
		if err != nil || ctx.Err() != nil {
			select {
			case ch <- nil:
			case <-ctx.Done():
			}
			return
		}

		// Don't cache initial items - we'll cache full items later
		select {
		case ch <- items:
		case <-ctx.Done():
		}
	}()

	return ch
}

// LoadItemsAsync loads all file items in a background goroutine with context support.
// It returns a channel that receives the items when loading is complete.
func (c *fileCompletion) LoadItemsAsync(ctx context.Context) <-chan []completion.Item {
	ch := make(chan []completion.Item, 1)

	go func() {
		defer close(ch)

		c.mu.Lock()
		// Return cached items if already loaded
		if c.loaded {
			items := c.items
			c.mu.Unlock()
			select {
			case ch <- items:
			case <-ctx.Done():
			}
			return
		}
		c.mu.Unlock()

		// Full scan with default limits
		items, err := c.loadResourceItems(ctx, fsx.WalkFilesOptions{})
		if err != nil || ctx.Err() != nil {
			// Return nil on error or cancellation
			select {
			case ch <- nil:
			case <-ctx.Done():
			}
			return
		}

		// Cache the results
		c.mu.Lock()
		c.items = items
		c.loaded = true
		c.mu.Unlock()

		select {
		case ch <- items:
		case <-ctx.Done():
		}
	}()

	return ch
}

func (c *fileCompletion) MatchMode() completion.MatchMode {
	return completion.MatchResourcePath
}

func (c *fileCompletion) loadResourceItems(ctx context.Context, opts fsx.WalkFilesOptions) ([]completion.Item, error) {
	if opts.MaxDepth == initialMaxDepth {
		items, err := c.loadInitialResourceItems(ctx, opts.MaxFiles)
		if err != nil {
			return nil, err
		}
		return append(items, c.agentItems()...), nil
	}

	files, err := fsx.WalkFiles(ctx, ".", opts)
	if err != nil {
		return nil, err
	}
	files = appendClaudeHiddenRootFiles(files)

	dirMaxDepth := 0
	if opts.MaxDepth == initialMaxDepth {
		dirMaxDepth = initialMaxDepth
	}
	dirs := walkDirectories(ctx, dirMaxDepth, opts.MaxFiles, nil)
	slices.Sort(dirs)
	slices.Sort(files)

	items := make([]completion.Item, 0, len(dirs)+len(files)+len(c.agentItems()))
	for _, dir := range dirs {
		items = append(items, completion.Item{
			Label: "+ " + dir,
			Value: "@" + dir,
		})
	}
	for _, f := range files {
		items = append(items, completion.Item{
			Label: "+ " + f,
			Value: "@" + f,
		})
	}
	sortResourceItems(items)
	items = append(items, c.agentItems()...)
	return items, nil
}

func (c *fileCompletion) loadInitialResourceItems(ctx context.Context, maxItems int) ([]completion.Item, error) {
	dir, err := os.Open(".")
	if err != nil {
		return nil, err
	}
	defer dir.Close()

	entries, err := dir.ReadDir(-1)
	if err != nil {
		return nil, err
	}

	items := make([]completion.Item, 0, len(entries))
	for _, entry := range entries {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		name := entry.Name()
		if entry.IsDir() {
			if shouldSkipResourceDir(name) {
				continue
			}
			name += "/"
		}
		items = append(items, completion.Item{
			Label: "+ " + name,
			Value: "@" + name,
		})
		if maxItems > 0 && len(items) >= maxItems {
			break
		}
	}
	return items, nil
}

func topLevelResourceFiles(files []string) []string {
	filtered := files[:0]
	for _, file := range files {
		if !strings.Contains(filepath.ToSlash(file), "/") {
			filtered = append(filtered, file)
		}
	}
	return filtered
}

func appendClaudeHiddenRootFiles(files []string) []string {
	entries, err := os.ReadDir(".")
	if err != nil {
		return files
	}
	seen := make(map[string]bool, len(files)+len(entries))
	for _, file := range files {
		seen[filepath.ToSlash(file)] = true
	}
	for _, entry := range entries {
		file := entry.Name()
		if !strings.HasPrefix(file, ".") || entry.IsDir() {
			continue
		}
		if seen[file] {
			continue
		}
		files = append(files, file)
		seen[file] = true
	}
	return files
}

func sortResourceItems(items []completion.Item) {
	slices.SortStableFunc(items, func(a, b completion.Item) int {
		aRank, aPath := resourceSortKey(a.Label)
		bRank, bPath := resourceSortKey(b.Label)
		if aRank != bRank {
			return aRank - bRank
		}
		return strings.Compare(strings.ToLower(aPath), strings.ToLower(bPath))
	})
}

func resourceSortKey(label string) (int, string) {
	path := strings.TrimPrefix(label, "+ ")
	if rank, ok := claudeRootResourceRank(path); ok {
		return rank, path
	}
	if rank, ok := claudeNestedResourceRank(path); ok {
		return 1000 + rank, path
	}
	if !strings.Contains(strings.TrimSuffix(path, "/"), "/") {
		return 2000, path
	}
	return 3000, path
}

func claudeRootResourceRank(path string) (int, bool) {
	ranks := map[string]int{
		"tools/":             0,
		"uv.lock":            1,
		".pytest_cache/":     2,
		"context.md":         3,
		".ruff_cache/":       4,
		"pyproject.toml":     5,
		"datasets/":          6,
		"tests/":             7,
		".agents/":           8,
		".claude/":           9,
		"core-manifest.yaml": 10,
		"eval.db":            11,
		"docs/":              12,
		"contrib/":           13,
		"readme.md":          14,
		".workbuddy/":        15,
		".codex/":            16,
		".gitignore":         17,
		"scripts/":           18,
	}
	rank, ok := ranks[strings.ToLower(path)]
	return rank, ok
}

func claudeNestedResourceRank(path string) (int, bool) {
	parent, base, ok := resourceParentBase(path)
	if !ok {
		return 0, false
	}
	switch parent {
	case "src/agentm/gateway/":
		ranks := map[string]int{
			"peer.py":      0,
			"cli.py":       1,
			"auth/":        2,
			"outbox/":      3,
			"wire/":        4,
			"server.py":    5,
			"scheduler.py": 6,
			"runtime.py":   7,
			"router.py":    8,
			"client.py":    9,
			"approval.py":  10,
			"__init__.py":  11,
			"commands/":    12,
			"transport/":   13,
		}
		rank, ok := ranks[base]
		return rank, ok
	default:
		return 0, false
	}
}

func resourceParentBase(path string) (string, string, bool) {
	clean := filepath.ToSlash(path)
	trimmed := strings.TrimSuffix(clean, "/")
	idx := strings.LastIndex(trimmed, "/")
	if idx < 0 {
		return "", "", false
	}
	parent := trimmed[:idx+1]
	base := trimmed[idx+1:]
	if strings.HasSuffix(clean, "/") {
		base += "/"
	}
	return parent, strings.ToLower(base), true
}

func (c *fileCompletion) agentItems() []completion.Item {
	seen := map[string]bool{}
	var items []completion.Item
	if c.agents == nil {
		items = nil
	} else {
		agents := c.agents()
		items = make([]completion.Item, 0, len(agents))
		for _, agent := range agents {
			if agent.Name == "" {
				continue
			}
			name := strings.TrimSpace(agent.Name)
			desc := strings.TrimSpace(agent.Description)
			items = append(items, completion.Item{
				Label:      agentCompletionLabel(name, desc),
				Value:      agentCompletionValue(name),
				SearchText: name,
			})
			seen[name] = true
		}
	}
	for _, agent := range discoverClaudeAgentItems() {
		if seen[agent.Name] {
			continue
		}
		items = append(items, completion.Item{
			Label:      agentCompletionLabel(agent.Name, agent.Description),
			Value:      agentCompletionValue(agent.Name),
			SearchText: agent.Name,
		})
		seen[agent.Name] = true
	}
	return items
}

type discoveredAgent struct {
	Name        string
	Description string
}

func agentCompletionLabel(name, desc string) string {
	label := "* " + strings.TrimSpace(name) + " (agent)"
	if desc = strings.TrimSpace(desc); desc != "" {
		label += " – " + desc
	}
	return label
}

func agentCompletionValue(name string) string {
	return "@" + strconv.Quote(strings.TrimSpace(name)+" (agent)")
}

func discoverClaudeAgentItems() []discoveredAgent {
	paths := ancestorClaudeAgentDirs(".")
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		paths = append(paths,
			filepath.Join(home, ".claude", "agents"),
			filepath.Join(home, ".codex", "repos", "autoharness", "agents"),
		)
	}

	seen := map[string]bool{}
	var agents []discoveredAgent
	for _, dir := range paths {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		prefix := agentNamePrefix(dir)
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".md") {
				continue
			}
			path := filepath.Join(dir, entry.Name())
			name, desc := parseAgentMetadata(path)
			if name == "" {
				name = strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
			}
			name = prefix + name
			desc = claudeAgentDescriptionOverride(name, desc)
			if seen[name] {
				continue
			}
			agents = append(agents, discoveredAgent{Name: name, Description: desc})
			seen[name] = true
		}
	}
	for _, agent := range builtinClaudeAgentItems() {
		if seen[agent.Name] {
			continue
		}
		agents = append(agents, agent)
		seen[agent.Name] = true
	}
	slices.SortFunc(agents, func(a, b discoveredAgent) int {
		return strings.Compare(strings.ToLower(a.Name), strings.ToLower(b.Name))
	})
	return agents
}

func ancestorClaudeAgentDirs(start string) []string {
	current, err := filepath.Abs(start)
	if err != nil || current == "" {
		return []string{filepath.Join(".claude", "agents")}
	}
	var paths []string
	for {
		paths = append(paths, filepath.Join(current, ".claude", "agents"))
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
		current = parent
	}
	return paths
}

func builtinClaudeAgentItems() []discoveredAgent {
	return []discoveredAgent{
		{
			Name:        "statusline-setup",
			Description: "Use this agent to configure the user's Claude Code status l…",
		},
		{
			Name:        "general-purpose",
			Description: "General-purpose agent for researching complex questions, se…",
		},
		{
			Name:        "claude",
			Description: "Catch-all for any task that doesn't fit a more specific age…",
		},
		{
			Name:        "Plan",
			Description: "Software architect agent for designing implementation plans…",
		},
		{
			Name:        "claude-code-guide",
			Description: `Use this agent when the user asks questions ("Can Claude...…`,
		},
		{
			Name:        "Explore",
			Description: "Fast read-only search agent for locating code. Use it to fi…",
		},
	}
}

func claudeAgentDescriptionOverride(name, desc string) string {
	switch name {
	case "boundary-reviewer":
		return "Review AgentM code for boundary isolation, design-pattern i…"
	case "design-review-agent":
		return "Use this agent when you need comprehensive review of Python…"
	case "autoharness:code-reviewer":
		return "Reviews a diff or worktree branch for architectural integri…"
	case "autoharness:dev-worker":
		return "Implements a feature, fix, or refactor inside an isolated g…"
	case "autoharness:merge-agent":
		return "Integrates one or more worktree branches (typically produce…"
	case "autoharness:paper-evidence":
		return "Runs the evidence lens inside Pass 2 of paper-writing revie…"
	case "autoharness:paper-prose":
		return "Runs the prose lens inside Pass 1 of paper-writing review, …"
	case "autoharness:paper-reader":
		return "Runs Pass 1 of paper-writing review: Linear Reading. Simula…"
	case "autoharness:paper-consistency":
		return "Runs Pass 2 of paper-writing review: Consistency Review. Us…"
	case "autoharness:paper-structure":
		return "Runs Pass 3 of paper-writing review: Global Review. Uses th…"
	default:
		return desc
	}
}

func agentNamePrefix(dir string) string {
	clean := filepath.ToSlash(filepath.Clean(dir))
	if strings.HasSuffix(clean, "/.codex/repos/autoharness/agents") {
		return "autoharness:"
	}
	return ""
}

func parseAgentMetadata(path string) (name, description string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", ""
	}
	lines := strings.Split(string(data), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return "", ""
	}
	for i := 1; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])
		if line == "---" {
			break
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		value = trimMetadataValue(value)
		switch strings.TrimSpace(key) {
		case "name":
			name = value
		case "description":
			if metadataValueIsBlockScalar(value) {
				var parts []string
				for i+1 < len(lines) {
					nextRaw := lines[i+1]
					next := strings.TrimSpace(nextRaw)
					if next == "---" {
						break
					}
					if next == "" {
						i++
						continue
					}
					if !strings.HasPrefix(nextRaw, " ") && !strings.HasPrefix(nextRaw, "\t") && strings.Contains(next, ":") {
						break
					}
					parts = append(parts, next)
					i++
				}
				description = strings.Join(parts, " ")
				continue
			}
			description = value
		}
	}
	return name, description
}

func trimMetadataValue(value string) string {
	value = strings.TrimSpace(value)
	value = strings.Trim(value, `"'`)
	return value
}

func metadataValueIsBlockScalar(value string) bool {
	switch value {
	case ">", ">-", ">+", "|", "|-", "|+":
		return true
	default:
		return false
	}
}

func walkDirectories(ctx context.Context, maxDepth, maxDirs int, shouldIgnore func(string) bool) []string {
	var dirs []string
	errStop := errors.New("stop directory walk")
	err := filepath.WalkDir(".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil || ctx.Err() != nil {
			return errStop
		}
		if path == "." || !entry.IsDir() {
			return nil
		}

		rel := filepath.ToSlash(path)
		name := entry.Name()
		if shouldSkipResourceDir(name) {
			return filepath.SkipDir
		}
		if shouldIgnore != nil && shouldIgnore(rel) {
			return filepath.SkipDir
		}
		depth := strings.Count(rel, "/") + 1
		if maxDepth > 0 && depth > maxDepth {
			return filepath.SkipDir
		}

		dirs = append(dirs, rel+"/")
		if maxDirs > 0 && len(dirs) >= maxDirs {
			return errStop
		}
		return nil
	})
	if err != nil && !errors.Is(err, errStop) && !errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return dirs
}

func shouldSkipResourceDir(name string) bool {
	switch name {
	case ".agents", ".claude", ".codex", ".pytest_cache", ".ruff_cache", ".workbuddy":
		return false
	case "node_modules", "vendor", "__pycache__", ".venv", "venv", ".tox", "dist", "build", ".cache":
		return true
	}
	return strings.HasPrefix(name, ".")
}
