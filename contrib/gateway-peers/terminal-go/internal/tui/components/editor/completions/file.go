package completions

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/fsx"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/completion"
)

// Initial loading limits for snappy UX
const (
	initialMaxFiles = 100
	initialMaxDepth = 2
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
	return completion.MatchFuzzy
}

func (c *fileCompletion) loadResourceItems(ctx context.Context, opts fsx.WalkFilesOptions) ([]completion.Item, error) {
	vcsMatcher, _ := fsx.NewVCSMatcher(".")
	var shouldIgnore func(string) bool
	if vcsMatcher != nil {
		shouldIgnore = vcsMatcher.ShouldIgnore
	}
	opts.ShouldIgnore = shouldIgnore

	files, err := fsx.WalkFiles(ctx, ".", opts)
	if err != nil {
		return nil, err
	}

	dirs := walkDirectories(ctx, opts.MaxDepth, opts.MaxFiles, shouldIgnore)
	slices.Sort(dirs)
	slices.Sort(files)

	items := make([]completion.Item, 0, len(dirs)+len(files)+len(c.agentItems()))
	for _, f := range files {
		items = append(items, completion.Item{
			Label:       "+ " + f,
			Description: "file",
			Value:       "@" + f,
		})
	}
	for _, dir := range dirs {
		items = append(items, completion.Item{
			Label:       "+ " + dir,
			Description: "directory",
			Value:       "@" + dir,
		})
	}
	items = append(items, c.agentItems()...)
	return items, nil
}

func (c *fileCompletion) agentItems() []completion.Item {
	if c.agents == nil {
		return nil
	}
	agents := c.agents()
	items := make([]completion.Item, 0, len(agents))
	for _, agent := range agents {
		if agent.Name == "" {
			continue
		}
		desc := strings.TrimSpace(agent.Description)
		if desc == "" {
			desc = "agent"
		}
		items = append(items, completion.Item{
			Label:       "* " + agent.Name + " (agent)",
			Description: desc,
			Value:       "@agent:" + agent.Name,
		})
	}
	return items
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
	case ".git", ".github", ".gitlab", ".agents", ".claude":
		return false
	case "node_modules", "vendor", "__pycache__", ".venv", "venv", ".tox", "dist", "build", ".cache":
		return true
	}
	return strings.HasPrefix(name, ".")
}
