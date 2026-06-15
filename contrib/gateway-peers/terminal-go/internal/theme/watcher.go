package theme

import (
	"bufio"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/fsnotify/fsnotify"
)

// ThemeChangedMsg is sent to the bubbletea loop when the watched theme
// config file changes on disk.
type ThemeChangedMsg struct {
	Name string // "dark" or "light"
}

// Watcher watches a theme config file for changes and emits
// ThemeChangedMsg through a bubbletea Cmd. The watched file is a plain
// text file whose first line contains "dark" or "light".
type Watcher struct {
	path     string
	debounce time.Duration
	done     chan struct{}
	watcher  *fsnotify.Watcher
}

// NewWatcher creates a Watcher for the given config file path.
// The file does not need to exist at creation time.
func NewWatcher(path string) *Watcher {
	return &Watcher{
		path:     path,
		debounce: 500 * time.Millisecond,
		done:     make(chan struct{}),
	}
}

// Watch returns a tea.Cmd that blocks until the theme file changes, then
// returns a ThemeChangedMsg. The caller must re-invoke Watch() after
// receiving the message to continue watching (standard bubbletea
// subscription pattern).
func (w *Watcher) Watch() tea.Cmd {
	return func() tea.Msg {
		fw, err := fsnotify.NewWatcher()
		if err != nil {
			log.Printf("[theme/watcher] fsnotify.NewWatcher failed: %v", err)
			// Block forever rather than spin.
			<-w.done
			return nil
		}
		w.watcher = fw

		// Watch the parent directory so we catch file creation if the
		// config file does not yet exist.
		dir := filepath.Dir(w.path)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			log.Printf("[theme/watcher] mkdir %s: %v", dir, err)
		}
		if err := fw.Add(dir); err != nil {
			log.Printf("[theme/watcher] watch %s: %v", dir, err)
			fw.Close()
			<-w.done
			return nil
		}

		basename := filepath.Base(w.path)
		var debounceTimer *time.Timer
		notify := make(chan struct{}, 1)

		for {
			select {
			case <-w.done:
				if debounceTimer != nil {
					debounceTimer.Stop()
				}
				fw.Close()
				return nil

			case ev, ok := <-fw.Events:
				if !ok {
					return nil
				}
				// Only care about events targeting our file.
				if filepath.Base(ev.Name) != basename {
					continue
				}
				relevant := ev.Op&(fsnotify.Write|fsnotify.Create|fsnotify.Rename) != 0
				if !relevant {
					continue
				}
				// Reset debounce timer on every relevant event.
				if debounceTimer != nil {
					debounceTimer.Stop()
				}
				debounceTimer = time.AfterFunc(w.debounce, func() {
					select {
					case notify <- struct{}{}:
					default:
					}
				})

			case <-notify:
				if debounceTimer != nil {
					debounceTimer.Stop()
				}
				fw.Close()
				name := readThemeFile(w.path)
				return ThemeChangedMsg{Name: name}

			case err, ok := <-fw.Errors:
				if !ok {
					return nil
				}
				log.Printf("[theme/watcher] fsnotify error: %v", err)
			}
		}
	}
}

// Close stops the watcher. Safe to call multiple times.
func (w *Watcher) Close() {
	select {
	case <-w.done:
		// Already closed.
	default:
		close(w.done)
	}
	if w.watcher != nil {
		w.watcher.Close()
	}
}

// readThemeFile reads the first non-empty line from path and returns
// "dark" or "light". Returns "dark" on any error or unrecognised value.
func readThemeFile(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return "dark"
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.EqualFold(line, "light") {
			return "light"
		}
		return "dark"
	}
	return "dark"
}

// ThemeConfPath returns the default theme config file path, respecting
// the AGENTM_HOME environment variable.
func ThemeConfPath() string {
	home := os.Getenv("AGENTM_HOME")
	if home == "" {
		userHome, err := os.UserHomeDir()
		if err != nil {
			return filepath.Join(".agentm", "theme.conf")
		}
		home = filepath.Join(userHome, ".agentm")
	}
	return filepath.Join(home, "theme.conf")
}
