package tui

import (
	"cmp"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	goruntime "runtime"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/notification"
)

// exitFunc is the function called by the shutdown safety net when the
// graceful exit times out. It defaults to os.Exit but can be replaced in tests.
var exitFunc = os.Exit

var shutdownTimeout = 5 * time.Second

// cleanupAll cleans up all sessions, editors, and resources.
func (m *appModel) cleanupAll() {
	m.transcriber.Stop()
	m.closeTranscriptCh()
	for _, ed := range m.editors {
		ed.Cleanup()
	}

	// Safety net: bubbletea's renderer can deadlock on shutdown if stdout
	// is wedged. Race Wait() against a deadline and force-exit if shutdown
	// stalls. Clear m.program so repeated cleanup calls are no-ops.
	program := m.program
	if program == nil {
		return
	}
	m.program = nil
	timeout := shutdownTimeout
	exit := exitFunc
	go func() {
		done := make(chan struct{})
		go func() {
			program.Wait()
			close(done)
		}()

		select {
		case <-done:
		case <-time.After(timeout):
			slog.Warn("Graceful shutdown timed out, forcing exit")
			go func() { _ = program.ReleaseTerminal() }()
			exit(0)
		}
	}()
}

// openExternalEditor opens the current editor content in an external editor.
func (m *appModel) openExternalEditor() (tea.Model, tea.Cmd) {
	content := m.editor.Value()

	tmpFile, err := os.CreateTemp("", "cagent-*.md")
	if err != nil {
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to create temp file: %v", err))
	}
	tmpPath := tmpFile.Name()

	if _, err := tmpFile.WriteString(content); err != nil {
		_ = tmpFile.Close()
		_ = os.Remove(tmpPath)
		return m, notification.ErrorCmd(fmt.Sprintf("Failed to write temp file: %v", err))
	}
	_ = tmpFile.Close()

	editorCmd := cmp.Or(os.Getenv("VISUAL"), os.Getenv("EDITOR"))
	if editorCmd == "" {
		if goruntime.GOOS == "windows" {
			editorCmd = "notepad"
		} else {
			editorCmd = "vi"
		}
	}

	parts := strings.Fields(editorCmd)
	args := append(parts[1:], tmpPath)
	// External editor is owned by tea.ExecProcess, so exec.Command is intentional.
	cmd := exec.Command(parts[0], args...) //nolint:noctx // owned by tea.ExecProcess

	ed := m.editor
	return m, tea.ExecProcess(cmd, func(err error) tea.Msg {
		if err != nil {
			os.Remove(tmpPath)
			return notification.ShowMsg{Text: fmt.Sprintf("Editor error: %v", err), Type: notification.TypeError}
		}

		updatedContent, readErr := os.ReadFile(tmpPath)
		os.Remove(tmpPath)

		if readErr != nil {
			return notification.ShowMsg{Text: fmt.Sprintf("Failed to read edited file: %v", readErr), Type: notification.TypeError}
		}

		c := strings.TrimSuffix(string(updatedContent), "\n")

		if strings.TrimSpace(c) == "" {
			ed.SetValue("")
		} else {
			ed.SetValue(c)
		}

		return nil
	})
}
