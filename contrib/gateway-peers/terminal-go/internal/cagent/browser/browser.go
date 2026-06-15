// Package browser opens URLs in the user's default browser. Copied verbatim
// from cagent (stdlib-only, no transitive deps).
package browser

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
)

// Open opens the given URL in the platform's default browser.
func Open(ctx context.Context, urlToOpen string) error {
	var cmd string
	var args []string

	switch runtime.GOOS {
	case "windows":
		cmd = "rundll32"
		args = []string{"url.dll,FileProtocolHandler", urlToOpen}
	case "darwin":
		cmd = "open"
		args = []string{urlToOpen}
	case "linux":
		cmd = "xdg-open"
		args = []string{urlToOpen}
	default:
		return fmt.Errorf("unsupported platform: %s", runtime.GOOS)
	}

	err := exec.CommandContext(ctx, cmd, args...).Start()
	if err != nil {
		return fmt.Errorf("failed to open browser: %w", err)
	}
	return nil
}
