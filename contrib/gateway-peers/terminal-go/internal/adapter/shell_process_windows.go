//go:build windows

package adapter

import (
	"context"
	"os/exec"
)

func shellCommandContext(ctx context.Context, name string, args ...string) *exec.Cmd {
	return exec.CommandContext(ctx, name, args...)
}
