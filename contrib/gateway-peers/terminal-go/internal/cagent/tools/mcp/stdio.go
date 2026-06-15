package mcp

import (
	"context"
	"os/exec"

	gomcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

type stdioMCPClient struct {
	sessionClient

	command string
	args    []string
	env     []string
	cwd     string
}

func newStdioCmdClient(command string, args, env []string, cwd string) *stdioMCPClient {
	return &stdioMCPClient{
		command: command,
		args:    args,
		env:     env,
		cwd:     cwd,
	}
}

func (c *stdioMCPClient) Initialize(ctx context.Context, _ *gomcp.InitializeRequest) (*gomcp.InitializeResult, error) {
	// The upstream client probed Docker Desktop reachability here to produce a
	// nicer error. That probe is backend behaviour that never runs on the
	// wire-render path (and was already skipped on Linux, where Docker runs
	// natively without Docker Desktop), so the pre-flight check is dropped.

	toolChanged, promptChanged := c.notificationHandlers()

	// Create client options with elicitation, sampling, and notification support
	opts := &gomcp.ClientOptions{
		ElicitationHandler:       c.handleElicitationRequest,
		CreateMessageHandler:     c.handleSamplingRequest,
		ToolListChangedHandler:   toolChanged,
		PromptListChangedHandler: promptChanged,
	}

	client := gomcp.NewClient(&gomcp.Implementation{
		Name:    "docker agent",
		Version: "1.0.0",
	}, opts)

	cmd := exec.CommandContext(ctx, c.command, c.args...)
	cmd.Env = c.env
	cmd.Dir = c.cwd
	session, err := client.Connect(ctx, &gomcp.CommandTransport{
		Command: cmd,
	}, nil)
	if err != nil {
		return nil, err
	}

	c.setSession(session)

	return session.InitializeResult(), nil
}
