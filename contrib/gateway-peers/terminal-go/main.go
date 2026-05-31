package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/app"
	"github.com/AoyangSpace/agentm-terminal/internal/wire"
)

func main() {
	connectURL := flag.String("connect", "", "Gateway URL (unix:///path or ws://host:port)")
	token := flag.String("token", "", "Bearer token for ws/wss")
	chatID := flag.String("chat-id", "terminal", "Chat/session ID")
	senderID := flag.String("sender-id", "local", "Sender ID")
	scenario := flag.String("scenario", "", "Scenario name (first message only)")
	themeName := flag.String("theme", "dark", "Theme: dark or light")
	mockMode := flag.Bool("mock", false, "Run with mock data (no gateway)")
	flag.Parse()

	cfg := app.Config{
		SenderID: *senderID,
		ChatID:   *chatID,
		Scenario: *scenario,
		Theme:    *themeName,
	}

	if !*mockMode && *connectURL == "" {
		// Check AGENTM_SOCKET env var
		if envURL := os.Getenv("AGENTM_SOCKET"); envURL != "" {
			*connectURL = envURL
		}
	}

	if !*mockMode && *connectURL != "" {
		transport, err := wire.ResolveTransport(*connectURL, "")
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		client := wire.NewWireClient(transport, "terminal-go", *token)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := client.Connect(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "connect error: %v\n", err)
			os.Exit(1)
		}
		defer client.Close()

		cfg.WireClient = client
	}

	m := app.NewModel(cfg)
	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
