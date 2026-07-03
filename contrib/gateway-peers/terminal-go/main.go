// Command agentm-terminal is the AgentM gateway's terminal chat-client peer.
// It renders a cagent-derived TUI adapted to the AgentM gateway wire protocol
// (internal/wire) via internal/adapter, instead of a local agent runtime.
package main

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/adapter"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/tui"
	tuiinput "github.com/AoyangSpace/agentm-terminal/internal/tui/input"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
	"github.com/AoyangSpace/agentm-terminal/internal/wire"
)

func main() {
	connectURL := flag.String("connect", "", "Gateway URL (unix:///path or ws://host:port)")
	token := flag.String("token", "", "Bearer token for ws/wss")
	chatID := flag.String("chat-id", "", "Chat/session ID (default: working directory basename)")
	senderID := flag.String("sender-id", "local", "Sender ID")
	scenario := flag.String("scenario", "", "Scenario name (first message only)")
	themeName := flag.String("theme", "dark", "Theme: dark or light")
	mockMode := flag.Bool("mock", false, "Run with mock data (no gateway)")
	simpleMode := flag.Bool("simple", false, "Run a simplified chat layout for narrow terminals")
	leanMode := flag.Bool("lean", false, "Alias for --simple")
	hideSidebar := flag.Bool("hide-sidebar", false, "Keep the Claude-style full layout without the legacy right sidebar")
	logFile := flag.String("log", "", "Log file path (default: /tmp/agentm-terminal.log)")
	flag.Parse()

	// Default chat-id to "<basename>-<sha1(abspath)[:12]>" so different directories
	// produce different session keys even when they share the same basename,
	// while the same directory always gets the same key (conversation continuity).
	if *chatID == "" {
		wd, _ := os.Getwd()
		if wd != "" {
			parts := strings.Split(strings.TrimRight(wd, "/"), "/")
			base := parts[len(parts)-1]
			h := sha1.Sum([]byte(wd))
			suffix := hex.EncodeToString(h[:])[:12]
			*chatID = base + "-" + suffix
		} else {
			*chatID = "terminal"
		}
	}

	// File logging: bubbletea owns stdout and stderr is unreliable in alt-screen.
	logPath := *logFile
	if logPath == "" {
		logPath = "/tmp/agentm-terminal.log"
	}
	if f, err := tea.LogToFile(logPath, "agentm-terminal"); err == nil {
		defer f.Close()
	}

	// Apply the requested theme before constructing the model.
	styles.ApplyThemeRef(*themeName)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Resolve the gateway URL: explicit flag > env > default socket.
	if !*mockMode && *connectURL == "" {
		if envURL := os.Getenv("AGENTM_SOCKET"); envURL != "" {
			*connectURL = envURL
		} else {
			*connectURL = wire.DefaultSocketURL()
		}
	}

	wd, _ := os.Getwd()

	var initialApp *app.App
	var initialSession *session.Session
	var spawner tui.SessionSpawner
	var liveAdapter *adapter.Adapter

	switch {
	case *mockMode:
		// No gateway: render the TUI against an empty session with no backend.
		// Methods are inert (the App has no Controller), so the UI is visible
		// but does not talk to anything. Useful for layout/theme inspection.
		initialSession = session.New(session.WithTitle("agentm (mock)"))
		initialApp = app.New(ctx, initialSession)
		spawner = adapter.ErrorSpawner()

	default:
		transport, err := wire.ResolveTransport(*connectURL, "")
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		client := wire.NewWireClient(transport, "terminal-go", *token, wire.WithCwd(wd))

		// The timeout covers only the dial + handshake, not the long-lived
		// connection.
		dialCtx, dialCancel := context.WithTimeout(ctx, 10*time.Second)
		err = client.Connect(dialCtx)
		dialCancel()
		if err != nil {
			fmt.Fprintf(os.Stderr, "connect error: %v\n", err)
			os.Exit(1)
		}
		defer client.Close()

		// Compose the session_key per the §3.4 default rule (<channel>:<chat_id>).
		id := adapter.Identity{
			Channel:    "terminal",
			ChatID:     *chatID,
			SenderID:   *senderID,
			SessionKey: "terminal:" + *chatID,
			Scenario:   *scenario,
		}

		ad := adapter.New(client, id, "")
		ad.Start(ctx)

		initialApp = ad.App
		initialSession = ad.Session
		spawner = ad.Spawner()
		liveAdapter = ad
	}

	// Mirror cagent's runTUI program wiring: coalesce mouse-wheel events, build
	// the model via tui.New, hand the *tea.Program back to the model so it can
	// send itself messages, then run.
	coalescer := tuiinput.NewWheelCoalescer()
	filter := func(_ tea.Model, msg tea.Msg) tea.Msg {
		wheelMsg, ok := msg.(tea.MouseWheelMsg)
		if !ok {
			return msg
		}
		if coalescer.Handle(wheelMsg) {
			return nil
		}
		return msg
	}

	var tuiOpts []tui.Option
	if *simpleMode || *leanMode {
		tuiOpts = append(tuiOpts, tui.WithLeanMode())
	}
	if *hideSidebar {
		tuiOpts = append(tuiOpts, tui.WithHideSidebar())
	}

	model := tui.New(ctx, spawner, initialApp, wd, func() {}, tuiOpts...)

	p := tea.NewProgram(model, tea.WithContext(ctx), tea.WithFilter(filter))
	coalescer.SetSender(p.Send)

	if m, ok := model.(interface{ SetProgram(p *tea.Program) }); ok {
		m.SetProgram(p)
	}

	// Hand the program to the adapter so its child manager can drive the TUI's
	// new-tab path when a sub-agent session starts.
	if liveAdapter != nil {
		liveAdapter.SetProgram(p)
	}

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
