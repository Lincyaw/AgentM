// Command agentm-terminal is the AgentM gateway's terminal chat-client peer.
// It renders a cagent-derived TUI adapted to the AgentM gateway wire protocol
// (internal/wire) via internal/adapter, instead of a local agent runtime.
package main

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/adapter"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/version"
	"github.com/AoyangSpace/agentm-terminal/internal/tui"
	tuiinput "github.com/AoyangSpace/agentm-terminal/internal/tui/input"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
	"github.com/AoyangSpace/agentm-terminal/internal/wire"
)

func main() {
	flags := flag.NewFlagSet("ag", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.Usage = func() {}

	connectURL := flags.String("connect", "", "Gateway URL (unix:///path or ws://host:port)")
	token := flags.String("token", "", "Bearer token for ws/wss")
	sessionID := flags.String("session-id", "", "Session ID (default: new session per terminal)")
	senderID := flags.String("sender-id", "local", "Sender ID")
	scenario := flags.String("scenario", "", "Scenario name (first message only)")
	themeName := flags.String("theme", "dark", "Theme: dark or light")
	mockMode := flags.Bool("mock", false, "Run with mock data (no gateway)")
	logFile := flags.String("log", "", "Log file path (default: /tmp/agentm-terminal.log)")
	showVersion := flags.Bool("version", false, "Print version and exit")
	rawArgs := os.Args[1:]
	if wantsHelp(rawArgs) {
		printUsage(os.Stdout)
		return
	}
	if err := flags.Parse(stripDeprecatedBoolFlag(rawArgs, "hide-sidebar")); err != nil {
		fmt.Fprintf(os.Stderr, "error: %s\n\n", formatFlagError(err, rawArgs))
		printUsage(os.Stderr)
		os.Exit(2)
	}
	if *showVersion {
		fmt.Printf("ag %s (%s)\n", version.Version, version.Commit)
		return
	}

	// Default session-id to a per-process value so multiple terminal windows in
	// the same directory become independent sessions. Pass -session-id
	// explicitly to reconnect to a known session key.
	if *sessionID == "" {
		wd, _ := os.Getwd()
		*sessionID = defaultSessionID(wd)
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
			ChatID:     *sessionID,
			SenderID:   *senderID,
			SessionKey: "terminal:" + *sessionID,
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

	model := tui.New(ctx, spawner, initialApp, wd, func() {}, tui.WithHideSidebar())

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

func defaultSessionID(wd string) string {
	base := "terminal"
	if wd != "" {
		parts := strings.Split(strings.TrimRight(wd, "/"), "/")
		if parts[len(parts)-1] != "" {
			base = parts[len(parts)-1]
		}
	}
	buf := make([]byte, 6)
	if _, err := cryptorand.Read(buf); err == nil {
		return base + "-" + hex.EncodeToString(buf)
	}
	return fmt.Sprintf("%s-%d-%d", base, os.Getpid(), time.Now().UnixNano())
}

func printUsage(out io.Writer) {
	fmt.Fprintln(out, "Usage: ag [--scenario <name>] [options]")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "AgentM terminal client. Starts one interactive gateway-backed chat session.")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "Options:")
	fmt.Fprintln(out, "  --scenario <name>     Scenario for the first message, e.g. trainticket")
	fmt.Fprintln(out, "  --session-id <id>     Reconnect to a known terminal session id")
	fmt.Fprintln(out, "  --connect <url>       Gateway URL (unix:///path or ws://host:port)")
	fmt.Fprintln(out, "  --token <token>       Bearer token for ws/wss gateways")
	fmt.Fprintln(out, "  --sender-id <id>      Sender id for gateway routing (default: local)")
	fmt.Fprintln(out, "  --theme <dark|light>  Terminal theme (default: dark)")
	fmt.Fprintln(out, "  --log <path>          Log file path (default: /tmp/agentm-terminal.log)")
	fmt.Fprintln(out, "  --mock                Run the TUI without a gateway, for layout inspection")
	fmt.Fprintln(out, "  --version             Print version and exit")
	fmt.Fprintln(out, "  --help                Show this help")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "Examples:")
	fmt.Fprintln(out, "  ag")
	fmt.Fprintln(out, "  ag --scenario trainticket")
	fmt.Fprintln(out, "  ag --connect unix:///tmp/agentm-gateway.sock --scenario chatbot")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "For non-interactive or structured use, call the Python agentm CLI or SDK.")
}

func wantsHelp(args []string) bool {
	for _, arg := range args {
		switch arg {
		case "-h", "--help", "-help":
			return true
		}
	}
	return false
}

func formatFlagError(err error, args []string) string {
	text := err.Error()
	const unknownPrefix = "flag provided but not defined: -"
	if name, ok := strings.CutPrefix(text, unknownPrefix); ok {
		if usedLongFlag(args, name) {
			return "unknown option --" + name
		}
		return "unknown option -" + name
	}
	const missingPrefix = "flag needs an argument: -"
	if name, ok := strings.CutPrefix(text, missingPrefix); ok {
		return "missing value for --" + name
	}
	return strings.ReplaceAll(text, " -", " --")
}

func usedLongFlag(args []string, name string) bool {
	for _, arg := range args {
		if arg == "--"+name || strings.HasPrefix(arg, "--"+name+"=") {
			return true
		}
	}
	return false
}

func stripDeprecatedBoolFlag(args []string, name string) []string {
	cleaned := make([]string, 0, len(args))
	short := "-" + name
	long := "--" + name
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == short || arg == long {
			if i+1 < len(args) && (args[i+1] == "true" || args[i+1] == "false") {
				i++
			}
			continue
		}
		if strings.HasPrefix(arg, short+"=") || strings.HasPrefix(arg, long+"=") {
			continue
		}
		cleaned = append(cleaned, arg)
	}
	return cleaned
}
