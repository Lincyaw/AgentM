package commands

import (
	"context"
	"slices"
	"strings"

	tea "charm.land/bubbletea/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
)

// ExecuteFunc is a function that executes a command with an optional argument.
type ExecuteFunc func(arg string) tea.Cmd

// Category represents a category of commands
type Category struct {
	Name     string
	Commands []Item
}

// Item represents a single command in the palette
type Item struct {
	ID           string
	Label        string
	Description  string
	Category     string
	SlashCommand string
	Execute      ExecuteFunc
	Hidden       bool // Hidden commands work as slash commands but don't appear in the palette
	// Immediate marks commands that should run as soon as they are submitted
	// instead of being treated as ordinary queued chat input.
	Immediate bool
}

func builtInSessionCommands() []Item {
	cmds := []Item{
		{
			ID:           "session.background",
			Label:        "Background",
			SlashCommand: "/background",
			Description:  "Send this session to the background and free the terminal",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.BackgroundSessionMsg{})
			},
		},
		{
			ID:           "session.branch",
			Label:        "Branch",
			SlashCommand: "/branch",
			Description:  "Create a branch of the current conversation at this point",
			Category:     "Session",
			Immediate:    true,
			Execute: func(arg string) tea.Cmd {
				input := "/branch"
				if arg = strings.TrimSpace(arg); arg != "" {
					input += " " + arg
				}
				return core.CmdHandler(messages.SendMsg{Content: input, BypassQueue: true})
			},
		},
		{
			ID:           "session.clear",
			Label:        "Clear",
			SlashCommand: "/clear",
			Description:  "Start a new session with empty context; previous session stays on disk (resumable with /resume)",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ClearSessionMsg{})
			},
		},
		{
			ID:           "session.attach",
			Label:        "Attach",
			SlashCommand: "/attach",
			Description:  "Attach a file to your message (usage: /attach [path])",
			Category:     "Session",
			Immediate:    true,
			Execute: func(arg string) tea.Cmd {
				return core.CmdHandler(messages.AttachFileMsg{FilePath: arg})
			},
		},
		{
			ID:           "session.compact",
			Label:        "Compact",
			SlashCommand: "/compact",
			Description:  "Free up context by summarizing the conversation so far",
			Category:     "Session",
			Immediate:    true,
			Execute: func(arg string) tea.Cmd {
				return core.CmdHandler(messages.CompactSessionMsg{AdditionalPrompt: arg})
			},
		},
		{
			ID:           "session.clipboard",
			Label:        "Copy",
			SlashCommand: "/copy",
			Description:  "Copy Claude's last response to clipboard (or /copy N for the Nth-latest)",
			Category:     "Session",
			Immediate:    true,
			Execute: func(arg string) tea.Cmd {
				return core.CmdHandler(messages.CopySessionToClipboardMsg{Argument: arg})
			},
		},
		{
			ID:           "session.cost",
			Label:        "Cost",
			SlashCommand: "/cost",
			Description:  "Show detailed cost breakdown for this session",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowCostDialogMsg{})
			},
		},
		{
			ID:           "session.context",
			Label:        "Context",
			SlashCommand: "/context",
			Description:  "Visualize current context usage as a colored grid",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowContextDialogMsg{})
			},
		},
		{
			ID:           "session.color",
			Label:        "Color",
			SlashCommand: "/color",
			Description:  "Set the prompt bar color for this session",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.CycleSessionColorMsg{})
			},
		},
		{
			ID:           "session.config",
			Label:        "Config",
			SlashCommand: "/config",
			Description:  "Open settings",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowConfigDialogMsg{})
			},
		},
		{
			ID:           "session.settings",
			Label:        "Settings",
			SlashCommand: "/settings",
			Description:  "Open settings",
			Category:     "Session",
			Immediate:    true,
			Hidden:       true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowSettingsDialogMsg{})
			},
		},
		{
			ID:           "session.usage",
			Label:        "Usage",
			SlashCommand: "/usage",
			Description:  "Show session cost, plan usage, and activity stats",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowUsageDialogMsg{})
			},
		},
		{
			ID:           "session.help",
			Label:        "Help",
			SlashCommand: "/help",
			Description:  "Show help and available commands",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowHelpMsg{})
			},
		},
		{
			ID:           "session.exit",
			Label:        "Exit",
			SlashCommand: "/exit",
			Description:  "Exit the CLI",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ExitSessionMsg{})
			},
		},
		{
			ID:           "session.export",
			Label:        "Export",
			SlashCommand: "/export",
			Description:  "Export the current conversation to a file or clipboard",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowExportDialogMsg{})
			},
		},
		{
			ID:           "session.quit",
			Label:        "Quit",
			SlashCommand: "/quit",
			Description:  "Quit the application (alias for /exit)",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ExitSessionMsg{})
			},
		},
		{
			ID:           "session.q",
			Label:        "Quit",
			SlashCommand: "/q",
			Hidden:       true,
			Description:  "Quit the application (alias for /exit)",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ExitSessionMsg{})
			},
		},
		{
			ID:           "session.model",
			Label:        "Model",
			SlashCommand: "/model",
			Description:  "Change the model for the current agent",
			Category:     "Session",
			Immediate:    true,
			Execute: func(arg string) tea.Cmd {
				// `/model <name>` switches directly via the gateway's
				// switch_model command, which is always available with a
				// remote runtime. Only the bare `/model` (no argument) needs
				// the local picker, which depends on the gateway having
				// advertised a model list in session_ready.
				if name := strings.TrimSpace(arg); name != "" {
					return core.CmdHandler(messages.ChangeModelMsg{ModelRef: name})
				}
				return core.CmdHandler(messages.OpenModelPickerMsg{ShowTranscript: true})
			},
		},
		{
			ID:           "session.effort",
			Label:        "Effort",
			SlashCommand: "/effort",
			Description:  "Set effort level for model usage",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.OpenEffortPickerMsg{ShowTranscript: true})
			},
		},
		{
			ID:           "session.new",
			Label:        "New",
			SlashCommand: "/new",
			Description:  "Start a new conversation",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.NewSessionMsg{})
			},
		},
		{
			ID:           "session.permissions",
			Label:        "Permissions",
			SlashCommand: "/permissions",
			Description:  "Show tool permission rules and workspace policy",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowPermissionsDialogMsg{})
			},
		},
		{
			ID:           "session.resume",
			Label:        "Resume",
			SlashCommand: "/resume",
			Description:  "Resume a previous session",
			Category:     "Session",
			Immediate:    true,
			Execute: func(arg string) tea.Cmd {
				if name := strings.TrimSpace(arg); name != "" {
					return core.CmdHandler(messages.SendMsg{Content: "/resume " + name, BypassQueue: true})
				}
				return core.CmdHandler(messages.OpenSessionBrowserMsg{})
			},
		},
		{
			ID:           "session.shell",
			Label:        "Shell",
			SlashCommand: "/shell",
			Description:  "Start a shell",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.StartShellMsg{})
			},
		},
		{
			ID:           "session.status",
			Label:        "Status",
			SlashCommand: "/status",
			Description:  "Show Claude Code status including version, model, account, API connectivity, and tool statuses",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowStatusMsg{})
			},
		},

		{
			ID:           "session.tools",
			Label:        "Tools",
			SlashCommand: "/tools",
			Description:  "Claude Code compatibility: /tools is not a command",
			Category:     "Session",
			Hidden:       true,
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.UnknownCommandMsg{
					Command: "/tools",
				})
			},
		},
		{
			ID:           "session.skills",
			Label:        "Skills",
			SlashCommand: "/skills",
			Description:  "List available skills",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowSkillsDialogMsg{})
			},
		},
		{
			ID:           "session.yolo",
			Label:        "Yolo",
			SlashCommand: "/yolo",
			Description:  "Toggle automatic approval of tool calls",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ToggleYoloMsg{})
			},
		},
	}

	// Add speak command on supported platforms (macOS only)
	if speak := speakCommand(); speak != nil {
		cmds = append(cmds, *speak)
	}

	return cmds
}

func builtInSettingsCommands() []Item {
	return []Item{
		{
			ID:           "settings.split-diff",
			Label:        "Split Diff",
			SlashCommand: "/split-diff",
			Description:  "Toggle split diff view mode",
			Category:     "Settings",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ToggleSplitDiffMsg{})
			},
		},
		{
			ID:           "settings.theme",
			Label:        "Theme",
			SlashCommand: "/theme",
			Description:  "Change the color theme",
			Category:     "Settings",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.OpenThemePickerMsg{})
			},
		},
	}
}

// visibleOnly returns items that are not hidden.
func visibleOnly(items []Item) []Item {
	visible := make([]Item, 0, len(items))
	for _, item := range items {
		if !item.Hidden {
			visible = append(visible, item)
		}
	}
	return visible
}

// sortByLabel returns items sorted alphabetically by label.
func sortByLabel(items []Item) []Item {
	slices.SortFunc(items, func(a, b Item) int {
		return strings.Compare(strings.ToLower(a.Label), strings.ToLower(b.Label))
	})
	return items
}

// snapshotCommandIDs is the set of IDs that depend on the snapshot feature.
// They are stripped from the palette and the slash-command parser when
// snapshots are turned off.
var snapshotCommandIDs = map[string]bool{
	"session.undo":      true,
	"session.snapshots": true,
}

// removeByIDs returns items whose IDs are not in ids.
func removeByIDs(items []Item, ids map[string]bool) []Item {
	out := make([]Item, 0, len(items))
	for _, item := range items {
		if !ids[item.ID] {
			out = append(out, item)
		}
	}
	return out
}

// BuildCommandCategories builds the list of command categories for the command palette
func BuildCommandCategories(ctx context.Context, application *app.App) []Category {
	// Get session commands and filter based on model capabilities
	sessionCommands := builtInSessionCommands()
	if !application.SnapshotsEnabled() {
		sessionCommands = removeByIDs(sessionCommands, snapshotCommandIDs)
	}

	categories := []Category{
		{
			Name:     "Session",
			Commands: sessionCommands,
		},
	}

	agentCommands := application.CurrentAgentCommands(ctx)
	if len(agentCommands) > 0 {
		var commands []Item
		names := make([]string, 0, len(agentCommands))
		for name := range agentCommands {
			names = append(names, name)
		}
		slices.Sort(names)
		for _, name := range names {
			cmd := agentCommands[name]
			commandName := name
			description := toolcommon.TruncateText(cmd.DisplayText(), 60)
			if description == "" {
				description = "/" + commandName
			}
			commands = append(commands, Item{
				ID:           "agent.command." + commandName,
				Label:        commandName,
				Description:  description,
				Category:     "Agent Commands",
				SlashCommand: "/" + commandName,
				Immediate:    true,
				Execute: func(arg string) tea.Cmd {
					input := "/" + commandName
					if arg = strings.TrimSpace(arg); arg != "" {
						input += " " + arg
					}
					return core.CmdHandler(messages.AgentCommandMsg{Command: input})
				},
			})
		}

		categories = append(categories, Category{
			Name:     "Agent Commands",
			Commands: commands,
		})
	}

	// Add skill commands if skills are enabled for the current agent
	skillsList := application.CurrentAgentSkills()
	if len(skillsList) > 0 {
		skillCommands := make([]Item, 0, len(skillsList))
		for _, skill := range skillsList {
			skillName := skill.Name
			description := toolcommon.TruncateText(skill.Description, 55)

			skillCommands = append(skillCommands, Item{
				ID:           "skill." + skillName,
				Label:        skillName,
				Description:  description,
				Category:     "Skills",
				SlashCommand: "/" + skillName,
				Immediate:    true,
				Execute: func(arg string) tea.Cmd {
					input := "/" + skillName
					if arg = strings.TrimSpace(arg); arg != "" {
						input += " " + arg
					}
					return core.CmdHandler(messages.SendMsg{Content: input, BypassQueue: true})
				},
			})
		}

		categories = append(categories, Category{
			Name:     "Skills",
			Commands: skillCommands,
		})
	}

	categories = append(categories, Category{
		Name:     "Settings",
		Commands: builtInSettingsCommands(),
	})

	// Keep hidden commands in the parser while palette/completion renderers
	// filter them out at presentation time.
	for i := range categories {
		categories[i].Commands = sortByLabel(categories[i].Commands)
	}

	return categories
}

type Parser struct {
	categories []Category
}

func NewParser(categories ...Category) *Parser {
	return &Parser{
		categories: categories,
	}
}

func (p *Parser) Parse(input string) tea.Cmd {
	return p.parse(input, false)
}

func (p *Parser) ParseUnknown(input string) tea.Cmd {
	cmd, _, ok := splitSlashCommand(input)
	if !ok || cmd == "/" || p.hasSlashCommand(cmd) {
		return nil
	}
	return core.CmdHandler(messages.UnknownCommandMsg{
		Command:    cmd,
		Suggestion: p.unknownCommandSuggestion(cmd),
	})
}

func (p *Parser) ParseHidden(input string) tea.Cmd {
	return p.parse(input, true)
}

func (p *Parser) parse(input string, hiddenOnly bool) tea.Cmd {
	cmd, arg, ok := splitSlashCommand(input)
	if !ok {
		return nil
	}

	// Search through all categories and commands
	for _, category := range p.categories {
		for _, item := range category.Commands {
			if hiddenOnly && !item.Hidden {
				continue
			}
			if item.SlashCommand == cmd && item.Immediate {
				return item.Execute(arg)
			}
		}
	}

	return nil
}

func (p *Parser) hasSlashCommand(cmd string) bool {
	for _, category := range p.categories {
		for _, item := range category.Commands {
			if item.SlashCommand == cmd {
				return true
			}
		}
	}
	return false
}

func splitSlashCommand(input string) (cmd string, arg string, ok bool) {
	if input == "" || input[0] != '/' {
		return "", "", false
	}
	cmd, arg, _ = strings.Cut(input, " ")
	return cmd, arg, true
}

func (p *Parser) unknownCommandSuggestion(cmd string) string {
	query := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(cmd)), "/")
	if len(query) < 3 {
		return ""
	}

	bestCommand := ""
	bestDistance := 3
	bestLength := 0
	for _, category := range p.categories {
		for _, item := range category.Commands {
			if item.SlashCommand == "" {
				continue
			}
			candidate := strings.TrimPrefix(strings.ToLower(item.SlashCommand), "/")
			distance := boundedCommandDistance(query, candidate, bestDistance)
			if distance > 2 {
				continue
			}
			if bestCommand == "" ||
				distance < bestDistance ||
				(distance == bestDistance && len(candidate) < bestLength) {
				bestCommand = item.SlashCommand
				bestDistance = distance
				bestLength = len(candidate)
			}
		}
	}
	return bestCommand
}

func boundedCommandDistance(a, b string, maxDistance int) int {
	if a == b {
		return 0
	}
	if commandAbs(len(a)-len(b)) > maxDistance {
		return maxDistance + 1
	}

	ar := []rune(a)
	br := []rune(b)
	prevPrev := make([]int, len(br)+1)
	prev := make([]int, len(br)+1)
	curr := make([]int, len(br)+1)
	for j := range prev {
		prev[j] = j
	}

	for i := 1; i <= len(ar); i++ {
		curr[0] = i
		rowBest := curr[0]
		for j := 1; j <= len(br); j++ {
			cost := 0
			if ar[i-1] != br[j-1] {
				cost = 1
			}
			best := min(
				prev[j]+1,
				curr[j-1]+1,
				prev[j-1]+cost,
			)
			if i > 1 && j > 1 && ar[i-1] == br[j-2] && ar[i-2] == br[j-1] {
				best = min(best, prevPrev[j-2]+1)
			}
			curr[j] = best
			rowBest = min(rowBest, best)
		}
		if rowBest > maxDistance {
			return maxDistance + 1
		}
		prevPrev, prev, curr = prev, curr, prevPrev
	}

	return prev[len(br)]
}

func commandAbs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
