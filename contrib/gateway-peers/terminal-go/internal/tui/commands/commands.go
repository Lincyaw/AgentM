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
			ID:           "session.clear",
			Label:        "Clear",
			SlashCommand: "/clear",
			Description:  "Clear the current tab and start a new session",
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
			Description:  "Summarize the current conversation (usage: /compact [additional instructions])",
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
			Description:  "Copy the current conversation to the clipboard",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.CopySessionToClipboardMsg{})
			},
		},
		{
			ID:           "session.copy_last_response",
			Label:        "Copy Last Response",
			SlashCommand: "/copy-last",
			Description:  "Copy the last assistant message to the clipboard",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.CopyLastResponseToClipboardMsg{})
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
			ID:           "session.exit",
			Label:        "Exit",
			SlashCommand: "/exit",
			Description:  "Exit the application",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ExitSessionMsg{})
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
				return core.CmdHandler(messages.OpenModelPickerMsg{})
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
			ID:           "session.star",
			Label:        "Star",
			SlashCommand: "/star",
			Description:  "Toggle star on current session",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ToggleSessionStarMsg{})
			},
		},

		{
			ID:           "session.tools",
			Label:        "Tools",
			SlashCommand: "/tools",
			Description:  "Show tools advertised by the gateway",
			Category:     "Session",
			Immediate:    true,
			Execute: func(string) tea.Cmd {
				return core.CmdHandler(messages.ShowToolsDialogMsg{})
			},
		},
		{
			ID:           "session.skills",
			Label:        "Skills",
			SlashCommand: "/skills",
			Description:  "List skills available to the current agent",
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

	// Filter out hidden commands and sort by label in all categories.
	for i := range categories {
		categories[i].Commands = sortByLabel(visibleOnly(categories[i].Commands))
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
	if input == "" || input[0] != '/' {
		return nil
	}

	// Split into command and argument
	cmd, arg, _ := strings.Cut(input, " ")

	// Search through all categories and commands
	for _, category := range p.categories {
		for _, item := range category.Commands {
			if item.SlashCommand == cmd && item.Immediate {
				return item.Execute(arg)
			}
		}
	}

	return nil
}
