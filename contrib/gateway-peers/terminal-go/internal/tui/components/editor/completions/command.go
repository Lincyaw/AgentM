package completions

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/commands"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/completion"
)

type commandCompletion struct {
	categories []commands.Category
}

func NewCommandCompletion(categories []commands.Category) Completion {
	return &commandCompletion{
		categories: categories,
	}
}

func (c *commandCompletion) RequiresEmptyEditor() bool {
	return true
}

func (c *commandCompletion) Trigger() string {
	return "/"
}

func (c *commandCompletion) Items() []completion.Item {
	available := make(map[string]bool)
	for _, cmd := range c.categories {
		for _, command := range cmd.Commands {
			if command.Hidden {
				continue
			}
			available[command.SlashCommand] = true
		}
	}

	items := claudeCompatibleCommandItems(available)
	seen := make(map[string]bool, len(items))
	for _, item := range items {
		seen[item.Value] = true
	}

	for _, cmd := range c.categories {
		for _, command := range cmd.Commands {
			if command.Hidden || seen[command.SlashCommand] || hiddenClaudeCommandCompletion(command.SlashCommand) {
				continue
			}
			item := completion.Item{
				Label:       command.SlashCommand,
				Description: command.Description,
				Value:       command.SlashCommand,
			}
			items = append(items, item)
			seen[item.Value] = true
		}
	}

	return items
}

func hiddenClaudeCommandCompletion(command string) bool {
	switch command {
	case "/attach":
		return true
	default:
		return false
	}
}

func claudeCompatibleCommandItems(available map[string]bool) []completion.Item {
	specs := []struct {
		name        string
		label       string
		description string
		descPrefix  string
		path        string
		prefix      string
		search      string
	}{
		{
			name:        "init",
			description: "Initialize a new CLAUDE.md file with codebase documentation",
		},
		{
			name:        "paper-review",
			description: "Run the paper-writing three-pass reading pipeline and produce Reading Notes plus a final prioritized review report.",
			descPrefix:  "(autoharness) ",
			path:        filepath.Join(userHomeDir(), ".codex", "skills", "paper-review", "SKILL.md"),
			prefix:      "autoharness:",
		},
		{
			name: "rca-failure-analysis",
			path: filepath.Join(userHomeDir(), ".agents", "skills", "rca-failure-analysis", "SKILL.md"),
		},
		{
			name: "lark-doc",
			path: filepath.Join(userHomeDir(), ".agents", "skills", "lark-doc", "SKILL.md"),
		},
		{
			name:   "lark-wiki",
			path:   filepath.Join(userHomeDir(), ".agents", "skills", "lark-wiki", "SKILL.md"),
			search: "h",
		},
		{
			name:        "agents",
			description: "(removed) Ask Claude to create/manage subagents, or edit .claude/agents/",
		},
		{
			name:        "background",
			description: "Send this session to the background and free the terminal",
		},
		{
			name:        "branch",
			description: "Create a branch of the current conversation at this point",
		},
		{
			name:        "cd",
			description: "Move this session to a new working directory",
		},
		{
			name:        "clear",
			description: "Start a new session with empty context; previous session stays on disk (resumable with /resume)",
		},
		{
			name:        "color",
			description: "Set the prompt bar color for this session",
		},
		{
			name:        "compact",
			description: "Free up context by summarizing the conversation so far",
		},
		{
			name:        "config",
			label:       "/config (settings)",
			description: "Open settings",
			search:      "settings",
		},
		{
			name:        "context",
			description: "Visualize current context usage as a colored grid",
		},
		{
			name:        "model",
			description: "Set the AI model for Claude Code (currently Opus 4.8 (1M context))",
		},
		{
			name:        "claude-api",
			description: "Reference for the Claude API / Anthropic SDK — model ids, pricing, params, streaming, tool use, MCP, agents, caching, token counting, model migration. TRIGGER when configuring Claude API calls.",
			search:      "mo model",
		},
		{
			name:        "agentm-sdk",
			description: "AgentM SDK development guide — manifest-as-agent-unit philosophy, SDK programmatic invocation, dynamic workflow orchestration, atom contract, Operations abstraction, event system, service communication, CLI conventions, scenario authoring, logging, structured output, and config resolution.",
			path:        filepath.Join(userHomeDir(), "workspace", "AgentM", ".agents", "skills", "agentm-sdk", "SKILL.md"),
			search:      "mo model",
		},
		{
			name:       "grill",
			descPrefix: "(autoharness) ",
			path:       filepath.Join(userHomeDir(), ".codex", "skills", "grill", "SKILL.md"),
			search:     "mo model",
		},
		{
			name:        "effort",
			description: "Set effort level for model usage",
			search:      "mo model",
		},
		{
			name:        "status",
			description: "Show Claude Code status including version, model, account, API connectivity, and tool statuses",
			search:      "mo model",
		},
		{
			name:   "deployment-awareness",
			path:   filepath.Join(userHomeDir(), "workspace", "AgentM", ".agents", "skills", "deployment-awareness", "SKILL.md"),
			search: "mo model",
		},
		{
			name:        "update-config",
			description: `Use this skill to configure the Claude Code harness via settings.json. Automated behaviors ("from now on when X", "each time X", "whenever X", "before/after X") must be encoded as settings, not prompt memory.`,
			search:      "mo model",
		},
		{
			name:   "verifier-result-analysis",
			path:   filepath.Join(userHomeDir(), "workspace", "AgentM", ".agents", "skills", "verifier-result-analysis", "SKILL.md"),
			search: "mo model",
		},
		{
			name:        "fast",
			description: "Toggle fast mode (Opus 4.8)",
			search:      "mo model",
		},
		{
			name:        "plan",
			description: "Enable plan mode or view the current session plan",
			search:      "mo model",
		},
		{
			name:        "mobile",
			description: "Show QR code to download the Claude mobile app",
		},
		{
			name:        "skills",
			description: "List available skills",
		},
		{
			name:       "skill-evolve",
			descPrefix: "(autoharness) ",
			path:       filepath.Join(userHomeDir(), ".codex", "skills", "skill-evolve", "SKILL.md"),
		},
		{
			name:        "skill-creator",
			description: "Create new skills, modify and improve existing skills, and measure skill performance.",
			descPrefix:  "(skill-creator) ",
			path:        filepath.Join(userHomeDir(), ".codex", "plugins", "cache", "claude-plugins-official", "skill-creator", "local", "skills", "skill-creator", "SKILL.md"),
		},
		{
			name:       "skill-feedback",
			descPrefix: "(autoharness) ",
			path:       filepath.Join(userHomeDir(), ".codex", "skills", "skill-feedback", "SKILL.md"),
		},
		{
			name:        "reload-skills",
			description: "Pick up skills added or changed on disk during this session",
		},
		{
			name:       "long-horizon",
			descPrefix: "(autoharness) ",
			path:       filepath.Join(userHomeDir(), ".codex", "skills", "long-horizon", "SKILL.md"),
			search:     "mo",
		},
	}

	items := make([]completion.Item, 0, len(specs))
	for _, spec := range specs {
		name := spec.name
		description := spec.description
		if spec.path != "" {
			if parsedName, parsedDescription := parseAgentMetadata(spec.path); parsedName != "" || parsedDescription != "" {
				if parsedName != "" {
					name = parsedName
				}
				if description == "" && parsedDescription != "" {
					description = parsedDescription
				}
			}
		}
		if spec.descPrefix != "" && description != "" && !strings.HasPrefix(description, spec.descPrefix) {
			description = spec.descPrefix + description
		}
		name = spec.prefix + name
		value := "/" + name
		if !available[value] {
			continue
		}
		label := value
		if spec.label != "" {
			label = spec.label
		}
		items = append(items, completion.Item{
			Label:       label,
			Description: description,
			Value:       value,
			SearchText:  spec.search,
		})
	}
	return items
}

func userHomeDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return home
}

func (c *commandCompletion) MatchMode() completion.MatchMode {
	return completion.MatchFuzzyPrefixPriority
}
