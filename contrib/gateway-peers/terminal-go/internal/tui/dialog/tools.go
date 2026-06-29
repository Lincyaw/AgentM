package dialog

import (
	"fmt"
	"slices"
	"strings"

	"charm.land/lipgloss/v2"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/tools"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

// toolsDialog renders the tools advertised by the gateway.
type toolsDialog struct {
	readOnlyScrollDialog

	tools []tools.Tool
}

// NewToolsDialog creates the /tools dialog from the gateway capability view.
func NewToolsDialog(toolList []tools.Tool) Dialog {
	// Sort tools by category then name.
	sortedTools := make([]tools.Tool, len(toolList))
	copy(sortedTools, toolList)
	slices.SortFunc(sortedTools, func(a, b tools.Tool) int {
		if c := strings.Compare(strings.ToLower(a.Category), strings.ToLower(b.Category)); c != 0 {
			return c
		}
		return strings.Compare(strings.ToLower(a.DisplayName()), strings.ToLower(b.DisplayName()))
	})

	d := &toolsDialog{
		tools: sortedTools,
	}
	d.readOnlyScrollDialog = newReadOnlyScrollDialog(
		readOnlyScrollDialogSize{widthPercent: 70, minWidth: 60, maxWidth: 120, heightPercent: 80, heightMax: 40},
		d.renderLines,
	)
	return d
}

func (d *toolsDialog) renderLines(contentWidth, _ int) []string {
	title := fmt.Sprintf("Tools (%d)", len(d.tools))
	lines := []string{
		RenderTitle(title, contentWidth, styles.DialogTitleStyle),
		RenderSeparator(contentWidth),
		"",
	}

	lines = append(lines, d.renderTools(contentWidth)...)

	return lines
}

func (d *toolsDialog) renderTools(contentWidth int) []string {
	out := []string{sectionHeader("Tools"), ""}

	if len(d.tools) == 0 {
		out = append(out, "  "+styles.MutedStyle.Render("No tools available."), "")
		return out
	}

	var lastCategory string
	for i := range d.tools {
		t := &d.tools[i]
		cat := t.Category
		if cat == "" {
			cat = "Other"
		}
		if cat != lastCategory {
			if lastCategory != "" {
				out = append(out, "")
			}
			out = append(out, "  "+lipgloss.NewStyle().Bold(true).Foreground(styles.TextSecondary).Render(cat))
			lastCategory = cat
		}

		name := lipgloss.NewStyle().Foreground(styles.Highlight).Render("    " + t.DisplayName())
		if desc, _, _ := strings.Cut(t.Description, "\n"); desc != "" {
			separator := " • "
			separatorWidth := lipgloss.Width(separator)
			nameWidth := lipgloss.Width(name)
			availableWidth := contentWidth - nameWidth - separatorWidth
			if availableWidth > 0 {
				truncated := toolcommon.TruncateText(desc, availableWidth)
				name += styles.MutedStyle.Render(separator + truncated)
			}
		}
		out = append(out, name)
	}
	out = append(out, "")

	return out
}

// sectionHeader returns the styled top-level section header used inside
// the dialog ("Toolsets", "Tools"). Kept private to make the dialog
// layout self-contained.
func sectionHeader(label string) string {
	return lipgloss.NewStyle().Bold(true).Foreground(styles.TextSecondary).Render(label)
}
