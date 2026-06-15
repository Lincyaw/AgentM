package dialog

import (
	"fmt"
	"os"
	"strings"

	"charm.land/lipgloss/v2"

	pathx "github.com/AoyangSpace/agentm-terminal/internal/cagent/path"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/skills"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/toolcommon"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

type skillsDialog struct {
	readOnlyScrollDialog

	skills []skills.Skill
}

// NewSkillsDialog creates the /skills dialog showing every skill exposed
// to the current agent.
func NewSkillsDialog(skillList []skills.Skill) Dialog {
	d := &skillsDialog{
		skills: skillList,
	}
	d.readOnlyScrollDialog = newReadOnlyScrollDialog(
		readOnlyScrollDialogSize{widthPercent: 70, minWidth: 60, maxWidth: 120, heightPercent: 80, heightMax: 40},
		d.renderLines,
	)
	return d
}

func (d *skillsDialog) renderLines(contentWidth, _ int) []string {
	title := fmt.Sprintf("Skills (%d)", len(d.skills))
	lines := []string{
		RenderTitle(title, contentWidth, styles.DialogTitleStyle),
		RenderSeparator(contentWidth),
		"",
	}

	if len(d.skills) == 0 {
		lines = append(lines, "  "+styles.MutedStyle.Render("No skills available."), "")
		return lines
	}

	for i := range d.skills {
		lines = append(lines, formatSkill(&d.skills[i], contentWidth)...)
	}

	return lines
}

func formatSkill(s *skills.Skill, contentWidth int) []string {
	name := lipgloss.NewStyle().Foreground(styles.Highlight).Render("  " + s.Name)
	name += " " + skillSourceBadge(s)
	if s.IsFork() {
		name += " " + styles.MutedStyle.Render("[fork]")
	}

	out := []string{name}

	if desc, _, _ := strings.Cut(s.Description, "\n"); desc != "" {
		indent := "    "
		availableWidth := contentWidth - lipgloss.Width(indent)
		if availableWidth > 0 {
			out = append(out, indent+styles.MutedStyle.Render(toolcommon.TruncateText(desc, availableWidth)))
		}
	}

	if path := skillLoadedFrom(s); path != "" {
		indent := "    "
		prefix := "from: "
		availableWidth := contentWidth - lipgloss.Width(indent) - lipgloss.Width(prefix)
		if availableWidth > 0 {
			out = append(out, indent+styles.MutedStyle.Render(prefix+toolcommon.TruncateText(path, availableWidth)))
		}
	}

	out = append(out, "")
	return out
}

func skillLoadedFrom(s *skills.Skill) string {
	if s.IsInline() {
		return ""
	}

	path := s.BaseDir
	if path == "" {
		path = s.FilePath
	}
	if path == "" {
		return ""
	}

	if s.Local {
		if skills.IsHomeSkillPath(path) {
			return pathx.ShortenHome(path)
		}
		cwd, _ := os.Getwd()
		return pathx.RelativeTo(path, cwd)
	}

	return pathx.ShortenHome(path)
}

func skillSourceBadge(s *skills.Skill) string {
	switch {
	case s.IsInline():
		return styles.MutedStyle.Render("[inline]")
	case s.Local:
		return styles.SuccessStyle.Render("[local]")
	default:
		return styles.WarningStyle.Render("[remote]")
	}
}
