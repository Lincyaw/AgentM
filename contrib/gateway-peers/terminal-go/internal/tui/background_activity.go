package tui

import (
	"fmt"
	"slices"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
)

const (
	// Render a short bounded surface while retaining a larger recent set for a
	// useful hidden count without unbounded stale rows in long terminal sessions.
	maxVisibleBackgroundActivities    = 4
	maxRetainedBackgroundActivities   = 32
	backgroundActivityKeySeparator    = "\x00"
	backgroundActivityStatusRunning   = "running"
	backgroundActivityStatusError     = "error"
	backgroundActivityStatusFailed    = "failed"
	backgroundActivityStatusCanceled  = "canceled"
	backgroundActivityStatusCancelled = "cancelled"
)

type backgroundActivity struct {
	sessionID string
	source    string
	id        string
	label     string
	status    string
	note      string
	startedAt time.Time
	updatedAt time.Time
	finished  bool
}

type normalizedBackgroundActivity struct {
	sessionID  string
	activityID string
	source     string
	label      string
	status     string
	note       string
}

func backgroundActivityKey(sessionID, activityID string) string {
	if sessionID == "" {
		return activityID
	}
	return sessionID + backgroundActivityKeySeparator + activityID
}

func normalizeBackgroundActivityStatus(status string) string {
	status = strings.ToLower(strings.TrimSpace(status))
	if status == "" {
		return backgroundActivityStatusRunning
	}
	return status
}

func retainFinishedBackgroundActivity(status string) bool {
	switch status {
	case backgroundActivityStatusError,
		backgroundActivityStatusFailed:
		return true
	default:
		return false
	}
}

func (m *appModel) backgroundActivityCountText() string {
	total := len(m.backgroundActivities)
	if total == 0 {
		return ""
	}
	sourceCounts := map[string]int{}
	for _, activity := range m.backgroundActivities {
		sourceCounts[cmpNonEmpty(activity.source, "background")]++
	}
	backgroundCount := sourceCounts["background"]
	monitorCount := sourceCounts["monitor"]
	otherCount := total - backgroundCount - monitorCount

	parts := make([]string, 0, 3)
	if backgroundCount > 0 {
		parts = append(parts, backgroundStatusCountLabel(backgroundCount, "background task", "background tasks"))
	}
	if monitorCount > 0 {
		parts = append(parts, backgroundStatusCountLabel(monitorCount, "monitor", "monitors"))
	}
	if otherCount > 0 {
		parts = append(parts, backgroundStatusCountLabel(otherCount, "activity", "activities"))
	}
	return strings.Join(parts, " · ")
}

func (m *appModel) backgroundShellCount() int {
	count := 0
	for _, activity := range m.backgroundActivities {
		if activity.finished {
			continue
		}
		if activity.source == "background" && strings.HasPrefix(activity.label, "bash:") {
			count++
		}
	}
	return count
}

func (m *appModel) backgroundShellText() string {
	count := m.backgroundShellCount()
	switch count {
	case 0:
		return ""
	case 1:
		return "1 shell"
	default:
		return fmt.Sprintf("%d shells", count)
	}
}

func isBackgroundShellActivity(activity backgroundActivity) bool {
	return !activity.finished && activity.source == "background" && strings.HasPrefix(activity.label, "bash:")
}

func (m *appModel) selectedBackgroundShellActivity() (backgroundActivity, bool) {
	var selected backgroundActivity
	for _, activity := range m.sortedBackgroundActivities() {
		if !isBackgroundShellActivity(activity) {
			continue
		}
		selected = activity
	}
	return selected, selected.id != ""
}

func backgroundShellTaskID(activity backgroundActivity) string {
	return strings.TrimPrefix(activity.id, "background:")
}

func backgroundStatusCountLabel(count int, singular, plural string) string {
	if count == 1 {
		return fmt.Sprintf("1 %s", singular)
	}
	return fmt.Sprintf("%d %s", count, plural)
}

func joinBackgroundStatusParts(parts ...string) string {
	nonEmpty := make([]string, 0, len(parts))
	for _, part := range parts {
		if strings.TrimSpace(part) != "" {
			nonEmpty = append(nonEmpty, part)
		}
	}
	return strings.Join(nonEmpty, " · ")
}

func normalizeBackgroundActivity(ev *runtime.BackgroundActivityEvent) (normalizedBackgroundActivity, bool) {
	source := strings.TrimSpace(ev.Source)
	label := strings.TrimSpace(ev.Label)
	activityID := strings.TrimSpace(ev.ActivityID)
	if activityID == "" {
		activityID = source + ":" + label
	}
	if activityID == ":" {
		return normalizedBackgroundActivity{}, false
	}
	return normalizedBackgroundActivity{
		sessionID:  strings.TrimSpace(ev.SessionID),
		activityID: activityID,
		source:     cmpNonEmpty(source, "background"),
		label:      cmpNonEmpty(label, "background"),
		status:     normalizeBackgroundActivityStatus(ev.Status),
		note:       strings.TrimSpace(ev.Note),
	}, true
}

func (m *appModel) handleBackgroundActivity(ev *runtime.BackgroundActivityEvent) (tea.Model, tea.Cmd) {
	if ev == nil {
		return m, nil
	}
	activityChanged := m.updateBackgroundActivity(ev)
	if m.backgroundShellCount() == 0 {
		m.backgroundActivityPrompt = false
		m.backgroundActivityDetail = false
	}
	m.statusBar.SetActivity(m.backgroundActivityText())
	if !activityChanged {
		return m, nil
	}
	return m, m.resizeAllIfBottomSurfaceChanged()
}

func (m *appModel) updateBackgroundActivity(ev *runtime.BackgroundActivityEvent) bool {
	if m.backgroundActivities == nil {
		m.backgroundActivities = map[string]backgroundActivity{}
	}
	activity, ok := normalizeBackgroundActivity(ev)
	if !ok {
		return false
	}
	key := backgroundActivityKey(activity.sessionID, activity.activityID)
	if ev.Terminal && !retainFinishedBackgroundActivity(activity.status) {
		_, existed := m.backgroundActivities[key]
		delete(m.backgroundActivities, key)
		return existed
	}
	now := time.Now()
	previous, existed := m.backgroundActivities[key]
	startedAt := now
	if existed && !previous.startedAt.IsZero() {
		startedAt = previous.startedAt
	}
	next := backgroundActivity{
		sessionID: activity.sessionID,
		source:    activity.source,
		id:        activity.activityID,
		label:     activity.label,
		status:    activity.status,
		note:      activity.note,
		startedAt: startedAt,
		updatedAt: now,
		finished:  ev.Terminal,
	}
	m.backgroundActivities[key] = next
	m.pruneBackgroundActivities()
	// Existing activity rows keep the same bottom-surface height; repaint is enough.
	return !existed
}

func (m *appModel) pruneBackgroundActivities() {
	if len(m.backgroundActivities) <= maxRetainedBackgroundActivities {
		return
	}
	for _, activity := range m.sortedBackgroundActivities() {
		if len(m.backgroundActivities) <= maxRetainedBackgroundActivities {
			return
		}
		if !activity.finished {
			continue
		}
		delete(m.backgroundActivities, backgroundActivityKey(activity.sessionID, activity.id))
	}
}

func (m *appModel) removeBackgroundActivitiesForSession(sessionID string) {
	if sessionID == "" || len(m.backgroundActivities) == 0 {
		return
	}
	for id, activity := range m.backgroundActivities {
		if activity.sessionID == sessionID {
			delete(m.backgroundActivities, id)
		}
	}
}

func (m *appModel) sortedBackgroundActivities() []backgroundActivity {
	if len(m.backgroundActivities) == 0 {
		return nil
	}
	rows := make([]backgroundActivity, 0, len(m.backgroundActivities))
	for _, activity := range m.backgroundActivities {
		rows = append(rows, activity)
	}
	slices.SortFunc(rows, func(a, b backgroundActivity) int {
		if cmp := a.updatedAt.Compare(b.updatedAt); cmp != 0 {
			return cmp
		}
		return strings.Compare(a.id, b.id)
	})
	return rows
}

func (m *appModel) backgroundActivityRows() ([]backgroundActivity, int) {
	rows := m.sortedBackgroundActivities()
	if len(rows) == 0 {
		return nil, 0
	}
	if len(rows) <= maxVisibleBackgroundActivities {
		return rows, 0
	}
	hidden := len(rows) - maxVisibleBackgroundActivities
	return rows[hidden:], hidden
}

func (m *appModel) renderBackgroundActivityRow(row backgroundActivity, width int) string {
	prefix := "  "
	icon := "◯"
	if row.finished {
		icon = "✻"
	}
	left := fmt.Sprintf("%s%s %-16s %s", prefix, icon, cmpNonEmpty(row.source, "background"), cmpNonEmpty(row.label, "task"))
	status := cmpNonEmpty(row.status, backgroundActivityStatusRunning)
	right := status
	if row.note != "" {
		right = status + " · " + row.note
	}
	age := formatWorkflowAge(row.updatedAt)
	if age != "" {
		right += " · " + age
	}
	gap := max(1, width-lipgloss.Width(left)-lipgloss.Width(right))
	line := left + strings.Repeat(" ", gap) + styles.MutedStyle.Render(right)
	if lipgloss.Width(line) > width {
		line = ansi.Truncate(line, width, "…")
	}
	if row.finished {
		return styles.SecondaryStyle.Render(line)
	}
	return styles.MutedStyle.Render(line)
}

func (m *appModel) renderBackgroundShellDetails(width int) string {
	activity, ok := m.selectedBackgroundShellActivity()
	if !ok {
		return ""
	}
	innerWidth := max(20, width-appPaddingHorizontal)
	command := strings.TrimSpace(strings.TrimPrefix(activity.label, "bash:"))
	if command == "" {
		command = "bash"
	}
	runtimeText := formatWorkflowAge(activity.startedAt)
	if runtimeText == "" {
		runtimeText = "0s"
	}
	status := cmpNonEmpty(activity.status, backgroundActivityStatusRunning)
	bodyLines := []string{
		"Shell details",
		"",
		fmt.Sprintf("Status:   %s", status),
		fmt.Sprintf("Runtime:  %s", runtimeText),
		"Command:  " + command,
		"",
		"Output:",
		"No output available",
		"",
	}
	lines := make([]string, 0, len(bodyLines))
	for _, line := range bodyLines {
		lines = append(lines, styles.SecondaryStyle.Render(ansi.Truncate(line, innerWidth, "…")))
	}
	return lipgloss.NewStyle().Padding(0, styles.AppPadding).Render(strings.Join(lines, "\n"))
}

func renderHiddenBackgroundActivitiesRow(count, width int) string {
	noun := "activity"
	if count != 1 {
		noun = "activities"
	}
	line := fmt.Sprintf("  + %d older background %s", count, noun)
	if lipgloss.Width(line) > width {
		line = ansi.Truncate(line, width, "…")
	}
	return styles.MutedStyle.Render(line)
}
