package message

import (
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/markdown"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/components/spinner"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/core/layout"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/styles"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/types"
)

const (
	maxUserMessageLines       = 30
	collapsedUserMessageLines = 5
	claudeActivityLoadingText = "Accomplishing…"
)

var claudeActivityGlyphs = []string{"✻", "✢", "✳", "✶", "✽", "✦", "✧", "·"}

var claudeActivityMessages = []string{
	"Accomplishing",
	"Cascading",
	"Channeling",
	"Computing",
	"Determining",
	"Envisioning",
	"Gusting",
	"Infusing",
	"Manifesting",
	"Nucleating",
	"Tempering",
	"Tinkering",
	"Whisking",
}

// Model represents a view that can render a message
type Model interface {
	layout.Model
	layout.Sizeable
	SetMessage(msg *types.Message)
	SetSelected(selected bool)
	SetHovered(hovered bool)
	CodeBlocks() []markdown.CodeBlock
	// Finalize releases per-message render state that is only needed while the
	// message is actively streaming. The message content and code-block metadata
	// are preserved; calling View() afterwards still produces correct output
	// without retaining a per-view render cache or IncrementalRenderer.
	Finalize()
	// HasLiveRenderState reports whether this view currently retains a
	// populated renderCache or an IncrementalRenderer instance. Used by tests
	// to assert that finalized views have actually released their per-message
	// render state without reaching into unexported fields via reflection.
	HasLiveRenderState() bool
}

// messageModel implements Model
type messageModel struct {
	message  *types.Message
	previous *types.Message

	width    int
	height   int
	focused  bool
	selected bool
	hovered  bool
	expanded bool
	spinner  spinner.Spinner

	// renderCache memoizes the output of Render(width) keyed by the inputs
	// that affect its output. During streaming, View() and Height() are called
	// in pairs for each new chunk, and the chat list also re-renders for hover
	// tracking and scroll updates; without this cache each call would re-parse
	// the entire accumulated markdown from scratch.
	renderCache renderCache

	// codeBlocks holds the fenced code blocks emitted by the last call to
	// render() for assistant messages, with Line indices translated into the
	// messageModel's own View() output coordinate system (i.e. zero-indexed
	// from the first line of View()).
	codeBlocks []markdown.CodeBlock

	// mdRenderer is reused across renders of an assistant message so that
	// streamed-in chunks only re-render the trailing block instead of the whole
	// accumulated markdown each time.
	mdRenderer *markdown.IncrementalRenderer

	// finalized is set by Finalize() once the message is no longer the active
	// streaming view. After it is set, Render() still produces correct output,
	// but does not store anything in renderCache and does not retain an
	// IncrementalRenderer between calls — both are pure caches whose memory
	// dominates a long session, and they are not worth keeping for messages
	// that are unlikely to be re-rendered hot.
	finalized bool
}

// renderCache stores the most recent Render result keyed by the inputs that
// can change its output. The key is small enough (a string and a few flags)
// that comparing it is much cheaper than rendering markdown.
type renderCache struct {
	valid     bool
	content   string
	msgType   types.MessageType
	width     int
	selected  bool
	hovered   bool
	expanded  bool
	sameAgent bool
	result    string
}

// New creates a new message view
func New(msg, previous *types.Message) *messageModel {
	return &messageModel{
		message:  msg,
		previous: previous,
		width:    80, // Default width
		height:   1,  // Will be calculated
		focused:  false,
		spinner:  spinner.New(spinner.ModeBoth, styles.SpinnerDotsAccentStyle),
	}
}

// Bubble Tea Model methods

// Init initializes the message view
func (mv *messageModel) Init() tea.Cmd {
	if mv.message.Type == types.MessageTypeSpinner || mv.message.Type == types.MessageTypeLoading {
		return mv.spinner.Init()
	}
	return nil
}

func (mv *messageModel) SetMessage(msg *types.Message) {
	// Un-finalize when the underlying message is changed (e.g. streaming
	// resumes into this view). Finalize is meant for views that have
	// permanently lost their actively-streaming status; mutating the message
	// re-arms the per-message caches so subsequent renders are fast again.
	mv.finalized = false
	// If the new content is not an extension of the previous one (different
	// message, or the message was edited), drop the IncrementalRenderer's
	// cached prefix so its memory is released immediately rather than on the
	// next render. The renderer detects mismatches on its own and falls back
	// to a full render either way, so this is purely an optimization.
	if mv.mdRenderer != nil && mv.message != nil && msg != nil && !strings.HasPrefix(msg.Content, mv.message.Content) {
		mv.mdRenderer.Reset()
	}
	mv.message = msg
	mv.renderCache.valid = false
}

func (mv *messageModel) SetSelected(selected bool) {
	if mv.selected != selected {
		mv.selected = selected
		mv.renderCache.valid = false
	}
}

func (mv *messageModel) SetHovered(hovered bool) {
	if mv.hovered != hovered {
		mv.hovered = hovered
		mv.renderCache.valid = false
	}
}

// Update handles messages and updates the message view state
func (mv *messageModel) Update(msg tea.Msg) (layout.Model, tea.Cmd) {
	if mv.message.Type == types.MessageTypeSpinner || mv.message.Type == types.MessageTypeLoading {
		s, cmd := mv.spinner.Update(msg)
		mv.spinner = s.(spinner.Spinner)
		return mv, cmd
	}
	return mv, nil
}

// Toggle switches between expanded and collapsed state.
func (mv *messageModel) Toggle() {
	mv.expanded = !mv.expanded
	mv.renderCache.valid = false
}

// IsToggleLine returns true if the line contains the expand/collapse affordance.
func (mv *messageModel) IsToggleLine(lineIdx int) bool {
	if mv.message == nil || mv.message.Type != types.MessageTypeUser {
		return false
	}
	content := strings.TrimRight(mv.message.Content, "\n\r\t ")
	if strings.Count(content, "\n")+1 <= maxUserMessageLines {
		return false
	}

	// The indicator is placed at the end of the message view with a leading \n\n.
	// Depending on edit state, the view has 0 or 1 lines of top padding,
	// and 1 line of bottom padding.
	// height-1 is the bottom padding.
	// height-2 is the text of the indicator ("[-] click to collapse").
	// height-3 is the empty line above it.
	// By checking >= height-3, we provide a generous clickable area exactly on the toggle.
	height := mv.Height(mv.width)
	return lineIdx >= height-3
}

// View renders the message view
func (mv *messageModel) View() string {
	return mv.Render(mv.width)
}

// Render renders the message view content. Results are memoized so repeated
// calls with the same inputs (very common during streaming, hover tracking,
// and from Height()) skip the expensive markdown parse.
func (mv *messageModel) Render(width int) string {
	msg := mv.message

	// Spinner-driven types (MessageTypeSpinner, MessageTypeLoading, and an empty
	// MessageTypeAssistant placeholder) animate on every tick, so the result is
	// not cacheable. Everything else is a pure function of the inputs tracked in
	// renderCache below.
	// Spinner-driven messages animate every tick and are not cacheable.
	// Finalized messages skip writing into renderCache so the per-view
	// retained ANSI string does not pile up across long sessions; the chat
	// list's bounded LRU still memoizes their rendered output.
	cacheable := !mv.isSpinnerDriven() && !mv.finalized
	if cacheable {
		c := &mv.renderCache
		if c.valid &&
			c.width == width &&
			c.msgType == msg.Type &&
			c.selected == mv.selected &&
			c.hovered == mv.hovered &&
			c.expanded == mv.expanded &&
			c.content == msg.Content &&
			c.sameAgent == mv.sameAgentAsPrevious(msg) {
			return c.result
		}
	}

	result := mv.render(width)

	if cacheable {
		mv.renderCache = renderCache{
			valid:     true,
			content:   msg.Content,
			msgType:   msg.Type,
			width:     width,
			selected:  mv.selected,
			hovered:   mv.hovered,
			expanded:  mv.expanded,
			sameAgent: mv.sameAgentAsPrevious(msg),
			result:    result,
		}
	}
	return result
}

// isSpinnerDriven reports whether the rendered output animates on every tick
// and therefore cannot be cached across renders.
func (mv *messageModel) isSpinnerDriven() bool {
	switch mv.message.Type {
	case types.MessageTypeSpinner, types.MessageTypeLoading:
		return true
	case types.MessageTypeAssistant:
		return mv.message.Content == ""
	}
	return false
}

// render is the uncached rendering core. Render() wraps it with memoization.
func (mv *messageModel) render(width int) string {
	msg := mv.message
	switch msg.Type {
	case types.MessageTypeSpinner:
		return "\n" + renderClaudeWorking()
	case types.MessageTypeUser:
		if bang := formatBangTranscript(msg.Content); bang != "" {
			return bang
		}

		formatUserContent := func(c string) string {
			c = strings.TrimRight(c, "\n\r\t ")
			if c == "" {
				return msg.Content
			}

			totalLines := strings.Count(c, "\n") + 1
			if totalLines > maxUserMessageLines {
				if !mv.expanded {
					parts := strings.SplitN(c, "\n", collapsedUserMessageLines+1)
					visibleLines := strings.Join(parts[:collapsedUserMessageLines], "\n")
					hiddenCount := totalLines - collapsedUserMessageLines
					indicator := "\n\n" + styles.MutedStyle.Render(fmt.Sprintf("[+] expand %d more lines", hiddenCount))
					return visibleLines + indicator
				}
				indicator := "\n\n" + styles.MutedStyle.Render("[-] collapse")
				return c + indicator
			}
			return c
		}

		return formatClaudeUserTranscript(formatUserContent(msg.Content), width+4)
	case types.MessageTypeAssistant:
		if msg.Content == "" {
			return "\n" + renderClaudeWorking()
		}

		messageStyle := styles.AssistantMessageStyle
		if mv.selected {
			messageStyle = styles.SelectedMessageStyle
		}

		innerRenderWidth := width - messageStyle.GetHorizontalFrameSize()
		rendered, codeBlocks, err := mv.renderAssistantMarkdown(preserveLineBreaks(msg.Content), innerRenderWidth)
		if err != nil {
			rendered = msg.Content
			codeBlocks = nil
		}

		prefixAssistant := false
		if !mv.sameAgentAsPrevious(msg) {
			prefixAssistant = msg.Sender != ""
		}

		innerWidth := width - messageStyle.GetHorizontalFrameSize()
		body := rendered
		if prefixAssistant {
			body = addClaudeAssistantPrefix(body)
			messageStyle = messageStyle.PaddingLeft(0).BorderLeft(false)
		}
		controlRows := 0
		if mv.hovered || mv.selected {
			copyIcon := styles.MutedStyle.Render(types.AssistantMessageCopyLabel)
			iconWidth := ansi.StringWidth(types.AssistantMessageCopyLabel)
			padding := max(innerWidth-iconWidth, 0)
			topRow := strings.Repeat(" ", padding) + copyIcon
			body = topRow + "\n" + rendered
			controlRows = 1
		}
		leadingRows := 0
		if prefixAssistant && !strings.HasPrefix(body, "\n") {
			body = "\n" + body
			leadingRows = 1
		}

		// Translate the markdown-relative line indices into messageModel View()
		// coordinates. The rendered markdown is preceded by the sender prefix
		// (when shown) and the optional copy-control row, so the first line of
		// `rendered` lands at this offset.
		lineOffset := leadingRows + controlRows
		if len(codeBlocks) > 0 {
			mv.codeBlocks = make([]markdown.CodeBlock, len(codeBlocks))
			for i, cb := range codeBlocks {
				mv.codeBlocks[i] = markdown.CodeBlock{
					Content: cb.Content,
					Line:    cb.Line + lineOffset,
				}
			}
		} else {
			mv.codeBlocks = nil
		}

		return messageStyle.Width(width).Render(body)
	case types.MessageTypeShellOutput:
		if rendered, err := markdown.NewRenderer(width).Render(fmt.Sprintf("```console\n%s\n```", msg.Content)); err == nil {
			return rendered
		}
		return msg.Content
	case types.MessageTypeCancelled:
		return renderNoticeMessage("Interrupted · What should Claude do instead?", width)
	case types.MessageTypeWelcome:
		messageStyle := styles.WelcomeMessageStyle
		// Convert explicit newlines to markdown hard line breaks (two trailing spaces)
		// This preserves line breaks from YAML multiline syntax (|) while still
		// allowing markdown formatting like **bold** and *italic*
		content := preserveLineBreaks(msg.Content)
		rendered, err := markdown.NewRenderer(width - messageStyle.GetHorizontalFrameSize()).Render(content)
		if err != nil {
			rendered = msg.Content
		}
		return messageStyle.Width(width - 1).Render(strings.TrimRight(rendered, "\n\r\t "))
	case types.MessageTypeSystem:
		messageStyle := styles.SystemMessageStyle
		if isPreformattedSystemContent(msg.Content) && msg.Sender == "" {
			return renderPreformattedSystemContent(msg.Content, width)
		}
		// preserveLineBreaks keeps single newlines (control-command output is
		// often a `- key: value` list) from collapsing under markdown.
		body := preserveLineBreaks(msg.Content)
		if msg.Sender != "" {
			body = "**" + msg.Sender + "**\n\n" + body
		}
		rendered, err := markdown.NewRenderer(width - messageStyle.GetHorizontalFrameSize()).Render(body)
		if err != nil {
			rendered = msg.Content
		}
		return messageStyle.Width(width - 1).Render(strings.TrimRight(rendered, "\n\r\t "))
	case types.MessageTypeNotice:
		return renderNoticeMessage(msg.Content, width)
	case types.MessageTypeError:
		return styles.ErrorMessageStyle.Width(width - 1).Render(msg.Content)
	case types.MessageTypeLoading:
		glyph, description := "✶", msg.Content
		if msg.Content == claudeActivityLoadingText {
			glyph, description = claudeActivityStatus(mv.spinner.FrameIndex(), mv.spinner.CurrentMessage())
		}
		prefix := styles.SpinnerDotsAccentStyle.Render(glyph)
		prefixWidth := ansi.StringWidth(glyph) + 1
		maxDescWidth := width - prefixWidth
		if maxDescWidth > 0 && ansi.StringWidth(description) > maxDescWidth {
			description = ansi.Truncate(description, maxDescWidth, "…")
		}
		return prefix + " " + styles.MutedStyle.Render(description)
	default:
		return msg.Content
	}
}

func renderNoticeMessage(content string, width int) string {
	content = strings.TrimRight(content, "\n\r\t ")
	prefix := "  ⎿  "
	continuation := strings.Repeat(" ", lipgloss.Width(prefix))
	wrapWidth := max(1, width-lipgloss.Width(prefix)-1)
	var out []string
	first := true
	for _, logicalLine := range strings.Split(content, "\n") {
		wrapped := wrapNoticeLine(logicalLine, wrapWidth)
		if len(wrapped) == 0 {
			wrapped = []string{""}
		}
		for _, line := range wrapped {
			linePrefix := continuation
			if first {
				linePrefix = prefix
				first = false
			}
			out = append(out, styles.MutedStyle.Render(linePrefix+line))
		}
	}
	if len(out) == 0 {
		return styles.MutedStyle.Render(prefix)
	}
	return strings.Join(out, "\n")
}

func wrapNoticeLine(line string, width int) []string {
	line = strings.TrimRight(line, " ")
	if line == "" {
		return []string{""}
	}
	var wrapped []string
	remaining := line
	for remaining != "" {
		if lipgloss.Width(remaining) <= width {
			wrapped = append(wrapped, remaining)
			break
		}
		cut := hardWrapCut(remaining, width)
		if cut <= 0 {
			wrapped = append(wrapped, ansi.Truncate(remaining, width, ""))
			break
		}
		wrapped = append(wrapped, strings.TrimRight(remaining[:cut], " "))
		remaining = strings.TrimLeft(remaining[cut:], " ")
	}
	return wrapped
}

func hardWrapCut(text string, width int) int {
	cut := 0
	currentWidth := 0
	for idx, r := range text {
		runeWidth := lipgloss.Width(string(r))
		if currentWidth+runeWidth > width {
			break
		}
		currentWidth += runeWidth
		cut = idx + len(string(r))
	}
	return cut
}

func isPreformattedSystemContent(content string) bool {
	trimmed := strings.TrimLeft(ansi.Strip(content), "\n\r\t ")
	return strings.HasPrefix(trimmed, "Settings  Status") ||
		strings.HasPrefix(trimmed, "Help  General") ||
		strings.HasPrefix(trimmed, "Context Usage") ||
		strings.HasPrefix(trimmed, "Export conversation") ||
		strings.HasPrefix(trimmed, "Skills") ||
		strings.Contains(trimmed, "\n  Skills\n") ||
		strings.HasPrefix(trimmed, "Permissions  Recently denied") ||
		strings.HasPrefix(trimmed, "Add allow permission rule") ||
		strings.HasPrefix(trimmed, "Add ask permission rule") ||
		strings.HasPrefix(trimmed, "Add deny permission rule") ||
		strings.HasPrefix(trimmed, "Add workspace permission rule")
}

func indentPreformattedSystemContent(content string) string {
	content = strings.TrimRight(content, "\n\r\t ")
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		lines[i] = "  " + line
	}
	return strings.Join(lines, "\n")
}

func renderPreformattedSystemContent(content string, width int) string {
	separator := strings.Repeat("─", max(1, width))
	return separator + "\n" + indentPreformattedSystemContent(content)
}

func claudeActivityStatus(frame int, message string) (string, string) {
	if frame < 0 {
		frame = 0
	}
	glyph := claudeActivityGlyphs[frame%len(claudeActivityGlyphs)]
	if message == "" {
		message = claudeActivityMessages[0]
	}
	return glyph, message + "…"
}

func formatBangTranscript(content string) string {
	content = strings.TrimRight(content, "\n\r\t ")
	if !strings.HasPrefix(content, "! ") {
		return ""
	}
	return content
}

func formatClaudeUserTranscript(content string, width int) string {
	content = strings.TrimRight(content, "\n\r\t ")
	if content == "" {
		return "❯"
	}
	var out []string
	first := true
	for _, logicalLine := range strings.Split(content, "\n") {
		prefix := "  "
		if first {
			prefix = "❯ "
			first = false
		}
		wrapped := wrapTranscriptLine(logicalLine, max(1, width-lipgloss.Width(prefix)))
		if len(wrapped) == 0 {
			out = append(out, prefix)
			continue
		}
		for i, line := range wrapped {
			if i > 0 {
				prefix = "  "
			}
			out = append(out, prefix+line)
		}
	}
	return strings.Join(out, "\n")
}

func wrapTranscriptLine(line string, width int) []string {
	line = strings.TrimRight(line, " ")
	if line == "" {
		return []string{""}
	}
	var wrapped []string
	remaining := line
	for remaining != "" {
		if lipgloss.Width(remaining) <= width {
			wrapped = append(wrapped, remaining)
			break
		}
		cut := transcriptWrapCut(remaining, width)
		if cut <= 0 {
			wrapped = append(wrapped, ansi.Truncate(remaining, width, ""))
			break
		}
		wrapped = append(wrapped, strings.TrimRight(remaining[:cut], " "))
		remaining = strings.TrimLeft(remaining[cut:], " ")
	}
	return wrapped
}

func transcriptWrapCut(text string, width int) int {
	cut := 0
	lastSpace := -1
	currentWidth := 0
	for idx, r := range text {
		runeWidth := lipgloss.Width(string(r))
		if currentWidth+runeWidth > width {
			break
		}
		currentWidth += runeWidth
		next := idx + len(string(r))
		cut = next
		if r == ' ' {
			lastSpace = next
		}
	}
	if lastSpace > 0 {
		return lastSpace
	}
	return cut
}

func addClaudeAssistantPrefix(rendered string) string {
	rendered = strings.TrimLeft(rendered, " ")
	if rendered == "" {
		return styles.SecondaryStyle.Render("⏺")
	}
	lines := strings.Split(rendered, "\n")
	lines[0] = styles.SecondaryStyle.Render("⏺") + " " + strings.TrimLeft(lines[0], " ")
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(ansi.Strip(lines[i])) == "" {
			continue
		}
		lines[i] = "  " + lines[i]
	}
	return strings.Join(lines, "\n")
}

// renderAssistantMarkdown renders streamed assistant content using a per-message
// IncrementalRenderer. The renderer remembers the last rendered stable prefix
// so each new chunk only re-parses the trailing region. The first render at a
// given width is equivalent to a fresh full render.
//
// For finalized messages we use a transient renderer that is discarded after
// each call. Finalized messages are no longer streamed, so the prefix-cache
// inside an IncrementalRenderer is not earning its keep — keeping it resident
// across the lifetime of every historical message in a session is the
// dominant source of retained memory in long sessions. The parent message
// list's bounded rendered-item LRU can still memoize finalized message output
// without storing an additional per-view copy.
//
// It also returns the list of fenced code blocks emitted by the renderer so
// that callers can map clicks on the per-block copy affordance back to the
// underlying raw code.
func (mv *messageModel) renderAssistantMarkdown(content string, width int) (string, []markdown.CodeBlock, error) {
	if mv.finalized {
		r := markdown.NewIncrementalRenderer(width)
		return r.RenderWithCodeBlocks(content)
	}
	if mv.mdRenderer == nil {
		mv.mdRenderer = markdown.NewIncrementalRenderer(width)
	} else {
		mv.mdRenderer.SetWidth(width)
	}
	return mv.mdRenderer.RenderWithCodeBlocks(content)
}

func (mv *messageModel) senderPrefix(sender string) string {
	if sender == "" {
		return ""
	}
	return styles.AgentBadgeStyleFor(sender).MarginLeft(2).Render(sender) + "\n"
}

// sameAgentAsPrevious returns true if the previous message was from the same agent
func (mv *messageModel) sameAgentAsPrevious(msg *types.Message) bool {
	if mv.previous == nil || mv.previous.Sender != msg.Sender {
		return false
	}
	if standaloneAssistantNotice(mv.previous) || standaloneAssistantNotice(msg) {
		return false
	}
	switch mv.previous.Type {
	case types.MessageTypeAssistant,
		types.MessageTypeAssistantReasoningBlock,
		types.MessageTypeToolCall,
		types.MessageTypeToolResult:
		return true
	default:
		return false
	}
}

func standaloneAssistantNotice(msg *types.Message) bool {
	return msg != nil &&
		msg.Type == types.MessageTypeAssistant &&
		strings.HasPrefix(strings.TrimSpace(msg.Content), "Unknown command: ")
}

// Height calculates the height needed for this message view. Render() is
// memoized, so calling it from here does not duplicate work when View() is
// invoked for the same inputs.
func (mv *messageModel) Height(width int) int {
	content := mv.Render(width)
	return strings.Count(content, "\n") + 1
}

// Message returns the underlying message
func (mv *messageModel) Message() *types.Message {
	return mv.message
}

// CodeBlocks returns the fenced code blocks emitted by the most recent render
// of this message, with Line indices expressed in View() output coordinates.
// Returns nil when the message has no code blocks or has not been rendered
// yet (e.g. non-assistant messages).
func (mv *messageModel) CodeBlocks() []markdown.CodeBlock {
	return mv.codeBlocks
}

// Layout.Sizeable methods

// StopAnimation stops the spinner animation and unregisters from the animation coordinator.
// This must be called when the view is removed from the UI to avoid leaked animation subscriptions.
func (mv *messageModel) StopAnimation() {
	if mv.message.Type == types.MessageTypeSpinner || mv.message.Type == types.MessageTypeLoading {
		mv.spinner.Stop()
	}
}

// Finalize releases per-message render state that no longer needs to be kept
// resident once the message is no longer the actively streaming view. This is
// called by the parent message list when a new top-level message arrives, and
// for every historical view loaded from a session.
//
// Finalize is a no-op for non-assistant message types: only assistant views
// allocate an IncrementalRenderer and accumulate large rendered ANSI strings
// during streaming, so user messages, tool calls, error/welcome banners and
// the like have nothing to release. Setting `finalized = true` on those views
// would only have the side-effect of permanently disabling renderCache for
// selected/hovered states (which bypass the parent's bounded LRU), forcing a
// fresh re-render on every animation tick. Restricting the disable to
// assistant views keeps the leak fix scoped to the type that actually leaks.
//
// The retained payload of an assistant view is dominated by the renderCache
// (a copy of the rendered ANSI string) and the IncrementalRenderer's internal
// caches (last rendered prefix, glamour AST state). Both are pure render
// state — they can be regenerated from mv.message on demand. We deliberately
// leave mv.message, mv.codeBlocks and the spinner untouched so that View()
// keeps returning correct output, click-targeting on code blocks still works,
// and the spinner-driven types continue to animate.
//
// Finalize is idempotent and durable: subsequent renders do not re-populate
// renderCache or store an IncrementalRenderer on the struct. This is
// important because the parent message list invalidates its own LRU on
// several events (spinner removal, theme change, window resize) and would
// otherwise re-render every previously finalized view, putting the per-
// message render state right back where it was.
func (mv *messageModel) Finalize() {
	if mv.message == nil || mv.message.Type != types.MessageTypeAssistant {
		return
	}
	mv.renderCache = renderCache{}
	if mv.mdRenderer != nil {
		mv.mdRenderer.Reset()
		mv.mdRenderer = nil
	}
	mv.finalized = true
}

func renderClaudeWorking() string {
	return styles.SpinnerDotsAccentStyle.Render("✻") + " " + styles.MutedStyle.Render("Working…")
}

// HasLiveRenderState reports whether this view still retains per-message
// render state — either a populated renderCache or an IncrementalRenderer
// instance. Used as a structural assertion in regression tests that verify
// Finalize() actually released what it was supposed to release.
func (mv *messageModel) HasLiveRenderState() bool {
	return mv.renderCache.result != "" || mv.mdRenderer != nil
}

// SetSize sets the dimensions of the message view
func (mv *messageModel) SetSize(width, height int) tea.Cmd {
	if mv.width != width {
		mv.renderCache.valid = false
	}
	mv.width = width
	mv.height = height
	return nil
}

// GetSize returns the current dimensions
func (mv *messageModel) GetSize() (width, height int) {
	return mv.width, mv.height
}

// preserveLineBreaks preserves leading indentation by converting leading spaces
// to non-breaking spaces (U+00A0) which won't be stripped by markdown parsers.
// Line breaks are handled by glamour.WithPreservedNewLines().
func preserveLineBreaks(s string) string {
	if !strings.Contains(s, "\n") {
		return preserveIndentation(s)
	}
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		lines[i] = preserveIndentation(line)
	}
	return strings.Join(lines, "\n")
}

// preserveIndentation converts leading spaces in a line to non-breaking spaces (U+00A0).
// This prevents markdown parsers from stripping leading whitespace while maintaining
// the same visual appearance in terminal output.
func preserveIndentation(line string) string {
	if line == "" || line[0] != ' ' {
		return line
	}
	leadingSpaces := 0
	for _, c := range line {
		if c == ' ' {
			leadingSpaces++
		} else {
			break
		}
	}
	if leadingSpaces == 0 {
		return line
	}
	return strings.Repeat("\u00A0", leadingSpaces) + line[leadingSpaces:]
}
