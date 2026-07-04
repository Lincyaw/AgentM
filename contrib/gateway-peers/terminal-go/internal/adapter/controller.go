package adapter

import (
	"context"
	"log"
	"strings"
	"sync"

	"github.com/AoyangSpace/agentm-terminal/internal/cagent/app"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/runtime"
	"github.com/AoyangSpace/agentm-terminal/internal/cagent/session"
	"github.com/AoyangSpace/agentm-terminal/internal/tui/messages"
	"github.com/AoyangSpace/agentm-terminal/internal/wire"
)

// Controller implements app.Controller.
var _ app.Controller = (*Controller)(nil)

// Identity carries the platform identifiers a chat-client peer stamps on every
// inbound (§2.4). For the terminal peer these come from CLI flags.
type Identity struct {
	Channel    string // "terminal"
	ChatID     string // -session-id
	SenderID   string // -sender-id
	SessionKey string // composed conversation identity (§3.4)
	Scenario   string // -scenario, sent on the first inbound only
}

// Controller is the wire-backed implementation of app.Controller. It turns the
// TUI's user actions into `inbound` wire envelopes and tracks whether the first
// inbound for this session has been sent (so `scenario` rides only the first).
type Controller struct {
	client     *wire.WireClient
	id         Identity
	translator *Translator

	mu           sync.Mutex
	scenarioSent bool
	firstMessage string
	hasFirstMsg  bool

	// prevRunCtx tracks the ctx from the most recent Run/RunWithMessage/
	// CompactSession call. When the TUI cancels it (ESC or new message) and a
	// new Run follows, we send {control:"interrupt"} synchronously before the
	// new message so the gateway stops the old turn. A background goroutine
	// watches it for the ESC-only case (no new Run follows).
	prevRunCtx context.Context
}

// NewController builds a Controller. firstMessage is the optional prompt to send
// automatically on launch (empty == none).
func NewController(client *wire.WireClient, id Identity, tr *Translator, firstMessage string) *Controller {
	return &Controller{
		client:       client,
		id:           id,
		translator:   tr,
		firstMessage: firstMessage,
		hasFirstMsg:  firstMessage != "",
	}
}

// NewChildController builds the Controller a sub-agent tab drives: its inbounds
// carry the child's session_id as session_key, so the gateway routes a typed
// message into that live child's inbox (interactive-subagent design) instead of
// the main conversation. The child session already exists server-side, so the
// scenario must NOT ride its inbounds — scenarioSent starts true. id.SessionKey
// MUST be the child's session id (== the wire child_id).
func NewChildController(client *wire.WireClient, id Identity, tr *Translator) *Controller {
	return &Controller{
		client:       client,
		id:           id,
		translator:   tr,
		scenarioSent: true,
	}
}

// FirstMessage returns the queued launch prompt once; subsequent calls report
// none so the message is not re-sent on session reset.
func (c *Controller) FirstMessage() (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.hasFirstMsg {
		return "", false
	}
	c.hasFirstMsg = false
	return c.firstMessage, true
}

// sendInbound composes and sends an `inbound` envelope, attaching `scenario`
// only to the first one for this session (§2.4).
func (c *Controller) sendInbound(body map[string]any) {
	c.mu.Lock()
	scenario := ""
	if _, hasBodyContent := body["content"]; hasBodyContent {
		if _, ok := body["request_id"]; !ok {
			body["request_id"] = wire.NewID()
		}
		if _, hasAction := body["action"]; !hasAction {
			content, _ := body["content"].(string)
			if _isSlashCommand(content) {
				body["action"] = "run_command"
			} else {
				body["action"] = "submit"
				body["policy"] = "interrupt_first"
			}
		}
	}
	if !c.scenarioSent {
		scenario = c.id.Scenario
		c.scenarioSent = true
	}
	c.mu.Unlock()

	body["channel"] = c.id.Channel
	body["chat_id"] = c.id.ChatID
	if c.id.SenderID != "" {
		body["sender_id"] = c.id.SenderID
	}
	if err := c.client.SendInbound(body, c.id.SessionKey, scenario); err != nil {
		log.Printf("[adapter] send inbound failed: %v", err)
		if c.translator != nil {
			c.translator.emit(runtime.Error("failed to reach gateway: " + err.Error()))
		}
	}
}

func _isSlashCommand(content string) bool {
	return strings.HasPrefix(content, "/") && !strings.HasPrefix(content, "//")
}

// sendInterrupt sends a control=interrupt inbound to stop the current gateway
// turn. The gateway's router (§3.2) dispatches it to session.interrupt().
func (c *Controller) sendInterrupt() {
	log.Printf("[adapter] sending interrupt for session_key=%s", c.id.SessionKey)
	c.sendInbound(map[string]any{"control": "interrupt"})
}

// interruptIfPending checks whether the previous Run's ctx was cancelled by the
// TUI (ESC or new-message-replaces-old). If so, sends an interrupt to the
// gateway before the next message, ensuring the old turn is stopped promptly.
func (c *Controller) interruptIfPending() {
	c.mu.Lock()
	prev := c.prevRunCtx
	c.mu.Unlock()
	if prev == nil {
		return
	}
	select {
	case <-prev.Done():
		c.sendInterrupt()
	default:
	}
}

// trackRun stores the current Run's ctx and spawns a goroutine that sends an
// interrupt if the ctx is cancelled (ESC without a following new message).
// At most one goroutine per Controller is "leaked" (the last completed turn's
// watcher); subsequent Run calls supersede the old watcher via prevRunCtx.
func (c *Controller) trackRun(ctx context.Context) {
	c.mu.Lock()
	c.prevRunCtx = ctx
	c.mu.Unlock()

	go func() {
		<-ctx.Done()
		c.mu.Lock()
		isCurrent := c.prevRunCtx == ctx
		c.mu.Unlock()
		if isCurrent {
			c.sendInterrupt()
		}
	}()
}

// Run sends one user turn to the gateway.
func (c *Controller) Run(ctx context.Context, cancel context.CancelFunc, message string, attachments []messages.Attachment) {
	_ = cancel
	_ = attachments
	if message == "" {
		return
	}
	c.interruptIfPending()
	c.echoUserMessage(message)
	c.sendInbound(map[string]any{"content": message})
	c.trackRun(ctx)
}

// RunCooperative sends one user turn without interrupting the current gateway
// run. The gateway queues the content in the core SessionInbox; if a foreground
// tool is wrapped by background_exec, that pending inbox item can soft-preempt
// the tool into the background.
func (c *Controller) RunCooperative(ctx context.Context, cancel context.CancelFunc, message string, attachments []messages.Attachment) {
	_ = ctx
	_ = cancel
	_ = attachments
	if message == "" {
		return
	}
	c.sendInbound(map[string]any{
		"content": message,
		"action":  "submit",
		"policy":  "cooperative",
	})
}

// echoUserMessage renders the user's turn into the transcript locally. The
// gateway does not echo inbound user messages back over the wire (there is no
// user_message outbound kind), and unlike the in-process cagent runtime the
// wire-backed controller has no other path that surfaces the prompt. Without
// this the user's own message never appears in the TUI. ReplaceLoadingWithUser
// tolerates the absence of a loading placeholder, so a plain UserMessageEvent
// is enough to paint the bubble.
func (c *Controller) echoUserMessage(content string) {
	if c.translator == nil || content == "" {
		return
	}
	c.translator.emit(runtime.UserMessage(content, c.translator.sessionID(), nil))
}

// RunWithMessage sends a pre-built message. The wire protocol carries plain
// text, so the message's textual content is forwarded.
func (c *Controller) RunWithMessage(ctx context.Context, cancel context.CancelFunc, msg *session.Message) {
	_ = cancel
	if msg == nil {
		return
	}
	content := msg.Message.Content
	if content == "" {
		return
	}
	c.interruptIfPending()
	c.echoUserMessage(content)
	c.sendInbound(map[string]any{"content": content})
	c.trackRun(ctx)
}

// CompactSession asks the gateway to compact history via the /compact command.
func (c *Controller) CompactSession(ctx context.Context, cancel context.CancelFunc, additionalPrompt string) {
	_ = cancel
	content := "/compact"
	if additionalPrompt != "" {
		content += " " + additionalPrompt
	}
	c.interruptIfPending()
	c.echoUserMessage(content)
	c.sendInbound(map[string]any{"content": content})
	c.trackRun(ctx)
}

// Resume delivers a tool-confirmation decision by sending an inbound carrying
// the approval button_value the gateway minted on the approval_request card.
func (c *Controller) Resume(req runtime.ResumeRequest) {
	approvalID := ""
	if c.translator != nil {
		approvalID = c.translator.PendingApprovalID()
	}
	if approvalID == "" {
		log.Printf("[adapter] Resume with no pending approval; dropping %+v", req)
		return
	}
	decision := "approve"
	switch req.Type {
	case runtime.ResumeTypeReject:
		decision = "deny"
	case runtime.ResumeTypeApproveSession:
		decision = "approve_session"
	case runtime.ResumeTypeApprove,
		runtime.ResumeTypeApproveTool:
		decision = "approve"
	}
	c.sendInbound(map[string]any{"button_value": approvalID + ":" + decision})
}

// UpdateSessionTitle sets the local session title and echoes a SessionTitle
// event so the TUI updates immediately. The gateway owns no per-session title,
// so this is a client-local rename.
func (c *Controller) UpdateSessionTitle(ctx context.Context, title string) error {
	_ = ctx
	if c.translator != nil {
		if c.translator.sess != nil {
			c.translator.sess.Title = title
		}
		c.translator.emit(runtime.SessionTitle(c.translator.sessionID(), title))
	}
	return nil
}

// RunBangCommand forwards a shell command as a "!"-prefixed inbound so the
// gateway scenario can run it. The terminal peer does not execute shell locally.
func (c *Controller) RunBangCommand(ctx context.Context, command string) {
	_ = ctx
	if command == "" {
		return
	}
	c.echoUserMessage("!" + command)
	c.sendInbound(map[string]any{"content": "!" + command})
}

// NewSession starts a fresh gateway session via the /new command.
func (c *Controller) NewSession() {
	c.echoUserMessage("/new")
	c.sendInbound(map[string]any{"content": "/new"})
}

// SwitchModel asks the gateway to switch the active model profile by sending a
// "/model <name>" command (the gateway's switch_model command). A blank name is
// ignored so an empty picker selection is a no-op.
func (c *Controller) SwitchModel(name string) {
	if name == "" {
		return
	}
	c.echoUserMessage("/model " + name)
	c.sendInbound(map[string]any{"content": "/model " + name})
}
