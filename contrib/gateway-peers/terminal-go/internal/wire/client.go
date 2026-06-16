package wire

import (
	"context"
	"fmt"
	"io"
	"log"
	"sync"
)

// WireClient manages a connection to the AgentM gateway.
type WireClient struct {
	transport Transport
	peerName  string
	token     string

	conn      io.ReadWriteCloser
	mu        sync.Mutex // protects writes
	outbound  chan *Envelope
	done      chan struct{}
	closeOnce sync.Once
	lastErr   error // last error from readLoop, readable after Done() fires

	// capabilities is the static capability block the gateway stamped on the
	// welcome frame (models, active model, scenario, gateway command catalog).
	// Captured at Connect so the TUI can populate its model picker and command
	// palette before any session exists. Nil if the gateway sent none.
	capabilities map[string]any
}

// Capabilities returns the static capability block from the welcome handshake
// (or nil). Safe to read after Connect returns.
func (c *WireClient) Capabilities() map[string]any {
	return c.capabilities
}

// NewWireClient creates a new client (does not connect yet).
func NewWireClient(transport Transport, peerName string, token string) *WireClient {
	return &WireClient{
		transport: transport,
		peerName:  peerName,
		token:     token,
		outbound:  make(chan *Envelope, 64),
		done:      make(chan struct{}),
	}
}

// Connect establishes the connection and performs the hello/welcome handshake.
func (c *WireClient) Connect(ctx context.Context) error {
	conn, err := c.transport.Connect(ctx)
	if err != nil {
		return fmt.Errorf("transport connect: %w", err)
	}
	c.conn = conn

	hello := &Envelope{
		V:    2,
		ID:   NewID(),
		Kind: KindHello,
		TS:   Now(),
		Body: map[string]any{
			"peer_name":    c.peerName,
			"peer_version": "0.1.0",
			"capabilities": map[string]any{},
		},
	}
	if c.token != "" {
		hello.Body["auth"] = map[string]any{"token": c.token}
	}

	if err := WriteFrame(c.conn, hello); err != nil {
		conn.Close()
		return fmt.Errorf("hello failed: %w", err)
	}

	resp, err := ReadFrame(c.conn)
	if err != nil {
		conn.Close()
		return fmt.Errorf("welcome read failed: %w", err)
	}
	if resp.Kind == KindError {
		conn.Close()
		msg := ""
		if m, ok := resp.Body["message"].(string); ok {
			msg = m
		}
		return fmt.Errorf("gateway rejected: %s", msg)
	}
	if resp.Kind != KindWelcome {
		conn.Close()
		return fmt.Errorf("unexpected response kind: %s", resp.Kind)
	}

	if caps, ok := resp.Body["capabilities"].(map[string]any); ok {
		c.capabilities = caps
	}
	log.Printf("[wire] connected, welcome from server_version=%v", resp.Body["server_version"])
	go c.readLoop()
	return nil
}

// readLoop reads frames from the connection and dispatches them.
func (c *WireClient) readLoop() {
	defer func() {
		log.Printf("[wire] readLoop exiting")
		close(c.done)
	}()
	for {
		env, err := ReadFrame(c.conn)
		if err != nil {
			log.Printf("[wire] readLoop read error: %v", err)
			c.lastErr = err
			return
		}
		log.Printf("[wire] recv kind=%s id=%s", env.Kind, env.ID)
		switch env.Kind {
		case KindPing:
			c.sendPong(env.ID)
		case KindOutbound, KindError:
			select {
			case c.outbound <- env:
			default:
				log.Printf("[wire] outbound channel full, dropping frame kind=%s", env.Kind)
			}
		}
	}
}

// SendInbound sends a user message to the gateway.
func (c *WireClient) SendInbound(body map[string]any, sessionKey, scenario string) error {
	env := &Envelope{
		V:          2,
		ID:         NewID(),
		Kind:       KindInbound,
		TS:         Now(),
		SessionKey: sessionKey,
		Body:       body,
	}
	if scenario != "" {
		env.Scenario = scenario
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	log.Printf("[wire] send inbound id=%s session_key=%s", env.ID, env.SessionKey)
	return WriteFrame(c.conn, env)
}

// SendAck acknowledges a received envelope.
func (c *WireClient) SendAck(envelopeID string) error {
	env := &Envelope{
		V:    2,
		ID:   NewID(),
		Kind: KindAck,
		TS:   Now(),
		Body: map[string]any{"envelope_id": envelopeID},
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return WriteFrame(c.conn, env)
}

func (c *WireClient) sendPong(pingID string) {
	env := &Envelope{
		V:    2,
		ID:   NewID(),
		Kind: KindPong,
		TS:   Now(),
		Body: map[string]any{},
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	// Best-effort; ignore errors on pong.
	_ = WriteFrame(c.conn, env)
}

// Outbound returns the channel of received outbound envelopes.
func (c *WireClient) Outbound() <-chan *Envelope {
	return c.outbound
}

// Done returns a channel that is closed when the read loop exits.
func (c *WireClient) Done() <-chan struct{} {
	return c.done
}

// Err returns the error that caused the read loop to exit, if any.
func (c *WireClient) Err() error {
	return c.lastErr
}

// Close shuts down the connection.
func (c *WireClient) Close() error {
	c.closeOnce.Do(func() {
		if c.conn != nil {
			c.conn.Close()
		}
	})
	return nil
}
