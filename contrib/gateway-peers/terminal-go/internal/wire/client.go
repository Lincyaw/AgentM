package wire

import (
	"context"
	"fmt"
	"io"
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

	go c.readLoop()
	return nil
}

// readLoop reads frames from the connection and dispatches them.
func (c *WireClient) readLoop() {
	defer close(c.done)
	for {
		env, err := ReadFrame(c.conn)
		if err != nil {
			return
		}
		switch env.Kind {
		case KindPing:
			c.sendPong(env.ID)
		case KindOutbound, KindError:
			select {
			case c.outbound <- env:
			default:
				// Drop if channel full (backpressure)
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

// Close shuts down the connection.
func (c *WireClient) Close() error {
	c.closeOnce.Do(func() {
		if c.conn != nil {
			c.conn.Close()
		}
	})
	return nil
}
