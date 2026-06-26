package wire

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"sync"
	"time"
)

// ErrDisconnected is returned by SendInbound / SendAck when the client is
// disconnected and the reconnect loop has been stopped (e.g. via Close).
var ErrDisconnected = errors.New("wire: client disconnected")

// reconnectBaseDelay is the initial backoff delay for reconnection attempts.
const reconnectBaseDelay = 100 * time.Millisecond

// reconnectMaxDelay caps the exponential backoff.
const reconnectMaxDelay = 30 * time.Second

// WireClient manages a connection to the AgentM gateway.
type WireClient struct {
	transport Transport
	peerName  string
	token     string
	cwd       string // working directory stamped on hello

	conn      io.ReadWriteCloser
	mu        sync.Mutex // protects writes + conn swap
	outbound  chan *Envelope
	done      chan struct{}
	closeOnce sync.Once
	lastErr   error // last error from readLoop, readable after Done() fires

	// capabilities is the static capability block the gateway stamped on the
	// welcome frame (models, active model, scenario, gateway command catalog).
	// Captured at Connect so the TUI can populate its model picker and command
	// palette before any session exists. Nil if the gateway sent none.
	capabilities map[string]any

	// reconnecting is closed when the readLoop enters its reconnect loop and
	// re-opened (a fresh channel) when a reconnect succeeds. The adapter's pump
	// selects on this to know the client is temporarily offline rather than dead.
	reconnecting chan struct{}

	// reconnected is closed when the current reconnect loop succeeds. The
	// adapter uses this to block until Outbound() points at the fresh channel.
	reconnected chan struct{}
	reconnectMu sync.RWMutex

	// stopReconnect is closed by Close() to abort a running reconnect loop.
	stopReconnect chan struct{}

	// OnReconnect, if set, is called after a successful reconnect inside the
	// readLoop goroutine. The adapter uses this to re-seed capabilities.
	OnReconnect func()
}

// Capabilities returns the static capability block from the welcome handshake
// (or nil). Safe to read after Connect returns.
func (c *WireClient) Capabilities() map[string]any {
	return c.capabilities
}

// NewWireClient creates a new client (does not connect yet).
func NewWireClient(transport Transport, peerName string, token string, opts ...ClientOption) *WireClient {
	c := &WireClient{
		transport:     transport,
		peerName:      peerName,
		token:         token,
		outbound:      make(chan *Envelope, 64),
		done:          make(chan struct{}),
		reconnecting:  make(chan struct{}),
		reconnected:   make(chan struct{}),
		stopReconnect: make(chan struct{}),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// ClientOption configures a WireClient at construction time.
type ClientOption func(*WireClient)

// WithCwd stamps the peer's working directory on the hello frame.
func WithCwd(cwd string) ClientOption {
	return func(c *WireClient) { c.cwd = cwd }
}

// Connect establishes the connection and performs the hello/welcome handshake.
func (c *WireClient) Connect(ctx context.Context) error {
	conn, err := c.transport.Connect(ctx)
	if err != nil {
		return fmt.Errorf("transport connect: %w", err)
	}
	c.conn = conn

	helloBody := map[string]any{
		"peer_name":    c.peerName,
		"peer_version": "0.1.0",
		"capabilities": map[string]any{},
	}
	if c.cwd != "" {
		helloBody["cwd"] = c.cwd
	}
	hello := &Envelope{
		V:    2,
		ID:   NewID(),
		Kind: KindHello,
		TS:   Now(),
		Body: helloBody,
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

// readLoop reads frames from the connection and dispatches them. On transient
// read errors it enters a reconnect loop with exponential backoff. The done
// channel is NOT closed on successful reconnects — only when reconnect gives
// up or Close() is called.
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
			if c.reconnect() {
				continue
			}
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

// reconnect attempts to re-establish the connection with exponential backoff.
// It is called from readLoop when a read error occurs. On success it creates a
// fresh outbound channel, swaps the connection, signals readiness, and returns
// true so readLoop can continue on the new connection. On failure (e.g.
// gateway rejects auth) it returns false and the caller's defer closes done.
func (c *WireClient) reconnect() bool {
	// Signal that we are reconnecting. The adapter's pump can observe this.
	c.reconnectMu.Lock()
	c.reconnected = make(chan struct{})
	close(c.reconnecting)
	c.reconnectMu.Unlock()

	delay := reconnectBaseDelay
	for {
		// Check if Close() was called.
		select {
		case <-c.stopReconnect:
			log.Printf("[wire] reconnect aborted by Close")
			return false
		default:
		}

		log.Printf("[wire] reconnecting in %v ...", delay)

		// Wait for the backoff delay, but abort if Close() is called.
		select {
		case <-time.After(delay):
		case <-c.stopReconnect:
			log.Printf("[wire] reconnect aborted by Close during backoff")
			return false
		}

		// Attempt to connect.
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		conn, err := c.transport.Connect(ctx)
		cancel()
		if err != nil {
			log.Printf("[wire] reconnect attempt failed: %v", err)
			delay *= 2
			if delay > reconnectMaxDelay {
				delay = reconnectMaxDelay
			}
			continue
		}

		// Send hello.
		reconnectBody := map[string]any{
			"peer_name":    c.peerName,
			"peer_version": "0.1.0",
			"capabilities": map[string]any{},
		}
		if c.cwd != "" {
			reconnectBody["cwd"] = c.cwd
		}
		hello := &Envelope{
			V:    2,
			ID:   NewID(),
			Kind: KindHello,
			TS:   Now(),
			Body: reconnectBody,
		}
		if c.token != "" {
			hello.Body["auth"] = map[string]any{"token": c.token}
		}

		if err := WriteFrame(conn, hello); err != nil {
			log.Printf("[wire] reconnect hello failed: %v", err)
			conn.Close()
			delay *= 2
			if delay > reconnectMaxDelay {
				delay = reconnectMaxDelay
			}
			continue
		}

		// Read welcome.
		resp, err := ReadFrame(conn)
		if err != nil {
			log.Printf("[wire] reconnect welcome read failed: %v", err)
			conn.Close()
			delay *= 2
			if delay > reconnectMaxDelay {
				delay = reconnectMaxDelay
			}
			continue
		}

		// Treat error responses as fatal during reconnect too — the gateway is
		// telling us we're not welcome.
		if resp.Kind == KindError {
			msg := ""
			if m, ok := resp.Body["message"].(string); ok {
				msg = m
			}
			log.Printf("[wire] reconnect rejected by gateway: %s", msg)
			conn.Close()
			c.lastErr = fmt.Errorf("reconnect rejected: %s", msg)
			return false
		}

		if resp.Kind != KindWelcome {
			log.Printf("[wire] reconnect unexpected response kind: %s", resp.Kind)
			conn.Close()
			delay *= 2
			if delay > reconnectMaxDelay {
				delay = reconnectMaxDelay
			}
			continue
		}

		// Reconnect succeeded.
		log.Printf("[wire] reconnected, server_version=%v", resp.Body["server_version"])

		// Update capabilities.
		if caps, ok := resp.Body["capabilities"].(map[string]any); ok {
			c.capabilities = caps
		}
		c.lastErr = nil

		// Atomically swap the connection and outbound channel.
		c.mu.Lock()
		oldConn := c.conn
		c.conn = conn
		oldOutbound := c.outbound
		c.outbound = make(chan *Envelope, 64)
		c.mu.Unlock()

		// Close old resources outside the lock.
		oldConn.Close()
		draining := true
		for draining {
			select {
			case <-oldOutbound:
				continue
			default:
				close(oldOutbound)
				draining = false
			}
		}

		// Open a fresh reconnecting channel for the next cycle, then signal
		// waiters that Outbound() now points at the recovered connection.
		c.reconnectMu.Lock()
		c.reconnecting = make(chan struct{})
		close(c.reconnected)
		c.reconnectMu.Unlock()

		// Notify the adapter.
		if c.OnReconnect != nil {
			c.OnReconnect()
		}

		return true
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
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.outbound
}

// Done returns a channel that is closed when the read loop exits permanently.
func (c *WireClient) Done() <-chan struct{} {
	return c.done
}

// Reconnecting returns a channel that is closed when the client enters its
// reconnect loop. A new channel is returned after each successful reconnect.
// The adapter's pump can select on this to know the client is temporarily
// offline rather than dead.
func (c *WireClient) Reconnecting() <-chan struct{} {
	c.reconnectMu.RLock()
	defer c.reconnectMu.RUnlock()
	return c.reconnecting
}

// Reconnected returns a channel that is closed when the active reconnect loop
// succeeds. It is meaningful after Reconnecting() has fired.
func (c *WireClient) Reconnected() <-chan struct{} {
	c.reconnectMu.RLock()
	defer c.reconnectMu.RUnlock()
	return c.reconnected
}

// Err returns the error that caused the read loop to exit, if any.
func (c *WireClient) Err() error {
	return c.lastErr
}

// Close shuts down the connection and aborts any in-progress reconnect.
func (c *WireClient) Close() error {
	c.closeOnce.Do(func() {
		close(c.stopReconnect)
		if c.conn != nil {
			c.conn.Close()
		}
	})
	return nil
}
