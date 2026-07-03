package wire

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"
)

// Envelope is the v2 wire protocol envelope.
type Envelope struct {
	V          int            `json:"v"`
	ID         string         `json:"id"`
	Kind       string         `json:"kind"`
	TS         float64        `json:"ts"`
	SessionKey string         `json:"session_key,omitempty"`
	Scenario   string         `json:"scenario,omitempty"`
	Body       map[string]any `json:"body"`
}

// WireVersion is the current AgentM gateway wire protocol version.
const WireVersion = 2

// Kind constants for the v2 wire protocol.
const (
	KindHello    = "hello"
	KindInbound  = "inbound"
	KindAck      = "ack"
	KindPong     = "pong"
	KindWelcome  = "welcome"
	KindOutbound = "outbound"
	KindError    = "error"
	KindPing     = "ping"
)

// NewID generates a 12-char hex envelope ID using crypto/rand.
func NewID() string {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		panic(fmt.Sprintf("crypto/rand failed: %v", err))
	}
	return hex.EncodeToString(b)
}

// Now returns the current Unix timestamp with sub-second precision.
func Now() float64 {
	return float64(time.Now().UnixNano()) / 1e9
}
