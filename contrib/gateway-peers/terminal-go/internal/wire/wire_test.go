package wire

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"net"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestNewID_Length(t *testing.T) {
	id := NewID()
	if len(id) != 12 {
		t.Errorf("NewID() length = %d, want 12", len(id))
	}
}

func TestNewID_HexOnly(t *testing.T) {
	id := NewID()
	if _, err := hex.DecodeString(id); err != nil {
		t.Errorf("NewID() = %q is not valid hex: %v", id, err)
	}
}

func TestNewID_Unique(t *testing.T) {
	seen := make(map[string]bool)
	for i := 0; i < 100; i++ {
		id := NewID()
		if seen[id] {
			t.Fatalf("NewID() produced duplicate: %s", id)
		}
		seen[id] = true
	}
}

func TestFramingRoundTrip(t *testing.T) {
	original := &Envelope{
		V:          2,
		ID:         "abcdef012345",
		Kind:       KindInbound,
		TS:         1748400000.42,
		SessionKey: "feishu:oc_xxx",
		Scenario:   "general_purpose",
		Body: map[string]any{
			"content":   "hello world",
			"sender_id": "ou_zzz",
		},
	}

	var buf bytes.Buffer
	if err := WriteFrame(&buf, original); err != nil {
		t.Fatalf("WriteFrame: %v", err)
	}

	decoded, err := ReadFrame(&buf)
	if err != nil {
		t.Fatalf("ReadFrame: %v", err)
	}

	if decoded.V != original.V {
		t.Errorf("V = %d, want %d", decoded.V, original.V)
	}
	if decoded.ID != original.ID {
		t.Errorf("ID = %q, want %q", decoded.ID, original.ID)
	}
	if decoded.Kind != original.Kind {
		t.Errorf("Kind = %q, want %q", decoded.Kind, original.Kind)
	}
	if decoded.TS != original.TS {
		t.Errorf("TS = %f, want %f", decoded.TS, original.TS)
	}
	if decoded.SessionKey != original.SessionKey {
		t.Errorf("SessionKey = %q, want %q", decoded.SessionKey, original.SessionKey)
	}
	if decoded.Scenario != original.Scenario {
		t.Errorf("Scenario = %q, want %q", decoded.Scenario, original.Scenario)
	}
	if decoded.Body["content"] != original.Body["content"] {
		t.Errorf("Body[content] = %v, want %v", decoded.Body["content"], original.Body["content"])
	}
}

func TestFramingEmptyBody(t *testing.T) {
	env := &Envelope{
		V:    2,
		ID:   "aabbccddeeff",
		Kind: KindPong,
		TS:   1.0,
		Body: map[string]any{},
	}

	var buf bytes.Buffer
	if err := WriteFrame(&buf, env); err != nil {
		t.Fatalf("WriteFrame: %v", err)
	}

	decoded, err := ReadFrame(&buf)
	if err != nil {
		t.Fatalf("ReadFrame: %v", err)
	}
	if len(decoded.Body) != 0 {
		t.Errorf("Body has %d entries, want 0", len(decoded.Body))
	}
}

func TestFramingTooLarge(t *testing.T) {
	// Write a length prefix that exceeds MaxFrameSize.
	var buf bytes.Buffer
	length := uint32(MaxFrameSize + 1)
	if err := binary.Write(&buf, binary.BigEndian, length); err != nil {
		t.Fatal(err)
	}

	_, err := ReadFrame(&buf)
	if err == nil {
		t.Fatal("expected error for oversized frame")
	}
	if !strings.Contains(err.Error(), "frame too large") {
		t.Errorf("error = %q, want it to contain 'frame too large'", err.Error())
	}
}

func TestFramingExactlyMaxSize(t *testing.T) {
	// A frame at exactly MaxFrameSize should be accepted.
	// We build a valid JSON payload that is exactly MaxFrameSize bytes.
	env := &Envelope{
		V:    2,
		ID:   "aabbccddeeff",
		Kind: KindOutbound,
		TS:   1.0,
		Body: map[string]any{},
	}
	data, _ := json.Marshal(env)

	// Pad the body to reach exactly MaxFrameSize total JSON bytes.
	// We add a string field with enough padding.
	needed := MaxFrameSize - len(data) - len(`,"pad":""}`) + len("}")
	if needed > 0 {
		env.Body["pad"] = strings.Repeat("x", needed)
	}
	data, _ = json.Marshal(env)

	var buf bytes.Buffer
	length := uint32(len(data))
	if length > MaxFrameSize {
		t.Skipf("padded envelope is %d bytes, exceeds max", length)
	}
	if err := binary.Write(&buf, binary.BigEndian, length); err != nil {
		t.Fatal(err)
	}
	buf.Write(data)

	decoded, err := ReadFrame(&buf)
	if err != nil {
		t.Fatalf("ReadFrame on max-size frame: %v", err)
	}
	if decoded.Kind != KindOutbound {
		t.Errorf("Kind = %q, want %q", decoded.Kind, KindOutbound)
	}
}

func TestEnvelopeOmitsEmptyFields(t *testing.T) {
	env := &Envelope{
		V:    2,
		ID:   "aabbccddeeff",
		Kind: KindPong,
		TS:   1.0,
		Body: map[string]any{},
	}

	data, err := json.Marshal(env)
	if err != nil {
		t.Fatal(err)
	}

	// session_key and scenario should be omitted when empty.
	if strings.Contains(string(data), "session_key") {
		t.Error("JSON should omit empty session_key")
	}
	if strings.Contains(string(data), "scenario") {
		t.Error("JSON should omit empty scenario")
	}
}

func TestEnvelopeIncludesSessionKey(t *testing.T) {
	env := &Envelope{
		V:          2,
		ID:         "aabbccddeeff",
		Kind:       KindInbound,
		TS:         1.0,
		SessionKey: "feishu:oc_xxx",
		Body:       map[string]any{"content": "hi"},
	}

	data, err := json.Marshal(env)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(string(data), `"session_key":"feishu:oc_xxx"`) {
		t.Errorf("JSON should include session_key, got: %s", string(data))
	}
}

func TestResolveTransportUnix(t *testing.T) {
	tr, err := ResolveTransport("unix:///tmp/agentm.sock", "")
	if err != nil {
		t.Fatalf("ResolveTransport unix: %v", err)
	}
	ut, ok := tr.(*UnixTransport)
	if !ok {
		t.Fatalf("expected *UnixTransport, got %T", tr)
	}
	if ut.Path != "/tmp/agentm.sock" {
		t.Errorf("Path = %q, want /tmp/agentm.sock", ut.Path)
	}
}

func TestResolveTransportWS(t *testing.T) {
	tr, err := ResolveTransport("ws://localhost:8080/ws", "")
	if err != nil {
		t.Fatalf("ResolveTransport ws: %v", err)
	}
	wt, ok := tr.(*WSTransport)
	if !ok {
		t.Fatalf("expected *WSTransport, got %T", tr)
	}
	if wt.URL != "ws://localhost:8080/ws" {
		t.Errorf("URL = %q, want ws://localhost:8080/ws", wt.URL)
	}
	if wt.TLSConfig != nil {
		t.Error("TLSConfig should be nil for ws://")
	}
}

func TestResolveTransportWSS(t *testing.T) {
	// Without CA cert, should succeed with nil TLSConfig.
	tr, err := ResolveTransport("wss://example.com/ws", "")
	if err != nil {
		t.Fatalf("ResolveTransport wss: %v", err)
	}
	wt, ok := tr.(*WSTransport)
	if !ok {
		t.Fatalf("expected *WSTransport, got %T", tr)
	}
	if wt.URL != "wss://example.com/ws" {
		t.Errorf("URL = %q, want wss://example.com/ws", wt.URL)
	}
}

func TestResolveTransportUnsupported(t *testing.T) {
	_, err := ResolveTransport("tcp://localhost:9090", "")
	if err == nil {
		t.Fatal("expected error for unsupported scheme")
	}
	if !strings.Contains(err.Error(), "unsupported scheme") {
		t.Errorf("error = %q, want it to contain 'unsupported scheme'", err.Error())
	}
}

func TestResolveTransportWSSWithBadCA(t *testing.T) {
	_, err := ResolveTransport("wss://example.com/ws", "/nonexistent/ca.pem")
	if err == nil {
		t.Fatal("expected error for nonexistent CA cert")
	}
}

func TestMultipleFramesRoundTrip(t *testing.T) {
	envs := []*Envelope{
		{V: 2, ID: "aaaaaaaaaaaa", Kind: KindHello, TS: 1.0, Body: map[string]any{"peer_name": "test"}},
		{V: 2, ID: "bbbbbbbbbbbb", Kind: KindWelcome, TS: 2.0, Body: map[string]any{"server_version": "1.0"}},
		{V: 2, ID: "cccccccccccc", Kind: KindPing, TS: 3.0, Body: map[string]any{}},
	}

	var buf bytes.Buffer
	for _, env := range envs {
		if err := WriteFrame(&buf, env); err != nil {
			t.Fatalf("WriteFrame: %v", err)
		}
	}

	for i, want := range envs {
		got, err := ReadFrame(&buf)
		if err != nil {
			t.Fatalf("ReadFrame[%d]: %v", i, err)
		}
		if got.ID != want.ID {
			t.Errorf("[%d] ID = %q, want %q", i, got.ID, want.ID)
		}
		if got.Kind != want.Kind {
			t.Errorf("[%d] Kind = %q, want %q", i, got.Kind, want.Kind)
		}
	}
}

func TestNowReturnsReasonableTimestamp(t *testing.T) {
	ts := Now()
	// Should be after 2020-01-01 (1577836800) and before 2100-01-01 (4102444800).
	if ts < 1577836800 || ts > 4102444800 {
		t.Errorf("Now() = %f, expected a timestamp between 2020 and 2100", ts)
	}
}

type reconnectTestTransport struct {
	t             *testing.T
	mu            sync.Mutex
	connects      int
	releaseSecond chan struct{}
}

func (tr *reconnectTestTransport) Connect(ctx context.Context) (io.ReadWriteCloser, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	client, server := net.Pipe()
	tr.mu.Lock()
	connectIndex := tr.connects
	tr.connects++
	tr.mu.Unlock()

	go func() {
		defer server.Close()
		_ = server.SetDeadline(time.Now().Add(2 * time.Second))
		hello, err := ReadFrame(server)
		if err != nil {
			tr.t.Errorf("server ReadFrame hello: %v", err)
			return
		}
		if hello.Kind != KindHello {
			tr.t.Errorf("hello kind = %q, want %q", hello.Kind, KindHello)
			return
		}
		if err := WriteFrame(server, &Envelope{
			V:    2,
			ID:   "bbbbbbbbbbbb",
			Kind: KindWelcome,
			TS:   2.0,
			Body: map[string]any{"server_version": "test"},
		}); err != nil {
			tr.t.Errorf("server WriteFrame welcome: %v", err)
			return
		}
		if connectIndex == 0 {
			return
		}
		_ = server.SetDeadline(time.Time{})
		<-tr.releaseSecond
	}()

	return client, nil
}

func TestWireClientKeepsDoneOpenAfterSuccessfulReconnect(t *testing.T) {
	transport := &reconnectTestTransport{
		t:             t,
		releaseSecond: make(chan struct{}),
	}
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() {
		close(transport.releaseSecond)
	})
	client := NewWireClient(transport, "peer", "")
	reconnected := make(chan struct{})
	var reconnectOnce sync.Once
	client.OnReconnect = func() {
		reconnectOnce.Do(func() {
			close(reconnected)
		})
	}

	if err := client.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	select {
	case <-client.Done():
		t.Fatal("Done closed during a successful reconnect")
	case <-reconnected:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for reconnect")
	}

	select {
	case <-client.Done():
		t.Fatal("Done closed after reconnect succeeded")
	default:
	}

	releaseOnce.Do(func() {
		close(transport.releaseSecond)
	})
	if err := client.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	select {
	case <-client.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for Done after Close")
	}
}
