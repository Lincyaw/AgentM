package wire

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"net"
	"os"

	"nhooyr.io/websocket"
)

// Transport abstracts a bidirectional connection to the gateway.
type Transport interface {
	Connect(ctx context.Context) (io.ReadWriteCloser, error)
}

// UnixTransport connects via Unix domain socket.
type UnixTransport struct {
	Path string
}

// Connect dials the Unix socket and returns the connection.
func (t *UnixTransport) Connect(ctx context.Context) (io.ReadWriteCloser, error) {
	var d net.Dialer
	conn, err := d.DialContext(ctx, "unix", t.Path)
	if err != nil {
		return nil, fmt.Errorf("unix dial %s: %w", t.Path, err)
	}
	return conn, nil
}

// WSTransport connects via WebSocket.
type WSTransport struct {
	URL       string
	Token     string
	TLSConfig *tls.Config
}

// Connect establishes a WebSocket connection and returns an io.ReadWriteCloser adapter.
func (t *WSTransport) Connect(ctx context.Context) (io.ReadWriteCloser, error) {
	opts := &websocket.DialOptions{}
	if t.TLSConfig != nil {
		opts.HTTPClient = newTLSHTTPClient(t.TLSConfig)
	}
	if t.Token != "" {
		opts.HTTPHeader = make(map[string][]string)
		opts.HTTPHeader["Authorization"] = []string{"Bearer " + t.Token}
	}

	conn, _, err := websocket.Dial(ctx, t.URL, opts)
	if err != nil {
		return nil, fmt.Errorf("websocket dial %s: %w", t.URL, err)
	}

	return &wsReadWriteCloser{conn: conn, ctx: ctx}, nil
}

// wsReadWriteCloser adapts a websocket.Conn to io.ReadWriteCloser.
// Each WebSocket message contains exactly one length-prefixed frame.
type wsReadWriteCloser struct {
	conn *websocket.Conn
	ctx  context.Context
	buf  bytes.Buffer
}

func (w *wsReadWriteCloser) Read(p []byte) (int, error) {
	if w.buf.Len() > 0 {
		return w.buf.Read(p)
	}
	_, data, err := w.conn.Read(w.ctx)
	if err != nil {
		return 0, err
	}
	w.buf.Write(data)
	return w.buf.Read(p)
}

func (w *wsReadWriteCloser) Write(p []byte) (int, error) {
	err := w.conn.Write(w.ctx, websocket.MessageBinary, p)
	if err != nil {
		return 0, err
	}
	return len(p), nil
}

func (w *wsReadWriteCloser) Close() error {
	return w.conn.Close(websocket.StatusNormalClosure, "bye")
}

// LoadTLSCACert reads a CA certificate file and returns a tls.Config
// with the cert added to the root CA pool.
func LoadTLSCACert(caPath string) (*tls.Config, error) {
	caCert, err := os.ReadFile(caPath)
	if err != nil {
		return nil, fmt.Errorf("read CA cert %s: %w", caPath, err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caCert) {
		return nil, fmt.Errorf("failed to parse CA cert from %s", caPath)
	}
	return &tls.Config{RootCAs: pool}, nil
}
