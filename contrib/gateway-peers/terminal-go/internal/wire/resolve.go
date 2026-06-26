package wire

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultSocketURL mirrors the Python gateway's default_socket_url():
// $XDG_RUNTIME_DIR/agentm-gw.sock when set, else /tmp/agentm-gw-<uid>.sock.
func DefaultSocketURL() string {
	if runtime := os.Getenv("XDG_RUNTIME_DIR"); runtime != "" {
		if info, err := os.Stat(runtime); err == nil && info.IsDir() {
			return "unix://" + filepath.Join(runtime, "agentm-gw.sock")
		}
	}
	return fmt.Sprintf("unix:///tmp/agentm-gw-%d.sock", os.Getuid())
}

// ResolveTransport parses a URL string into a Transport.
// Supported schemes: unix://, ws://, wss://.
func ResolveTransport(url string, tlsCACert string) (Transport, error) {
	switch {
	case strings.HasPrefix(url, "unix://"):
		path := strings.TrimPrefix(url, "unix://")
		return &UnixTransport{Path: path}, nil

	case strings.HasPrefix(url, "ws://"):
		return &WSTransport{URL: url}, nil

	case strings.HasPrefix(url, "wss://"):
		t := &WSTransport{URL: url}
		if tlsCACert != "" {
			cfg, err := LoadTLSCACert(tlsCACert)
			if err != nil {
				return nil, fmt.Errorf("load TLS CA: %w", err)
			}
			t.TLSConfig = cfg
		}
		return t, nil

	default:
		return nil, fmt.Errorf("unsupported scheme in URL: %s", url)
	}
}
