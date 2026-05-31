package wire

import (
	"fmt"
	"strings"
)

// DefaultSocketURL is the default Unix socket path for the gateway.
const DefaultSocketURL = "unix:///tmp/agentm-gateway.sock"

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
