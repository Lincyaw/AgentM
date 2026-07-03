package wire

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultSocketURL mirrors the Python gateway daemon default:
// $AGENTM_RUNTIME_DIR/gateway.sock when set, else
// $TMPDIR/agentm-<uid>-<AGENTM_HOME hash>/gateway.sock.
func DefaultSocketURL() string {
	runtime := os.Getenv("AGENTM_RUNTIME_DIR")
	if runtime == "" {
		home := os.Getenv("AGENTM_HOME")
		if home == "" {
			if userHome, err := os.UserHomeDir(); err == nil && userHome != "" {
				home = filepath.Join(userHome, ".agentm")
			} else {
				home = ".agentm"
			}
		}
		sum := sha1.Sum([]byte(home))
		digest := hex.EncodeToString(sum[:])[:8]
		runtime = filepath.Join(os.TempDir(), fmt.Sprintf("agentm-%d-%s", os.Getuid(), digest))
	}
	return "unix://" + filepath.Join(runtime, "gateway.sock")
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
