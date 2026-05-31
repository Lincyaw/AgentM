package wire

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
)

// MaxFrameSize is the maximum allowed frame payload (10 MB).
const MaxFrameSize = 10_000_000

// WriteFrame writes a length-prefixed JSON frame to w.
// Format: 4-byte big-endian length + N bytes UTF-8 JSON body.
func WriteFrame(w io.Writer, env *Envelope) error {
	data, err := json.Marshal(env)
	if err != nil {
		return fmt.Errorf("marshal envelope: %w", err)
	}
	length := uint32(len(data))
	if err := binary.Write(w, binary.BigEndian, length); err != nil {
		return fmt.Errorf("write length prefix: %w", err)
	}
	if _, err := w.Write(data); err != nil {
		return fmt.Errorf("write payload: %w", err)
	}
	return nil
}

// ReadFrame reads one length-prefixed JSON frame from r.
func ReadFrame(r io.Reader) (*Envelope, error) {
	var length uint32
	if err := binary.Read(r, binary.BigEndian, &length); err != nil {
		return nil, fmt.Errorf("read length prefix: %w", err)
	}
	if length > MaxFrameSize {
		return nil, fmt.Errorf("frame too large: %d bytes (max %d)", length, MaxFrameSize)
	}
	data := make([]byte, length)
	if _, err := io.ReadFull(r, data); err != nil {
		return nil, fmt.Errorf("read payload: %w", err)
	}
	var env Envelope
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, fmt.Errorf("unmarshal envelope: %w", err)
	}
	return &env, nil
}
