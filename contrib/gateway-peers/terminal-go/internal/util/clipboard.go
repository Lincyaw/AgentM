package util

import (
	"encoding/base64"
	"fmt"
	"os"
)

// CopyToClipboard writes text to the system clipboard using the OSC 52
// terminal escape sequence. This works over SSH and tmux without needing
// a clipboard binary. The escape is written to stderr so it doesn't
// interfere with stdout (which bubbletea owns).
func CopyToClipboard(text string) {
	b64 := base64.StdEncoding.EncodeToString([]byte(text))
	fmt.Fprintf(os.Stderr, "\033]52;c;%s\a", b64)
}
