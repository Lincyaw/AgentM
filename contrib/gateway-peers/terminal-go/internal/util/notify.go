package util

import (
	"fmt"
	"os"
)

// Notify sends a desktop notification via terminal escape sequences.
// Uses OSC 777 (iTerm2/Konsole) and OSC 99 (kitty) for broad compatibility.
// The notification is written to stderr to avoid interfering with bubbletea.
func Notify(title, body string) {
	// OSC 777 (iTerm2-style)
	fmt.Fprintf(os.Stderr, "\033]777;notify;%s;%s\a", title, body)
	// OSC 99 (kitty-style)
	fmt.Fprintf(os.Stderr, "\033]99;i=1:d=0:p=body;%s\033\\", body)
}
