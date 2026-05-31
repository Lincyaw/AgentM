package util

import (
	"fmt"
	"regexp"
)

var urlPattern = regexp.MustCompile(`https?://[^\s<>\]\)]+`)

// Linkify wraps URLs in the text with OSC 8 hyperlink escape sequences
// so compatible terminals make them clickable.
func Linkify(text string) string {
	return urlPattern.ReplaceAllStringFunc(text, func(url string) string {
		return fmt.Sprintf("\033]8;;%s\033\\%s\033]8;;\033\\", url, url)
	})
}
