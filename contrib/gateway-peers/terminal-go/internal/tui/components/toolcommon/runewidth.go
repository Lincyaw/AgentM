package toolcommon

import (
	"github.com/clipperhouse/displaywidth"
	"github.com/clipperhouse/uax29/v2/graphemes"
)

// runeWidth returns the display width of a rune for terminal rendering.
// It optimizes for common cases (ASCII, Latin-1) to avoid expensive library calls.
func runeWidth(r rune) int {
	// Fast path: printable ASCII (most common case)
	if r >= 32 && r < 127 {
		return 1
	}
	return runeWidthSlow(r)
}

// runeWidthSlow handles non-ASCII runes. Kept separate to allow the fast path
// to be inlined while this function handles the complex cases.
//
//go:noinline
func runeWidthSlow(r rune) int {
	// Control characters (C0 and DEL)
	if r < 32 || r == 127 {
		return 0
	}

	// C1 control characters (0x80-0x9F)
	if r >= 0x80 && r < 0xA0 {
		return 0
	}

	// Latin-1 Supplement printable (0xA0-0xFF) and Latin Extended A/B (0x100-0x24F)
	// These are all single-width characters (accented letters, etc.)
	if r <= 0x24F {
		return 1
	}

	// For everything else (CJK, emoji, etc.), use the library
	cluster := graphemes.FromString(string(r)).First()
	return displaywidth.String(cluster)
}
