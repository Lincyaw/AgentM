package util

import (
	"strings"
	"unicode"
)

// TokenKind classifies a token produced by tokenize.
type TokenKind int

const (
	TokenIdentifier  TokenKind = iota // letters, digits, underscores
	TokenWhitespace                   // spaces, tabs, etc.
	TokenPunctuation                  // everything else (one rune each)
)

// Token is a contiguous substring tagged with its kind.
type Token struct {
	Text string
	Kind TokenKind
}

// tokenize splits s into semantic tokens. Runs of identifier characters
// (unicode letters, digits, underscore) coalesce into one token; runs of
// whitespace coalesce similarly; every other rune becomes its own
// single-character punctuation token. The concatenation of all token texts
// always equals the original string.
func tokenize(s string) []Token {
	if s == "" {
		return nil
	}
	var tokens []Token
	var buf strings.Builder
	curKind := TokenKind(-1)

	flush := func() {
		if buf.Len() > 0 {
			tokens = append(tokens, Token{Text: buf.String(), Kind: curKind})
			buf.Reset()
		}
	}

	for _, r := range s {
		k := classifyRune(r)
		if k != curKind || k == TokenPunctuation {
			flush()
			curKind = k
		}
		buf.WriteRune(r)
	}
	flush()
	return tokens
}

func classifyRune(r rune) TokenKind {
	switch {
	case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_':
		return TokenIdentifier
	case unicode.IsSpace(r):
		return TokenWhitespace
	default:
		return TokenPunctuation
	}
}

// DiffSegment is a contiguous piece of text within a line, with a flag
// indicating whether it differs from the corresponding position in the
// paired line. Callers render changed segments in a highlight style and
// unchanged segments in the base diff color.
type DiffSegment struct {
	Text    string
	Changed bool
}

// WordDiff tokenizes old and new, runs an LCS match on the token
// sequences, and returns per-side segment slices marking which parts
// changed. The concatenation of segment texts on each side reconstructs
// the original string exactly.
func WordDiff(old, new string) (oldSegs, newSegs []DiffSegment) {
	if old == new {
		seg := []DiffSegment{{Text: old, Changed: false}}
		return seg, seg
	}

	oldToks := tokenize(old)
	newToks := tokenize(new)

	if len(oldToks) == 0 || len(newToks) == 0 {
		oldSegs = []DiffSegment{{Text: old, Changed: old != ""}}
		newSegs = []DiffSegment{{Text: new, Changed: new != ""}}
		return
	}

	lcsSeq := lcsTokens(oldToks, newToks)

	// Walk both token sequences and the LCS simultaneously, emitting
	// changed segments for tokens not in the LCS and unchanged segments
	// for tokens that are.
	oi, ni, li := 0, 0, 0
	var oBuf, nBuf strings.Builder
	flushChanged := func() {
		if oBuf.Len() > 0 {
			oldSegs = append(oldSegs, DiffSegment{Text: oBuf.String(), Changed: true})
			oBuf.Reset()
		}
		if nBuf.Len() > 0 {
			newSegs = append(newSegs, DiffSegment{Text: nBuf.String(), Changed: true})
			nBuf.Reset()
		}
	}

	for li < len(lcsSeq) {
		target := lcsSeq[li]
		// Consume old tokens until we hit the LCS token.
		for oi < len(oldToks) && oldToks[oi].Text != target.Text {
			oBuf.WriteString(oldToks[oi].Text)
			oi++
		}
		// Consume new tokens until we hit the LCS token.
		for ni < len(newToks) && newToks[ni].Text != target.Text {
			nBuf.WriteString(newToks[ni].Text)
			ni++
		}
		flushChanged()
		// Emit the matched token as unchanged on both sides.
		seg := DiffSegment{Text: target.Text, Changed: false}
		oldSegs = append(oldSegs, seg)
		newSegs = append(newSegs, seg)
		oi++
		ni++
		li++
	}
	// Remaining tokens after the last LCS match.
	for oi < len(oldToks) {
		oBuf.WriteString(oldToks[oi].Text)
		oi++
	}
	for ni < len(newToks) {
		nBuf.WriteString(newToks[ni].Text)
		ni++
	}
	flushChanged()

	return
}

// lcsTokens computes the longest common subsequence of two token slices
// using the classic O(n*m) DP table. Token counts per line are small so
// this is adequate.
func lcsTokens(a, b []Token) []Token {
	n, m := len(a), len(b)
	// dp[i][j] = LCS length of a[:i] and b[:j]
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}
	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if a[i-1].Text == b[j-1].Text {
				dp[i][j] = dp[i-1][j-1] + 1
			} else if dp[i-1][j] >= dp[i][j-1] {
				dp[i][j] = dp[i-1][j]
			} else {
				dp[i][j] = dp[i][j-1]
			}
		}
	}
	// Backtrack to recover the subsequence.
	result := make([]Token, dp[n][m])
	k := len(result) - 1
	for i, j := n, m; i > 0 && j > 0; {
		if a[i-1].Text == b[j-1].Text {
			result[k] = a[i-1]
			k--
			i--
			j--
		} else if dp[i-1][j] >= dp[i][j-1] {
			i--
		} else {
			j--
		}
	}
	return result
}
