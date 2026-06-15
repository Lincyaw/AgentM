package path

import (
	"os"
	"regexp"
)

// jsEnvRef matches the JS-template form `${env.VAR}` (optional surrounding
// whitespace). Path fields historically only understood shell-style
// expansion via os.ExpandEnv, so the JS form was passed through as a literal.
// We normalize it to `${VAR}` so both syntaxes resolve identically here.
//
// Only the plain `${env.VAR}` reference is recognized; richer JS expressions
// such as `${env.VAR || 'default'}` are not evaluated in path fields (that
// would require the goja engine, which pkg/path cannot import).
//
// Kept in sync with jsEnvRefStrict in pkg/config/expansion_warnings.go: that
// helper warns about the same `${env.VAR}` form in fields that, unlike paths,
// cannot resolve it. The pattern is duplicated rather than shared to avoid an
// import cycle (pkg/environment already imports pkg/path).
var jsEnvRef = regexp.MustCompile(`\$\{\s*env\.([A-Za-z_][A-Za-z0-9_]*)\s*\}`)

// ExpandPath expands shell-like patterns in a file path:
//   - ~ or ~/ at the start is replaced with the user's home directory
//   - Environment variables like ${HOME} or $HOME are expanded
//   - The JS-template form ${env.HOME} is accepted as an alias for ${HOME}
func ExpandPath(p string) string {
	if p == "" {
		return p
	}

	// Normalize ${env.VAR} to ${VAR} so the JS-template syntax used elsewhere
	// in the config also works in path fields (issue #2615).
	p = jsEnvRef.ReplaceAllString(p, "${$1}")

	// Expand environment variables
	p = os.ExpandEnv(p)

	if expanded, err := ExpandHomeDir(p); err == nil {
		return expanded
	}

	return p
}
