package adapter

import "testing"

func TestSlashCommandDetectionMatchesGatewayRouterRule(t *testing.T) {
	cases := []struct {
		input string
		expect bool
	}{
		{"/", true},
		{"/help", true},
		{"//", false},
		{"//tmp/file.go", false},
		{"hello", false},
		{" /leading-space", false},
	}

	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			if got := _isSlashCommand(tc.input); got != tc.expect {
				t.Fatalf("_isSlashCommand(%q)=%v want %v", tc.input, got, tc.expect)
			}
		})
	}
}
