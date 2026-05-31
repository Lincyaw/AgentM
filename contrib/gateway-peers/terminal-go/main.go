package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/AoyangSpace/agentm-terminal/internal/app"
	"github.com/AoyangSpace/agentm-terminal/internal/theme"
)

func main() {
	th := theme.DarkTheme()
	m := app.NewModel(th)
	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
