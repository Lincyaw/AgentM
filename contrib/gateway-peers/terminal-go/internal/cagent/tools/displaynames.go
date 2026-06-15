package tools

import "cmp"

func (t *Tool) DisplayName() string {
	return cmp.Or(t.Annotations.Title, t.Name)
}
