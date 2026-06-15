package animation

// Stopper is implemented by views that register with the animation
// coordinator. When such a view is removed from the UI, StopAnimation must
// be called to unregister its subscription; a leaked subscription keeps the
// tick stream alive forever.
type Stopper interface {
	StopAnimation()
}

// StopView stops the animation subscription of a view being removed, when
// it has one. Safe to call on any view.
func StopView(view any) {
	if stopper, ok := view.(Stopper); ok {
		stopper.StopAnimation()
	}
}
