package concurrent

import (
	"slices"
	"sync"
)

type Slice[V any] struct {
	mu     sync.RWMutex
	values []V
}

func NewSlice[V any]() *Slice[V] {
	return &Slice[V]{
		values: []V{},
	}
}

func (s *Slice[V]) Append(value V) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.values = append(s.values, value)
}

func (s *Slice[V]) Get(index int) (V, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if index < 0 || index >= len(s.values) {
		var zero V
		return zero, false
	}
	return s.values[index], true
}

func (s *Slice[V]) Set(index int, value V) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if index < 0 || index >= len(s.values) {
		return false
	}
	s.values[index] = value
	return true
}

func (s *Slice[V]) Length() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return len(s.values)
}

func (s *Slice[V]) All() []V {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return slices.Clone(s.values)
}

// Range calls f for every element in the slice. Iteration stops early if f
// returns false.
//
// f is invoked while a read lock is held on the slice. Callbacks must not
// call methods that acquire the write lock (Append, Set, Update, Clear) on
// the same Slice, or a deadlock will occur.
func (s *Slice[V]) Range(f func(index int, value V) bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for i, v := range s.values {
		if !f(i, v) {
			break
		}
	}
}

// Find returns the first element for which predicate returns true, along with
// its index, or the zero value and -1 if no element matches.
//
// predicate is invoked while a read lock is held on the slice. It must not
// call methods that acquire the write lock (Append, Set, Update, Clear) on
// the same Slice, or a deadlock will occur.
func (s *Slice[V]) Find(predicate func(V) bool) (V, int) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for i, v := range s.values {
		if predicate(v) {
			return v, i
		}
	}
	var zero V
	return zero, -1
}

// Update replaces the element at index with the result of f applied to the
// current value, returning true on success. If index is out of range, Update
// returns false and f is not called.
//
// f is invoked while the write lock is held on the slice. It must not call
// any other method on the same Slice, or a deadlock will occur.
func (s *Slice[V]) Update(index int, f func(V) V) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if index < 0 || index >= len(s.values) {
		return false
	}
	s.values[index] = f(s.values[index])
	return true
}

func (s *Slice[V]) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.values = nil
}
