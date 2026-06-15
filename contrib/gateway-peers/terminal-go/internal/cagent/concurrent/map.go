package concurrent

import (
	"maps"
	"sync"
)

type Map[K comparable, V any] struct {
	mu     sync.RWMutex
	values map[K]V
}

func NewMap[K comparable, V any]() *Map[K, V] {
	return &Map[K, V]{
		values: make(map[K]V),
	}
}

func (m *Map[K, V]) Load(key K) (V, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	val, ok := m.values[key]
	return val, ok
}

func (m *Map[K, V]) Store(key K, value V) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.values == nil {
		m.values = make(map[K]V)
	}
	m.values[key] = value
}

func (m *Map[K, V]) Delete(key K) {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.values, key)
}

func (m *Map[K, V]) Length() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.values)
}

// LoadOrStore returns the existing value for key if present; otherwise it
// stores and returns value. The loaded result is true if the value was
// loaded, false if stored.
func (m *Map[K, V]) LoadOrStore(key K, value V) (V, bool) {
	m.mu.RLock()
	if existing, ok := m.values[key]; ok {
		m.mu.RUnlock()
		return existing, true
	}
	m.mu.RUnlock()

	m.mu.Lock()
	defer m.mu.Unlock()

	// Re-check under the write lock: another goroutine may have stored
	// the key between releasing the read lock and acquiring the write lock.
	if existing, ok := m.values[key]; ok {
		return existing, true
	}
	if m.values == nil {
		m.values = make(map[K]V)
	}
	m.values[key] = value
	return value, false
}

// Clear removes all entries from the map.
func (m *Map[K, V]) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.values = make(map[K]V)
}

// Range calls f for every key/value pair in the map. Iteration stops early if
// f returns false.
//
// Range iterates over a snapshot of the map taken under a read lock; f is
// invoked without holding any lock, which means callbacks may safely call
// other methods on the same Map (including Store and Delete) without
// deadlocking. As a consequence, mutations performed during iteration are not
// reflected in the values seen by f.
func (m *Map[K, V]) Range(f func(key K, value V) bool) {
	m.mu.RLock()
	snapshot := maps.Clone(m.values)
	m.mu.RUnlock()

	for k, v := range snapshot {
		if !f(k, v) {
			break
		}
	}
}
