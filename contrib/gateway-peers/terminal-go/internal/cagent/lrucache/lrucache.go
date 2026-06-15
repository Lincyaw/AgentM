// Package lrucache provides a small generic Least-Recently-Used cache.
//
// The cache is bounded to a fixed maximum number of entries and evicts the
// least recently used entry when full. It is NOT safe for concurrent use;
// callers must provide their own synchronization.
//
// Note that Get is a mutating operation: it promotes the looked-up entry to
// most-recently-used. Callers using a sync.RWMutex must therefore acquire the
// write lock around Get, not the read lock.
package lrucache

import "container/list"

// LRU is a generic LRU cache holding at most maxSize entries.
//
// Front of the internal list is the most-recently-used entry; back is the
// least-recently-used and the next eviction candidate.
type LRU[K comparable, V any] struct {
	maxSize int
	items   map[K]*list.Element
	order   *list.List
}

type entry[K comparable, V any] struct {
	key   K
	value V
}

// New creates an LRU cache that holds at most maxSize entries. A non-positive
// maxSize is clamped to 1 so the cache always retains at least one entry.
func New[K comparable, V any](maxSize int) *LRU[K, V] {
	if maxSize < 1 {
		maxSize = 1
	}
	return &LRU[K, V]{
		maxSize: maxSize,
		items:   make(map[K]*list.Element, maxSize),
		order:   list.New(),
	}
}

// Get retrieves a value, promoting it to most-recently-used. Returns the
// value and true if found, or the zero value and false otherwise.
//
// Get mutates the cache (it updates the recency list), so callers using an
// RWMutex must hold the write lock, not the read lock.
func (c *LRU[K, V]) Get(key K) (V, bool) {
	if elem, ok := c.items[key]; ok {
		c.order.MoveToFront(elem)
		return entryOf[K, V](elem).value, true
	}
	var zero V
	return zero, false
}

// Put adds or updates a key-value pair. If the cache is at capacity the least
// recently used entry is evicted to make room.
func (c *LRU[K, V]) Put(key K, value V) {
	if elem, ok := c.items[key]; ok {
		c.order.MoveToFront(elem)
		entryOf[K, V](elem).value = value
		return
	}

	if c.order.Len() >= c.maxSize {
		oldest := c.order.Back()
		c.order.Remove(oldest)
		delete(c.items, entryOf[K, V](oldest).key)
	}

	c.items[key] = c.order.PushFront(&entry[K, V]{key: key, value: value})
}

// Delete removes the entry for key, if any.
func (c *LRU[K, V]) Delete(key K) {
	elem, ok := c.items[key]
	if !ok {
		return
	}
	c.order.Remove(elem)
	delete(c.items, key)
}

// Clear removes all entries while retaining the underlying capacity.
func (c *LRU[K, V]) Clear() {
	clear(c.items)
	c.order.Init()
}

// Len returns the number of entries currently in the cache.
func (c *LRU[K, V]) Len() int {
	return c.order.Len()
}

// Range calls fn on every entry, from least-recently-used to
// most-recently-used. Iteration stops if fn returns false. Mutating the cache
// from fn is not supported.
func (c *LRU[K, V]) Range(fn func(key K, value V) bool) {
	for elem := c.order.Back(); elem != nil; elem = elem.Prev() {
		e := entryOf[K, V](elem)
		if !fn(e.key, e.value) {
			return
		}
	}
}

func entryOf[K comparable, V any](elem *list.Element) *entry[K, V] {
	return elem.Value.(*entry[K, V])
}
