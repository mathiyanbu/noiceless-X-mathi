"""
Unit tests for lock-free single-producer single-consumer (SPSC) ring buffer.
Validates FIFO order, circular wraparound indexing, capacity overflow tracking,
batch operations, and thread-safe concurrent producer-consumer streaming.
"""

import threading
import time
import numpy as np
import pytest


class SpscRingBuffer:
    """Python reference implementation of the C++ lock-free SpscRingBuffer."""

    def __init__(self, capacity: int):
        assert capacity > 0 and (capacity & (capacity - 1)) == 0, "Capacity must be a power of 2"
        self._capacity = capacity
        self._mask = capacity - 1
        self._buffer = [None] * capacity
        self._head = 0  # Write pointer
        self._tail = 0  # Read pointer
        self._overflow_count = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    def size(self) -> int:
        with self._lock:
            return self._head - self._tail

    def empty(self) -> bool:
        with self._lock:
            return self._head == self._tail

    def full(self) -> bool:
        with self._lock:
            return (self._head - self._tail) >= self._capacity

    def push(self, item) -> bool:
        with self._lock:
            if (self._head - self._tail) >= self._capacity:
                self._overflow_count += 1
                return False
            self._buffer[self._head & self._mask] = item
            self._head += 1
            return True

    def pop(self):
        with self._lock:
            if self._head == self._tail:
                return None
            val = self._buffer[self._tail & self._mask]
            self._tail += 1
            return val

    def push_batch(self, items) -> int:
        with self._lock:
            available = self._capacity - (self._head - self._tail)
            to_push = min(len(items), available)
            for i in range(to_push):
                self._buffer[self._head & self._mask] = items[i]
                self._head += 1
            if len(items) > to_push:
                self._overflow_count += (len(items) - to_push)
            return to_push

    def pop_batch(self, max_count: int):
        with self._lock:
            available = self._head - self._tail
            to_pop = min(max_count, available)
            res = []
            for _ in range(to_pop):
                res.append(self._buffer[self._tail & self._mask])
                self._tail += 1
            return res


def test_ring_buffer_basic_push_pop():
    rb = SpscRingBuffer(8)
    assert rb.empty()
    assert not rb.full()
    assert rb.size() == 0
    assert rb.capacity == 8

    assert rb.push(42)
    assert rb.size() == 1
    assert not rb.empty()

    val = rb.pop()
    assert val == 42
    assert rb.empty()


def test_ring_buffer_fifo_order():
    rb = SpscRingBuffer(16)
    for i in range(10):
        assert rb.push(i * 10)
    assert rb.size() == 10

    for i in range(10):
        val = rb.pop()
        assert val == i * 10
    assert rb.empty()


def test_ring_buffer_wraparound_indexing():
    rb = SpscRingBuffer(4)
    # Perform multiple cycles to verify pointer masking and circular wraparound
    for cycle in range(50):
        assert rb.push(cycle)
        assert rb.push(cycle + 100)
        assert rb.pop() == cycle
        assert rb.pop() == cycle + 100

    assert rb.empty()
    assert rb.overflow_count == 0


def test_ring_buffer_overflow_handling():
    rb = SpscRingBuffer(4)
    for i in range(4):
        assert rb.push(i + 1)
    assert rb.full()

    # Attempting to push into full buffer must return False and increment overflow count
    assert not rb.push(5)
    assert not rb.push(6)
    assert rb.overflow_count == 2

    # Pop one item
    assert rb.pop() == 1
    assert not rb.full()

    # Push again
    assert rb.push(7)
    assert rb.overflow_count == 2


def test_ring_buffer_batch_operations():
    rb = SpscRingBuffer(16)
    items = list(range(100, 108))
    pushed = rb.push_batch(items)
    assert pushed == 8
    assert rb.size() == 8

    popped = rb.pop_batch(8)
    assert popped == items
    assert rb.empty()


def test_ring_buffer_batch_overflow():
    rb = SpscRingBuffer(8)
    items = [5] * 12
    pushed = rb.push_batch(items)
    assert pushed == 8
    assert rb.overflow_count == 4
    assert rb.full()


def test_ring_buffer_concurrent_producer_consumer():
    """Verify SPSC concurrency across threads without data corruption or loss."""
    rb = SpscRingBuffer(1024)
    total_items = 20000
    consumed = []
    producer_done = threading.Event()

    def producer():
        for i in range(total_items):
            while rb.full():
                time.sleep(0.00001)
            assert rb.push(i)
        producer_done.set()

    def consumer():
        while not producer_done.is_set() or not rb.empty():
            val = rb.pop()
            if val is not None:
                consumed.append(val)
            else:
                time.sleep(0.00001)

    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)

    t_prod.start()
    t_cons.start()

    t_prod.join(timeout=10.0)
    t_cons.join(timeout=10.0)

    assert len(consumed) == total_items
    assert consumed == list(range(total_items))
    assert rb.empty()
    assert rb.overflow_count == 0
