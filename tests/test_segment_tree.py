import operator

import numpy as np

from agilerl.components.segment_tree import MinSegmentTree, SegmentTree, SumSegmentTree


# Create a SegmentTree object with valid capacity, operation, and init_value.
def test_valid_segment_tree_creation():
    capacity = 8
    operation = operator.add
    init_value = 0.0

    segment_tree = SegmentTree(capacity, operation, init_value)

    assert segment_tree.capacity == capacity
    assert segment_tree.tree == [init_value] * (2 * capacity)
    assert segment_tree.operation == operation


# Set a value in the tree using __setitem__ and retrieve it using __getitem__.
def test_set_and_retrieve_value():
    capacity = 8
    operation = operator.add
    init_value = 0.0

    segment_tree = SegmentTree(capacity, operation, init_value)

    index = 3
    value = 5.0

    segment_tree[index] = value

    assert segment_tree[index] == value


def test_tree_set():
    tree = SumSegmentTree(4)

    tree[2] = 1.0
    tree[3] = 3.0

    assert np.isclose(tree.sum(), 4.0)
    assert np.isclose(tree.sum(0, 2), 0.0)
    assert np.isclose(tree.sum(0, 3), 1.0)
    assert np.isclose(tree.sum(2, 3), 1.0)
    assert np.isclose(tree.sum(2, -1), 1.0)
    assert np.isclose(tree.sum(2, 4), 4.0)


def test_tree_set_overlap():
    tree = SumSegmentTree(4)

    tree[2] = 1.0
    tree[2] = 3.0

    assert np.isclose(tree.sum(), 3.0)
    assert np.isclose(tree.sum(2, 3), 3.0)
    assert np.isclose(tree.sum(2, -1), 3.0)
    assert np.isclose(tree.sum(2, 4), 3.0)
    assert np.isclose(tree.sum(1, 2), 0.0)


def test_prefixsum_idx():
    tree = SumSegmentTree(4)

    tree[2] = 1.0
    tree[3] = 3.0

    assert tree.retrieve(0.0) == 2
    assert tree.retrieve(0.5) == 2
    assert tree.retrieve(0.99) == 2
    assert tree.retrieve(1.01) == 3
    assert tree.retrieve(3.00) == 3
    assert tree.retrieve(4.00) == 3


def test_prefixsum_idx2():
    tree = SumSegmentTree(4)

    tree[0] = 0.5
    tree[1] = 1.0
    tree[2] = 1.0
    tree[3] = 3.0

    assert tree.retrieve(0.00) == 0
    assert tree.retrieve(0.55) == 1
    assert tree.retrieve(0.99) == 1
    assert tree.retrieve(1.51) == 2
    assert tree.retrieve(3.00) == 3
    assert tree.retrieve(5.50) == 3


def test_max_interval_tree():
    tree = MinSegmentTree(4)

    tree[0] = 1.0
    tree[2] = 0.5
    tree[3] = 3.0

    assert np.isclose(tree.min(), 0.5)
    assert np.isclose(tree.min(0, 2), 1.0)
    assert np.isclose(tree.min(0, 3), 0.5)
    assert np.isclose(tree.min(0, -1), 0.5)
    assert np.isclose(tree.min(2, 4), 0.5)
    assert np.isclose(tree.min(3, 4), 3.0)

    tree[2] = 0.7

    assert np.isclose(tree.min(), 0.7)
    assert np.isclose(tree.min(0, 2), 1.0)
    assert np.isclose(tree.min(0, 3), 0.7)
    assert np.isclose(tree.min(0, -1), 0.7)
    assert np.isclose(tree.min(2, 4), 0.7)
    assert np.isclose(tree.min(3, 4), 3.0)

    tree[2] = 4.0

    assert np.isclose(tree.min(), 1.0)
    assert np.isclose(tree.min(0, 2), 1.0)
    assert np.isclose(tree.min(0, 3), 1.0)
    assert np.isclose(tree.min(0, -1), 1.0)
    assert np.isclose(tree.min(2, 4), 3.0)
    assert np.isclose(tree.min(2, 3), 4.0)
    assert np.isclose(tree.min(2, -1), 4.0)
    assert np.isclose(tree.min(3, 4), 3.0)
