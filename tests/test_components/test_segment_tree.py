import operator

import numpy as np

from agilerl.components.segment_tree import MinSegmentTree, SegmentTree, SumSegmentTree


class TestSegmentTreeInit:
    # Create a SegmentTree object with valid capacity, operation, and init_value.
    def test_valid_segment_tree_creation(self):
        capacity = 8
        operation = operator.add
        init_value = 0.0

        segment_tree = SegmentTree(capacity, operation, init_value)

        assert segment_tree.capacity == capacity
        assert segment_tree.tree.tolist() == [init_value] * (2 * capacity)
        assert segment_tree.operation == operation


class TestSegmentTreeSetItem:
    # Set a value in the tree using __setitem__ and retrieve it using __getitem__.
    def test_set_and_retrieve_value(self):
        capacity = 8
        operation = operator.add
        init_value = 0.0

        segment_tree = SegmentTree(capacity, operation, init_value)

        index = 3
        value = 5.0

        segment_tree[index] = value

        assert segment_tree[index] == value


class TestSumSegmentTreeSum:
    def test_tree_set(self):
        tree = SumSegmentTree(4)

        tree[2] = 1.0
        tree[3] = 3.0

        assert np.isclose(tree.sum(), 4.0)
        assert np.isclose(tree.sum(0, 2), 0.0)
        assert np.isclose(tree.sum(0, 3), 1.0)
        assert np.isclose(tree.sum(2, 3), 1.0)
        assert np.isclose(tree.sum(2, -1), 1.0)
        assert np.isclose(tree.sum(2, 4), 4.0)

    def test_tree_set_overlap(self):
        tree = SumSegmentTree(4)

        tree[2] = 1.0
        tree[2] = 3.0

        assert np.isclose(tree.sum(), 3.0)
        assert np.isclose(tree.sum(2, 3), 3.0)
        assert np.isclose(tree.sum(2, -1), 3.0)
        assert np.isclose(tree.sum(2, 4), 3.0)
        assert np.isclose(tree.sum(1, 2), 0.0)


class TestSumSegmentTreeRetrieve:
    def test_prefixsum_idx(self):
        tree = SumSegmentTree(4)

        tree[2] = 1.0
        tree[3] = 3.0

        assert tree.retrieve(0.0) == 2
        assert tree.retrieve(0.5) == 2
        assert tree.retrieve(0.99) == 2
        assert tree.retrieve(1.01) == 3
        assert tree.retrieve(3.00) == 3
        assert tree.retrieve(4.00) == 3

    def test_prefixsum_idx2(self):
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


class TestMinSegmentTreeMin:
    def test_max_interval_tree(self):
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


class TestSegmentTreeBatchOps:
    """The vectorised batch helpers must agree with the scalar API."""

    def test_get_batch_matches_getitem(self):
        tree = SumSegmentTree(8)
        for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
            tree[i] = v
        idxs = np.array([0, 3, 3, 7, 5])
        assert np.allclose(tree.get_batch(idxs), [tree[int(i)] for i in idxs])

    def test_update_batch_matches_sequential_sum(self):
        cap = 16
        idxs = [0, 1, 2, 5, 9, 15]
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        ref = SumSegmentTree(cap)
        for i, v in zip(idxs, vals, strict=False):
            ref[i] = v

        batch = SumSegmentTree(cap)
        batch.update_batch(np.array(idxs), np.array(vals))

        assert np.allclose(batch.tree, ref.tree)
        assert np.isclose(batch.sum(), ref.sum())

    def test_update_batch_matches_sequential_min(self):
        cap = 8
        idxs = [0, 2, 3, 7]
        vals = [1.0, 0.5, 3.0, 0.2]

        ref = MinSegmentTree(cap)
        for i, v in zip(idxs, vals, strict=False):
            ref[i] = v

        batch = MinSegmentTree(cap)
        batch.update_batch(np.array(idxs), np.array(vals))

        assert np.allclose(batch.tree, ref.tree)
        assert np.isclose(batch.min(), ref.min())

    def test_retrieve_batch_matches_scalar(self):
        cap = 16
        tree = SumSegmentTree(cap)
        rng = np.random.default_rng(0)
        tree.update_batch(np.arange(cap), rng.random(cap) + 0.01)

        ubs = np.linspace(0.0, tree.sum() - 1e-6, 50)
        batch_idx = tree.retrieve_batch(ubs).tolist()
        scalar_idx = [tree.retrieve(float(u)) for u in ubs]
        assert batch_idx == scalar_idx

    def test_update_batch_empty_is_noop(self):
        tree = SumSegmentTree(8)
        tree.update_batch(np.array([], dtype=np.int64), np.array([]))
        assert tree.sum() == 0.0
