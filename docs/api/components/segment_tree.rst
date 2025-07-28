Segment Trees
=============

Segment trees are efficient data structures used for range queries and updates, particularly useful in prioritized experience replay.
The implementation is based on OpenAI's baselines repository and provides efficient operations for priority-based sampling in reinforcement learning.

A segment tree is a binary tree where each leaf represents an element in an array, and each internal node represents some operation
(like sum or minimum) over a range of elements. This allows for O(log n) query and update operations, making it ideal for efficiently
sampling experiences based on priorities in prioritized replay buffers.

The base ``SegmentTree`` class provides the foundation, while ``SumSegmentTree`` and ``MinSegmentTree`` provide specialized
implementations for sum and minimum operations respectively.

.. code-block:: python

    from agilerl.components.segment_tree import SumSegmentTree, MinSegmentTree

    # Create a sum segment tree for priority-based sampling
    sum_tree = SumSegmentTree(capacity=1024)

    # Create a min segment tree for finding minimum priorities
    min_tree = MinSegmentTree(capacity=1024)

Classes
-------

.. autoclass:: agilerl.components.segment_tree.SegmentTree
  :members:
  :inherited-members:

.. autoclass:: agilerl.components.segment_tree.SumSegmentTree
  :members:
  :inherited-members:

.. autoclass:: agilerl.components.segment_tree.MinSegmentTree
  :members:
  :inherited-members:
