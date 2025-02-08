import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "reinforce-the-puck"))

from components.data_structures import SumTree


@pytest.fixture
def sum_tree():
    """Fixture to create a SumTree instance with a fixed capacity."""
    capacity = 4
    return SumTree(capacity)


def test_add_and_retrieve_data(sum_tree):
    """Test adding data and retrieving it from the SumTree."""
    sum_tree.add(1.0, "data1")
    sum_tree.add(2.0, "data2")
    sum_tree.add(3.0, "data3")
    sum_tree.add(4.0, "data4")

    # Check if data is stored correctly
    assert sum_tree.data[0] == "data1"
    assert sum_tree.data[1] == "data2"
    assert sum_tree.data[2] == "data3"
    assert sum_tree.data[3] == "data4"

    # Check if the tree is updated correctly
    assert sum_tree.tree[-4] == pytest.approx(1.0)
    assert sum_tree.tree[-3] == pytest.approx(2.0)
    assert sum_tree.tree[-2] == pytest.approx(3.0)
    assert sum_tree.tree[-1] == pytest.approx(4.0)
    assert sum_tree.tree[0] == pytest.approx(10.0)  # Root node sum


def test_update_priority(sum_tree):
    """Update the priority of any node in the tree."""
    sum_tree.add(1.0, "data1")
    sum_tree.add(2.0, "data2")
    sum_tree.add(3.0, "data3")
    sum_tree.add(4.0, "data4")

    # Update priority of the first leaf
    sum_tree.update(-4, 5.0)

    # Check if the tree is updated correctly
    assert sum_tree.tree[-4] == pytest.approx(5.0)  # Updated leaf
    assert sum_tree.tree[0] == pytest.approx(14.0)  # Updated root sum


def test_update_leaf_priority(sum_tree):
    """Test updating the priority of a leaf node."""
    sum_tree.add(1.0, "data1")
    sum_tree.add(2.0, "data2")
    sum_tree.add(3.0, "data3")
    sum_tree.add(4.0, "data4")

    # Update priority of the first leaf
    sum_tree.update_leaf(0, 5.0)

    # Check if the tree is updated correctly
    assert sum_tree.tree[-4] == pytest.approx(5.0)  # Updated leaf
    assert sum_tree.tree[0] == pytest.approx(14.0)  # Updated root sum


def test_sampling(sum_tree):
    """Test sampling data based on priorities."""
    sum_tree.add(1.0, "data1")
    sum_tree.add(2.0, "data2")
    sum_tree.add(3.0, "data3")
    sum_tree.add(4.0, "data4")

    # Sample a value
    sampled_index, sampled_priority, sampled_data = sum_tree.sample(7.5)

    # Check if the sampled data is correct
    assert sampled_data == "data4"
    assert sampled_priority == pytest.approx(4.0)


def test_circular_buffer(sum_tree):
    """Test the circular buffer behavior when capacity is exceeded."""
    sum_tree.add(1.0, "data1")
    sum_tree.add(2.0, "data2")
    sum_tree.add(3.0, "data3")
    sum_tree.add(4.0, "data4")
    sum_tree.add(5.0, "data5")  # Overwrites "data1"

    # Check if the oldest data is replaced
    assert sum_tree.data[0] == "data5"
    assert sum_tree.tree[-4] == pytest.approx(5.0)  # Updated leaf


def test_tree_structure(sum_tree):
    """Test the structure of the tree after multiple operations."""
    sum_tree.add(1.0, "data1")
    sum_tree.add(2.0, "data2")
    sum_tree.add(3.0, "data3")
    sum_tree.add(4.0, "data4")

    # Check intermediate nodes
    assert sum_tree.tree[1] == pytest.approx(3.0)  # Sum of first two leaves
    assert sum_tree.tree[2] == pytest.approx(7.0)  # Sum of last two leaves
    assert sum_tree.tree[0] == pytest.approx(10.0)  # Root node sum


def test_batch_sampling(sum_tree):
    """Test selecting a transition based on priority."""
    sum_tree.add(1.0, "data1")
    sum_tree.add(2.0, "data2")
    sum_tree.add(3.0, "data3")
    sum_tree.add(4.0, "data4")

    # Sample a batch of transitions
    indices, priorities, data = sum_tree.sample_batch(4)

    # Check if sampled indices are valid
    assert len(indices) == 4
    assert len(priorities) == 4
    assert len(data) == 4

    # Check if sampled data is valid
    for d in data:
        assert d in ["data1", "data2", "data3", "data4"]


def test_get_max_priority(sum_tree):
    """Test getting the maximum priority from the tree."""
    sum_tree.add(1.0, "data1")
    sum_tree.add(2.0, "data2")
    sum_tree.add(10.0, "data3")
    sum_tree.add(4.0, "data4")

    # Check if the max priority is correct
    assert sum_tree.max() == pytest.approx(10.0)


def test_empty_tree_sampling(sum_tree):
    """Test sampling from an empty tree."""
    with pytest.raises(ValueError):
        sum_tree.sample_batch(1)  # Sampling from an empty tree should raise an error


def test_add_batch(sum_tree):
    """Test adding a batch of data to the SumTree."""
    priorities = [1.0, 2.0, 3.0, 4.0]
    data = ["data1", "data2", "data3", "data4"]

    sum_tree.add_batch(priorities, data)

    # Check if data is stored correctly
    assert sum_tree.data[0] == "data1"
    assert sum_tree.data[1] == "data2"
    assert sum_tree.data[2] == "data3"
    assert sum_tree.data[3] == "data4"

    # Check if the tree is updated correctly
    assert sum_tree.tree[-4] == pytest.approx(1.0)
    assert sum_tree.tree[-3] == pytest.approx(2.0)
    assert sum_tree.tree[-2] == pytest.approx(3.0)
    assert sum_tree.tree[-1] == pytest.approx(4.0)
    assert sum_tree.tree[0] == pytest.approx(10.0)  # Root node sum
