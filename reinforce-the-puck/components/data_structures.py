import numpy as np


class SumTree:
    """Data structure to store priorities of transitions. Recommended by:
    Schaul, Tom. "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952 (2015).

    Following properties:
    - Binary tree
    - Parent node = Sum of child nodes
    - Leaf nodes = Priorities of transitions
    - Update in O(log n)
    - Sample in O(log n)
    """

    def __init__(self, capacity):
        """Initialize the SumTree object.

        Variables:
        - capacity: Capacity of the SumTree
        - tree: Array to store the SumTree, where the index represents a node
                shape = 2 * capacity - 1 (max size of the tree)
        - data: Array to store the transitions/data behind each leaf node
        - size: Current size of the SumTree (max = capacity)
        - current_idx: Current index to add new transitions (circular buffer)

        Args:
            capacity (int): Capacity of the SumTree.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.current_idx = 0

    def add(self, priority, data):
        """Add a new priority and data to the SumTree.
        If the SumTree is full, the oldest transition is replaced.
        Priority is added to the leaf node and propagated to the parent nodes.

        Args:
            priority (float): Priority of the transition.
            data (object): Data of the transition.
        """
        idx = self.current_idx + self.capacity - 1
        self.data[self.current_idx] = data
        self.update(idx, priority)
        self.current_idx = (self.current_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def leaf2global(self, leaf):
        """Convert a leaf index to a global index.

        Args:
            leaf (int): Index of the leaf node.

        Returns:
            int: Index of the global node.
        """
        return leaf + self.capacity - 1

    def update(self, index, priority):
        """Update the priority of a transition.
        Where the change = new priority - old priority.
        Propagate the change to the parent nodes.

        Args:
            index (int): Index of the transition.
            priority (float): New priority of the transition.
        """
        # Handle negative index
        if index < 0:
            index = len(self.tree) + index

        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def update_leaf(self, leaf, priority):
        """Update the priority of a transition based on the leaf index.

        Args:
            leaf (int): Index of the leaf node.
            priority (float): New priority of the transition.
        """
        index = self.leaf2global(leaf)
        self.update(index, priority)

    def _propagate(self, index, change):
        """Propagate the change in priority to the parent nodes.
        Takes the parent node and adds the change to the node.

        New parent = (index - 1) // 2 --> Based on array

        Args:
            index (int): Index of the transition.
            change (float): Change in priority.
        """
        parent = (index - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def _retrieve(self, index, s):
        """Retrieve the index of a transition based on a sample.

        Args:
            index (int): Index of the transition.
            s (float): Sample value.
        """
        left = 2 * index + 1
        right = left + 1

        while left < len(self.tree):
            if s <= self.tree[left]:
                index = left
            else:
                s -= self.tree[left]
                index = right
            left = 2 * index + 1
            right = left + 1

        return index

    def sample(self, s):
        """Retrieve the index, priority, and data of a transition based on a sample value.

        Args:
            s (float): A random value between 0 and the total sum of priorities.

        Returns:
            tuple: (index, priority, data) of the sampled transition.
        """
        index = 0  # Start at the root node

        # Traverse the tree to find the leaf node
        while index < self.capacity - 1:  # While the index is not a leaf node
            left_child = 2 * index + 1
            right_child = left_child + 1

            if s <= self.tree[left_child]:
                index = left_child
            else:
                s -= self.tree[left_child]
                index = right_child

        # Convert the leaf index to the data index
        leaf_index = index - (self.capacity - 1)
        priority = self.tree[index]
        data = self.data[leaf_index]

        return index, priority, data

    def sample_batch(self, batch_size: int) -> tuple[list, list, list]:
        """Sample a batch of transitions from the SumTree.


        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Batch of transitions (indices, priorities, data).
        """
        if self.size < batch_size:
            raise ValueError("The memory buffer is too small for the batch size.")

        batch = [[], [], []]  # Index, Priority, Data
        segment = self.tree[0] / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            index, priority, data = self.sample(s)
            batch[0].append(index)
            batch[1].append(priority)
            batch[2].append(data)

        return (*batch,)

    def get_sum(self):
        """Get the total sum of priorities."""
        return self.tree[0]

    def display_tree(self):
        """Display a NumPy array as a tree structure using for loops."""
        array = self.tree
        if len(array) == 0:
            print("The tree is empty.")
            return

        # Initialize the current level and the index for the next node to print
        current_level = 0
        index = 0
        level_count = 1  # Number of nodes at the current level

        while index < len(array):
            # Print the current level nodes
            for i in range(level_count):
                if index < len(array):
                    print(array[index], end=" ")
                    index += 1

            print()

            # Update the level count for the next level (2^current_level)
            level_count *= 2
            current_level += 1
