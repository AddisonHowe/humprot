"""Compact Trie class

"""

import os
import numpy as np
from collections import deque


class CompactTrie:
    """Compact Trie with optional memory-mapped storage."""

    def __init__(
            self, 
            filename=None, 
            mode="r+", 
            initial_nodes=1024, 
            alphabet_size=20,
    ):
        """
        Parameters
        ----------
        filename : str or None
            If provided, use a memory-mapped file at this path.
            If None, the trie is stored fully in memory.
        mode : str
            File mode for np.memmap. One of: 'r+', 'w+', 'c'.
        initial_nodes : int
            Initial capacity of the trie.
        alphabet_size : int
            Number of possible child symbols per node.
        """
        self.alphabet_size = alphabet_size
        self.filename = filename
        self.mode = mode
        self.size = 0
        self.depth=0

        dtype = np.dtype([
            ("key", np.int8),            # symbol leading to this node
            ("value", np.int32),
            ("children", np.int32, alphabet_size),
        ])

        # --------------------------------------------------
        # Initialize array (in-memory or memory-mapped)
        # --------------------------------------------------
        if filename is None:
            # Regular in-memory array
            self.nodes = np.zeros(initial_nodes, dtype=dtype)
        else:
            # Memory-mapped array
            if not os.path.exists(filename) or mode in ("w+", "r+"):
                # Create or overwrite file
                self.nodes = np.memmap(
                    filename, dtype=dtype, mode="w+", shape=(initial_nodes,)
                )
            else:
                # Reuse existing file
                self.nodes = np.memmap(
                    filename, dtype=dtype, mode=mode
                )

        self.nodes["children"][:] = -1
        self.nodes["key"][:] = -1
        self.next_free = 1  # node 0 = root

    # --------------------------------------------------
    # Node management
    # --------------------------------------------------

    def _ensure_capacity(self, min_size):
        """Ensure enough space for at least min_size nodes."""
        if min_size <= len(self.nodes):
            return

        new_size = max(min_size, len(self.nodes) * 2)
        dtype = self.nodes.dtype

        if self.filename is None:
            # Resize in memory
            new_nodes = np.zeros(new_size, dtype=dtype)
            new_nodes["children"][:] = -1
            new_nodes["key"][:] = -1
            new_nodes[:len(self.nodes)] = self.nodes
            self.nodes = new_nodes
        else:
            # Resize memory-mapped file
            self.nodes.flush()
            os.truncate(self.filename, new_size * dtype.itemsize)
            new_nodes = np.memmap(
                self.filename, dtype=dtype, mode="r+", 
                shape=(new_size,)
            )
            new_nodes[:len(self.nodes)] = self.nodes
            # Reinitialize unallocated region
            new_nodes["children"][len(self.nodes):] = -1
            new_nodes["key"][len(self.nodes):] = -1
            self.nodes = new_nodes

    def add_kmer(self, kmer, value=1):
        """Add a k-mer sequence (list/array of ints) to the trie."""
        k = len(kmer)
        if k > self.depth:
            self.depth = k
        node_idx = 0  # root
        self.nodes["value"][node_idx] += value

        for symbol in kmer:
            children = self.nodes["children"][node_idx]
            child_idx = children[symbol]

            if child_idx == -1:
                # Create new node
                child_idx = self.next_free
                self._ensure_capacity(child_idx + 1)
                children[symbol] = child_idx

                self.nodes["key"][child_idx] = symbol
                self.nodes["value"][child_idx] = value
                self.nodes["children"][child_idx][:] = -1

                self.next_free += 1
            else:
                self.nodes["value"][child_idx] += value

            node_idx = child_idx
        self.size += value

    # --------------------------------------------------
    # Accessors
    # --------------------------------------------------

    def get_key(self, node_idx: int):
        """Return the integer key leading into this node, or None for root."""
        k = int(self.nodes["key"][node_idx])
        return None if k == -1 else k

    def get_value(self, node_idx: int) -> int:
        """Return the stored value/count at this node."""
        return int(self.nodes["value"][node_idx])

    # --------------------------------------------------
    # Traversal
    # --------------------------------------------------

    def bfs(self):
        """Return node indices in breadth-first order."""
        q = deque([0])
        order = []
        while q:
            node = q.popleft()
            order.append(node)
            for child_idx in self.nodes["children"][node]:
                if child_idx != -1:
                    q.append(child_idx)
        return order

    def map_bfs(self, func, depth=None):
        """Apply function `func(node_idx)` to nodes in BFS order."""
        q = deque([(0, 0)])  # (node_idx, depth)
        results = []
        while q:
            node, d = q.popleft()
            if depth is None or d <= depth:
                results.append(func(node))
                if depth is None or d < depth:
                    for child_idx in self.nodes["children"][node]:
                        if child_idx != -1:
                            q.append((child_idx, d + 1))
        return results
    
    def query(self, kmer, node=0):
        """
        Return the value/count for the given kmer.
        Returns 0 if the kmer is not present in the trie.
        """
        node_idx = node  # start at given node index
        for symbol in kmer:
            if symbol < 0 or symbol >= self.alphabet_size:
                return 0
            child_idx = self.nodes["children"][node_idx, symbol]
            if child_idx == -1:
                return 0
            node_idx = child_idx
        return int(self.nodes["value"][node_idx])
    
    def deep_query(self, kmer, depth=None, node=0):
        """
        Return the value/count for the given kmer throughout the entire tree.
        Returns 0 if the kmer is not present in the trie.
        Descends through the tree, checking if kmer is found starting from any 
        node.
        """
        node_idx = node  # start at given node index
        k = len(kmer)
        if depth is None:
            depth = self.depth - k + 1
        if k >= self.depth:
            return self.query(kmer)
        
        child_idxs = self.nodes["children"][node_idx]
        idxs = child_idxs[child_idxs >= 0]
        count = self.query(kmer, node)
        for child_idx in idxs:
            count += self.deep_query(kmer, depth=depth - 1, node=child_idx)
        return count
        # for symbol in kmer:
        #     if symbol < 0 or symbol >= self.alphabet_size:
        #         return 0
        #     # Count number at current level
        #     child_idx = self.nodes["children"][node_idx, symbol]

        #     if child_idx == -1:
        #         return 0
        #     node_idx = child_idx
        # return int(self.nodes["value"][node_idx])
    
    def query_masked(self, kmer, mask, node_idx=None):
        """Query the trie with masked positions.

        Parameters
        ----------
        kmer : list/array of ints
            Sequence to query.
        mask : int
            Symbol treated as a wildcard (matches any child).
        node_idx : int or None
            Current node index for recursion (default: root).

        Returns
        -------
        int
            Sum of counts for all matching paths.
        """
        if node_idx is None:
            node_idx = 0  # start at root

        if len(kmer) == 0:
            return int(self.nodes["value"][node_idx])

        aa = kmer[0]

        if aa == mask:
            # sum over all children
            total = 0
            for child_idx in self.nodes["children"][node_idx]:
                if child_idx != -1:
                    total += self.query_masked(kmer[1:], mask, node_idx=child_idx)
            return total
        else:
            child_idx = self.nodes["children"][node_idx, aa]
            if child_idx == -1:
                return 0
            return self.query_masked(kmer[1:], mask, node_idx=child_idx)
        
        # aa = kmer[0]
        # next_aa = None if len(kmer) <= 1 else kmer[1]

        # if aa == mask:
        #     # Identify next non-mask
        #     non_mask_screen = np.where(kmer[1:] != mask)[0]
        #     if len(non_mask_screen) == 0:
        #         # only masks remain, so return count of current node
        #         return int(self.nodes["value"][node_idx])
            
        #     # sum over all children at the next 
        #     next_non_mask_pos = non_mask_screen[0]
        #     total = 0
        #     for child_idx in self.nodes["children"][node_idx]:
        #         if child_idx != -1:
        #             total += self.query_masked(kmer[1:], mask, node_idx=child_idx)
        #     return total
        # else:
        #     child_idx = self.nodes["children"][node_idx, aa]
        #     if child_idx == -1:
        #         return 0
        #     return self.query_masked(kmer[1:], mask, node_idx=child_idx)

    # --------------------------------------------------
    # Debug / Info
    # --------------------------------------------------

    @classmethod
    def _load_tree_data(cls, fpath):
        data = np.load(fpath, allow_pickle=False)
        check_keys = [
            "nodes", "alphabet_size", "next_free", "size", "depth",
        ]
        for key in check_keys:
            if key not in data:
                msg = f"Cannot load tree data from {fpath}.\nMissing key {key}"
                raise RuntimeError(msg)
        return data
    
    @classmethod
    def load(cls, fpath):
        data = cls._load_tree_data(fpath)
        tree = cls(alphabet_size=data["alphabet_size"])
        tree.nodes = data["nodes"]
        tree.next_free = int(data["next_free"])
        tree.size = data["size"]
        tree.depth = data["depth"]
        return tree
    
    def save(self, fpath):
        with open(fpath, "wb") as f:
            np.savez_compressed(
                f,
                nodes=self.nodes,
                next_free=self.next_free,
                alphabet_size=self.alphabet_size,
                size=self.size,
                depth=self.depth,
            )
        return
    
    def __len__(self):
        """Return number of allocated nodes."""
        return self.size

    def flush(self):
        """Flush memmap to disk (if applicable)."""
        if isinstance(self.nodes, np.memmap):
            self.nodes.flush()

    # def save(self, fname, compressed=False):
    #     if compressed:
    #         np.savez_compressed(fname, self.nodes)
    #     else:
    #         np.save(fname, self.nodes)

    def __repr__(self):
        return f"CompactTrie(nodes={self.size}, alphabet_size={self.alphabet_size}, mode={'memmap' if self.filename else 'memory'})"
    
    def get_size_in_memory(self, node_idx=None, depth=None):
        """
        Return the total memory used by nodes in the trie (or subtree) in bytes.

        Parameters
        ----------
        node_idx : int or None
            Root of the subtree to measure. If None, use the root of the trie.
        depth : int or None
            Maximum depth to include. If None, include all descendants.

        Returns
        -------
        int
            Total memory in bytes.
        """
        if node_idx is None:
            node_idx = 0  # root

        # Function to get memory usage of a single node
        def node_memory(n_idx):
            # Memory of this node's fields
            node_dtype = self.nodes.dtype
            node_size = node_dtype.itemsize
            return node_size

        # Use BFS traversal to sum node memory
        return sum(self.map_bfs(node_memory, depth=depth))

