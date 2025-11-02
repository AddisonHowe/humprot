"""MultiKmerTree

"""

import os
import numpy as np
from numpy.typing import NDArray
from collections import deque


class MultiKmerTree:
    """kmer tree with optional memory-mapped storage."""

    DTYPE_KEY = np.int8
    DTYPE_VALUE = np.int32
    DTYPE_CHILDREN = np.int32

    def __init__(
            self, *,
            alphabet_size,
            initial_nodes=128, 
            filename=None, 
            mode="r+", 
    ):
        """
        Args:
            alphabet_size (int) : Number of possible child symbols per node.
            initial_nodes (int) : Initial capacity of the tree.
            filename (str or None) : If provided, use a memory-mapped file at 
                this path. If None, the tree is stored fully in memory.
            mode (str) : File mode for np.memmap. One of: 'r+', 'w+', 'c'.
        """
        self.alphabet_size = alphabet_size
        self.filename = filename
        self.mode = mode
        self.size = 0
        self.depth = 0

        # Each node has a key, value, and children.
        dtype = np.dtype([
            ("key", self.DTYPE_KEY),
            ("value", self.DTYPE_VALUE),
            ("children", self.DTYPE_CHILDREN, alphabet_size),
        ])

        # Initialize array (in-memory or memory-mapped)
        if filename is None:
            # Regular in-memory array
            self.nodes = np.zeros(initial_nodes, dtype=dtype)
        else:
            msg = "Memory-mapped storage not implemented."
            msg += " Leave `filename` as None. `mode` argument ignored." 
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

        # Initialize nodes
        self.nodes["children"][:] = -1
        self.nodes["key"][:] = -1
        self.next_free = 1  # nodes[0] -> root

        self.layer_sizes = np.array([1, 0], dtype=int)
        self.layer_value_sums = np.array([0, 0], dtype=int)

    # --------------------------------------------------
    # Accessors
    # --------------------------------------------------

    def __len__(self):
        """Return number of nodes in the tree."""
        return self.size

    def __repr__(self):
        repstr = "MultiKmerTree[alphabet_size={}]".format(
            self.alphabet_size,
        )
        return repstr
    
    def __str__(self):
        s = "MultiKmerTree(alphabet_size={}, size={}, depth={})".format(
            self.alphabet_size, self.size, self.depth,
        )
        return s
    
    def get_depth(self):
        """Depth of the tree."""
        return self.depth
    
    def get_layer_size(self, depth: int) -> int:
        """Number of nodes at given depth"""
        return int(self.layer_sizes[depth])
    
    def get_layer_value_sum(self, depth: int) -> int:
        """Sum of node values at given depth"""
        return int(self.layer_value_sums[depth])
    
    def get_layer_sizes(self) -> NDArray[np.int_]:
        """Number of nodes at each depth"""
        layer_sizes = np.array([
            self.get_layer_size(d) for d in range(1, self.depth + 1)
        ], dtype=int)
        return layer_sizes
    
    def get_layer_value_sums(self) -> NDArray[np.int_]:
        """Sum of node values at each depth"""
        layer_value_sums = np.array([
            self.get_layer_value_sum(d) for d in range(1, self.depth + 1)
        ], dtype=int)
        return layer_value_sums
    
    # --------------------------------------------------
    # Node access
    # --------------------------------------------------
    
    def _get_key_at_node(self, node_idx: int) -> MultiKmerTree.DTYPE_KEY:
        """Return the integer key leading into this node, or -1 for root.
        
        Args:
            node_idx (int) : index of node, in breadth first order, with root 0
        
        Returns:
            (int; MultiKmerTree.DTYPE_KEY) Node key.
        """
        return self.nodes["key"][node_idx]

    def _get_value_at_node(self, node_idx: int) -> MultiKmerTree.DTYPE_VALUE:
        """Return the stored value/count at this node.
        
        Args:
            node_idx (int) : index of node, in breadth first order, with root 0
        
        Returns:
            (int; MultiKmerTree.DTYPE_KEY) Node value (i.e. count).
        """
        return self.nodes["value"][node_idx]
    
    def _get_numchildren_at_node(self, node_idx: int) -> int:
        """Return the numer of children of the given node"""
        return np.sum(self.nodes["children"][node_idx] >= 0, axis=-1, dtype=int)
    
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
        """Add a k-mer sequence (list/array of ints) to the tree."""
        k = len(kmer)
        if k < self.depth:
            msg = f"Cannot add {k}-mer to tree with current depth {self.depth}"
            raise RuntimeError(msg)
        elif k > self.depth + 1:
            msg = f"Cannot add {k}-mer to tree with current depth {self.depth}"
            raise RuntimeError(msg)
        elif k == self.depth + 1:
            self.depth = k
            self.layer_sizes = np.append(self.layer_sizes, 0)
            self.layer_value_sums = np.append(self.layer_value_sums, 0)
        
        node_idx = 0  # root
        for symidx, symbol in enumerate(kmer):
            depth = symidx + 1
            children = self.nodes["children"][node_idx]
            child_idx = children[symbol]

            if symidx == k - 1:
                # Need to update the children nodes
                if child_idx == -1:
                    # Create new node
                    child_idx = self.next_free
                    self._ensure_capacity(child_idx + 1)
                    children = self.nodes["children"][node_idx]
                    children[symbol] = child_idx
                    self.nodes["key"][child_idx] = symbol
                    self.nodes["value"][child_idx] = value
                    self.nodes["children"][child_idx][:] = -1
                    self.next_free += 1
                    self.size += 1
                    self.layer_sizes[depth] += 1
                    self.layer_value_sums[depth] += value
                else:
                    self.nodes["value"][child_idx] += value
                    self.layer_value_sums[depth] += value
            else:
                # Need to descend
                if child_idx == -1:
                    # Create new node
                    child_idx = self.next_free
                    self._ensure_capacity(child_idx + 1)
                    children = self.nodes["children"][node_idx]
                    children[symbol] = child_idx
                    self.nodes["key"][child_idx] = symbol
                    self.nodes["value"][child_idx] = value
                    self.nodes["children"][child_idx][:] = -1
                    self.next_free += 1
                    self.size += 1
                    self.layer_sizes[depth] += 1
                    self.layer_value_sums[depth] += value
            node_idx = child_idx

    def trim_free_nodes(self):
        self.nodes = self.nodes[:self.next_free + 1]

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
    
    def _iternodes(self, max_depth=None, node_func=None, by_layer=False):
        return self._iternodes_v2(
            max_depth=max_depth, node_func=node_func, by_layer=by_layer,
        )
    
    def _iternodes_v1(self, max_depth=None, node_func=None, by_layer=False):
        """Iterate over nodes in breadth-first order.
        
        Begin with the first child of the root node, and proceed through all 
        layers up to and including the max_depth, if specified. That is, if 
        max_depth=1, then only the first layer of the graph is traversed.
        """
        if by_layer:
            raise NotImplementedError("_iternodes_v1 does not support by_layer")
        if max_depth is None:
            max_depth = self.depth
        start_idx = 0
        queue = [start_idx]
        # Don't include root in yielded values. Start by adding its children.
        idx = queue.pop(0)
        children = self.nodes["children"][idx]
        for child_idx in children:
            if child_idx != -1:
                queue.append(child_idx)
        
        current_depth = 1
        running_count_in_layer = 0
        layer_size = self.get_layer_size(current_depth)
        while queue:
            idx = queue.pop(0)
            if node_func:
                yield node_func(idx)
            else:
                yield idx
            running_count_in_layer += 1
            if running_count_in_layer == layer_size:
                current_depth += 1
                running_count_in_layer = 0
                layer_size = self.get_layer_size(current_depth)
            # Enqueue all non-empty children
            children = self.nodes["children"][idx]
            if current_depth < max_depth:
                for child_idx in children:
                    if child_idx != -1:
                        queue.append(child_idx)
    
    def _iternodes_v2(self, max_depth=None, node_func=None, by_layer=False):
        """Iterate over nodes in breadth-first order.
        
        Begin with the first child of the root node, and proceed through all 
        layers up to and including the max_depth, if specified. That is, if 
        max_depth=1, then only the first layer of the graph is traversed.
        """
        if max_depth is None:
            max_depth = self.depth
        start_idx = 0
        # Don't include root in yielded values. Start by adding its children.
        idx = start_idx
        children = self.nodes["children"][idx]
        children = children[children > 0]
        
        current_depth = 1
        while current_depth <= max_depth:
            if node_func:
                values = node_func(children)
            else:
                values = children

            if by_layer:
                yield values
            else:
                for x in values:
                    yield x
                        
            current_depth += 1
            # Queue next layer of children
            children = self.nodes["children"][children].flatten()
            children = children[children > 0]

    def iternodes(self, max_depth=None):
        """Iterate over nodes in breadth-first order.
        
        Begin with the first child of the root node, and proceed through all 
        layers up to and including the max_depth, if specified. That is, if 
        max_depth=1, then only the first layer of the graph is traversed.
        """
        return self._iternodes(
            max_depth=max_depth, 
            node_func=None, 
            by_layer=False,
        )

    def iternodekeys(self, max_depth=None):
        """Iterate over keys of nodes in breadth-first order.
        
        Begin with the first child of the root node, and proceed through all 
        layers up to and including the max_depth, if specified. That is, if 
        max_depth=1, then only the first layer of the graph is traversed.
        """
        return self._iternodes(
            max_depth=max_depth, 
            node_func=self._get_key_at_node, 
            by_layer=False,
        )
    
    def iternodevalues(self, max_depth=None):
        """Iterate over keys of nodes in breadth-first order.
        
        Begin with the first child of the root node, and proceed through all 
        layers up to and including the max_depth, if specified. That is, if 
        max_depth=1, then only the first layer of the graph is traversed.
        """
        return self._iternodes(
            max_depth=max_depth, 
            node_func=self._get_value_at_node, 
            by_layer=False,
        )
    
    def iternodes_in_layers(self, max_depth=None, node_func=None):
        return self._iternodes(
            max_depth=max_depth, 
            by_layer=True, 
            node_func=node_func,
        )
        
    
    # --------------------------------------------------
    # Queries
    # --------------------------------------------------
    
    def query(self, kmer, node=0):
        """
        Return the value/count for the given kmer.
        Returns 0 if the kmer is not present in the tree.
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
    
    def query_masked(self, kmer, mask, node_idx=None):
        """Query the tree with masked positions.

        Args:
            kmer : list/array of ints
                Sequence to query.
            mask : int
                Symbol treated as a wildcard (matches any child).
            node_idx : int or None
                Current node index for recursion (default: root).

        Returns:
            (int) Sum of counts for all matching paths.
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

    # --------------------------------------------------
    # Debug / Info
    # --------------------------------------------------

    @classmethod
    def _load_tree_data(cls, fpath):
        data = np.load(fpath, allow_pickle=False)
        check_keys = [
            "alphabet_size", "nodes", "next_free", "size", "depth",
            "layer_sizes", "layer_value_sums"
        ]
        for key in check_keys:
            if key not in data:
                msg = f"Cannot load tree data from {fpath}.\nMissing key {key}"
                raise RuntimeError(msg)
        return data
    
    # --------------------------------------------------
    # I/O
    # --------------------------------------------------

    @classmethod
    def load(cls, fpath) -> MultiKmerTree:
        """Load a MultiKmerTree from a file."""
        data = cls._load_tree_data(fpath)
        tree = cls(alphabet_size=data["alphabet_size"])
        tree.nodes = data["nodes"]
        tree.next_free = int(data["next_free"])
        tree.size = int(data["size"])
        tree.depth = int(data["depth"])
        tree.layer_sizes = np.array(data["layer_sizes"], dtype=int)
        tree.layer_value_sums = np.array(data["layer_value_sums"], dtype=int)
        return tree
    
    def save(self, fpath, trim_unused_nodes=False):
        """Save the tree to a binary file."""
        if trim_unused_nodes:
            self.trim_free_nodes()
        with open(fpath, "wb") as f:
            np.savez_compressed(
                f,
                nodes=self.nodes,
                next_free=self.next_free,
                alphabet_size=self.alphabet_size,
                size=self.size,
                depth=self.depth,
                layer_sizes=self.layer_sizes,
                layer_value_sums=self.layer_value_sums,
            )
        return

    def flush(self):
        """Flush memmap to disk (if applicable)."""
        if isinstance(self.nodes, np.memmap):
            self.nodes.flush()
