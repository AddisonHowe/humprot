"""Tree classes

"""   

import sys
import numpy as np
from collections import deque


class Node:

    __slots__ = ("key", "value", "children")
    
    def __init__(self, key: int = None, value: int = 0):
        self.key = key
        self.value = np.uint32(value)
        self.children: dict[int, "Node"] = {}

    def add_child(self, key: int, value: int = 0):
        """Add a child node under this node with the given key."""
        if key not in self.children:
            self.children[key] = Node(key, value)
        return self.children[key]
    
    def get_key(self):
        """Return the key associated at the node."""
        return self.key
    
    def get_value(self):
        """Return the value associated at the node."""
        return self.value
    
    def get_child(self, key: int):
        """Return the child node for a key, or None if it doesn't exist."""
        return self.children.get(key)
    
    def get_memory_usage(self):
        size = sys.getsizeof(self)
        size += sys.getsizeof(self.key)
        size += sys.getsizeof(self.value)
        size += sys.getsizeof(self.children)
        size += np.sum([sys.getsizeof(c) for c in self.children])
        return size
    
    def increment(self, amount: int = 1):
        """Increment this node's value."""
        self.value += amount

    def __repr__(self):
        s = f"Node(key={self.key}, value={self.value}, "
        s += f"children={np.sort(list(self.children.keys()))})"
        return s
        

class KmerTree:
    
    def __init__(self):
        self.root = Node(value=0)
    
    def add_kmer(self, kmer, value=1):
        current_node = self.root
        current_node.increment(value)
        for i in range(len(kmer)):
            v = kmer[i]
            child = current_node.get_child(v)
            if child:
                child.increment(value)
            else:
                child = current_node.add_child(v, value)
            current_node = child

    def query(self, kmer):
        n = len(kmer)
        current_node = self.root
        for i in range(n):
            aa = kmer[i]  # get amino acid in i-th position
            child = current_node.get_child(aa)
            if child is None:
                return 0
            current_node = child
        return current_node.value
    
    def query_masked(self, kmer, mask, node=None):
        n = len(kmer)
        current_node = self.root if node is None else node
        for i in range(n):
            aa = kmer[i]  # get amino acid in i-th position
            if aa == mask:
                return np.sum(
                    [self.query_masked(kmer[i+1:], mask, node=c) 
                     for c in current_node.children.values()]
                )
            else:
                child = current_node.get_child(aa)
                if child is None:
                    return 0
                current_node = child
        return current_node.value
    
    def map_bfs(self, f: callable, node="root", depth=3):
        if depth is None:
            depth = np.inf
        if node is None or depth < 0:
            return None
        node = self.root if node == "root" else node

        queue = deque([(node, 0)])
        values = []
        while queue:
            node, d = queue.popleft()
            if d <= depth:
                value = f(node)  # apply function to node
                values.append(value)
                for child in np.sort(list(node.children.keys())):
                    queue.append((node.children[child], d + 1))
        return values
        
    def get_size_in_memory(self, node="root", depth=None):
        node = self.root if node == "root" else node
        return np.sum(
            self.map_bfs(lambda n: n.get_memory_usage(), node=node, depth=depth)
        )
    
    def print(self, node="root", depth=3):
        self.map_bfs(print, node=node, depth=depth)



# class Node:

#     ALPHA_SIZE = 22

#     __slots__ = ("key", "value", "children")
    
#     def __init__(self, key: int = None, value: int = 0):
#         self.key = key
#         self.value = np.uint32(value)
#         self.children = np.full(self.ALPHA_SIZE, None, dtype=object)

#     def add_child(self, key: int, value: int = 0):
#         """Add a child node under this node with the given key."""
#         if self.children[key] is None:
#             self.children[key] = self.__class__(key, value)
#         else:
#             self.children[key].value += value
#         return self.children[key]
    
#     def get_key(self):
#         """Return the key associated at the node."""
#         return self.key
    
#     def get_value(self):
#         """Return the value associated at the node."""
#         return self.value
    
#     def get_child(self, key: int):
#         """Return the child node for a key, or None if it doesn't exist."""
#         return self.children[key]
    
#     def get_memory_usage(self):
#         size = sys.getsizeof(self)
#         size += sys.getsizeof(self.key)
#         size += sys.getsizeof(self.value)
#         size += sys.getsizeof(self.children)
#         size += np.sum([sys.getsizeof(c) for c in self.children])
#         return size
    
#     def increment(self, amount: int = 1):
#         """Increment this node's value."""
#         self.value += np.uint32(amount)

#     def __repr__(self):
#         s = f"Node(key={self.key}, value={self.value}, "
#         s += f"children={np.sort(list(self.children.keys()))})"
#         return s
    
#     def iter_children(self):
#         for child in self.children:
#             if child is not None:
#                 yield child

# class KmerTree:
    
#     def __init__(self):
#         self.root = Node(value=0)
    
#     def add_kmer(self, kmer, value=1):
#         node = self.root
#         node.increment(value)
#         for k in kmer:
#             node = node.add_child(k, value)

#     def query(self, kmer):
#         node = self.root
#         for k in kmer:
#             node = node.get_child(k)
#             if node is None:
#                 return 0
#         return int(node.value)
    
#     def query_masked(self, kmer, mask, node=None):
#         n = len(kmer)
#         current_node = self.root if node is None else node
#         for i in range(n):
#             aa = kmer[i]  # get amino acid in i-th position
#             if aa == mask:
#                 return np.sum(
#                     [self.query_masked(kmer[i+1:], mask, node=c) 
#                      for c in current_node.iter_children()]
#                 )
#             else:
#                 child = current_node.get_child(aa)
#                 if child is None:
#                     return 0
#                 current_node = child
#         return current_node.value
    
#     def map_bfs(self, f, node="root", depth=np.inf):
#         if depth is None:
#             depth = np.inf
#         if node is None or depth < 0:
#             return None
#         node = self.root if node == "root" else node
#         q = deque([(node, 0)])
#         out = []
#         while q:
#             node, d = q.popleft()
#             if d <= depth:
#                 out.append(f(node))
#                 for child in node.iter_children():
#                     q.append((child, d + 1))
#         return out
        
#     def get_size_in_memory(self, node="root", depth=None):
#         node = self.root if node == "root" else node
#         return np.sum(
#             self.map_bfs(lambda n: n.get_memory_usage(), node=node, depth=depth)
#         )
    
#     def print(self, node="root", depth=3):
#         self.map_bfs(print, node=node, depth=depth)
