from collections import defaultdict, deque
from typing import (
    Callable,
    Generator,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

T = TypeVar("T")


class Graph(Generic[T]):
    """A `Graph` represents a directed graph of dependencies that is used to back the
    `JobGraph` and the `TaskGraph` representation."""

    def __init__(self, nodes: Optional[Mapping[T, Sequence[T]]] = {}):
        self._graph = defaultdict(list)
        self._parent_graph = defaultdict(list)

        for node, children in nodes.items():
            self.add_node(node, *children)

    def add_node(self, node: T, *children: T):
        """Adds the node to the graph along with the given children.

        Args:
            node: The node to be added to the graph.
            children: The children of the node, if any.
        """
        self._graph[node].extend([])
        for child in children:
            self.add_child(node, child)

    def add_child(self, node: T, child: T):
        """Adds a child to the given node.

        Args:
            node: The node to whom the child must be added.
            child: The child to be added.

        Raises:
            `ValueError` if the node is not found in the graph.
        """
        if node not in self._graph:
            raise ValueError(f"The node {node} was not found in the graph.")

        self._graph[node].append(child)
        self._graph[child].extend([])
        self._parent_graph[child].append(node)

    def get_children(self, node: T) -> Sequence[T]:
        """Retrieves the children nodes of the given node.

        Args:
            node: The node whose children need to be retrieved.

        Returns:
            The children of the node, if any.

        Raises:
            `ValueError` if the node is not found in the graph.
        """
        if node not in self._graph:
            raise ValueError(f"The node {node} was not found in the graph.")

        return self._graph[node]

    def get_parents(self, node: T) -> Sequence[T]:
        """Retrieves the parent nodes of the given node.

        Args:
            node: The node whose parents need to be retrieved.

        Returns:
            The parents of the node, if any.

        Raises:
            `ValueError` if the node is not found in the graph.
        """
        if node not in self._graph:
            raise ValueError(f"The node {node} was not found in the graph.")

        return self._parent_graph[node]

    def get_sources(self) -> Sequence[T]:
        """Retrieves the sources from the graph.

        Returns:
            The source nodes from the given graph.
        """
        sources = []
        for node in self._graph:
            if len(self.get_parents(node)) == 0:
                sources.append(node)
        return sources

    def get_nodes(self) -> Sequence[T]:
        """Retrieves the nodes stored in the given graph.

        Returns:
            The nodes stored in the graph.
        """
        return self._graph.keys()

    def get_node_depth(self, node: T) -> int:
        """Retrieves the depth of the node from the graph.

        Args:
            node: The node whose depth needs to be retrieved.

        Returns:
            The depth of the node with the depth of a source node being 1.

        Raises:
            `ValueError` if the node is not found in the graph.
        """
        if node not in self._graph:
            raise ValueError(f"The node {node} was not found in the graph.")

        return (
            1
            if self.is_source(node)
            else max(self.get_node_depth(parent) for parent in self.get_parents(node))
            + 1
        )

    def is_source(self, node: T) -> bool:
        """Checks whether the given node is a source.

        Args:
            node: The node to check.

        Returns:
            `True` if the node is a source node, and `False` otherwise.

        Raises:
            `ValueError` if the node is not found in the graph.
        """
        if node not in self._graph:
            raise ValueError(f"The node {node} was not found in the graph.")

        return len(self.get_parents(node)) == 0

    def filter(self, function: Callable[[T], bool]) -> List[T]:
        """Filter the graph nodes and return nodes that return True on the given
        function.

        Args:
            function: The function to apply to each node.

        Returns:
            A `List[T]` that contains the nodes that return True on the function.
        """
        return list(filter(function, self.get_nodes()))

    def remove(self, node: T):
        """Removes the node from the graph.

        Args:
            node: The node to remove from the graph.
        """
        for child in self.get_children(node):
            self._parent_graph[child].remove(node)
        del self._graph[node]

    def breadth_first(self) -> Generator[T, None, None]:
        """Iterates over the graph in a breadth-first manner."""
        visited_nodes = set()
        frontier = deque(self.get_sources())
        while len(frontier) > 0:
            job = frontier.popleft()
            visited_nodes.add(job)
            for child in self.get_children(job):
                if all(parent in visited_nodes for parent in self.get_parents(child)):
                    frontier.append(child)
            yield job

    def __iter__(self):
        for item in self.breadth_first():
            yield item

    def __len__(self):
        return len(self._graph)

    def __str__(self):
        return f"Graph({self._graph})"

    def __repr__(self):
        return str(self)
