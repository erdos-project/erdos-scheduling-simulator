from collections import defaultdict, deque
from typing import (
    Callable,
    Generator,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

T = TypeVar("T")


class Graph(Generic[T]):
    """A `Graph` represents a directed acyclic graph of dependencies that is used to
    back the `JobGraph` and the `TaskGraph` representation."""

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

    def get_edges(self) -> Sequence[Tuple[T, T]]:
        """Retrieves the edges stored in the given graph.

        Returns:
            The edges stored in the graph.
        """
        edges = []
        for node in self.get_nodes():
            for child in self.get_children(node):
                edges.append((node, child))
        return edges

    def get_node_depth(self, node: T, func=max) -> int:
        """Retrieves the depth of the node from the directed acyclic graph.

        Args:
            node: The node whose depth needs to be retrieved.
            func: min/max to choose which depth to retrieve of the node.

        Returns:
            The depth of the node with the depth of a source node being 1.

        Raises:
            `ValueError` if the node is not found in the graph, and `RuntimeError` if
            the graph has a cycle.
        """
        if node not in self._graph:
            raise ValueError(f"The node {node} was not found in the graph.")

        node_to_depth = defaultdict(lambda: 1)
        for _node in self.topological_sort():
            if len(self.get_parents(_node)) > 0:
                node_to_depth[_node] = (
                    func([node_to_depth[parent] for parent in self.get_parents(_node)])
                    + 1
                )

            if node == _node:
                return node_to_depth[node]

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

    def breadth_first(self, node: T = None) -> Generator[T, None, None]:
        """Iterates over the graph in a breadth-first manner.

        Args:
            node: The node to start the breadth-first search from.
        """
        visited_nodes = set()
        if node:
            # Start from the location provided by the caller.
            frontier = deque([node])
        else:
            # Start from the root of the graph.
            frontier = deque(self.get_sources())

        while len(frontier) > 0:
            current_node = frontier.popleft()
            visited_nodes.add(current_node)
            for child in self.get_children(current_node):
                if node is None and all(
                    parent in visited_nodes for parent in self.get_parents(child)
                ):
                    # If no starting node was provided, we should have visited all the
                    # parent nodes of a child before we add it to the frontier.
                    frontier.append(child)
                elif node is not None and all(
                    parent in visited_nodes
                    for parent in self.get_parents(child)
                    if self.are_dependent(node, parent)
                ):
                    # If the breadth-first traversal was initialized from a node, we
                    # can have children whose parents can never be visited. Hence, we
                    # ensure that a child gets added when all the parents that are
                    # dependent on the node have been visited.
                    frontier.append(child)
            yield current_node

    def depth_first(self, node: T = None) -> Generator[T, None, None]:
        """Iterates over the graph in a depth-first manner.

        Args:
            node: The node to start the depth-first search from.
        """
        visited_nodes = set()
        frontier = deque(self.get_sources() if node is None else [node])
        while len(frontier) > 0:
            node = frontier.pop()
            visited_nodes.add(node)
            yield node
            for child in self.get_children(node):
                if child not in visited_nodes:
                    frontier.append(child)

    def topological_sort(self) -> List[T]:
        """Compute the topological sort ordering of the directed acyclic graph.

        This method can be used to check if there are cycles in the graph.

        Returns:
            The topological sorting of the given graph.

        Raises:
            `RuntimeError` if the graph has a cycle.
        """
        node_marks = {node: "Unmarked" for node in self.get_nodes()}
        topological_sort = []

        def visit(node):
            if node_marks[node] == "Permanent":
                return
            elif node_marks[node] == "Temporary":
                raise RuntimeError("The given graph is not a DAG.")
            node_marks[node] = "Temporary"
            for child in self.get_children(node):
                visit(child)
            node_marks[node] = "Permanent"
            topological_sort.append(node)

        while any(mark != "Permanent" for mark in node_marks.values()):
            for node, mark in node_marks.items():
                if mark == "Unmarked":
                    visit(node)

        return topological_sort[::-1]

    def get_longest_path(self, weights=None) -> List[T]:
        """Retrieves the longest path in the graph that goes from a source node to a
        sink node.

        Returns:
            A `List[T]` that contains an ordered list of nodes that form the longest
            path from a source node to a sink node.
        """
        if weights is None:
            # If a function for the weights was not specified, use the
            # notion that if the longest path was just a source, it would be 1.
            # and then the addition of every node costs a +1 in length.
            weights = lambda node: 1 if self.is_source(node) else 2  # noqa: E731
        longest_path_length = {node: weights(node) for node in self.get_nodes()}
        predecessor = {}

        for node in self.topological_sort():
            for child in self.get_children(node):
                if longest_path_length[child] <= longest_path_length[node] + weights(
                    child
                ):
                    longest_path_length[child] = longest_path_length[node] + weights(
                        child
                    )
                    predecessor[child] = node

        start_node, cumulative_sum_length = max(
            longest_path_length.items(), key=lambda val: val[1]
        )
        longest_path = [start_node]
        cumulative_sum_length -= weights(start_node)
        while cumulative_sum_length > 0:
            start_node = predecessor[start_node]
            longest_path.append(start_node)
            cumulative_sum_length -= weights(start_node)
        return longest_path[::-1]

    def are_dependent(self, node_1: T, node_2: T) -> bool:
        """Check if `node_1` and `node_2` are dependent on each other.

        Args:
            node_1: The first node to check the dependency with.
            node_2: The second node to check the dependency with.

        Returns:
            `True` if either the first or the second node are dependent on each other
            and `False` otherwise.

        Raises:
            `RuntimeError` if the graph contains a cycle.
        """
        # Find depths of both of the nodes in the graph.
        node_1_depth = self.get_node_depth(node_1)
        node_2_depth = self.get_node_depth(node_2)

        def check_dependency(top_node, bottom_node):
            # Checks the dependency between a node of lower depth (`top_node`) and a
            # node of higher depth (`bottom_node`)
            for child_node in self.depth_first(top_node):
                if child_node == bottom_node:
                    return True
            return False

        if node_1_depth == node_2_depth:
            # If the two nodes are on the same depth, they can execute in parallel,
            # and are not dependent on each other.
            return False
        elif node_1_depth > node_2_depth:
            return check_dependency(node_2, node_1)
        else:
            return check_dependency(node_1, node_2)

    def to_dot(self, filename: str) -> None:
        """Save a DOT representation of the Graph to the given `filename`.

        Args:
            filename (`str`): The path where the DOT file to visualize this
                Graph needs to be stored.
        """
        with open(filename, "w") as output_file:
            output_file.write(f"digraph {self.__class__.__name__} {{\n")
            for node, _ in self._graph.items():
                output_file.write(f'  "{node.id}" [label="{node.name}"];\n')
            for node, children in self._graph.items():
                for child in children:
                    output_file.write(f'  "{node.id}" -> "{child.id}";\n')
            output_file.write("}")

    def __iter__(self):
        for item in self.breadth_first():
            yield item

    def __len__(self):
        return len(self._graph)

    def __str__(self):
        return f"Graph({self._graph})"

    def __repr__(self):
        return str(self)
