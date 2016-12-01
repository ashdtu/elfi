import uuid
import logging


logger = logging.getLogger(__name__)


class Graph:
    def __init__(self, name=None):
        """Directed acyclic graph (DAG) object consisting of nodes of the class `Node`.

        All names of the nodes in the graph must be unique.

        Parameters
        ----------
        name : hashable
        """
        self.name = name
        self.nodes = {}

    def add_node(self, node):
        """Adds a new node to graph. If a node with the same name already exists,
        renames the old one.

        Parameters
        ----------
        node : Node
        """
        if node not in self.nodes.values():
            if node.name in self.nodes:
                name_old = node.name
                while name_old in self.nodes:
                    name_old += "_old"
                logger.warning("Node with name {} exists. Renaming old to {}."
                               .format(node.name, name_old))
                self.nodes[node.name].rename(name_old)
                self.nodes[name_old] = self.nodes[node.name]
            self.nodes[node.name] = node

    def remove_node(self, node):
        if node.name in self.nodes:
            del self.nodes[node.name]
        else:
            raise IndexError("Node with name {} not found".format(node))


class Node(object):
    """A base class representing Nodes in a graphical model.
    This class is inherited by all types of nodes in the model.

    Attributes
    ----------
    name : string
    parents : list of Nodes
    children : list of Nodes
    """
    def __init__(self, name, *parents, graph=None):
        self.name = name
        self.parents = []
        self.children = []
        self.add_parents(parents)
        if graph is not None and not isinstance(graph, Graph):
            raise ValueError("Argument graph is not of type Graph")
        self.graph = graph
        if self.graph is not None:
            self.graph.add_node(self)

    def reset(self, *args, **kwargs):
        pass

    def rename(self, name):
        """Rename node.

        Parameters
        ----------
        name : string
        """
        self.name = name

    def add_parents(self, nodes):
        """Add multiple parents at once. See also `add_parent`.

        Parameters
        ----------
        nodes : list or tuple of parents
        """
        for n in self.node_list(nodes):
            self.add_parent(n)

    def add_parent(self, node, index=None, index_child=None):
        """Adds a parent and assigns itself as a child of node. Only add if new.

        Parameters
        ----------
        node : Node or None
            If None, this function will not do anything
        index : int
            Index in self.parents where to insert the new parent.
        index_child : int
            Index in self.children where to insert the new child.
        """
        if node is None:
            return
        node = self._ensure_node(node)
        if node in self.descendants:
            raise ValueError("Cannot have cyclic graph structure.")
        if node not in self.parents:
            if index is None:
                index = len(self.parents)
            else:
                if index < 0 or index > len(self.parents):
                    raise ValueError("Index out of bounds.")
            self.parents.insert(index, node)
        node._add_child(self, index_child)

    def _add_child(self, node, index=None):
        node = self._ensure_node(node)
        if not node in self.children:
            if index is None:
                index = len(self.children)
            else:
                if index < 0 or index > len(self.children):
                    raise ValueError("Index out of bounds.")
            self.children.insert(index, node)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_root(self):
        return len(self.parents) == 0

    def is_leaf(self):
        return len(self.children) == 0

    # TODO: how removing of a node will affect its status in its graph
    def remove(self, keep_parents=False, keep_children=False):
        """Remove references to self from parents and children.

        Parameters
        ----------
        parent_or_index : Node or int
        """
        if not keep_parents:
            while len(self.parents) > 0:
                self.remove_parent(0)
        if not keep_children:
            for c in self.children.copy():
                c.remove_parent(self)

    def remove_parent(self, parent_or_index):
        """Remove a parent.

        Self will not also be a child of the parent any longer.

        Parameters
        ----------
        parent_or_index : Node or int
        """
        index = parent_or_index
        if isinstance(index, Node):
            for i, p in enumerate(self.parents):
                if p == parent_or_index:
                    index = i
                    break
        if isinstance(index, Node):
            # TODO: add more informative error message (the __str__ in the future?)
            raise Exception("Could not find a parent")
        parent = self.parents[index]
        del self.parents[index]
        parent.children.remove(self)
        return index

    def change_to(self, node, transfer_parents=True, transfer_children=True):
        """Effectively changes self to another node. Reference to self is untouched.

        Parameters
        ----------
        node : Node
            The new Node to change self to.
        transfer_parents : boolean
            Whether to reuse current parents.
        transfer_children : boolean
            Whether to reuse current children, which will also be reset recursively.

        Returns
        -------
        node : Node
            The new node with parents and children associated.
        """
        if transfer_parents:
            parents = self.parents.copy()
            for p in parents:
                self.remove_parent(p)
            node.add_parents(parents)

        if transfer_children:
            children = self.children.copy()
            for c in children:
                index = c.remove_parent(self)
                c.add_parent(node, index=index)
                c.reset(propagate=True)

        return node

    @property
    def ancestors(self):
        _ancestors = self.parents.copy()
        for n in self.parents:
            for m in n.ancestors:
                if m not in _ancestors:
                    _ancestors.append(m)
        return _ancestors

    @property
    def descendants(self):
        _descendants = self.children.copy()
        for n in self.children:
            for m in n.descendants:
                if m not in _descendants:
                    _descendants.append(m)
        return _descendants

    @property
    def component(self):
        return [self] + self.ancestors + self.descendants

    @property
    def label(self):
        return self.name

    # TODO: is this needed?
    @property
    def neighbours(self):
        return self.children + self.parents

    """Private methods"""

    def _convert_to_node(self, obj, name):
        raise ValueError("No conversion to Node for value {}".format(obj))

    def _ensure_node(self, obj):
        if isinstance(obj, Node):
            return obj
        name = "_{}_{}".format(self.name, str(uuid.uuid4().hex[0:6]))
        return self._convert_to_node(obj, name)

    """Static methods"""

    # TODO: is this used any longer?
    @staticmethod
    def node_list(nodes):
        if isinstance(nodes, dict):
            nodes = nodes.values()
        elif isinstance(nodes, Node):
            nodes = [nodes]
        return nodes
