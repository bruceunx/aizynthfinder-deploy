from typing import List

import networkx as nx

from .node import MctsNode


class MctsSearchTree:

    def __init__(self, config, root_smiles=None):

        self.profiling = {
            "expansion_calls": 0,
            "reactants_generations": 0,
            "iterations": 0,
        }

        if root_smiles:
            self.root = MctsNode.create_root(smiles=root_smiles,
                                             tree=self,
                                             config=config)
        else:
            self.root = None
        self.config = config
        self._graph = None

    def backpropagate(self, from_node: MctsNode, value_estimate: float) -> None:
        current = from_node
        while current is not self.root:
            parent = current.parent
            # For mypy, parent should never by None unless current is the root
            assert parent is not None
            parent.backpropagate(current, value_estimate)
            current = parent

    def graph(self, recreate: bool = False) -> nx.DiGraph:
        if not self.root:
            raise ValueError("Root of search tree is not defined ")

        if not recreate and self._graph:
            return self._graph

        def add_node(node):
            self._graph.add_edge(node.parent, node, action=node.parent[node]["action"])
            for grandchild in node.children:
                add_node(grandchild)

        self._graph = nx.DiGraph()
        # Always add the root
        self._graph.add_node(self.root)
        for child in self.root.children:
            add_node(child)
        return self._graph

    def nodes(self) -> List[MctsNode]:
        return list(self.graph())

    def one_iteration(self) -> bool:
        self.profiling["iterations"] += 1
        leaf = self.select_leaf()
        leaf.expand()
        while not leaf.is_terminal():
            child = leaf.promising_child()
            if child is not None:
                child.expand()
                leaf = child
            else:
                break
        self.backpropagate(leaf, leaf.state.score)
        return leaf.state.is_solved

    def select_leaf(self) -> MctsNode:
        if not self.root:
            raise ValueError("Root of search tree is not defined ")

        current = self.root
        while current.is_expanded and not current.state.is_solved:
            promising_child = current.promising_child()
            # If promising_child returns None it means that the node
            # is unexpandable, and hence we should break the loop
            if promising_child:
                current = promising_child
        return current
