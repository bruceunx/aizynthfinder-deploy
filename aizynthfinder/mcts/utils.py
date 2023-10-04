from ..reactiontree import ReactionTreeLoader


class ReactionTreeFromSuperNode(ReactionTreeLoader):

    def _load(self, base_node):
        actions, nodes = route_to_node(base_node)
        self.tree.created_at_iteration = base_node.created_at_iteration
        root_mol = nodes[0].state.mols[0]
        self._unique_mols[id(root_mol)] = root_mol.make_unique()
        self._add_node(
            self._unique_mols[id(root_mol)],
            in_stock=nodes[0].state.is_solved,
        )
        for child, action in zip(nodes[1:], actions):
            self._add_bipartite(child, action)

    def _add_bipartite(self, child, action):

        reaction_obj = self._unique_reaction(action)
        self._add_node(reaction_obj, depth=2 * action.mol.transform + 1)
        self.tree.graph.add_edge(self._unique_mol(action.mol), reaction_obj)
        reactant_nodes = []
        for mol in child.state.mols:
            if mol.parent is action.mol:
                self._add_node(
                    self._unique_mol(mol),
                    depth=2 * mol.transform,
                    transform=mol.transform,
                    in_stock=mol in child.state.stock,
                )
                self.tree.graph.add_edge(reaction_obj, self._unique_mol(mol))
                reactant_nodes.append(self._unique_mol(mol))
        reaction_obj.reactants = (tuple(reactant_nodes), )


def route_to_node(from_node):
    actions = []
    nodes = []
    current = from_node

    while current is not None:
        parent = current.parent
        if parent is not None:
            action = parent[current]["action"]
            actions.append(action)
        nodes.append(current)
        current = parent
    return actions[::-1], nodes[::-1]
