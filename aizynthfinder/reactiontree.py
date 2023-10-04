from typing import Dict, Any, Iterable, Union, Tuple, Optional
import abc
import hashlib
import json
import operator

from PIL.Image import Image
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree

from .chem.mol import Molecule, UniqueMolecule, none_molecule
from .chem.reaction import RetroReaction, FixedRetroReaction
from .utils.image import RouteImageFactory

PilColor = Union[str, Tuple[int, int, int]]
FrameColors = Optional[Dict[bool, PilColor]]


class ReactionTree:

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.root = none_molecule()
        self.is_solved = False
        self.created_at_iteration: Optional[int] = None

    @property
    def metadata(self) -> Dict:
        return {
            "created_at_iteration": self.created_at_iteration,
            "is_solved": self.is_solved,
        }

    def depth(self, node: Union[UniqueMolecule, FixedRetroReaction]) -> int:
        return self.graph.nodes[node].get("depth", -1)

    def hash_key(self) -> str:
        return self._hash_func(self.root)

    def in_stock(self, node: Union[UniqueMolecule, FixedRetroReaction]) -> bool:
        return self.graph.nodes[node].get("in_stock", False)

    def is_branched(self) -> bool:
        nsteps = len(list(self.reactions()))
        max_depth = max(self.depth(leaf) for leaf in self.leafs())
        return nsteps != max_depth // 2

    def leafs(self) -> Iterable[UniqueMolecule]:
        for node in self.graph:
            if isinstance(node, UniqueMolecule) and not self.graph[node]:
                yield node

    def molecules(self) -> Iterable[UniqueMolecule]:
        for node in self.graph:
            if isinstance(node, UniqueMolecule):
                yield node

    def reactions(self) -> Iterable[FixedRetroReaction]:
        for node in self.graph:
            if not isinstance(node, Molecule):
                yield node

    def subtrees(self) -> Iterable["ReactionTree"]:

        def create_subtree(root_node):
            subtree = ReactionTree()
            subtree.root = root_node
            subtree.graph = dfs_tree(self.graph, root_node)
            for node in subtree.graph:
                prop = dict(self.graph.nodes[node])
                prop["depth"] -= self.graph.nodes[root_node].get("depth", 0)
                if "transform" in prop:
                    prop["transform"] -= self.graph.nodes[root_node].get("transform", 0)
                subtree.graph.nodes[node].update(prop)
            subtree.is_solved = all(subtree.in_stock(node) for node in subtree.leafs())
            return subtree

        for node in self.molecules():
            if node is not self.root and self.graph[node]:
                yield create_subtree(node)

    def to_dict(self, include_metadata=False) -> Dict:
        return self._build_dict(self.root, include_metadata=include_metadata)

    def to_image(
        self,
        in_stock_colors: FrameColors = None,
        show_all: bool = True,
    ) -> Image:
        factory = RouteImageFactory(self.to_dict(),
                                    in_stock_colors=in_stock_colors,
                                    show_all=show_all)
        return factory.image

    def to_json(self, include_metadata=False) -> str:
        return json.dumps(self.to_dict(include_metadata=include_metadata),
                          sort_keys=False,
                          indent=2)

    def _build_dict(
        self,
        node: Union[UniqueMolecule, FixedRetroReaction],
        dict_: Dict = None,
        include_metadata=False,
    ) -> Dict:
        if dict_ is None:
            dict_ = {}

        if node is self.root and include_metadata:
            dict_["route_metadata"] = self.metadata

        dict_["type"] = "mol" if isinstance(node, Molecule) else "reaction"
        dict_["hide"] = self.graph.nodes[node].get("hide", False)
        dict_["smiles"] = node.smiles
        if isinstance(node, UniqueMolecule):
            dict_["is_chemical"] = True
            dict_["in_stock"] = self.in_stock(node)
        elif isinstance(node, FixedRetroReaction):
            dict_["is_reaction"] = True
            dict_["metadata"] = dict(node.metadata)
        else:
            raise ValueError(
                f"This is an invalid reaction tree. Unknown node type {type(node)}")

        dict_["children"] = []

        children = list(self.graph.successors(node))
        if isinstance(node, FixedRetroReaction):
            children.sort(key=operator.attrgetter("weight"))
        for child in children:
            child_dict = self._build_dict(child)
            dict_["children"].append(child_dict)

        if not dict_["children"]:
            del dict_["children"]
        return dict_

    def _hash_func(self, node: Union[FixedRetroReaction, UniqueMolecule]) -> str:
        if isinstance(node, UniqueMolecule):
            hash_ = hashlib.sha224(node.inchi_key.encode())
        else:
            hash_ = hashlib.sha224(node.hash_key().encode())
        child_hashes = sorted(
            self._hash_func(child) for child in self.graph.successors(node))
        for child_hash in child_hashes:
            hash_.update(child_hash.encode())
        return hash_.hexdigest()


class ReactionTreeLoader(abc.ABC):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._unique_mols: Dict[int, UniqueMolecule] = {}
        self._unique_reactions: Dict[int, FixedRetroReaction] = {}
        self.tree = ReactionTree()
        self._load(*args, **kwargs)

        self.tree.is_solved = all(
            self.tree.in_stock(node) for node in self.tree.leafs())

    def _add_node(
        self,
        node: Union[UniqueMolecule, FixedRetroReaction],
        depth: int = 0,
        transform: int = 0,
        in_stock: bool = False,
        hide: bool = False,
    ) -> None:
        attributes = {
            "hide": hide,
            "depth": depth,
        }
        if isinstance(node, Molecule):
            attributes.update({"transform": transform, "in_stock": in_stock})
            if not self.tree.root:
                self.tree.root = node
        self.tree.graph.add_node(node, **attributes)

    @abc.abstractmethod
    def _load(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _unique_mol(self, molecule: Molecule) -> UniqueMolecule:
        id_ = id(molecule)
        if id_ not in self._unique_mols:
            self._unique_mols[id_] = molecule.make_unique()
        return self._unique_mols[id_]

    def _unique_reaction(self, reaction: RetroReaction) -> FixedRetroReaction:
        id_ = id(reaction)
        if id_ not in self._unique_reactions:
            metadata = dict(reaction.metadata)
            if ":" in reaction.mapped_reaction_smiles():
                metadata["mapped_reaction_smiles"] = reaction.mapped_reaction_smiles()
            self._unique_reactions[id_] = FixedRetroReaction(
                self._unique_mol(reaction.mol),
                smiles=reaction.smiles,
                metadata=metadata,
            )
        return self._unique_reactions[id_]
