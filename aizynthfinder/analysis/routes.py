from typing import Any, Optional, Dict, Sequence

from PIL.Image import Image
import numpy as np

from .tree_analysis import TreeAnalysis
from .utils import RouteSelectionArguments
from ..reactiontree import ReactionTree


class RouteCollection:

    def __init__(self, reaction_trees: Sequence[ReactionTree], **kwargs) -> None:
        self._routes: Sequence[Dict] = [{} for _ in range(len(reaction_trees))]
        self.reaction_trees = reaction_trees
        self._update_route_dict(reaction_trees, "reaction_tree")
        self.route_metadata = [rt.metadata for rt in reaction_trees]
        self._update_route_dict(self.route_metadata, "route_metadata")

        self.nodes = self._unpack_kwarg_with_default("nodes", None, **kwargs)
        self.scores = self._unpack_kwarg_with_default("scores", np.nan, **kwargs)
        self.all_scores = self._unpack_kwarg_with_default("all_scores", dict, **kwargs)

        self._dicts: Optional[Sequence[Dict]] = self._unpack_kwarg("dicts", **kwargs)
        self._images: Optional[Sequence[Image]] = self._unpack_kwarg("images", **kwargs)
        self._jsons: Optional[Sequence[str]] = self._unpack_kwarg("jsons", **kwargs)

    @classmethod
    def from_analysis(cls, analysis: TreeAnalysis,
                      selection: RouteSelectionArguments) -> "RouteCollection":

        items, scores = analysis.sort(selection)
        all_scores = [{repr(analysis.scorer): score} for score in scores]
        kwargs = {"scores": scores, "all_scores": all_scores}
        kwargs["nodes"] = items
        reaction_trees = [from_node.to_reaction_tree() for from_node in items]

        return cls(reaction_trees, **kwargs)

    def __getitem__(self, index: int) -> Dict:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        return self._routes[index]

    def __len__(self) -> int:
        return len(self.reaction_trees)

    @property
    def dicts(self) -> Sequence[Dict]:
        if self._dicts is None:
            self._dicts = self.make_dicts()
        return self._dicts

    @property
    def images(self) -> Sequence[Image]:
        if self._images is None:
            self._images = self.make_images()
        return self._images

    @property
    def jsons(self) -> Sequence[str]:
        if self._jsons is None:
            self._jsons = self.make_jsons()
        return self._jsons

    def make_dicts(self) -> Sequence[Dict]:
        self._dicts = [tree.to_dict() for tree in self.reaction_trees]
        self._update_route_dict(self._dicts, "dict")
        return self._dicts

    def make_images(self) -> Sequence[Optional[Image]]:
        self._images = []
        for tree in self.reaction_trees:
            try:
                img = tree.to_image()
            except ValueError:
                self._images.append(None)
            else:
                self._images.append(img)
        self._update_route_dict(self._images, "image")
        return self._images

    def make_jsons(self) -> Sequence[str]:
        self._jsons = [tree.to_json() for tree in self.reaction_trees]
        self._update_route_dict(self._jsons, "json")
        return self._jsons

    def _unpack_kwarg(self, key: str, **kwargs: Any) -> Optional[Sequence[Any]]:
        if key not in kwargs:
            return None
        arr = kwargs[key]
        self._update_route_dict(arr, key[:-1])
        return arr

    def _unpack_kwarg_with_default(self, key: str, default: Any,
                                   **kwargs: Any) -> Sequence[Any]:
        arr = self._unpack_kwarg(key, **kwargs)
        if arr is not None:
            return arr
        return [
            default() if callable(default) else default
            for _ in range(len(self.reaction_trees))
        ]

    def _update_route_dict(self, arr: Sequence[Any], key: str) -> None:
        for i, value in enumerate(arr):
            self._routes[i][key] = value
