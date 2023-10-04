from typing import Any, List, Dict, Union

import abc


class ContextCollection(abc.ABC):

    _single_selection = False
    _collection_name = "collection"

    def __init__(self) -> None:
        self._items: Dict = {}
        self._selection: List[str] = []

    def __delitem__(self, key: str) -> None:
        if key not in self._items:
            raise KeyError(
                f"{self._collection_name.capitalize()} with name {key} not loaded.")
        del self._items[key]

    def __getitem__(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(
                f"{self._collection_name.capitalize()} with name {key} not loaded.")
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> List[str]:
        return list(self._items.keys())

    @property
    def selection(self) -> Union[List[str], str, None]:
        if self._single_selection:
            return self._selection[0] if self._selection else None
        return self._selection

    @selection.setter
    def selection(self, value: str) -> None:
        self.select(value)

    def deselect(self, key: str = "") -> None:
        if key == "":
            self._selection = []
            return
        if key not in self._selection:
            raise KeyError(f"Cannot deselect {key} because it is not selected")
        self._selection.remove(key)

    @abc.abstractmethod
    def load(self, source, /):
        """Load an item. Needs to be implemented by a sub-class"""

    @abc.abstractmethod
    def load_from_config(self, /, **config):
        """Load items from a configuration. Needs to be implemented by a sub-class"""

    def select(self, value: str, append: bool = False) -> None:
        self._selection = [value]

    def select_first(self) -> None:
        if self.items:
            self.select(self.items[0])

    def select_last(self) -> None:
        if self.items:
            self.select(self.items[-1])
