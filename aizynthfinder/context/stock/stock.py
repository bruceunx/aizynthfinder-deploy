from typing import Any, List, Set, Union, Dict
import copy
from collections import defaultdict

from ...chem.mol import Molecule
from ..collection import ContextCollection
from .queries import (
    InMemoryInchiKeyQuery, )
from ...utils.exceptions import StockException


class Stock(ContextCollection):
    _collection_name = "stock"

    def __init__(self) -> None:
        super().__init__()
        self._exclude: Set[str] = set()
        self._stop_criteria: Dict = {"amount": None, "price": None, "counts": {}}
        self._use_stop_criteria: bool = False

    def __contains__(self, mol: Molecule) -> bool:
        if not self.selection or mol.inchi_key in self._exclude:
            return False

        if self._use_stop_criteria:
            return self._apply_stop_criteria(mol)

        for key in self.selection:
            if mol in self[key]:
                return True
        return False

    def __len__(self) -> int:
        return sum(len(self[key]) for key in self.selection or [])

    @property
    def stop_criteria(self) -> dict:
        return copy.deepcopy(self._stop_criteria)

    def amount(self, mol: Molecule) -> float:
        amounts = self._mol_property(mol, "amount")
        if not amounts:
            raise StockException("Could not obtain amount of molecule")
        return max(amounts)

    def availability_list(self, mol: Molecule) -> List[str]:
        availability = []
        for key in self.selection or []:
            if mol not in self[key]:
                continue
            try:
                availability.append(self[key].availability_string(mol))
            except Exception:
                availability.append(key)
        return availability

    def availability_string(self, mol: Molecule) -> str:
        availability = self.availability_list(mol)
        if availability:
            return ",".join(availability)
        return "Not in stock"

    def exclude(self, mol: Molecule) -> None:
        self._exclude.add(mol.inchi_key)

    def load(self, source: Union[str, Any], key: str) -> None:  # type: ignore
        src_str = str(source)
        if "object at 0x" in src_str:
            src_str = source.__class__.__name__

        if isinstance(source, str):
            source = InMemoryInchiKeyQuery(source)
        self._items[key] = source

    def load_from_config(self, **config: Any) -> None:
        if "stop_criteria" in config:
            self.set_stop_criteria(config["stop_criteria"])

        for key, stockfile in config.get("files", {}).items():
            self.load(stockfile, key)

    def price(self, mol: Molecule) -> float:
        prices = self._mol_property(mol, "price")
        if not prices:
            raise StockException("Could not obtain price of molecule")
        return min(prices)

    def reset_exclusion_list(self) -> None:
        """Remove all molecules in the exclusion list"""
        self._exclude = set()

    def set_stop_criteria(self, criteria: Dict | None = None) -> None:
        criteria = criteria or {}
        self._stop_criteria = {
            "price": criteria.get("price"),
            "amount": criteria.get("amount"),
            "counts": copy.deepcopy(criteria.get("size", criteria.get("counts"))),
        }
        self._use_stop_criteria = any(self._stop_criteria.values())
        {key: value for key, value in self._stop_criteria.items() if value}

    def smiles_in_stock(self, smiles: str) -> bool:
        return Molecule(smiles=smiles) in self

    def _apply_amount_criteria(self, mol: Molecule) -> bool:
        if not self._stop_criteria["amount"]:
            return True
        try:
            amount = self.amount(mol)
        except StockException:
            return True
        return amount >= self._stop_criteria.get("amount", amount)

    def _apply_counts_criteria(self, mol: Molecule) -> bool:
        if not self._stop_criteria["counts"]:
            return True
        atom_counts: dict = defaultdict(int)
        for atom in mol.rd_mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1
        for symbol, threshold in self._stop_criteria["counts"].items():
            if atom_counts[symbol] > threshold:
                return False
        return True

    def _apply_price_criteria(self, mol: Molecule) -> bool:
        if not self._stop_criteria["price"]:
            return True
        try:
            price = self.price(mol)
        except StockException:
            return True
        return price <= self._stop_criteria.get("price", price)

    def _apply_stop_criteria(self, mol: Molecule) -> bool:
        if not self._apply_counts_criteria(mol):
            return False

        passes = False
        for key in self.selection or []:
            passes = passes or self[key].cached_search(mol)
        passes = passes and self._apply_amount_criteria(mol)
        passes = passes and self._apply_price_criteria(mol)

        for key in self.selection or []:
            self[key].clear_cache()
        return passes

    def _mol_property(self, mol, property_name):
        values = []
        for key in self.selection:
            try:
                func = getattr(self[key], property_name)
                values.append(func(mol))
            except StockException:
                pass
        return values
