from typing import Set
import os

import pandas as pd

from ...chem.mol import Molecule


class StockQueryMixin:

    def __len__(self) -> int:
        return 0

    def __contains__(self, mol: Molecule) -> bool:
        return False

    def amount(self, mol: Molecule) -> float:
        return NotImplemented

    def availability_string(self, mol: Molecule) -> str:
        raise ValueError()

    def cached_search(self, mol: Molecule) -> bool:
        return mol in self

    def clear_cache(self):
        raise ValueError()

    def price(self, mol: Molecule) -> float:
        raise ValueError()


class InMemoryInchiKeyQuery(StockQueryMixin):

    def __init__(self, filename: str) -> None:
        ext = os.path.splitext(filename)[1]
        if ext in [".h5", ".hdf5"]:
            stock = pd.read_hdf(filename, key="table")  # type: ignore
            inchis = stock.inchi_key.values  # type: ignore
        elif ext == ".csv":
            stock = pd.read_csv(filename)
            inchis = stock.inchi_key.values
        else:
            with open(filename, "r") as fileobj:
                inchis = fileobj.read().splitlines()
        self._stock_inchikeys = set(inchis)

    def __contains__(self, mol: Molecule) -> bool:
        return mol.inchi_key in self._stock_inchikeys

    def __len__(self) -> int:
        return len(self._stock_inchikeys)

    @property
    def stock_inchikeys(self) -> Set[str]:
        return self._stock_inchikeys
