from typing import Any, Union

import numpy as np

from ...chem.mol import TreeMolecule
from ...chem.reaction import RetroReaction


def _make_fingerprint(obj: Union[TreeMolecule, RetroReaction],
                      model: Any) -> np.ndarray:
    fingerprint = obj.fingerprint(radius=2, nbits=len(model))
    return fingerprint.reshape([1, len(model)])
