from dataclasses import dataclass


@dataclass
class RouteSelectionArguments:

    nmin: int = 1
    nmax: int = 8
    return_all: bool = False
