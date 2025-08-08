from dataclasses import dataclass

@dataclass
class Defaults:
    # Default algorithmic parameters
    r1: int = 3
    r2: int = 10
    k: int = 5
    epsilon: float = 1e-6
    delta: float = 0.05
    seed: int = 42

DEFAULTS = Defaults()
