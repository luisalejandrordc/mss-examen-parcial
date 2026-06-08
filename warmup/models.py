from dataclasses import dataclass
from typing import Any


@dataclass
class WarmupResult:
    """Stores Warm-up Point information"""

    warmup: int
    diagnostics: dict[str, Any]
