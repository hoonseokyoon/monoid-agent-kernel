from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class CancellationToken:
    _event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def cancel(self) -> None:
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()
