from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mad.objs.base import MovableObj

ComputerOrder = Enum("ComputerOrder", ["IDLE", "RELOAD", "LAUNCH", "MOVE"])


@dataclass
class ComputerCommand:
    order: ComputerOrder
    target: MovableObj | None = None
