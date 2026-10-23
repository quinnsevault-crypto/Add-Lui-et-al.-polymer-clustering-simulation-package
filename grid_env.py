"""Spatial environment for the Lui et al. (2013) polymer/QS lattice model.

Coordinates are integer (row, column) pairs with zero-based indices. Bacteria
store the top-left coordinate of their 2x2 footprint; polymers and signals
store a single coordinate and may overlap any entity, including one another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import ClassVar, TypeAlias

Coordinate: TypeAlias = tuple[int, int]


class Strain(str, Enum):
    """Strain-dependent starting signal condition."""

    MM32 = "MM32"
    BB170 = "BB170"


@dataclass
class Bacterium:
    position: Coordinate
    polymer_binding_sites: int = 120  # B_P
    signal_binding_sites: int = 60  # B_S
    bs_express: float = 0.0  # Luciferase-expression state (BS_express)


@dataclass
class Polymer:
    position: Coordinate
    bacteria_binding_sites: int = 2  # P_B
    signal_binding_sites: int = 5  # P_S


@dataclass
class Signal:
    position: Coordinate


@dataclass
class GridEnvironment:
    """A 50x50 vertex lattice containing bacteria, polymers, and signals."""

    width: int = 50
    height: int = 50
    initial_bacteria: int = 25
    max_bacteria: int = 80
    strain: Strain = Strain.MM32
    bacteria: list[Bacterium] = field(default_factory=list)
    polymers: list[Polymer] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)

    BACTERIUM_SIZE: ClassVar[int] = 2
    INITIAL_MM32_SIGNALS: ClassVar[int] = 10_000

    def __post_init__(self) -> None:
        if self.width < self.BACTERIUM_SIZE or self.height < self.BACTERIUM_SIZE:
            raise ValueError("Grid must fit a 2x2 bacterium footprint")
        if not 0 <= self.initial_bacteria <= self.max_bacteria:
            raise ValueError("initial_bacteria must be between 0 and max_bacteria")
        if not self.bacteria:
            self._seed_bacteria(self.initial_bacteria, random.Random())
        if not self.signals:
            self._seed_signals(
                self.INITIAL_MM32_SIGNALS if self.strain is Strain.MM32 else 0,
                random.Random(),
            )

    def contains(self, position: Coordinate) -> bool:
        """Return whether a cell coordinate lies within the lattice."""
        row, col = position
        return 0 <= row < self.height and 0 <= col < self.width

    def _footprint(self, position: Coordinate) -> tuple[Coordinate, ...]:
        row, col = position
        return tuple(
            (row + dr, col + dc)
            for dr in range(self.BACTERIUM_SIZE)
            for dc in range(self.BACTERIUM_SIZE)
        )

    def _bacterium_position_is_free(self, position: Coordinate) -> bool:
        footprint = self._footprint(position)
        if not all(self.contains(cell) for cell in footprint):
            return False
        occupied = {cell for bacterium in self.bacteria for cell in self._footprint(bacterium.position)}
        return not any(cell in occupied for cell in footprint)

    def add_bacterium(self, position: Coordinate) -> Bacterium:
        """Place a bacterium if its full 2x2 footprint is in bounds and free."""
        if len(self.bacteria) >= self.max_bacteria:
            raise ValueError("Maximum bacteria count reached")
        if not self._bacterium_position_is_free(position):
            raise ValueError(f"Bacterium footprint is out of bounds or occupied: {position}")
        bacterium = Bacterium(position)
        self.bacteria.append(bacterium)
        return bacterium

    def add_polymer(self, position: Coordinate) -> Polymer:
        """Add a point polymer; it may share a cell with any entity."""
        self._require_cell(position)
        polymer = Polymer(position)
        self.polymers.append(polymer)
        return polymer

    def add_signal(self, position: Coordinate) -> Signal:
        """Add a point signal; it may share a cell with any entity."""
        self._require_cell(position)
        signal = Signal(position)
        self.signals.append(signal)
        return signal

    def _require_cell(self, position: Coordinate) -> None:
        if not self.contains(position):
            raise ValueError(f"Coordinate outside {self.height}x{self.width} grid: {position}")

    def _seed_bacteria(self, count: int, rng: random.Random) -> None:
        """Place bacteria at distinct, non-overlapping 2x2 footprints."""
        attempts = 0
        max_attempts = max(10_000, count * self.width * self.height)
        while len(self.bacteria) < count and attempts < max_attempts:
            attempts += 1
            position = (
                rng.randrange(self.height - self.BACTERIUM_SIZE + 1),
                rng.randrange(self.width - self.BACTERIUM_SIZE + 1),
            )
            if self._bacterium_position_is_free(position):
                self.bacteria.append(Bacterium(position))
        if len(self.bacteria) < count:
            raise RuntimeError("Could not place all initial bacteria without overlap")

    def _seed_signals(self, count: int, rng: random.Random) -> None:
        for _ in range(count):
            self.signals.append(
                Signal((rng.randrange(self.height), rng.randrange(self.width)))
            )
