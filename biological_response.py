"""Strain-specific growth, signal production, and QS output models.

Division periods are sampled from a triangular distribution whose mean/mode
is T and whose total support width is ``division_width`` (T +/- width/2).
Expression is integrated exactly over each caller-supplied interval while the
number of sensed, receptor-bound signals is held constant for that interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from grid_env import Bacterium, Coordinate, GridEnvironment, Signal, Strain


@dataclass(frozen=True)
class ResponseParameters:
    mean_doubling_time: float = 15_000.0
    division_width: float = 20_000.0
    signal_production_rate: float = 0.005
    k_syn: float = 0.05
    k_degrad: float = 0.01
    luminescence_scale: float = 2_500.0
    threshold_mm32: float = 38.0
    threshold_bb170: float = 15.0

    def __post_init__(self) -> None:
        if self.mean_doubling_time <= 0:
            raise ValueError("mean_doubling_time must be positive")
        if self.division_width < 0 or self.signal_production_rate < 0:
            raise ValueError("division_width and signal_production_rate must be non-negative")
        if self.k_syn < 0 or self.k_degrad < 0 or self.luminescence_scale < 0:
            raise ValueError("response rates and scale must be non-negative")


@dataclass
class BiologicalResponse:
    """Biological response process for MM32 or BB170.

    ``advance_interval`` is called for a Gillespie interval ``dt``. Pass
    ``bound_signal_counts`` from the engine at the *start* of that interval;
    it updates continuous expression and processes timed divisions and BB170
    signal production. Binding counts can be obtained from the engine's
    receptor occupancy state by calling :meth:`signal_sensing`.
    """

    environment: GridEnvironment
    strain: Strain | None = None
    parameters: ResponseParameters = field(default_factory=ResponseParameters)
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _division_deadlines: dict[int, float] = field(init=False, default_factory=dict, repr=False)
    time: float = 0.0

    def __post_init__(self) -> None:
        self.strain = self.strain or self.environment.strain
        self._rng = random.Random(self.seed)
        for bacterium in self.environment.bacteria:
            self._schedule_division(bacterium, from_time=self.time)

    @property
    def threshold(self) -> float:
        return (
            self.parameters.threshold_mm32
            if self.strain is Strain.MM32
            else self.parameters.threshold_bb170
        )

    def signal_sensing(self, engine: object) -> dict[int, int]:
        """Return bound signal count per bacterium from a Gillespie engine."""
        bound = getattr(engine, "_bs_bound", None)
        if bound is None:
            raise TypeError("engine must expose its _bs_bound receptor occupancy map")
        return {id(b): int(bound.get(id(b), 0)) for b in self.environment.bacteria}

    def luminescence(self, bacterium: Bacterium) -> float:
        """Compute lum = Scale * (1 - exp(DeltaBS-1)/(exp(DeltaBS)+1))."""
        delta_bs = 0.25 * (self.threshold - bacterium.bs_express)
        # exp(DeltaBS-1)/(exp(DeltaBS)+1) is exp(-1)*sigmoid(DeltaBS).
        # This form avoids overflow for large positive DeltaBS.
        if delta_bs >= 0:
            sigmoid = 1.0 / (1.0 + math.exp(-delta_bs))
        else:
            exp_delta = math.exp(delta_bs)
            sigmoid = exp_delta / (1.0 + exp_delta)
        fraction = 1.0 - math.exp(-1.0) * sigmoid
        return self.parameters.luminescence_scale * fraction

    def population_luminescence(self) -> float:
        return sum(self.luminescence(b) for b in self.environment.bacteria)

    def advance_interval(
        self,
        dt: float,
        *,
        bound_signal_counts: dict[int, int] | None = None,
        engine: object | None = None,
    ) -> list[Bacterium]:
        """Advance biology by ``dt`` and return newly created daughter cells.

        Division times are sampled once per bacterium. A due division waits
        until an unoccupied 2x2 placement exists; the daughter starts with
        zero expressed protein and fresh, unoccupied binding sites. BB170
        signal production is sampled as a Poisson process at rate 0.005 per
        bacterium per time unit. MM32 does not produce signal.
        """
        if dt < 0:
            raise ValueError("dt must be non-negative")
        if dt == 0:
            return []
        end = self.time + dt
        counts = bound_signal_counts or {}
        bacteria_at_interval_start = tuple(self.environment.bacteria)

        # Integrate the stated piecewise expression/degradation ODE exactly
        # over this interval, using the signal occupancy at interval start.
        for bacterium in tuple(self.environment.bacteria):
            sensed = max(0, counts.get(id(bacterium), 0))
            self._advance_expression(bacterium, sensed, dt)

        daughters: list[Bacterium] = []
        # Process due division clocks in chronological order. A new daughter
        # receives its own sampled clock starting at its birth time.
        while True:
            due = [
                (deadline, bacterium)
                for bacterium in self.environment.bacteria
                if (deadline := self._division_deadlines.get(id(bacterium))) is not None
                and deadline <= end
            ]
            if not due:
                break
            deadline, parent = min(due, key=lambda item: item[0])
            if len(self.environment.bacteria) >= self.environment.max_bacteria:
                self._division_deadlines[id(parent)] = math.inf
                continue
            position = self._random_empty_bacterium_position()
            if position is None:
                # Keep the clock overdue and retry at the next interval.
                self._division_deadlines[id(parent)] = end
                break
            daughter = self.environment.add_bacterium(position)
            if engine is not None:
                register_bacterium = getattr(engine, "register_bacterium", None)
                if register_bacterium is None:
                    raise TypeError("engine must provide register_bacterium()")
                register_bacterium(daughter)
            daughters.append(daughter)
            self._schedule_division(parent, from_time=deadline)
            self._schedule_division(daughter, from_time=deadline)

        if self.strain is Strain.BB170:
            for bacterium in bacteria_at_interval_start:
                expected = self.parameters.signal_production_rate * dt
                for _ in range(self._poisson(expected)):
                    position = self._random_neighboring_cell(bacterium)
                    signal = self.environment.add_signal(position)
                    if engine is not None:
                        register_signal = getattr(engine, "register_signal", None)
                        if register_signal is None:
                            raise TypeError("engine must provide register_signal()")
                        register_signal(signal)

        self.time = end
        return daughters

    def _advance_expression(self, bacterium: Bacterium, sensed: int, dt: float) -> None:
        expressed = max(0.0, bacterium.bs_express)
        if sensed > expressed:
            rate = self.parameters.k_syn
            bacterium.bs_express = sensed + (expressed - sensed) * math.exp(-rate * dt)
        else:
            bacterium.bs_express = expressed * math.exp(-self.parameters.k_degrad * dt)

    def _schedule_division(self, bacterium: Bacterium, *, from_time: float) -> None:
        half_width = self.parameters.division_width / 2.0
        low = max(0.0, self.parameters.mean_doubling_time - half_width)
        high = self.parameters.mean_doubling_time + half_width
        duration = self._rng.triangular(low, high, self.parameters.mean_doubling_time)
        self._division_deadlines[id(bacterium)] = from_time + duration

    def _random_empty_bacterium_position(self) -> Coordinate | None:
        candidates = [
            (row, col)
            for row in range(self.environment.height - 1)
            for col in range(self.environment.width - 1)
            if self.environment._bacterium_position_is_free((row, col))
        ]
        return self._rng.choice(candidates) if candidates else None

    def _random_neighboring_cell(self, bacterium: Bacterium) -> Coordinate:
        row, col = bacterium.position
        neighbors = [
            (r, c)
            for r in range(row - 1, row + 3)
            for c in range(col - 1, col + 3)
            if (r < row or r > row + 1 or c < col or c > col + 1)
            and self.environment.contains((r, c))
        ]
        # For a bacterium touching a corner, in-bounds neighbors always exist
        # on a valid grid of at least 2x2.
        return self._rng.choice(neighbors)

    def _poisson(self, mean: float) -> int:
        """Draw a Poisson count without requiring a third-party dependency."""
        if mean <= 0:
            return 0
        if mean < 30.0:
            limit = math.exp(-mean)
            product = 1.0
            count = 0
            while product > limit:
                count += 1
                product *= self._rng.random()
            return count - 1
        # For larger means, use a normal proposal and reject negative counts;
        # the large-mean regime is not reached with the default event spacing.
        return max(0, int(round(self._rng.gauss(mean, math.sqrt(mean)))))
