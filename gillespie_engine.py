"""Gillespie stochastic updates for the Lui et al. (2013) lattice model.

The engine enumerates individual reaction/diffusion events and samples one
from their propensities. Polymer affinity is represented as K = k_on/k_off;
for P-B and P-S, the requested material did not specify numerical K values,
so they are parameters. Local signal crowding uses f(P)=1/(1+lambda*P_local),
where P_local is the number of polymers at the signal's lattice coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal

from grid_env import Bacterium, Coordinate, GridEnvironment, Polymer, Signal


@dataclass(frozen=True)
class Kinetics:
    """Rate and affinity parameters (time is in arbitrary model units)."""
    k_bs_on: float = 0.001
    k_bs_off: float = 0.002
    k_bs_affinity: float = 0.5
    k_pb_affinity: float = 1.0
    k_ps_affinity: float = 1.0
    k_polymer_on: float = 0.001
    d_bacteria: float = 0.0005
    d_polymer: float = 0.01
    d_signal: float = 0.05
    crowding_lambda: float = 8.0
    critical_polymer_concentration: int = 10

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        # A zero polymer affinity means that interaction is absent entirely.

    @property
    def k_polymer_off(self) -> float:
        # K = k_on/k_off; the same configurable on-rate is used for P-B and P-S.
        return self.k_polymer_on / self.k_pb_affinity if self.k_pb_affinity > 0 else 0.0

    @property
    def k_polymer_signal_off(self) -> float:
        return self.k_polymer_on / self.k_ps_affinity if self.k_ps_affinity > 0 else 0.0


@dataclass(frozen=True)
class Event:
    propensity: float
    action: Literal[
        "bind_bs", "unbind_bs", "bind_pb", "unbind_pb",
        "bind_ps", "unbind_ps", "move_bacteria", "move_polymer", "move_signal",
    ]
    entities: tuple[object, ...]
    destination: Coordinate | None = None


class _FenwickTree:
    """Dynamic non-negative weights with logarithmic updates and sampling."""

    def __init__(self, weights: list[float]) -> None:
        self.size = len(weights)
        self.tree = [0.0] * (self.size + 1)
        for index, weight in enumerate(weights, start=1):
            self.tree[index] += weight
            parent = index + (index & -index)
            if parent <= self.size:
                self.tree[parent] += self.tree[index]

    @property
    def total(self) -> float:
        index = self.size
        result = 0.0
        while index:
            result += self.tree[index]
            index -= index & -index
        return result

    def add(self, index: int, delta: float) -> None:
        i = index + 1
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def sample(self, target: float) -> int:
        """Return the index containing ``target`` in cumulative weight space."""
        total = self.total
        if total <= 0:
            raise ValueError("Cannot sample a zero-weight tree")
        target = min(max(0.0, target), math.nextafter(total, 0.0))
        index = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate <= self.size and self.tree[candidate] <= target:
                index = candidate
                target -= self.tree[candidate]
            bit >>= 1
        return min(index, self.size - 1)


class GillespieEngine:
    """Discrete-event simulator coupled to a :class:`GridEnvironment`.

    Bacterial adjacency for P-B binding is defined by a polymer lying in or
    immediately next to a bacterium's 2x2 footprint (8-neighbour contact).
    Clusters are connected components of bacteria linked through bound
    polymers; diffusion translates every member by one lattice step.
    """

    _STEPS: tuple[Coordinate, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(
        self,
        environment: GridEnvironment,
        kinetics: Kinetics | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.environment = environment
        self.kinetics = kinetics or Kinetics()
        self.rng = random.Random(seed)
        self.time = 0.0
        # Bound signal and polymer site counts, keyed by object identity.
        self._bs_bound: dict[int, int] = {}
        self._ps_bound: dict[int, int] = {}
        # Each tuple represents one occupied P_B--B_P bond.
        self._pb_bonds: list[tuple[Polymer, Bacterium]] = []

        # Spatial indexes are updated locally when agents move/bind. Signal
        # diffusion is sampled through a Fenwick tree instead of materializing
        # four Event objects for every one of the (often 10,000) signals.
        self._signals_by_position: dict[Coordinate, list[Signal]] = {}
        self._polymer_counts: dict[Coordinate, int] = {}
        self._bacteria_by_cell: dict[Coordinate, list[Bacterium]] = {}
        self._cell_neighbors: list[tuple[Coordinate, ...]] = []
        self._neighbor_index_table = None
        self._build_spatial_indexes()

    def _cell_index(self, position: Coordinate) -> int:
        row, col = position
        return row * self.environment.width + col

    def _position_from_index(self, index: int) -> Coordinate:
        return divmod(index, self.environment.width)

    def _build_spatial_indexes(self) -> None:
        env = self.environment
        for signal in env.signals:
            self._signals_by_position.setdefault(signal.position, []).append(signal)
        for polymer in env.polymers:
            self._polymer_counts[polymer.position] = self._polymer_counts.get(polymer.position, 0) + 1
        for bacterium in env.bacteria:
            self._index_bacterium(bacterium)

        neighbors_by_cell: list[tuple[Coordinate, ...]] = []
        weights = [0.0] * (env.width * env.height)
        for row in range(env.height):
            for col in range(env.width):
                position = (row, col)
                neighbors = tuple(
                    (row + dr, col + dc)
                    for dr, dc in self._STEPS
                    if env.contains((row + dr, col + dc))
                )
                index = self._cell_index(position)
                neighbors_by_cell.append(neighbors)
        self._cell_neighbors = neighbors_by_cell
        for row in range(env.height):
            for col in range(env.width):
                position = (row, col)
                index = self._cell_index(position)
                signal_count = len(self._signals_by_position.get(position, ()))
                weights[index] = signal_count * self._signal_weight_per_particle(position)
        self._signal_diffusion_tree = _FenwickTree(weights)

    def _index_bacterium(self, bacterium: Bacterium) -> None:
        for cell in self._footprint(bacterium):
            self._bacteria_by_cell.setdefault(cell, []).append(bacterium)

    def _unindex_bacterium(self, bacterium: Bacterium) -> None:
        for cell in self._footprint(bacterium):
            members = self._bacteria_by_cell.get(cell, [])
            members[:] = [member for member in members if member is not bacterium]
            if not members:
                self._bacteria_by_cell.pop(cell, None)

    def _signal_weight_per_particle(self, position: Coordinate) -> float:
        neighbors = self._cell_neighbors[self._cell_index(position)] if self._cell_neighbors else ()
        count = self._polymer_counts.get(position, 0)
        factor = 1.0 / (1.0 + self.kinetics.crowding_lambda * count)
        return self.kinetics.d_signal * factor * len(neighbors)

    def _change_signal_count(self, position: Coordinate, delta: float) -> None:
        self._signal_diffusion_tree.add(
            self._cell_index(position), delta * self._signal_weight_per_particle(position)
        )

    def register_bacterium(self, bacterium: Bacterium) -> None:
        """Index a bacterium already appended to the environment (e.g. daughter)."""
        self._index_bacterium(bacterium)

    def register_signal(self, signal: Signal) -> None:
        """Index a signal already appended to the environment."""
        self._signals_by_position.setdefault(signal.position, []).append(signal)
        self._change_signal_count(signal.position, 1)

    def register_polymer(self, polymer: Polymer) -> None:
        """Index a polymer already appended to the environment."""
        position = polymer.position
        old_weight = self._signal_weight_per_particle(position)
        self._polymer_counts[position] = self._polymer_counts.get(position, 0) + 1
        new_weight = self._signal_weight_per_particle(position)
        signal_count = len(self._signals_by_position.get(position, ()))
        self._signal_diffusion_tree.add(self._cell_index(position), signal_count * (new_weight - old_weight))

    def add_signal(self, position: Coordinate) -> Signal:
        signal = self.environment.add_signal(position)
        self.register_signal(signal)
        return signal

    def _remove_signal(self, signal: Signal) -> None:
        for index, member in enumerate(self.environment.signals):
            if member is signal:
                self.environment.signals.pop(index)
                break
        else:
            raise ValueError("Signal is not present in the environment")
        members = self._signals_by_position[signal.position]
        for index, member in enumerate(members):
            if member is signal:
                members.pop(index)
                break
        else:
            raise ValueError("Signal is missing from its spatial index")
        if not members:
            del self._signals_by_position[signal.position]
        self._change_signal_count(signal.position, -1)

    def _move_signal(self, signal: Signal, destination: Coordinate) -> None:
        old = signal.position
        members = self._signals_by_position[old]
        for index, member in enumerate(members):
            if member is signal:
                members.pop(index)
                break
        else:
            raise ValueError("Signal is missing from its spatial index")
        if not members:
            del self._signals_by_position[old]
        self._change_signal_count(old, -1)
        signal.position = destination
        self._signals_by_position.setdefault(destination, []).append(signal)
        self._change_signal_count(destination, 1)

    def _move_polymer(self, polymer: Polymer, destination: Coordinate) -> None:
        old = polymer.position
        old_weight = self._signal_weight_per_particle(old)
        new_weight = self._signal_weight_per_particle(destination)
        old_signal_count = len(self._signals_by_position.get(old, ()))
        new_signal_count = len(self._signals_by_position.get(destination, ()))
        self._polymer_counts[old] -= 1
        if self._polymer_counts[old] == 0:
            del self._polymer_counts[old]
        self._polymer_counts[destination] = self._polymer_counts.get(destination, 0) + 1
        polymer.position = destination
        old_weight_after = self._signal_weight_per_particle(old)
        new_weight_after = self._signal_weight_per_particle(destination)
        self._signal_diffusion_tree.add(
            self._cell_index(old), old_signal_count * (old_weight_after - old_weight)
        )
        self._signal_diffusion_tree.add(
            self._cell_index(destination), new_signal_count * (new_weight_after - new_weight)
        )

    def local_polymer_count(self, position: Coordinate) -> int:
        return self._polymer_counts.get(position, 0)

    def signal_diffusion_factor(self, position: Coordinate) -> float:
        """Crowding slowdown f(P); polymers at/above Pc also lose P_S sites."""
        count = self.local_polymer_count(position)
        return 1.0 / (1.0 + self.kinetics.crowding_lambda * count)

    def free_polymer_signal_sites(self, polymer: Polymer) -> int:
        local_count = self.local_polymer_count(polymer.position)
        free = polymer.signal_binding_sites - self._ps_bound.get(id(polymer), 0)
        if local_count >= self.kinetics.critical_polymer_concentration:
            return 0
        return max(0, free)

    def _footprint(self, bacterium: Bacterium, position: Coordinate | None = None) -> set[Coordinate]:
        row, col = position if position is not None else bacterium.position
        return {(row + dr, col + dc) for dr in range(2) for dc in range(2)}

    def _bacterium_free_bs(self, bacterium: Bacterium) -> int:
        return max(0, bacterium.signal_binding_sites - self._bs_bound.get(id(bacterium), 0))

    def _bacterium_free_bp(self, bacterium: Bacterium) -> int:
        occupied = sum(1 for _, b in self._pb_bonds if b is bacterium)
        return max(0, bacterium.polymer_binding_sites - occupied)

    def _polymer_free_pb(self, polymer: Polymer) -> int:
        occupied = sum(1 for p, _ in self._pb_bonds if p is polymer)
        return max(0, polymer.bacteria_binding_sites - occupied)

    def _signals_near(self, bacterium: Bacterium) -> list[Signal]:
        return [
            signal
            for cell in self._footprint(bacterium)
            for signal in self._signals_by_position.get(cell, ())
        ]

    def _bacteria_near(self, polymer: Polymer) -> list[Bacterium]:
        row, col = polymer.position
        candidates: dict[int, Bacterium] = {}
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                for bacterium in self._bacteria_by_cell.get((row + dr, col + dc), ()):
                    candidates[id(bacterium)] = bacterium
        return list(candidates.values())

    def _polymer_signals_near(self, polymer: Polymer) -> list[Signal]:
        return list(self._signals_by_position.get(polymer.position, ()))

    def _clusters(self) -> list[tuple[Bacterium, ...]]:
        bacteria = self.environment.bacteria
        adjacency: dict[int, set[int]] = {id(b): set() for b in bacteria}
        objects = {id(b): b for b in bacteria}
        by_polymer: dict[int, set[int]] = {}
        for polymer, bacterium in self._pb_bonds:
            by_polymer.setdefault(id(polymer), set()).add(id(bacterium))
        for linked in by_polymer.values():
            for left in linked:
                adjacency[left].update(linked - {left})
        clusters: list[tuple[Bacterium, ...]] = []
        seen: set[int] = set()
        for root in adjacency:
            if root in seen:
                continue
            stack, members = [root], []
            seen.add(root)
            while stack:
                current = stack.pop()
                members.append(objects[current])
                for neighbor in adjacency[current] - seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
            clusters.append(tuple(members))
        return clusters

    def _bacteria_placements(self, cluster: tuple[Bacterium, ...], delta: Coordinate) -> dict[int, Coordinate] | None:
        dr, dc = delta
        proposed = {id(b): (b.position[0] + dr, b.position[1] + dc) for b in cluster}
        cluster_ids = {id(b) for b in cluster}
        all_current = {
            cell
            for b in self.environment.bacteria
            if id(b) not in cluster_ids
            for cell in self._footprint(b)
        }
        all_new: set[Coordinate] = set()
        for b in cluster:
            pos = proposed[id(b)]
            footprint = self._footprint(b, pos)
            if not all(self.environment.contains(cell) for cell in footprint):
                return None
            if footprint & (all_current | all_new):
                return None
            all_new.update(footprint)
        return proposed

    def propensities(self, *, include_signal_diffusion: bool = True) -> list[Event]:
        """Enumerate feasible reaction and one-step diffusion events."""
        events: list[Event] = []
        k = self.kinetics
        # Bacteria-signal binding/unbinding.
        for bacterium in self.environment.bacteria:
            nearby = self._signals_near(bacterium)
            free_sites = self._bacterium_free_bs(bacterium)
            if nearby and free_sites:
                events.append(Event(k.k_bs_on * free_sites * len(nearby), "bind_bs", (bacterium, nearby)))
            bound = self._bs_bound.get(id(bacterium), 0)
            if bound:
                events.append(Event(k.k_bs_off * bound, "unbind_bs", (bacterium,)))
        # Polymer-bacterium binding and dissociation.
        for polymer in self.environment.polymers:
            nearby_b = self._bacteria_near(polymer)
            free_p = self._polymer_free_pb(polymer)
            if free_p and k.k_pb_affinity > 0:
                for bacterium in nearby_b:
                    free_b = self._bacterium_free_bp(bacterium)
                    if free_b:
                        events.append(
                            Event(
                                k.k_polymer_on * free_p * free_b,
                                "bind_pb",
                                (polymer, bacterium),
                            )
                        )
        for polymer, bacterium in self._pb_bonds:
            if k.k_pb_affinity > 0:
                events.append(Event(k.k_polymer_off, "unbind_pb", (polymer, bacterium)))
        # Polymer-signal binding, subject to the Pc steric cutoff.
        for polymer in self.environment.polymers:
            signals = self._polymer_signals_near(polymer)
            free_sites = self.free_polymer_signal_sites(polymer)
            if signals and free_sites and k.k_ps_affinity > 0:
                events.append(Event(k.k_polymer_on * free_sites * len(signals), "bind_ps", (polymer, signals)))
            bound = self._ps_bound.get(id(polymer), 0)
            if bound and k.k_ps_affinity > 0:
                events.append(Event(k.k_polymer_signal_off * bound, "unbind_ps", (polymer,)))
        # Cluster translations. Each cardinal direction is one possible event.
        for cluster in self._clusters():
            rate = k.d_bacteria / len(cluster)
            for delta in self._STEPS:
                placement = self._bacteria_placements(cluster, delta)
                if placement is not None:
                    events.append(Event(rate, "move_bacteria", (cluster, placement)))
        # Polymers diffuse as independent point particles.
        for polymer in self.environment.polymers:
            for dr, dc in self._STEPS:
                destination = (polymer.position[0] + dr, polymer.position[1] + dc)
                if self.environment.contains(destination):
                    events.append(Event(k.d_polymer, "move_polymer", (polymer,), destination))
        # Signal diffusion may be included as grouped cell events for callers
        # inspecting propensities. step() uses the Fenwick sampler directly.
        if include_signal_diffusion:
            for position, signals in self._signals_by_position.items():
                neighbors = self._cell_neighbors[self._cell_index(position)]
                propensity = (
                    len(signals)
                    * k.d_signal
                    * self.signal_diffusion_factor(position)
                    * len(neighbors)
                )
                if propensity:
                    events.append(
                        Event(propensity, "move_signal", (tuple(signals), neighbors))
                    )
        return events

    @staticmethod
    def _choose_weighted(events: list[Event], threshold: float) -> Event:
        cumulative = 0.0
        for event in events:
            cumulative += event.propensity
            if threshold < cumulative:
                return event
        return events[-1]

    def step(
        self,
        *,
        include_signal_diffusion: bool = True,
        max_wait: float | None = None,
    ) -> tuple[float, Event | None] | None:
        """Take a Gillespie step, optionally excluding signal diffusion.

        ``max_wait`` is used by the split-step approximate runner. If the next
        reaction would occur later than this bound, the clock advances by the
        bound and no reaction is applied.
        """
        events = self.propensities(include_signal_diffusion=False)
        other_total = sum(event.propensity for event in events)
        signal_total = self._signal_diffusion_tree.total if include_signal_diffusion else 0.0
        total = other_total + signal_total
        if total <= 0.0:
            return None
        r1 = self.rng.random()
        r2 = self.rng.random()
        threshold = r1 * total
        if threshold < other_total:
            event = self._choose_weighted(events, threshold)
        elif include_signal_diffusion:
            cell_index = self._signal_diffusion_tree.sample(threshold - other_total)
            position = self._position_from_index(cell_index)
            signals = self._signals_by_position[position]
            neighbors = self._cell_neighbors[cell_index]
            signal = self.rng.choice(signals)
            destination = self.rng.choice(neighbors)
            event = Event(
                len(signals) * self._signal_weight_per_particle(position),
                "move_signal",
                (signal, position),
                destination,
            )
        else:
            raise RuntimeError("Selected an excluded signal-diffusion event")
        tau = math.log(1.0 / r2) / total
        if max_wait is not None and tau >= max_wait:
            self.time += max_wait
            return max_wait, None
        self._apply(event)
        self.time += tau
        return tau, event

    def diffuse_signals_leap(self, dt: float, rng: object) -> int:
        """Approximate free-signal movement with a batched random-walk leap.

        For each free signal, draw the number of jumps from a Poisson count
        using its starting-cell diffusion rate, then vectorize the cardinal
        jumps. Local crowding rates are held fixed within the leap, so this is
        still an operator-splitting approximation. ``rng`` must provide NumPy
        Generator ``poisson``, ``integers`` and array ``choice`` methods.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not self.environment.signals:
            return 0
        import numpy as np

        signals = self.environment.signals
        width = self.environment.width
        flat_positions = np.fromiter(
            (s.position[0] * width + s.position[1] for s in signals),
            dtype=np.int64,
            count=len(signals),
        )
        degrees = np.fromiter(
            (len(self._cell_neighbors[int(i)]) for i in flat_positions),
            dtype=np.int64,
            count=len(signals),
        )
        local_polymer_counts = np.fromiter(
            (self._polymer_counts.get(self._position_from_index(int(i)), 0) for i in flat_positions),
            dtype=np.float64,
            count=len(signals),
        )
        factors = 1.0 / (1.0 + self.kinetics.crowding_lambda * local_polymer_counts)
        jump_counts = rng.poisson(self.kinetics.d_signal * factors * degrees * dt)
        max_jumps = int(jump_counts.max(initial=0))
        if self._neighbor_index_table is None:
            neighbor_table = np.full((width * self.environment.height, 4), -1, dtype=np.int64)
            for cell_index, neighbors in enumerate(self._cell_neighbors):
                for neighbor_index, neighbor in enumerate(neighbors):
                    neighbor_table[cell_index, neighbor_index] = self._cell_index(neighbor)
            self._neighbor_index_table = neighbor_table
        neighbor_table = self._neighbor_index_table
        current = flat_positions.copy()
        for jump_number in range(max_jumps):
            active = jump_counts > jump_number
            if not np.any(active):
                break
            active_positions = current[active]
            active_degrees = np.fromiter(
                (len(self._cell_neighbors[int(i)]) for i in active_positions),
                dtype=np.int64,
                count=len(active_positions),
            )
            ranks = rng.integers(0, active_degrees)
            current[active] = neighbor_table[active_positions, ranks]

        moved_total = int(np.count_nonzero(current != flat_positions))
        self._signals_by_position = {}
        for signal, index in zip(signals, current):
            index_int = int(index)
            signal.position = (index_int // width, index_int % width)
            self._signals_by_position.setdefault(signal.position, []).append(signal)
        # Rebuild the weighted sampler once per leap instead of once per moved
        # molecule. This also restores consistency for later exact signal use.
        weights = [0.0] * (self.environment.width * self.environment.height)
        for position, signals in self._signals_by_position.items():
            weights[self._cell_index(position)] = (
                len(signals) * self._signal_weight_per_particle(position)
            )
        self._signal_diffusion_tree = _FenwickTree(weights)
        return moved_total

    def run(self, *, until: float, max_steps: int | None = None) -> int:
        """Run until simulation time reaches ``until`` or ``max_steps`` fires."""
        if until < self.time:
            raise ValueError("until must be >= current simulation time")
        steps = 0
        while self.time < until and (max_steps is None or steps < max_steps):
            result = self.step()
            if result is None:
                break
            steps += 1
        return steps

    def _apply(self, event: Event) -> None:
        action = event.action
        if action == "bind_bs":
            bacterium, signals = event.entities
            signal = self.rng.choice([s for s in signals if isinstance(s, Signal)])
            self._remove_signal(signal)
            self._bs_bound[id(bacterium)] = self._bs_bound.get(id(bacterium), 0) + 1
        elif action == "unbind_bs":
            (bacterium,) = event.entities
            self._bs_bound[id(bacterium)] -= 1
            if not self._bs_bound[id(bacterium)]:
                del self._bs_bound[id(bacterium)]
            self.add_signal(bacterium.position)
        elif action == "bind_pb":
            polymer, bacterium = event.entities
            self._pb_bonds.append((polymer, bacterium))
        elif action == "unbind_pb":
            polymer, bacterium = event.entities
            for index, (bound_polymer, bound_bacterium) in enumerate(self._pb_bonds):
                if bound_polymer is polymer and bound_bacterium is bacterium:
                    self._pb_bonds.pop(index)
                    break
            else:
                raise ValueError("Polymer-bacterium bond is not present")
        elif action == "bind_ps":
            polymer, signals = event.entities
            signal = self.rng.choice([s for s in signals if isinstance(s, Signal)])
            self._remove_signal(signal)
            self._ps_bound[id(polymer)] = self._ps_bound.get(id(polymer), 0) + 1
        elif action == "unbind_ps":
            (polymer,) = event.entities
            self._ps_bound[id(polymer)] -= 1
            if not self._ps_bound[id(polymer)]:
                del self._ps_bound[id(polymer)]
            self.add_signal(polymer.position)
        elif action == "move_bacteria":
            cluster, placement = event.entities
            for bacterium in cluster:
                self._unindex_bacterium(bacterium)
            for bacterium in cluster:
                bacterium.position = placement[id(bacterium)]
                self._index_bacterium(bacterium)
        elif action == "move_polymer":
            (polymer,) = event.entities
            self._move_polymer(polymer, event.destination)  # type: ignore[arg-type]
        elif action == "move_signal":
            signal = event.entities[0]
            if isinstance(signal, tuple):
                signal = self.rng.choice(signal)
            destination = event.destination
            if destination is None and len(event.entities) > 1:
                destinations = event.entities[1]
                if isinstance(destinations, tuple) and destinations and isinstance(destinations[0], tuple):
                    destination = self.rng.choice(destinations)
            if not isinstance(signal, Signal) or destination is None:
                raise ValueError("Malformed signal diffusion event")
            self._move_signal(signal, destination)
        else:  # pragma: no cover - guards future additions to Event.action
            raise ValueError(f"Unknown action: {action}")

