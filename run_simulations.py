"""Run seeded ensembles of the Lui et al. polymer/QS lattice model.

Example (from the project root):
    python outputs/run_simulations.py --output-dir outputs/results --max-time 30000
    python outputs/run_simulations.py --sweep-affinities --output-dir outputs/sweep

Polymer concentration is mapped to a number of point-agent polymers by
``round(concentration / polymer_units_per_agent)``. This scale is configurable
because the lattice model does not define a conversion from experimental
concentration units to polymer-agent count.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Iterable

# Support both `python outputs/run_simulations.py` and import as a package.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from biological_response import BiologicalResponse
from grid_env import Bacterium, GridEnvironment, Polymer, Signal, Strain
from gillespie_engine import GillespieEngine, Kinetics


DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
CONCENTRATIONS = list(range(0, 7001, 1000))
POLYMER_TYPES = {
    "P1": (5.0, 0.0),
    "P2": (0.0, 0.1),
    "P3": (5.0, 10.0),
}


@dataclass(frozen=True)
class Condition:
    strain: str
    polymer: str
    k_pb: float
    k_ps: float
    concentration: int
    seed: int

    @property
    def condition_id(self) -> str:
        return f"{self.strain}_{self.polymer}_KPB{self.k_pb:g}_KPS{self.k_ps:g}_P{self.concentration}"


def make_affinity_grid() -> list[tuple[float, float]]:
    """Return the requested 5x5 (25-condition) affinity grid from 0.1 to 10."""
    # Supplementary Information S26 explicitly lists these five values.
    axis = [0.1, 0.5, 1.0, 5.0, 10.0]
    return [(k_pb, k_ps) for k_pb in axis for k_ps in axis]


def condition_specs(sweep_affinities: bool) -> list[tuple[str, float, float]]:
    specs = [(name, k_pb, k_ps) for name, (k_pb, k_ps) in POLYMER_TYPES.items()]
    if sweep_affinities:
        specs.extend(("grid", k_pb, k_ps) for k_pb, k_ps in make_affinity_grid())
    return specs


def make_environment(strain: Strain, seed: int, polymer_count: int) -> GridEnvironment:
    """Create a seeded 50x50 initialization without implicit global randomness."""
    rng = random.Random(seed)
    bacteria: list[Bacterium] = []
    env = GridEnvironment(
        width=50,
        height=50,
        initial_bacteria=25,
        max_bacteria=80,
        strain=strain,
        bacteria=bacteria,
        # A non-empty list suppresses GridEnvironment's automatic seeding.
        signals=[Signal((0, 0))],
        polymers=[],
    )
    env.signals.clear()

    # Sample valid, non-overlapping 2x2 bacterial footprints.
    tries = 0
    while len(env.bacteria) < env.initial_bacteria and tries < 100_000:
        tries += 1
        position = (rng.randrange(env.height - 1), rng.randrange(env.width - 1))
        if env._bacterium_position_is_free(position):
            env.bacteria.append(Bacterium(position))
    if len(env.bacteria) != env.initial_bacteria:
        raise RuntimeError("Failed to place the initial bacterial population")

    initial_signal_count = 10_000 if strain is Strain.MM32 else 0
    env.signals.extend(
        Signal((rng.randrange(env.height), rng.randrange(env.width)))
        for _ in range(initial_signal_count)
    )
    env.polymers.extend(
        Polymer((rng.randrange(env.height), rng.randrange(env.width)))
        for _ in range(polymer_count)
    )
    return env


def sample_times(max_time: float, interval: float) -> list[float]:
    if max_time <= 0 or interval <= 0:
        raise ValueError("max_time and sample_interval must be positive")
    times = [0.0]
    n = 1
    while n * interval < max_time:
        times.append(n * interval)
        n += 1
    if times[-1] != max_time:
        times.append(max_time)
    return times


def simulate_one(
    condition: Condition,
    *,
    max_time: float,
    sample_interval: float,
    polymer_units_per_agent: float,
    max_steps: int | None = None,
    fast_diffusion_dt: float | None = None,
) -> list[dict[str, float | int | str]]:
    if fast_diffusion_dt is not None:
        return simulate_one_fast_diffusion(
            condition,
            max_time=max_time,
            sample_interval=sample_interval,
            polymer_units_per_agent=polymer_units_per_agent,
            diffusion_dt=fast_diffusion_dt,
            max_steps=max_steps,
        )
    polymer_count = int(round(condition.concentration / polymer_units_per_agent))
    env = make_environment(Strain(condition.strain), condition.seed, polymer_count)
    engine = GillespieEngine(
        env,
        Kinetics(k_pb_affinity=condition.k_pb, k_ps_affinity=condition.k_ps),
        seed=condition.seed + 10_000_019,
    )
    response = BiologicalResponse(
        env,
        strain=Strain(condition.strain),
        seed=condition.seed + 20_000_033,
    )
    times = sample_times(max_time, sample_interval)
    results: list[dict[str, float | int | str]] = []
    sample_index = 0

    def record(at_time: float) -> None:
        results.append({
            "strain": condition.strain,
            "polymer": condition.polymer,
            "k_pb": condition.k_pb,
            "k_ps": condition.k_ps,
            "concentration": condition.concentration,
            "seed": condition.seed,
            "time": at_time,
            "luminescence": response.population_luminescence(),
            "bacteria_count": len(env.bacteria),
            "signal_count": len(env.signals),
        })

    record(times[0])
    sample_index = 1
    steps = 0
    while engine.time < max_time:
        if max_steps is not None and steps >= max_steps:
            break
        # Sensed signal is held fixed across the waiting interval that is
        # sampled by Gillespie; this couples the continuous ODE to its clock.
        sensed = response.signal_sensing(engine)
        event_result = engine.step()
        if event_result is None:
            break
        dt, _event = event_result
        response.advance_interval(dt, bound_signal_counts=sensed, engine=engine)
        steps += 1
        while sample_index < len(times) and times[sample_index] <= engine.time:
            record(times[sample_index])
            sample_index += 1

    # Do not invent observations after a step cap or an absorbing state. The
    # output may end before max_time, and ensemble aggregation uses observed
    # replicate points only.
    return results


def simulate_one_fast_diffusion(
    condition: Condition,
    *,
    max_time: float,
    sample_interval: float,
    polymer_units_per_agent: float,
    diffusion_dt: float,
    max_steps: int | None = None,
) -> list[dict[str, float | int | str]]:
    """Run a split-step approximation that batches free-signal diffusion.

    Signal movement is leaped at fixed intervals; binding, unbinding,
    polymer/bacteria movement remain Gillespie events between leaps.
    Consequently, this is approximate and must not be presented as an exact
    direct-Gillespie reproduction.
    """
    if diffusion_dt <= 0:
        raise ValueError("fast_diffusion_dt must be positive")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for fast diffusion leaps") from exc

    polymer_count = int(round(condition.concentration / polymer_units_per_agent))
    env = make_environment(Strain(condition.strain), condition.seed, polymer_count)
    engine = GillespieEngine(
        env,
        Kinetics(k_pb_affinity=condition.k_pb, k_ps_affinity=condition.k_ps),
        seed=condition.seed + 10_000_019,
    )
    response = BiologicalResponse(
        env,
        strain=Strain(condition.strain),
        seed=condition.seed + 20_000_033,
    )
    leap_rng = np.random.default_rng(condition.seed + 30_000_037)
    times = sample_times(max_time, sample_interval)
    results: list[dict[str, float | int | str]] = []
    sample_index = 0

    def record(at_time: float) -> None:
        results.append({
            "strain": condition.strain,
            "polymer": condition.polymer,
            "k_pb": condition.k_pb,
            "k_ps": condition.k_ps,
            "concentration": condition.concentration,
            "seed": condition.seed,
            "time": at_time,
            "luminescence": response.population_luminescence(),
            "bacteria_count": len(env.bacteria),
            "signal_count": len(env.signals),
        })

    record(times[0])
    sample_index = 1
    steps = 0
    while engine.time < max_time:
        if max_steps is not None and steps >= max_steps:
            break
        next_boundary = times[sample_index] if sample_index < len(times) else max_time
        leap_end = min(engine.time + diffusion_dt, next_boundary, max_time)
        leap_size = leap_end - engine.time
        engine.diffuse_signals_leap(leap_size, leap_rng)

        while engine.time < leap_end:
            if max_steps is not None and steps >= max_steps:
                break
            sensed = response.signal_sensing(engine)
            remaining = leap_end - engine.time
            event_result = engine.step(
                include_signal_diffusion=False,
                max_wait=remaining,
            )
            if event_result is None:
                response.advance_interval(remaining, bound_signal_counts=sensed, engine=engine)
                engine.time = leap_end
                break
            dt, event = event_result
            response.advance_interval(dt, bound_signal_counts=sensed, engine=engine)
            if event is None:
                break
            steps += 1

        if max_steps is not None and steps >= max_steps:
            break
        while sample_index < len(times) and times[sample_index] <= engine.time:
            record(times[sample_index])
            sample_index += 1
    return results


def aggregate(raw: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[object, ...], list[dict[str, float | int | str]]] = {}
    keys = ("strain", "polymer", "k_pb", "k_ps", "concentration", "time")
    for row in raw:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)

    # Normalize each replicate to its own same-strain, same-time P=0 control.
    controls: dict[tuple[str, int, int], float] = {}
    for row in raw:
        if row["concentration"] == 0:
            controls[(str(row["strain"]), int(row["seed"]), int(row["time"]))] = float(row["luminescence"])

    output: list[dict[str, float | int | str]] = []
    epsilon = 1e-12
    for key, rows in groups.items():
        values = [float(r["luminescence"]) for r in rows]
        log_values = [math.log10(max(epsilon, v)) for v in values]
        folds = [
            value / max(epsilon, controls.get((str(r["strain"]), int(r["seed"]), int(r["time"])), epsilon))
            for value, r in zip(values, rows)
        ]
        normalized = [v / max(epsilon, controls.get((str(r["strain"]), int(r["seed"]), int(r["time"])), epsilon)) for v, r in zip(values, rows)]
        row_out: dict[str, float | int | str] = dict(zip(("strain", "polymer", "k_pb", "k_ps", "concentration", "time"), key))
        row_out.update({
            "n_runs": len(rows),
            "lum_mean": statistics.fmean(values),
            "lum_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "log10_lum_mean": statistics.fmean(log_values),
            "log10_lum_sd": statistics.stdev(log_values) if len(log_values) > 1 else 0.0,
            "normalized_mean": statistics.fmean(normalized),
            "normalized_sd": statistics.stdev(normalized) if len(normalized) > 1 else 0.0,
            "fold_change_mean": statistics.fmean(folds),
            "fold_change_sd": statistics.stdev(folds) if len(folds) > 1 else 0.0,
        })
        output.append(row_out)
    return output


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_ensembles(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    """Write normalized, log-luminescence, and fold-change plots by strain/type."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to generate plots. Install it with "
            "`python -m pip install matplotlib` and rerun."
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    group_fields = sorted({
        (str(r["strain"]), str(r["polymer"]), float(r["k_pb"]), float(r["k_ps"]))
        for r in rows
    })
    plot_specs = [
        ("normalized_mean", "normalized_sd", "Normalized luminescence", "normalized_luminescence"),
        ("log10_lum_mean", "log10_lum_sd", "log10 luminescence", "log_luminescence"),
        ("fold_change_mean", "fold_change_sd", "Fold change vs. no-polymer control", "fold_change"),
    ]
    for strain, polymer, k_pb, k_ps in group_fields:
        subset = [
            r for r in rows
            if r["strain"] == strain and r["polymer"] == polymer
            and float(r["k_pb"]) == k_pb and float(r["k_ps"]) == k_ps
        ]
        concentrations = sorted({int(r["concentration"]) for r in subset})
        for mean_field, sd_field, ylabel, filename in plot_specs:
            fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
            for concentration in concentrations:
                series = sorted(
                    (r for r in subset if int(r["concentration"]) == concentration),
                    key=lambda r: float(r["time"]),
                )
                x = [float(r["time"]) for r in series]
                y = [float(r[mean_field]) for r in series]
                sd = [float(r[sd_field]) for r in series]
                ax.plot(x, y, label=f"[P]={concentration}")
                ax.fill_between(x, [a - b for a, b in zip(y, sd)], [a + b for a, b in zip(y, sd)], alpha=0.16)
            ax.set_xlabel("Time (a.u.)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{strain} {polymer} (K_PB={k_pb:g}, K_PS={k_ps:g}): {ylabel}")
            ax.legend(title="Polymer concentration", fontsize="small", ncol=2)
            ax.grid(alpha=0.25)
            affinity_tag = f"_KPB{k_pb:g}_KPS{k_ps:g}" if polymer == "grid" else ""
            fig.savefig(output_dir / f"{strain}_{polymer}{affinity_tag}_{filename}.png", dpi=180)
            plt.close(fig)


def write_config(path: Path, args: argparse.Namespace, specs: list[tuple[str, float, float]]) -> None:
    config = {
        "seeds": args.seeds,
        "runs_per_condition": len(args.seeds),
        "strains": args.strains,
        "polymer_types": POLYMER_TYPES,
        "affinity_specs": specs,
        "concentrations": list(range(args.concentration_min, args.concentration_max + 1, args.concentration_step)),
        "max_time": args.max_time,
        "sample_interval": args.sample_interval,
        "polymer_units_per_agent": args.polymer_units_per_agent,
        "max_steps": args.max_steps,
        "fast_diffusion_dt": args.fast_diffusion_dt,
        "simulation_method": "split-step approximate signal diffusion" if args.fast_diffusion_dt else "exact Gillespie direct method",
        "normalization": "paired to the same seed, strain, time, and zero-polymer condition",
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_SCRIPT_DIR / "simulation_results")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--strains", choices=("MM32", "BB170", "both"), default="both")
    parser.add_argument("--sweep-affinities", action="store_true", help="also run the 5x5 K_PB/K_PS grid")
    parser.add_argument("--concentration-min", type=int, default=0)
    parser.add_argument("--concentration-max", type=int, default=7000)
    parser.add_argument("--concentration-step", type=int, default=1000)
    parser.add_argument("--polymer-units-per-agent", type=float, default=100.0)
    parser.add_argument("--max-time", type=float, default=30_000.0)
    parser.add_argument("--sample-interval", type=float, default=1_000.0)
    parser.add_argument("--max-steps", type=int, default=None, help="optional cap per run for exploratory runs")
    parser.add_argument(
        "--fast-diffusion-dt",
        type=float,
        default=None,
        help="batch free-signal diffusion every this many a.u.; approximate split-step mode",
    )
    args = parser.parse_args(argv)
    if not args.seeds:
        parser.error("At least one random seed is required")
    if args.concentration_step <= 0 or args.concentration_min < 0 or args.concentration_max < args.concentration_min:
        parser.error("invalid concentration range")
    if args.polymer_units_per_agent <= 0:
        parser.error("polymer-units-per-agent must be positive")
    if args.fast_diffusion_dt is not None and args.fast_diffusion_dt <= 0:
        parser.error("fast-diffusion-dt must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Matplotlib is required to run this script and save plots. "
            "Install it with `python -m pip install matplotlib`."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    strains = ["MM32", "BB170"] if args.strains == "both" else [args.strains]
    specs = condition_specs(args.sweep_affinities)
    concentrations = list(range(args.concentration_min, args.concentration_max + 1, args.concentration_step))
    write_config(args.output_dir / "configuration.json", args, specs)

    conditions = [
        Condition(strain, polymer_name, k_pb, k_ps, concentration, seed)
        for strain in strains
        for polymer_name, k_pb, k_ps in specs
        for concentration in concentrations
        for seed in args.seeds
    ]
    raw: list[dict[str, float | int | str]] = []
    total = len(conditions)
    for index, condition in enumerate(conditions, start=1):
        print(f"[{index}/{total}] {condition.condition_id}, seed={condition.seed}", flush=True)
        raw.extend(simulate_one(
            condition,
            max_time=args.max_time,
            sample_interval=args.sample_interval,
            polymer_units_per_agent=args.polymer_units_per_agent,
            max_steps=args.max_steps,
            fast_diffusion_dt=args.fast_diffusion_dt,
        ))

    summary = aggregate(raw)
    save_csv(args.output_dir / "replicate_results.csv", raw)
    save_csv(args.output_dir / "ensemble_summary.csv", summary)
    plot_ensembles(summary, args.output_dir / "plots")
    print(f"Saved simulation data and plots to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
