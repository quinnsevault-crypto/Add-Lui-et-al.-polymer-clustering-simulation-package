"""Run a resumable, reduced-condition 30,000-a.u. core ensemble.

The signal diffusion operator is split into batched random-walk leaps. This is
an approximate solver, selected to make the long ensemble more tractable.
Conditions are no-polymer and P1/P2/P3 at 7,000 a.u. for MM32 and BB170,
with 10 explicit seeds per condition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_simulations import Condition, DEFAULT_SEEDS, POLYMER_TYPES, aggregate, make_environment, plot_ensembles, sample_times, save_csv
from biological_response import BiologicalResponse
from grid_env import Strain
from gillespie_engine import Event, GillespieEngine, Kinetics


def simulate_tau_leap(condition: Condition, *, max_time: float, sample_interval: float,
                      polymer_units_per_agent: float, leap: float) -> list[dict[str, float | int | str]]:
    """Coarse operator-split stochastic simulation, avoiding per-molecule events."""
    import math
    import numpy as np

    env = make_environment(Strain(condition.strain), condition.seed,
                            int(round(condition.concentration / polymer_units_per_agent)))
    engine = GillespieEngine(env, Kinetics(k_pb_affinity=condition.k_pb, k_ps_affinity=condition.k_ps),
                             seed=condition.seed + 10_000_019)
    response = BiologicalResponse(env, strain=Strain(condition.strain), seed=condition.seed + 20_000_033)
    rng = np.random.default_rng(condition.seed + 30_000_037)
    k = engine.kinetics
    rows: list[dict[str, float | int | str]] = []
    sample_times_list = sample_times(max_time, sample_interval)
    next_sample = 0

    def rebuild_indexes() -> None:
        engine._signals_by_position.clear()
        engine._polymer_counts.clear()
        engine._bacteria_by_cell.clear()
        engine._cell_neighbors.clear()
        engine._build_spatial_indexes()

    def record(t: float) -> None:
        rows.append({"strain": condition.strain, "polymer": condition.polymer,
                     "k_pb": condition.k_pb, "k_ps": condition.k_ps,
                     "concentration": condition.concentration, "seed": condition.seed,
                     "time": t, "luminescence": response.population_luminescence(),
                     "bacteria_count": len(env.bacteria), "signal_count": len(env.signals)})

    def remove_batch(signals: list, remove_ids: set[int]) -> None:
        if remove_ids:
            env.signals[:] = [s for s in env.signals if id(s) not in remove_ids]
            rebuild_indexes()

    while next_sample < len(sample_times_list) and sample_times_list[next_sample] == 0:
        record(0.0)
        next_sample += 1
    time = 0.0
    while time < max_time:
        dt = min(leap, max_time - time)
        # Free signal diffusion is batched as independent random walks.
        engine.diffuse_signals_leap(dt, rng)

        # Reversible signal binding to bacteria: binomial tau-leap per receptor pool.
        bs_remove: set[int] = set()
        bs_release: list[tuple[int, tuple[int, int]]] = []
        for b in tuple(env.bacteria):
            bid = id(b)
            bound = engine._bs_bound.get(bid, 0)
            released = int(rng.binomial(bound, 1.0 - math.exp(-k.k_bs_off * dt))) if bound else 0
            bound -= released
            engine._bs_bound[bid] = bound
            if released:
                bs_release.extend((bid, b.position) for _ in range(released))
            nearby = engine._signals_near(b)
            free = max(0, b.signal_binding_sites - bound)
            if free and nearby:
                p = 1.0 - math.exp(-k.k_bs_on * len(nearby) * dt)
                n = min(int(rng.binomial(free, p)), len(nearby))
                if n:
                    chosen = [nearby[int(i)] for i in rng.choice(len(nearby), size=n, replace=False)]
                    bs_remove.update(id(s) for s in chosen)
                    bound += n
            engine._bs_bound[bid] = bound
        remove_batch(env.signals, bs_remove)
        for _, pos in bs_release:
            env.add_signal(pos)
        if bs_release:
            rebuild_indexes()

        # Polymer-signal interactions obey the crowding cutoff. These use the
        # post bacterial-binding free-signal field for this leap.
        ps_remove: set[int] = set()
        ps_release: list[tuple[int, tuple[int, int]]] = []
        if k.k_ps_affinity > 0:
            for p_obj in tuple(env.polymers):
                pid = id(p_obj)
                bound = engine._ps_bound.get(pid, 0)
                released = int(rng.binomial(bound, 1.0 - math.exp(-k.k_polymer_signal_off * dt))) if bound else 0
                bound -= released
                engine._ps_bound[pid] = bound
                if released:
                    ps_release.extend((pid, p_obj.position) for _ in range(released))
                free = engine.free_polymer_signal_sites(p_obj)
                nearby = list(engine._signals_by_position.get(p_obj.position, ()))
                if free and nearby:
                    p_bind = 1.0 - math.exp(-k.k_polymer_on * len(nearby) * dt)
                    n = min(int(rng.binomial(free, p_bind)), len(nearby))
                    if n:
                        chosen = [nearby[int(i)] for i in rng.choice(len(nearby), size=n, replace=False)]
                        ps_remove.update(id(s) for s in chosen)
                        bound += n
                engine._ps_bound[pid] = bound
        remove_batch(env.signals, ps_remove)
        for _, pos in ps_release:
            env.add_signal(pos)
        if ps_release:
            rebuild_indexes()

        # P-B bond association/dissociation is leaped by polymer-bacterium pair.
        if k.k_pb_affinity > 0:
            if engine._pb_bonds:
                p_off = 1.0 - math.exp(-k.k_polymer_off * dt)
                keep = rng.random(len(engine._pb_bonds)) >= p_off
                engine._pb_bonds[:] = [bond for bond, ok in zip(engine._pb_bonds, keep) if ok]
            for p_obj in tuple(env.polymers):
                nearby_bacteria = engine._bacteria_near(p_obj)
                for b in nearby_bacteria:
                    free_p = engine._polymer_free_pb(p_obj)
                    free_b = engine._bacterium_free_bp(b)
                    if free_p <= 0 or free_b <= 0:
                        continue
                    total_rate = k.k_polymer_on * free_p * free_b
                    n = min(int(rng.poisson(total_rate * dt)), free_p, free_b)
                    engine._pb_bonds.extend((p_obj, b) for _ in range(n))

        # Polymer random walks are batched; rebuild the indexes once per leap.
        moved_polymer = False
        for p_obj in env.polymers:
            degree = len(engine._cell_neighbors[engine._cell_index(p_obj.position)])
            jumps = int(rng.poisson(k.d_polymer * degree * dt))
            for _ in range(jumps):
                neighbors = engine._cell_neighbors[engine._cell_index(p_obj.position)]
                p_obj.position = neighbors[int(rng.integers(len(neighbors)))]
                moved_polymer = True
        if moved_polymer:
            rebuild_indexes()

        # Move each connected bacterial cluster at most once per leap.
        for cluster in engine._clusters():
            valid = []
            for delta in engine._STEPS:
                placement = engine._bacteria_placements(cluster, delta)
                if placement is not None:
                    valid.append((delta, placement))
            if not valid:
                continue
            probability = 1.0 - math.exp(-k.d_bacteria * len(valid) * dt / len(cluster))
            if rng.random() < probability:
                _, placement = valid[int(rng.integers(len(valid)))]
                engine._apply(Event(1.0, "move_bacteria", (cluster, placement)))

        sensed = response.signal_sensing(engine)
        response.advance_interval(dt, bound_signal_counts=sensed, engine=engine)
        time += dt
        engine.time = time
        while next_sample < len(sample_times_list) and sample_times_list[next_sample] <= time + 1e-10:
            record(sample_times_list[next_sample])
            next_sample += 1
    return rows


def read_checkpoint(path: Path) -> list[dict[str, float | int | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("k_pb", "k_ps", "concentration", "seed", "time", "luminescence", "bacteria_count", "signal_count"):
            row[key] = float(row[key])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_HERE / "long_core_results")
    parser.add_argument("--max-time", type=float, default=30_000.0)
    parser.add_argument("--sample-interval", type=float, default=1_000.0)
    parser.add_argument("--leap", type=float, default=100.0)
    parser.add_argument("--polymer-units-per-agent", type=float, default=100.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(DEFAULT_SEEDS)
    conditions = []
    for strain in ("MM32", "BB170"):
        conditions.extend(Condition(strain, "CONTROL", 0.0, 0.0, 0, seed) for seed in seeds)
        for polymer, (k_pb, k_ps) in POLYMER_TYPES.items():
            conditions.extend(Condition(strain, polymer, k_pb, k_ps, 7_000, seed) for seed in seeds)

    config = {
        "status": "running_or_resumable",
        "approximate": True,
        "method": "operator-split binomial/Poisson tau-leaping for signal binding, polymer binding and diffusion; coarse cluster movement",
        "tau_leap_au": args.leap,
        "max_time_au": args.max_time,
        "sample_interval_au": args.sample_interval,
        "seeds": seeds,
        "strains": ["MM32", "BB170"],
        "conditions": "no-polymer control and P1/P2/P3 at 7,000 a.u. for each strain",
        "unique_replicates": len(conditions),
        "polymer_units_per_agent": args.polymer_units_per_agent,
        "polymer_concentration_mapping_is_assumption": True,
        "interpretation": "Exploratory tau-leap ensemble; not exact direct-Gillespie or full concentration-series reproduction.",
    }
    (args.output_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    combined: list[dict[str, float | int | str]] = []
    for i, condition in enumerate(conditions, 1):
        tag = f"{condition.strain}_{condition.polymer}_KPB{condition.k_pb:g}_KPS{condition.k_ps:g}_P{condition.concentration}_seed{condition.seed}"
        checkpoint = checkpoint_dir / f"{tag}.csv"
        if checkpoint.exists():
            result = read_checkpoint(checkpoint)
            print(f"[{i}/{len(conditions)}] reuse {tag}", flush=True)
        else:
            print(f"[{i}/{len(conditions)}] run {tag}", flush=True)
            result = simulate_tau_leap(
                condition,
                max_time=args.max_time,
                sample_interval=args.sample_interval,
                polymer_units_per_agent=args.polymer_units_per_agent,
                leap=args.leap,
            )
            save_csv(checkpoint, result)
        combined.extend(result)
        # Update partial aggregate after every replicate so results remain useful
        # if the long run is interrupted.
        save_csv(args.output_dir / "replicate_results_partial.csv", combined)
    save_csv(args.output_dir / "replicate_results.csv", combined)
    save_csv(args.output_dir / "ensemble_summary.csv", aggregate(combined))
    plot_ensembles(aggregate(combined), args.output_dir / "plots")
    config["status"] = "complete"
    (args.output_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Long core ensemble complete: {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
