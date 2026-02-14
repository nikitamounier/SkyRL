#!/usr/bin/env python3
"""
Test: FastTextWorldSimulator vs Inform7 engine.

Runs both engines on the same games with the same walkthrough commands and
compares scores, rewards, and done/won flags at every step.
Also benchmarks speed (reset + full walkthrough) to confirm fast_sim is faster.
"""
import json
import time
from pathlib import Path
from typing import List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Paths to game directories (adjust if your layout differs)
# ---------------------------------------------------------------------------
TRIVIAL_GAMES_DIR = Path("/home/parsaidp/data/textworld_memo/games_trivial")
SMALL_GAMES_DIR = Path("/home/parsaidp/data/textworld_memo/games")
LONG_GAMES_DIR = Path("/home/parsaidp/data/textworld_memo_long/games")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_game_pairs(directory: Path, pattern: str = "*.json", limit: int = 5) -> List[Tuple[str, str]]:
    """Return list of (json_path, z8_path) pairs found in *directory*."""
    pairs = []
    for jpath in sorted(directory.glob(pattern))[:limit]:
        z8 = jpath.with_suffix(".z8")
        if z8.exists():
            pairs.append((str(jpath), str(z8)))
    return pairs


def _get_walkthrough(json_path: str) -> List[str]:
    """Extract the longest walkthrough command list from a game JSON."""
    with open(json_path) as f:
        spec = json.load(f)
    # Use the last non-empty quest's commands (it contains the full walkthrough)
    commands: List[str] = []
    for quest in spec.get("quests", []):
        cmds = quest.get("commands", [])
        if cmds:
            commands = cmds
    return commands


def _run_fast_sim(json_path: str, commands: List[str]):
    """Run fast simulator, return list of (obs, reward, done, score, won) per step."""
    from skyrl_gym.envs.textworld.fast_sim import FastTextWorldSimulator
    sim = FastTextWorldSimulator(json_path)
    obs, info = sim.reset()
    results = []
    for cmd in commands:
        obs, reward, done, info = sim.step(cmd)
        results.append({
            "reward": reward,
            "done": done,
            "score": info["score"],
            "won": info["won"],
        })
        if done:
            break
    return results


def _run_inform7(z8_path: str, commands: List[str]):
    """Run Inform7 engine, return list of (obs, reward, done, score, won) per step."""
    import textworld
    env = textworld.start(z8_path)
    gs = env.reset()
    results = []
    cumulative_reward = 0.0
    for cmd in commands:
        gs, reward, done = env.step(cmd)
        cumulative_reward += reward
        results.append({
            "reward": reward,
            "done": done,
            "score": getattr(gs, "score", 0) or 0,
            "won": getattr(gs, "won", False),
        })
        if done:
            break
    env.close()
    return results


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def _assert_same_trajectory(game_name, fast_results, inform7_results):
    """Compare step-by-step results from both engines."""
    # Both should have the same number of steps
    assert len(fast_results) == len(inform7_results), (
        f"[{game_name}] Step count mismatch: fast={len(fast_results)}, inform7={len(inform7_results)}"
    )

    for i, (fr, ir) in enumerate(zip(fast_results, inform7_results)):
        step = i + 1
        # Score should match exactly
        assert fr["score"] == ir["score"], (
            f"[{game_name}] Step {step}: score mismatch: fast={fr['score']}, inform7={ir['score']}"
        )
        # Reward should match
        assert abs(fr["reward"] - ir["reward"]) < 1e-6, (
            f"[{game_name}] Step {step}: reward mismatch: fast={fr['reward']}, inform7={ir['reward']}"
        )
        # Done and won flags
        assert fr["done"] == ir["done"], (
            f"[{game_name}] Step {step}: done mismatch: fast={fr['done']}, inform7={ir['done']}"
        )
        assert fr["won"] == ir["won"], (
            f"[{game_name}] Step {step}: won mismatch: fast={fr['won']}, inform7={ir['won']}"
        )


class TestCorrectnessTrivia:
    """Compare fast sim vs Inform7 on trivial (1-room, simple quest) games."""

    @pytest.fixture(params=_find_game_pairs(TRIVIAL_GAMES_DIR) if TRIVIAL_GAMES_DIR.exists() else [])
    def game(self, request):
        return request.param

    def test_walkthrough_matches(self, game):
        json_path, z8_path = game
        commands = _get_walkthrough(json_path)
        assert commands, f"No walkthrough found for {json_path}"

        fast_results = _run_fast_sim(json_path, commands)
        inform7_results = _run_inform7(z8_path, commands)

        name = Path(json_path).stem
        _assert_same_trajectory(name, fast_results, inform7_results)

        # Should have won at the end
        assert fast_results[-1]["won"], f"[{name}] fast_sim did not win"
        assert inform7_results[-1]["won"], f"[{name}] inform7 did not win"


class TestCorrectnessSmall:
    """Compare on small multi-room games."""

    @pytest.fixture(params=_find_game_pairs(SMALL_GAMES_DIR) if SMALL_GAMES_DIR.exists() else [])
    def game(self, request):
        return request.param

    def test_walkthrough_matches(self, game):
        json_path, z8_path = game
        commands = _get_walkthrough(json_path)
        assert commands, f"No walkthrough found for {json_path}"

        fast_results = _run_fast_sim(json_path, commands)
        inform7_results = _run_inform7(z8_path, commands)

        name = Path(json_path).stem
        _assert_same_trajectory(name, fast_results, inform7_results)

        assert fast_results[-1]["won"], f"[{name}] fast_sim did not win"


class TestCorrectnessLong:
    """Compare on longer, more complex games."""

    @pytest.fixture(params=_find_game_pairs(LONG_GAMES_DIR, limit=5) if LONG_GAMES_DIR.exists() else [])
    def game(self, request):
        return request.param

    def test_walkthrough_matches(self, game):
        json_path, z8_path = game
        commands = _get_walkthrough(json_path)
        assert commands, f"No walkthrough found for {json_path}"

        fast_results = _run_fast_sim(json_path, commands)
        inform7_results = _run_inform7(z8_path, commands)

        name = Path(json_path).stem
        _assert_same_trajectory(name, fast_results, inform7_results)

        assert fast_results[-1]["won"], f"[{name}] fast_sim did not win"


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

class TestSpeedBenchmark:
    """Benchmark fast sim vs Inform7 to confirm speed advantage."""

    def _benchmark_engine(self, run_fn, game_path, commands, n_runs=5):
        """Time n_runs of reset + full walkthrough."""
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            run_fn(game_path, commands)
            times.append(time.perf_counter() - t0)
        return min(times), sum(times) / len(times)

    @pytest.mark.skipif(
        not TRIVIAL_GAMES_DIR.exists(),
        reason="Trivial games directory not found",
    )
    def test_fast_sim_is_faster_trivial(self):
        pairs = _find_game_pairs(TRIVIAL_GAMES_DIR, limit=3)
        assert pairs, "No trivial game pairs found"

        for json_path, z8_path in pairs:
            commands = _get_walkthrough(json_path)
            if not commands:
                continue

            fast_min, fast_avg = self._benchmark_engine(
                _run_fast_sim, json_path, commands, n_runs=10
            )
            inform7_min, inform7_avg = self._benchmark_engine(
                _run_inform7, z8_path, commands, n_runs=5
            )

            name = Path(json_path).stem
            speedup = inform7_avg / fast_avg if fast_avg > 0 else float("inf")
            print(
                f"\n[{name}] {len(commands)} steps | "
                f"fast_sim: {fast_avg*1000:.1f}ms avg | "
                f"inform7: {inform7_avg*1000:.1f}ms avg | "
                f"speedup: {speedup:.1f}x"
            )

            # Fast sim should be at least 2x faster
            assert speedup > 2.0, (
                f"[{name}] Expected fast_sim to be >2x faster, got {speedup:.1f}x"
            )

    @pytest.mark.skipif(
        not SMALL_GAMES_DIR.exists(),
        reason="Small games directory not found",
    )
    def test_fast_sim_is_faster_small(self):
        pairs = _find_game_pairs(SMALL_GAMES_DIR, limit=3)
        assert pairs, "No small game pairs found"

        for json_path, z8_path in pairs:
            commands = _get_walkthrough(json_path)
            if not commands:
                continue

            fast_min, fast_avg = self._benchmark_engine(
                _run_fast_sim, json_path, commands, n_runs=10
            )
            inform7_min, inform7_avg = self._benchmark_engine(
                _run_inform7, z8_path, commands, n_runs=5
            )

            name = Path(json_path).stem
            speedup = inform7_avg / fast_avg if fast_avg > 0 else float("inf")
            print(
                f"\n[{name}] {len(commands)} steps | "
                f"fast_sim: {fast_avg*1000:.1f}ms avg | "
                f"inform7: {inform7_avg*1000:.1f}ms avg | "
                f"speedup: {speedup:.1f}x"
            )

            assert speedup > 2.0, (
                f"[{name}] Expected fast_sim to be >2x faster, got {speedup:.1f}x"
            )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Quick run without pytest for debugging."""
    print("=" * 70)
    print("Fast TextWorld Simulator vs Inform7 — Correctness & Speed Test")
    print("=" * 70)

    all_dirs = [
        ("TRIVIAL", TRIVIAL_GAMES_DIR),
        ("SMALL", SMALL_GAMES_DIR),
        ("LONG", LONG_GAMES_DIR),
    ]

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for label, gdir in all_dirs:
        if not gdir.exists():
            print(f"\n[{label}] Directory not found: {gdir} — SKIPPED")
            total_skip += 1
            continue

        pairs = _find_game_pairs(gdir, limit=5)
        if not pairs:
            print(f"\n[{label}] No game pairs found — SKIPPED")
            total_skip += 1
            continue

        print(f"\n{'='*70}")
        print(f"[{label}] Testing {len(pairs)} games from {gdir}")
        print(f"{'='*70}")

        for json_path, z8_path in pairs:
            name = Path(json_path).stem
            commands = _get_walkthrough(json_path)
            if not commands:
                print(f"  [{name}] No walkthrough — SKIPPED")
                total_skip += 1
                continue

            try:
                # Correctness
                fast_results = _run_fast_sim(json_path, commands)
                inform7_results = _run_inform7(z8_path, commands)
                _assert_same_trajectory(name, fast_results, inform7_results)

                assert fast_results[-1]["won"], "fast_sim didn't win"
                assert inform7_results[-1]["won"], "inform7 didn't win"

                # Speed
                t0 = time.perf_counter()
                for _ in range(10):
                    _run_fast_sim(json_path, commands)
                fast_time = (time.perf_counter() - t0) / 10

                t0 = time.perf_counter()
                for _ in range(3):
                    _run_inform7(z8_path, commands)
                inform7_time = (time.perf_counter() - t0) / 3

                speedup = inform7_time / fast_time if fast_time > 0 else float("inf")

                print(
                    f"  [{name}] PASS | {len(commands)} steps | "
                    f"score: {fast_results[-1]['score']} | "
                    f"fast: {fast_time*1000:.1f}ms | "
                    f"inform7: {inform7_time*1000:.1f}ms | "
                    f"speedup: {speedup:.1f}x"
                )
                total_pass += 1

            except Exception as e:
                print(f"  [{name}] FAIL | {e}")
                total_fail += 1

    print(f"\n{'='*70}")
    print(f"Results: {total_pass} passed, {total_fail} failed, {total_skip} skipped")
    print(f"{'='*70}")
