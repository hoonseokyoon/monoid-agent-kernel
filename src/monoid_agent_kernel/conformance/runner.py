"""External conformance runner with JSON and JUnit reports."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from monoid_agent_kernel.conformance.harness import ConformanceHarness, MinimalAgentHarness
from monoid_agent_kernel.conformance.profiles.minimal_agent import run_minimal_agent_profile
from monoid_agent_kernel.conformance.report import ConformanceReport

SUPPORTED_RUNNER_PROFILES = ("minimal-agent",)


def run_conformance(harness: ConformanceHarness, profile_id: str) -> ConformanceReport:
    """Run one supported profile and return a typed report."""

    if profile_id not in SUPPORTED_RUNNER_PROFILES:
        raise ValueError(f"profile {profile_id!r} is not executable by the external runner")
    if profile_id not in harness.supported_profiles:
        raise ValueError(f"harness {harness.harness_id!r} does not declare profile {profile_id!r}")
    started_at = time.time()
    started = time.perf_counter()
    if profile_id == "minimal-agent":
        if not isinstance(harness, MinimalAgentHarness):
            raise TypeError("minimal-agent requires MinimalAgentHarness")
        outcomes = run_minimal_agent_profile(harness)
    else:  # pragma: no cover - guarded by SUPPORTED_RUNNER_PROFILES
        raise AssertionError(profile_id)
    return ConformanceReport(
        harness_id=harness.harness_id,
        profile_id=profile_id,
        outcomes=outcomes,
        started_at=started_at,
        duration_s=time.perf_counter() - started,
    )


def load_harness(factory_ref: str) -> ConformanceHarness:
    """Load ``module:factory`` and construct an external harness."""

    module_name, separator, attribute = factory_ref.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("harness must use module:factory syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"harness factory is not callable: {factory_ref}")
    harness = factory()
    if not isinstance(harness, ConformanceHarness):
        raise TypeError(f"factory did not return ConformanceHarness: {factory_ref}")
    return harness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m monoid_agent_kernel.conformance.runner")
    parser.add_argument(
        "--harness", required=True, help="External harness factory as module:factory"
    )
    parser.add_argument("--profile", choices=SUPPORTED_RUNNER_PROFILES, default="minimal-agent")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--junit-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    harness: ConformanceHarness | None = None
    try:
        harness = load_harness(args.harness)
        report = run_conformance(harness, args.profile)
        if args.json_out is not None:
            report.write_json(args.json_out)
        if args.junit_out is not None:
            report.write_junit(args.junit_out)
        print(json.dumps(report.to_json(), sort_keys=True))
        return 0 if report.passed else 1
    except Exception as exc:
        print(f"conformance runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        close = getattr(harness, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                print(
                    f"conformance runner close error: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
