"""
tests/eval/
===========

Evaluation harness skeleton for Sentinel.

This package provides a declarative, mockable evaluation framework that lets us
detect semantic regressions on prompt edits, routing logic changes, and
sandbox guard rails — without invoking a live LLM on every test run.

Scenarios are described in YAML (``scenarios.yaml``); a runner loads them and
executes them against a (real or mocked) ``AgentOrchestrator``; metric
functions in :mod:`tests.eval.metrics` provide pluggable scoring.

The harness is opt-in: only collected via ``pytest -m eval``. Full
DeepEval / Ragas integration arrives in a later phase — this module is
the skeleton + smoke tests.
"""
