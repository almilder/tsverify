"""pytest plugin: ``--ts-adapter module:Class`` selects the code under test for ``tsverify.conformance``."""

import importlib

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--ts-adapter",
        default=None,
        metavar="MODULE:CLASS",
        help="tsverify adapter of the code under test, e.g. tsverify.adapters.oracle:OracleAdapter",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: an expensive tsverify conformance test")


def load_adapter(spec):
    module, _, attr = spec.partition(":")
    if not attr:
        raise pytest.UsageError(f"--ts-adapter expects MODULE:CLASS, got {spec!r}")
    return getattr(importlib.import_module(module), attr)()


@pytest.fixture(scope="session")
def ts_adapter(request):
    spec = request.config.getoption("--ts-adapter")
    if spec is None:
        pytest.skip("no code under test: pass --ts-adapter MODULE:CLASS")
    return load_adapter(spec)
