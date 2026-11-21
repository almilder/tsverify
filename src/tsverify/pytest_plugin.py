"""pytest plugin: ``--ts-adapter module:Class`` selects the code under test for ``tsverify.conformance``;
``--ts-report`` and ``--ts-report-json`` write the conformance report."""

import importlib

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("tsverify")
    group.addoption(
        "--ts-adapter",
        default=None,
        metavar="MODULE:CLASS",
        help="tsverify adapter of the code under test, e.g. tsverify.adapters.oracle:OracleAdapter",
    )
    group.addoption("--ts-report", default=None, metavar="PATH", help="write the tsverify report as Markdown")
    group.addoption("--ts-report-json", default=None, metavar="PATH", help="write the tsverify report as JSON")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: an expensive tsverify conformance test")
    config.addinivalue_line("markers", "heavy: a tsverify conformance test of many fits (minutes per test)")


def pytest_sessionfinish(session):
    config = session.config
    markdown_path, json_path = config.getoption("--ts-report"), config.getoption("--ts-report-json")
    if not (markdown_path or json_path):
        return
    from . import report

    collected = report.collected()
    if not collected:
        return
    adapter = collected[0][0]
    names = {a.name for a, _ in collected}
    meta = report.Meta(code=", ".join(sorted(names)), route=f"adapter {type(adapter).__module__}:{type(adapter).__name__}",
                       conventions=adapter.conventions)
    report.write([r for _, r in collected], meta, markdown_path, json_path)


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
