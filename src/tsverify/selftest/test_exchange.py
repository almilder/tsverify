"""File exchange: inputs survive the files exactly, and the file route scores like the Python route."""

import dataclasses
import json

import numpy as np
import pytest

from tsverify import exchange
from tsverify.adapters.oracle import OracleAdapter
from tsverify.adapters.toy import ToyAdapter
from tsverify.cli import main
from tsverify.conventions import Conventions
from tsverify.execute import score as python_route
from tsverify.tasks import all_tasks

EVERYTHING_DIFFERENT = Conventions(
    flow_sign=-1, electron_drift="lab", relativistic="none", spectrum_per="frequency", dispersion_in_k=False,
    absolute=False,
)


@pytest.fixture(scope="module")
def different(tmp_path_factory):
    bundle = tmp_path_factory.mktemp("different")
    exchange.export(bundle, EVERYTHING_DIFFERENT)
    return bundle


@pytest.fixture(scope="module")
def canonical(tmp_path_factory):
    bundle = tmp_path_factory.mktemp("canonical")
    exchange.export(bundle, Conventions())
    return bundle


def _same(a, b):
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    return a == b


@pytest.mark.parametrize("fmt", exchange.FORMATS)
def test_inputs_read_back_exactly_in_the_code_s_conventions(different, fmt):
    for task in all_tasks(slow=False):
        if task.not_applicable(EVERYTHING_DIFFERENT) or any(run.needs for run in task.runs.values()):
            continue
        for name, run in task.runs.items():
            inputs = exchange.read_inputs(different / "runs" / exchange.run_ids(task)[name], run.kind, fmt)
            if run.kind == "fit":
                problem, read = run.problem(EVERYTHING_DIFFERENT, {}), inputs["problem"]
                assert read.start.plasma == EVERYTHING_DIFFERENT.express(problem.start.plasma), task.id
                assert read.detector == dataclasses.replace(problem.detector, peak_windows=()), task.id
                assert read.free == problem.free and read.start_amplitudes == problem.start_amplitudes, task.id
                for field in ("data", "variance", "mask"):
                    assert _same(getattr(read, field), getattr(problem, field)), (task.id, field)
            elif run.kind == "zprime":
                assert _same(inputs["zeta"], run.zeta)
            elif run.kind == "spectral_density":
                assert inputs["plasma"] == EVERYTHING_DIFFERENT.express(run.plasma), task.id
                assert _same(inputs["wavelengths_nm"], run.wavelengths_nm)
            else:
                expressed = dataclasses.replace(run.volume, plasma=EVERYTHING_DIFFERENT.express(run.volume.plasma))
                assert inputs["volume"] == expressed, task.id
                assert inputs["detector"] == dataclasses.replace(run.detector, peak_windows=()), task.id


def _round_trip(bundle, outputs, adapter, fmt):
    exchange.respond(bundle, adapter, outputs=outputs, fmt=fmt)
    results, _ = exchange.score(bundle, outputs=outputs, name=adapter.name)
    return {r.task: r for r in results}


@pytest.mark.parametrize(
    "adapter, fmt",
    [(OracleAdapter(EVERYTHING_DIFFERENT), "csv"), (ToyAdapter(), "json"), (ToyAdapter("thermal-speed"), "csv"),
     (ToyAdapter("zprime-table"), "csv")],
    ids=["oracle-other-conventions", "toy", "toy-thermal-speed", "toy-zprime-table"],
)
def test_the_file_route_scores_exactly_like_the_python_route(request, tmp_path, adapter, fmt):
    bundle = request.getfixturevalue("different" if adapter.conventions == EVERYTHING_DIFFERENT else "canonical")
    files = _round_trip(bundle, tmp_path, adapter, fmt)
    for task in all_tasks(slow=False):
        direct = python_route(adapter, task)
        via_files = files[task.id]
        if any(run.needs for run in task.runs.values()):
            assert via_files.status == "not supported" and exchange.PYTHON_ONLY in via_files.message, task.id
            continue
        assert via_files.status == direct.status, task.id
        assert via_files.measurements == direct.measurements, task.id
        if direct.status != "not supported":
            assert via_files.message == direct.message, task.id


def test_missing_and_malformed_outputs(tmp_path):
    exchange.export(tmp_path, Conventions(), formats=("csv",))
    exchange.respond(tmp_path, ToyAdapter())
    folder = tmp_path / "runs" / "L2_A1-epw-hydrogen@1_256"
    (folder / "output.csv").write_text("S\n1.0\n2.0\n", encoding="utf-8")
    (tmp_path / "runs" / "L1_zprime" / "output.csv").unlink()
    results = {r.task: r for r in exchange.score(tmp_path)[0]}
    assert results["L1/zprime"].status == "not supported"
    bad = results["L2/A1-epw-hydrogen@1/256"]
    assert bad.status == "failed" and "the output has 2 values, the run needs 256" in bad.message


def test_a_bundle_from_another_version_is_refused(canonical, tmp_path):
    manifest = json.loads((canonical / "manifest.json").read_text())
    manifest["tsverify"] = "0.0.1"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(exchange.BundleError, match="exported by tsverify 0.0.1"):
        exchange.score(tmp_path)


def test_command_line(canonical, tmp_path, capsys):
    bundle = tmp_path / "bundle"
    assert main(["export", str(bundle), "--format", "csv", "--relativistic", "none"]) == 0
    assert main(["export", str(bundle)]) == 2
    assert json.loads((bundle / "manifest.json").read_text())["conventions"]["relativistic"] == "none"
    good, bad = tmp_path / "good", tmp_path / "bad"
    assert main(["respond", str(canonical), "tsverify.adapters.toy:ToyAdapter", "--outputs", str(good)]) == 0
    report = tmp_path / "report.md"
    args = ["score", str(canonical), "--outputs", str(good), "--name", "toy", "--report", str(report)]
    assert main(args + ["--json", str(tmp_path / "r.json")]) == 0
    assert "# tsverify report: toy" in report.read_text(encoding="utf-8")
    exchange.respond(canonical, ToyAdapter("flow-sign"), outputs=bad)
    assert main(["score", str(canonical), "--outputs", str(bad)]) == 1
    assert "FAILED probe/flow" in capsys.readouterr().out
