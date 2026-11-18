"""The case catalog is well formed."""

from tsverify.cases import SPECTRUM_CASES


def test_case_ids_are_unique():
    ids = [case.id for case in SPECTRUM_CASES]
    assert len(ids) == len(set(ids))


def test_cases_name_a_known_channel():
    assert {case.detector.channel for case in SPECTRUM_CASES} <= {"EPW", "IAW"}
