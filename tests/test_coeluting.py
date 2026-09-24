"""Tests for the co-eluting multi-charge confirmation endpoint and solver."""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.solver import (
    VERDICT_AMBIGUOUS,
    VERDICT_UNIQUE,
    VERDICT_UNRESOLVED,
    CoelutingDeconvolver,
    Deconvolver,
    Peak,
    SearchSpaceExceededError,
)

client = TestClient(app)

COELUTING_URL = "/api/v1/deconvolve/coeluting"
DECONVOLVE_URL = "/api/v1/deconvolve"

# A single precursor of neutral mass 500 observed at z=1 and z=2.
UNIQUE_PEAKS = [
    {"mz": "250.000000", "intensity": 400},
    {"mz": "250.5016775", "intensity": 300},
    {"mz": "500.000000", "intensity": 1000},
    {"mz": "501.003355", "intensity": 800},
]

# The z=1 strong envelope (mass 500) is inconsistent with the only z=2
# envelope (mass 600); the feasible combination must trade intensity for
# mass consistency and pick the weak z=1 envelope at mass 600.
TRADEOFF_PEAKS = [
    {"mz": "300.000000", "intensity": 500},
    {"mz": "300.5016775", "intensity": 400},
    {"mz": "500.000000", "intensity": 1000},
    {"mz": "501.003355", "intensity": 800},
    {"mz": "600.000000", "intensity": 100},
    {"mz": "601.003355", "intensity": 90},
]

# Two equal-quality z=1 envelopes (mass 600 and 602) both fit the z=2
# envelope (mass 600) within mass_tolerance 1.0.
AMBIGUOUS_PEAKS = [
    {"mz": "300.000000", "intensity": 10},
    {"mz": "300.5016775", "intensity": 10},
    {"mz": "600.000000", "intensity": 100},
    {"mz": "601.003355", "intensity": 100},
    {"mz": "602.000000", "intensity": 100},
    {"mz": "603.003355", "intensity": 100},
]


def make_peaks(spec: list[tuple[str, int]]) -> list[Peak]:
    return [Peak(index=i, mz=Decimal(mz), intensity=it) for i, (mz, it) in enumerate(spec)]


def solve(spec, charges, tol, required, mass_tol, **kwargs):
    return CoelutingDeconvolver(
        make_peaks(spec), charges, Decimal(tol), required, Decimal(mass_tol), **kwargs
    ).solve()


def post(payload: dict, url: str = COELUTING_URL):
    return client.post(url, json=payload)


# ---------------------------------------------------------------------- #
# Solver-level tests
# ---------------------------------------------------------------------- #


def test_unique_two_charge_envelope():
    result = solve(
        [("250.000000", 400), ("250.5016775", 300), ("500.000000", 1000), ("501.003355", 800)],
        [1, 2],
        "0.0001",
        [1, 2],
        "0.5",
    )
    assert result.verdict == VERDICT_UNIQUE
    assert (result.explained_intensity, result.explained_peak_count, result.cluster_count) == (2500, 4, 2)
    assert result.secondary is None
    # Canonically sorted: cluster starting at the lowest peak index first.
    assert [(c.charge, c.peak_indices) for c in result.primary] == [ (2, (0, 1)), (1, (2, 3))]
    assert result.primary_mass_interval == (Decimal("499.500000"), Decimal("500.500000"))


def test_required_charge_order_is_irrelevant():
    spec = [("250.000000", 400), ("250.5016775", 300), ("500.000000", 1000), ("501.003355", 800)]
    forward = solve(spec, [1, 2], "0.0001", [1, 2], "0.5")
    reversed_ = solve(spec, [2, 1], "0.0001", [2, 1], "0.5")
    assert forward == reversed_


def test_common_mass_boundary_is_inclusive_and_exact():
    # Neutral-mass estimates 500.4 (z=2) and 500.0 (z=1): spread exactly 0.4.
    spec = [("250.200000", 10), ("250.7016775", 20), ("500.000000", 30), ("501.003355", 40)]
    on_boundary = solve(spec, [1, 2], "0.0001", [1, 2], "0.2")
    assert on_boundary.verdict == VERDICT_UNIQUE
    # The common intersection collapses to a single point.
    assert on_boundary.primary_mass_interval == (Decimal("500.200000"), Decimal("500.200000"))
    below = solve(spec, [1, 2], "0.0001", [1, 2], "0.1999")
    assert below.verdict == VERDICT_UNRESOLVED
    assert below.primary == ()
    assert below.primary_mass_interval is None


def test_global_tradeoff_beats_per_charge_optimum():
    spec = [
        ("300.000000", 500), ("300.5016775", 400),
        ("500.000000", 1000), ("501.003355", 800),
        ("600.000000", 100), ("601.003355", 90),
    ]
    # The unconstrained deconvolution explains everything (2890 over 3
    # clusters) but mixes incompatible neutral masses.
    plain = Deconvolver(make_peaks(spec), [1, 2], Decimal("0.0001")).solve()
    assert (plain.explained_intensity, plain.explained_peak_count, plain.cluster_count) == (2890, 6, 3)
    # The co-eluting search must not inherit that optimum: the strong z=1
    # envelope (mass 500) is inconsistent with the z=2 envelope (mass 600),
    # so the feasible optimum trades intensity for a common neutral mass.
    result = solve(spec, [1, 2], "0.0001", [1, 2], "0.3")
    assert result.verdict == VERDICT_UNIQUE
    assert (result.explained_intensity, result.explained_peak_count, result.cluster_count) == (1090, 4, 2)
    assert [(c.charge, c.peak_indices) for c in result.primary] == [(2, (0, 1)), (1, (4, 5))]
    assert result.primary_mass_interval == (Decimal("599.700000"), Decimal("600.300000"))


def test_ambiguous_tie_yields_distinct_second_witness():
    result = solve(
        [
            ("300.000000", 10), ("300.5016775", 10),
            ("600.000000", 100), ("601.003355", 100),
            ("602.000000", 100), ("603.003355", 100),
        ],
        [1, 2],
        "0.0001",
        [1, 2],
        "1.0",
    )
    assert result.verdict == VERDICT_AMBIGUOUS
    assert result.secondary is not None
    primary_sets = {c.peak_indices for c in result.primary}
    secondary_sets = {c.peak_indices for c in result.secondary}
    assert primary_sets != secondary_sets
    assert primary_sets | secondary_sets == {(0, 1), (2, 3), (4, 5)}
    # Exactly one cluster per required charge in every witness.
    assert len(result.primary) == len(result.secondary) == 2
    assert result.primary_mass_interval == (Decimal("599.000000"), Decimal("601.000000"))
    assert result.secondary_mass_interval == (Decimal("601.000000"), Decimal("601.000000"))
    for witness in (result.primary, result.secondary):
        assert sum(c.explained_intensity for c in witness) == result.explained_intensity


def test_more_than_two_optima_still_report_two_distinct_witnesses():
    result = solve(
        [
            ("300.000000", 10), ("300.5016775", 10),
            ("600.000000", 100), ("601.003355", 100),
            ("602.000000", 100), ("603.003355", 100),
            ("604.000000", 100), ("605.003355", 100),
        ],
        [1, 2],
        "0.0001",
        [1, 2],
        "2.0",
    )
    assert result.verdict == VERDICT_AMBIGUOUS
    assert {c.peak_indices for c in result.primary} == {(0, 1), (2, 3)}
    assert {c.peak_indices for c in result.secondary} == {(0, 1), (4, 5)}


def test_unresolved_when_masses_cannot_intersect():
    result = solve(
        [("250.000000", 10), ("250.5016775", 10), ("510.000000", 10), ("511.003355", 10)],
        [1, 2],
        "0.0001",
        [1, 2],
        "0.1",
    )
    assert result.verdict == VERDICT_UNRESOLVED
    assert (result.explained_intensity, result.explained_peak_count, result.cluster_count) == (0, 0, 0)
    assert result.primary == () and result.secondary is None
    assert result.primary_mass_interval is None and result.secondary_mass_interval is None


def test_unresolved_when_required_charge_has_no_cluster():
    result = solve(
        [("500.000000", 10), ("501.003355", 10)],
        [1, 2],
        "0.0001",
        [1, 2],
        "1.0",
    )
    assert result.verdict == VERDICT_UNRESOLVED


def test_overlapping_envelopes_cannot_serve_two_charge_states():
    # One pair matches both the z=1 and (under a wide tolerance) z=2 spacing,
    # but the same two peaks cannot cover both required charges.
    result = solve(
        [("500.000000", 10), ("501.003355", 10)],
        [1, 2],
        "0.6",
        [1, 2],
        "1000",
    )
    assert result.verdict == VERDICT_UNRESOLVED


def test_intensity_is_optimised_before_peak_count():
    # Combo A (z=1 pair, intensity 220, 4 peaks) vs combo B (z=1 triplet,
    # intensity 23, 5 peaks): intensity dominates peak count.
    result = solve(
        [
            ("300.000000", 10), ("300.5016775", 10),
            ("600.000000", 100), ("601.003355", 100),
            ("602.000000", 1), ("603.003355", 1), ("604.006710", 1),
        ],
        [1, 2],
        "0.0001",
        [1, 2],
        "1.0",
    )
    assert result.verdict == VERDICT_UNIQUE
    assert (result.explained_intensity, result.explained_peak_count) == (220, 4)
    assert [(c.charge, c.peak_indices) for c in result.primary] == [(2, (0, 1)), (1, (2, 3))]


def test_three_charge_states():
    result = solve(
        [
            ("200.000000", 5), ("200.3344517", 5),
            ("300.000000", 7), ("300.5016775", 6),
            ("600.000000", 11), ("601.003355", 10),
        ],
        [1, 2, 3],
        "0.000001",
        [1, 2, 3],
        "0.5",
    )
    assert result.verdict == VERDICT_UNIQUE
    assert (result.explained_intensity, result.explained_peak_count, result.cluster_count) == (44, 6, 3)
    assert [c.charge for c in result.primary] == [3, 2, 1]
    assert result.primary_mass_interval == (Decimal("599.500000"), Decimal("600.500000"))


def test_four_charge_states():
    result = solve(
        [
            ("300.000000", 4), ("300.25083875", 4),
            ("400.000000", 5), ("400.3344517", 5),
            ("600.000000", 6), ("600.5016775", 6),
            ("1200.000000", 7), ("1201.003355", 7),
        ],
        [1, 2, 3, 4],
        "0.000001",
        [4, 3, 2, 1],
        "0.5",
    )
    assert result.verdict == VERDICT_UNIQUE
    assert (result.explained_intensity, result.explained_peak_count, result.cluster_count) == (44, 8, 4)
    assert [c.charge for c in result.primary] == [4, 3, 2, 1]


def test_coeluting_solver_is_deterministic():
    spec = [("300.0", 10), ("300.5016775", 10), ("600.0", 100), ("601.003355", 100), ("602.0", 100), ("603.003355", 100)]
    first = solve(spec, [1, 2], "0.0001", [1, 2], "1.0")
    second = solve(spec, [1, 2], "0.0001", [1, 2], "1.0")
    assert first == second


def test_coeluting_search_budget_guard():
    spec = [("250.000000", 1), ("250.5016775", 1), ("500.000000", 1), ("501.003355", 1)]
    with pytest.raises(SearchSpaceExceededError):
        solve(spec, [1, 2], "0.0001", [1, 2], "0.5", max_search_ops=1)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"required_charges": [1, 3]}, "allowed charges"),
        ({"required_charges": [1, 1]}, "unique"),
        ({"required_charges": []}, "at least one"),
        ({"required_charges": [1, 2], "mass_tolerance": "-0.1"}, "non-negative"),
    ],
)
def test_coeluting_invalid_arguments_raise(kwargs, message):
    spec = [("250.000000", 1), ("250.5016775", 1), ("500.000000", 1), ("501.003355", 1)]
    args = {
        "charges": [1, 2],
        "tolerance": "0.0001",
        "required_charges": [1, 2],
        "mass_tolerance": "0.5",
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        CoelutingDeconvolver(
            make_peaks(spec),
            args["charges"],
            Decimal(args["tolerance"]),
            args["required_charges"],
            Decimal(args["mass_tolerance"]),
        )


# ---------------------------------------------------------------------- #
# API-level tests
# ---------------------------------------------------------------------- #


def test_api_unique_end_to_end():
    resp = post(
        {
            "peaks": UNIQUE_PEAKS,
            "charges": [1, 2],
            "tolerance": "0.0001",
            "required_charges": [1, 2],
            "mass_tolerance": "0.5",
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "UNIQUE"
    assert body["objectives"] == {
        "explained_intensity": 2500,
        "explained_peak_count": 4,
        "cluster_count": 2,
    }
    assert [(c["charge"], c["peak_indices"]) for c in body["clusters"]] == [(2, [0, 1]), (1, [2, 3])]
    assert body["common_mass_interval"] == {"lower": "499.500000", "upper": "500.500000"}
    assert body["unexplained_peaks"] == []
    assert body["second_witness"] is None
    summary = body["input_summary"]
    assert summary["required_charges"] == [1, 2]
    assert summary["mass_tolerance"] == "0.5"
    assert summary["isotope_spacing"] == "1.003355"


def test_api_common_mass_boundary():
    base = {
        "peaks": [
            {"mz": "250.200000", "intensity": 10},
            {"mz": "250.7016775", "intensity": 20},
            {"mz": "500.000000", "intensity": 30},
            {"mz": "501.003355", "intensity": 40},
        ],
        "charges": [1, 2],
        "tolerance": "0.0001",
        "required_charges": [1, 2],
    }
    resp = post({**base, "mass_tolerance": "0.2"})
    body = resp.json()
    assert resp.status_code == 200 and body["verdict"] == "UNIQUE"
    assert body["common_mass_interval"] == {"lower": "500.200000", "upper": "500.200000"}
    resp = post({**base, "mass_tolerance": "0.1999"})
    assert resp.status_code == 200 and resp.json()["verdict"] == "UNRESOLVED"


def test_api_global_tradeoff_and_plain_endpoint_contrast():
    payload = {"peaks": TRADEOFF_PEAKS, "charges": [1, 2], "tolerance": "0.0001"}
    coeluting = post({**payload, "required_charges": [1, 2], "mass_tolerance": "0.3"})
    assert coeluting.status_code == 200
    body = coeluting.json()
    assert body["verdict"] == "UNIQUE"
    assert body["objectives"] == {
        "explained_intensity": 1090,
        "explained_peak_count": 4,
        "cluster_count": 2,
    }
    assert [(c["charge"], c["peak_indices"]) for c in body["clusters"]] == [(2, [0, 1]), (1, [4, 5])]
    # The plain endpoint on the same peaks still returns its own optimum:
    # the co-eluting constraints must not leak into the legacy semantics.
    plain = post(payload, url=DECONVOLVE_URL)
    assert plain.status_code == 200
    assert plain.json()["objectives"] == {
        "explained_intensity": 2890,
        "explained_peak_count": 6,
        "cluster_count": 3,
    }


def test_api_ambiguous_returns_distinct_second_witness():
    resp = post(
        {
            "peaks": AMBIGUOUS_PEAKS,
            "charges": [1, 2],
            "tolerance": "0.0001",
            "required_charges": [1, 2],
            "mass_tolerance": "1.0",
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "AMBIGUOUS"
    witness = body["second_witness"]
    assert witness is not None
    primary_sets = {tuple(c["peak_indices"]) for c in body["clusters"]}
    witness_sets = {tuple(c["peak_indices"]) for c in witness["clusters"]}
    assert primary_sets != witness_sets
    assert primary_sets | witness_sets == {(0, 1), (2, 3), (4, 5)}
    assert witness["common_mass_interval"] == {"lower": "601.000000", "upper": "601.000000"}
    w_intensity = sum(c["explained_intensity"] for c in witness["clusters"])
    assert w_intensity == body["objectives"]["explained_intensity"]


def test_api_unresolved():
    resp = post(
        {
            "peaks": [
                {"mz": "250.000000", "intensity": 10},
                {"mz": "250.5016775", "intensity": 10},
                {"mz": "510.000000", "intensity": 10},
                {"mz": "511.003355", "intensity": 10},
            ],
            "charges": [1, 2],
            "tolerance": "0.0001",
            "required_charges": [1, 2],
            "mass_tolerance": "0.1",
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "UNRESOLVED"
    assert body["clusters"] == []
    assert body["common_mass_interval"] is None
    assert body["second_witness"] is None
    assert body["objectives"] == {
        "explained_intensity": 0,
        "explained_peak_count": 0,
        "cluster_count": 0,
    }
    assert [p["index"] for p in body["unexplained_peaks"]] == [0, 1, 2, 3]


def test_api_response_is_deterministic():
    payload = {
        "peaks": AMBIGUOUS_PEAKS,
        "charges": [1, 2],
        "tolerance": "0.0001",
        "required_charges": [1, 2],
        "mass_tolerance": "1.0",
    }
    assert {post(payload).text for _ in range(3)} == {post(payload).text}


def test_api_search_budget_exceeded_returns_503(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "MAX_SEARCH_OPS", 1)
    resp = post(
        {
            "peaks": UNIQUE_PEAKS,
            "charges": [1, 2],
            "tolerance": "0.0001",
            "required_charges": [1, 2],
            "mass_tolerance": "0.5",
        }
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "SEARCH_SPACE_EXCEEDED"
    assert "verdict" not in body


def test_plain_deconvolve_rejects_coeluting_fields():
    # The legacy request schema must be unchanged: co-eluting-only fields are
    # still rejected as unknown.
    resp = client.post(
        DECONVOLVE_URL,
        json={
            "peaks": UNIQUE_PEAKS,
            "charges": [1, 2],
            "tolerance": "0.0001",
            "required_charges": [1, 2],
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert "verdict" not in body
    locs = [f["loc"] for f in body["error"]["fields"]]
    assert any(loc == "required_charges" for loc in locs)


GOOD_COELUTING = {
    "peaks": UNIQUE_PEAKS,
    "charges": [1, 2],
    "tolerance": "0.0001",
    "required_charges": [1, 2],
    "mass_tolerance": "0.5",
}


@pytest.mark.parametrize(
    "patch,expected_loc",
    [
        ({"required_charges": [1]}, "required_charges"),
        ({"required_charges": [1, 2, 3, 4, 5], "charges": [1, 2, 3, 4, 5]}, "required_charges"),
        ({"required_charges": [1, 1]}, "required_charges"),
        ({"required_charges": [1, 3]}, "required_charges"),
        ({"required_charges": [0, 1]}, "required_charges.0"),
        ({"required_charges": [1.5, 2]}, "required_charges.0"),
        ({"required_charges": [1, 2], "mass_tolerance": "-0.1"}, "mass_tolerance"),
        ({"required_charges": [1, 2], "mass_tolerance": "1.5", "charges": [1, 1]}, "charges"),
        ({"required_charges": [1, 2], "tolerance": "-0.1"}, "tolerance"),
    ],
)
def test_api_invalid_input_is_422_with_located_field(patch, expected_loc):
    payload = {**GOOD_COELUTING, **patch}
    resp = post(payload)
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert "verdict" not in body
    fields = body["error"]["fields"]
    assert fields, "expected at least one located field error"
    locs = [f["loc"] for f in fields]
    assert any(loc == expected_loc or loc.startswith(expected_loc + ".") for loc in locs), (
        f"expected loc {expected_loc!r} in {locs}"
    )


@pytest.mark.parametrize("missing", ["required_charges", "mass_tolerance", "peaks", "charges", "tolerance"])
def test_api_missing_required_field_is_422(missing):
    payload = {k: v for k, v in GOOD_COELUTING.items() if k != missing}
    resp = post(payload)
    assert resp.status_code == 422
    body = resp.json()
    assert "verdict" not in body
    locs = [f["loc"] for f in body["error"]["fields"]]
    assert any(loc == missing for loc in locs)


def test_api_unknown_field_is_422():
    resp = post({**GOOD_COELUTING, "debug": True})
    assert resp.status_code == 422
    locs = [f["loc"] for f in resp.json()["error"]["fields"]]
    assert any(loc == "debug" for loc in locs)
